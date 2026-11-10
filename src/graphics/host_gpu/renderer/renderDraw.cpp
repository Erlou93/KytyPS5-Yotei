#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/gpuTimer.h"
#include "graphics/host_gpu/renderer/debugSample.h"
#include "graphics/host_gpu/gpuCheckpoints.h"
#include "graphics/host_gpu/renderer/renderDraw.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/envDefaults.h"
#include "common/envRead.h"
#include "common/file.h"
#include "common/liveAb.h"
#include "graphics/guest_gpu/chunkCensus.h"
#include "graphics/guest_gpu/chunkLead.h"
#include "graphics/guest_gpu/command_processor/passCensus.h"
#include "common/logging/log.h"
#include "common/onceValue.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/workaroundHits.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/drawRepeat.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/dispatchCensus.h"
#include "graphics/host_gpu/renderer/cleanDraws.h"
#include "graphics/guest_gpu/drawStateObserver.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/perfCounters.h"
#include "graphics/host_gpu/renderer/recordPackets.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/renderDrawPlan.h"
#include "graphics/host_gpu/renderer/renderDrawTrace.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <fmt/format.h>
#include <set>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <immintrin.h>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {


std::pair<int32_t, uint32_t> ResolveDrawOffsets(uint32_t index_offset,
	                                           const ShaderVertexInputInfo& vs_input_info) {
	auto     vertex_offset   = static_cast<int32_t>(index_offset);
	uint32_t instance_offset = 0;
	if (!vs_input_info.fetch_embedded) {
		return {vertex_offset, instance_offset};
	}

	EXIT_IF(!vs_input_info.stage);
	const auto& program   = *vs_input_info.stage.program;
	const auto& resources = *vs_input_info.stage.resources;
	if (index_offset == 0 &&
	    program.info.vertex_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.vertex_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			vertex_offset = static_cast<int32_t>(resources.user_data[index]);
		}
	}
	if (program.info.instance_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.instance_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			instance_offset = resources.user_data[index];
		}
	}

	return {vertex_offset, instance_offset};
}

std::pair<bool, bool> IndirectDrawNativeOffsets(const ShaderVertexInputInfo& vs_input_info) {
	const auto* program = vs_input_info.stage.program;
	if (!vs_input_info.draw_base_once || program == nullptr || program->stage != ShaderType::Vertex) {
		const bool native = !vs_input_info.fetch_embedded;
		return {native, native};
	}
	return {program->info.vertex_offset_sgpr < 0, program->info.instance_offset_sgpr < 0};
}

// Draw base once: a direct draw whose register index offset wins over the base SGPR the
// embedded-fetch plan names (ResolveDrawOffsets) while that SGPR is not 0. The vertex entry still
// takes the SGPR off v5, so the shader's add gives the vertex inputs' index (index offset only),
// not index offset + SGPR. The first 16 are logged: whether such draws exist at all.
static void TraceDrawBaseIndexOffset(uint32_t index_offset, const ShaderVertexInputInfo& vs_input_info) {
	if (index_offset == 0 || !vs_input_info.draw_base_once || !vs_input_info.stage) {
		return;
	}
	const auto& program = *vs_input_info.stage.program;
	const auto  sgpr    = program.info.vertex_offset_sgpr;
	if (sgpr < static_cast<int32_t>(program.user_data_base)) {
		return;
	}
	const auto  index     = static_cast<uint32_t>(sgpr) - program.user_data_base;
	const auto& user_data = vs_input_info.stage.resources->user_data;
	if (index >= user_data.size() || user_data[index] == 0) {
		return;
	}
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 16u) {
		std::fprintf(stderr,
		             "FIX_DRAW_BASE_ONCE index offset %u wins over base SGPR s%d=%u (taken off v5) vs=%016llx\n",
		             index_offset, sgpr, user_data[index],
		             static_cast<unsigned long long>(program.shader_hash));
	}
}

static float ConvertPolygonOffsetConstantFactor(float guest_factor, const HW::PolyOffset& offset,
                                                vk::Format host_depth_format) {
	if (offset.db_is_float_fmt) {
		return guest_factor;
	}

	int host_depth_bits = 0;
	switch (host_depth_format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eD16UnormS8Uint: host_depth_bits = 16; break;
		case vk::Format::eD24UnormS8Uint: host_depth_bits = 24; break;
		default:
			// A fixed-point guest bias cannot be represented exactly by a floating-point host
			// attachment without VK_EXT_depth_bias_control.
			return guest_factor;
	}
	return std::ldexp(guest_factor, host_depth_bits + offset.neg_num_db_bits);
}

// KYTY_FIX_DRAW_STATE_FILTER (default on, =0 off, logged once): a draw records only the dynamic-state
// commands and the graphics pipeline bind that change what its command buffer already holds
// (CommandBuffer::GraphicsState has the rules), and the draw state object is reused instead of
// being zero-initialised for every draw (DrawStateHolder). With it unset every command is recorded
// as before.
// KYTY_FIX_DRAW_STATE_FILTER_VERIFY=1 (needs the filter): records every command exactly as without
// the filter and only counts, per kind, the commands the filter would have skipped; the counters
// are printed every 100000 draws. Compare renderings of FILTER=1 and FILTER=1 with VERIFY=1 to
// check the filter.
namespace {
using DynState = CommandBuffer::GraphicsState;

constexpr uint32_t kDynStateKinds = DynState::KindCount;

constexpr const char* kDynStateNames[] = {
    "viewports",       "scissors",      "line_width",    "blend_constants", "depth_test",
    "depth_write",     "depth_compare", "depth_bounds_test", "depth_bounds", "depth_bias_enable",
    "depth_bias",      "stencil_test",  "stencil_front", "stencil_back",    "color_write",
    "feedback_loop",   "pipeline_bind"};
static_assert(std::size(kDynStateNames) == kDynStateKinds);

struct DrawStateFilterOptions {
	bool enabled = false;
	bool verify  = false;
};

const DrawStateFilterOptions& DrawStateFilterConfig() {
	static constinit Common::OnceObject<DrawStateFilterOptions> options_once;
	const DrawStateFilterOptions& options = options_once.Get([] {
		DrawStateFilterOptions result;
		const char*            value = std::getenv("KYTY_FIX_DRAW_STATE_FILTER");
		if (value == nullptr || value[0] != '0') {  // default on; =0 turns it off
			result.enabled = true;
			std::fprintf(stderr, "FIX_DRAW_STATE_FILTER enabled\n");
			const char* verify = std::getenv("KYTY_FIX_DRAW_STATE_FILTER_VERIFY");
			if (verify != nullptr && verify[0] == '1') {
				result.verify = true;
				std::fprintf(stderr,
				             "FIX_DRAW_STATE_FILTER_VERIFY enabled: every command is recorded, the "
				             "ones the filter would skip are counted\n");
			}
		}
		return result;
	});
	return options;
}

struct DrawStateFilterStats {
	uint64_t                                draws = 0;
	std::array<uint64_t, kDynStateKinds>    calls {};
	std::array<uint64_t, kDynStateKinds>    skippable {};
};

DrawStateFilterStats& DrawStateStats() {
	static thread_local DrawStateFilterStats stats;
	return stats;
}

void ReportDrawStateStats(const DrawStateFilterStats& stats) {
	// kind=would_skip/calls
	std::string line = fmt::format("FIX_DRAW_STATE_FILTER_VERIFY draws={}", stats.draws);
	for (uint32_t kind = 0; kind < kDynStateKinds; kind++) {
		if (stats.calls[kind] != 0) {
			line += fmt::format(" {}={}/{}", kDynStateNames[kind], stats.skippable[kind],
			                    stats.calls[kind]);
		}
	}
	std::fprintf(stderr, "%s\n", line.c_str());
}

// Bit for bit, so that NaNs and signed zeros are never taken for equal.
template<typename T>
bool SameBits(const T& a, const T& b) {
	return std::memcmp(&a, &b, sizeof(T)) == 0;
}
template<typename T>
bool SameBits(const T* a, const T* b, size_t count) {
	return std::memcmp(a, b, count * sizeof(T)) == 0;
}
// The vulkan-hpp structs below are compared as raw bytes: they must have no padding.
static_assert(sizeof(vk::Viewport) == 6 * sizeof(float));
static_assert(sizeof(vk::Rect2D) == 4 * sizeof(uint32_t));
static_assert(sizeof(vk::StencilOpState) == 7 * sizeof(uint32_t));

// Decides, for each command of a draw, whether it is recorded. `known`: what the command buffer holds (its GraphicsState,
// or with the record packets the recording thread's copy of it).
class DynamicStateRecorder {
public:
	explicit DynamicStateRecorder(DynState& known): m_options(DrawStateFilterConfig()), m_known(known) {}

	[[nodiscard]] DynState& Known() const noexcept { return m_known; }

	// Verify mode: counts the draw and reports the counters now and then.
	void StartDraw() const {
		if (!m_options.verify) {
			return;
		}
		auto& stats = DrawStateStats();
		if (++stats.draws % 100000u == 0) {
			ReportDrawStateStats(stats);
		}
	}

	// True when the command of that kind has to be recorded: always without the filter, otherwise
	// unless the command buffer holds the value already (same() says so). Verify mode records
	// every command and only counts the ones that could have been skipped. A caller that records
	// the command stores the value in Known() and calls Known().Set(kind).
	template<typename Same>
	[[nodiscard]] bool Needs(DynState::Kind kind, Same&& same) const {
		if (!m_options.enabled) {
			return true;
		}
		const bool skippable = m_known.Has(kind) && same();
		if (m_options.verify) {
			auto& stats = DrawStateStats();
			stats.calls[kind]++;
			stats.skippable[kind] += skippable ? 1u : 0u;
			return true;
		}
		return !skippable;
	}

private:
	const DrawStateFilterOptions& m_options;
	DynState&                     m_known;
};
} // namespace

namespace {

// A draw's dynamic state as its commit computes it from the draw's registers, depth state and render targets, and its
// attachment feedback loop aspects: the values the commands of EmitGraphicsDynamicParams set. The viewports and scissors
// (viewport_count of each) are kept apart. the record packets: the dynamic-state packet carries a copy.
struct DynamicParams {
	uint32_t                                             viewport_count = 1;
	float                                                line_width     = 1.0f;
	std::array<float, 4>                                 blend_constants {};
	vk::Bool32                                           depth_test        = VK_FALSE;
	vk::Bool32                                           depth_write       = VK_FALSE;
	vk::CompareOp                                        depth_compare     = vk::CompareOp::eNever;
	vk::Bool32                                           depth_bounds_test = VK_FALSE;
	std::array<float, 2>                                 depth_bounds {};
	vk::Bool32                                           depth_bias_flag = VK_FALSE;
	std::array<float, 3>                                 depth_bias {};
	vk::Bool32                                           stencil_test = VK_FALSE;
	std::array<vk::StencilOpState, 2>                    stencil {};
	uint32_t                                             color_write_count = 0;
	std::array<vk::Bool32, RENDER_COLOR_ATTACHMENTS_MAX> color_write {};
	vk::ImageAspectFlags                                 feedback;
	bool                                                 depth_bounds_enable = false; // the bounds are set only then
	bool                                                 depth_bias_enable   = false; // the bias is set only then
	bool                                                 stencil_enable      = false; // the faces are set only then
	bool                                                 feedback_set        = false; // attachment_feedback_loop_enabled
};
static_assert(std::is_trivially_copyable_v<DynamicParams>);

using DrawViewports = std::array<vk::Viewport, DynState::ViewportSlots>;
using DrawScissors  = std::array<vk::Rect2D, DynState::ViewportSlots>;

// The values of a draw's dynamic state, no commands (one function used to compute and record them).
static void MakeGraphicsDynamicParams(const CommandBuffer& buffer, const ShaderVertexInputInfo& vs_input_info,
                                      const RenderDepthInfo& depth, const RenderState& rendering,
                                      const DrawDynamicRegs& regs, vk::ImageAspectFlags feedback_aspects,
                                      DynamicParams& params, DrawViewports& viewports, DrawScissors& scissors) {
	KYTY_PROFILER_FUNCTION();

	const auto&        vp  = regs.viewport;
	const vk::Extent2D framebuffer_extent {rendering.width, rendering.height};
	const auto& outputs = vs_input_info.stage.program->info.outputs;
	const bool  indexed_viewports =
	    std::any_of(outputs.begin(), outputs.end(), [](const auto& output) {
		    return output.kind == ShaderRecompiler::IR::StageOutputKind::ViewportIndex;
	    });
	constexpr uint32_t viewport_slots = std::size(HW::ScreenViewport {}.viewports);
	static_assert(DynState::ViewportSlots == viewport_slots);
	const uint32_t viewport_count = indexed_viewports ? viewport_slots : 1;
	for (uint32_t i = 0; i < viewport_count; i++) {
		const auto& guest    = vp.viewports[i];
		auto&       viewport = viewports[i];
		viewport             = vk::Viewport {};
		if (regs.clip.clip_disable) {
			const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
			viewport.width  = static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u));
			viewport.height = static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u));
		} else {
			viewport.x      = guest.xoffset - guest.xscale;
			viewport.y      = guest.yoffset - guest.yscale;
			viewport.width  = guest.xscale * 2.0f;
			viewport.height = guest.yscale * 2.0f;
		}
		viewport.minDepth =
		    guest.zoffset - (regs.clip.dx_clip_space ? 0.0f : guest.zscale);
		viewport.maxDepth = guest.zscale + guest.zoffset;

		const auto final_scissor =
		    calc_final_scissor(vp, regs.scan, framebuffer_extent, i);
		auto& scissor  = scissors[i];
		scissor        = vk::Rect2D {};
		scissor.offset = {final_scissor.left, final_scissor.top};
		scissor.extent = {static_cast<uint32_t>(final_scissor.right - final_scissor.left),
		                  static_cast<uint32_t>(final_scissor.bottom - final_scissor.top)};
		if (viewport.width == 0.0f) {
			// Keep empty slots at their guest index; Vulkan requires a positive viewport width.
			viewport.width = 1.0f;
			scissor.extent = {0, 0};
		}
	}
	params.viewport_count = viewport_count;

	float line_width = regs.line_width;
	if (line_width != 1.0f) {
		static bool logged = false;
		if (!logged) {
			LOGF("Render: temporary: clamping Vulkan line width %f to 1.0 because wideLines is "
			     "not enabled\n",
			     line_width);
			logged = true;
		}
		line_width = 1.0f;
	}
	params.line_width      = line_width;
	const auto& blend      = regs.blend;
	params.blend_constants = {blend.red, blend.green, blend.blue, blend.alpha};
	params.depth_test      = depth.depth_test_enable ? VK_TRUE : VK_FALSE;
	params.depth_write     = depth.depth_write_enable ? VK_TRUE : VK_FALSE;
	params.depth_compare   = depth.depth_compare_op;
#if !defined(__APPLE__)
	params.depth_bounds_test   = depth.depth_bounds_test_enable ? VK_TRUE : VK_FALSE;
	params.depth_bounds_enable = depth.depth_bounds_test_enable;
	if (depth.depth_bounds_test_enable) {
		params.depth_bounds = {regs.bounds_min, regs.bounds_max};
	}
#endif

	const auto& mode              = regs.mode;
	const auto& poly_offset       = regs.poly;
	const bool  use_front         = mode.poly_offset_front_enable && !mode.cull_front;
	const bool  use_back          = mode.poly_offset_back_enable && !mode.cull_back;
	const bool  depth_bias_enable = use_front || use_back;
	params.depth_bias_flag        = depth_bias_enable ? VK_TRUE : VK_FALSE;
	params.depth_bias_enable      = depth_bias_enable;
	if (depth_bias_enable) {
		// Vulkan has one bias for both faces. Prefer a visible front face when both are enabled.
		const float guest_constant_factor =
		    use_front ? poly_offset.front_offset : poly_offset.back_offset;
		const float constant_factor = ConvertPolygonOffsetConstantFactor(
		    guest_constant_factor, poly_offset, depth.desc.view_info.format);
		const float slope_factor =
		    (use_front ? poly_offset.front_scale : poly_offset.back_scale) / 16.0f;
		params.depth_bias = {constant_factor, poly_offset.clamp, slope_factor};
	}

	params.stencil_test   = depth.stencil_test_enable ? VK_TRUE : VK_FALSE;
	params.stencil_enable = depth.stencil_test_enable;
	if (depth.stencil_test_enable) {
		params.stencil = {depth.stencil_front, depth.stencil_back};
	}

#if !defined(__APPLE__)
	// MoltenVK has no VK_EXT_color_write_enable (see EmitGraphicsDynamicParams).
	params.color_write_count = rendering.num_color_attachments;
	for (uint32_t slot = 0; slot < rendering.num_color_attachments; slot++) {
		params.color_write[slot] = rendering.color_attachments[slot].image_view != nullptr;
	}
#endif
	params.feedback_set = buffer.GetGraphics().attachment_feedback_loop_enabled;
	params.feedback     = feedback_aspects;
}

// KYTY_FIX_IMAGE_MEMORY_OWNER: the rectangle a draw can write in its colour targets, before they are acquired: the union
// over its viewports of the scissor and the viewport bounds, as MakeGraphicsDynamicParams computes them, with the
// smallest colour-target extent standing in for the attachments' (never smaller than theirs, so never a smaller
// rectangle). False without colour targets or viewports.
static bool OwnerDrawRect(const ShaderVertexInputInfo& vs_input_info, const RenderColorInfo* colors, uint32_t color_count,
                          const DrawDynamicRegs& regs, int32_t& x0, int32_t& y0, int32_t& x1, int32_t& y1) {
	vk::Extent2D extent {UINT32_MAX, UINT32_MAX};
	for (uint32_t i = 0; i < color_count; i++) {
		if (!colors[i].image_id) continue;
		const auto e  = colors[i].Extent();
		extent.width  = std::min(extent.width, e.width);
		extent.height = std::min(extent.height, e.height);
	}
	if (extent.width == UINT32_MAX) return false;
	const auto& vp      = regs.viewport;
	const auto& outputs = vs_input_info.stage.program->info.outputs;
	const bool  indexed = std::any_of(outputs.begin(), outputs.end(), [](const auto& output) {
        return output.kind == ShaderRecompiler::IR::StageOutputKind::ViewportIndex;
	});
	const uint32_t count  = indexed ? static_cast<uint32_t>(DynState::ViewportSlots) : 1u;
	const auto     to_int = [](float v) { return static_cast<int32_t>(std::clamp(v, -1048576.0f, 1048576.0f)); };
	x0 = y0 = INT32_MAX;
	x1 = y1 = INT32_MIN;
	for (uint32_t i = 0; i < count; i++) {
		const auto& guest = vp.viewports[i];
		float       vx0 = -1048576.0f, vy0 = -1048576.0f, vx1 = 1048576.0f, vy1 = 1048576.0f;
		if (!regs.clip.clip_disable) {
			if (guest.xscale == 0.0f) continue; // an empty slot (MakeGraphicsDynamicParams gives it no scissor)
			vx0 = std::min(guest.xoffset - guest.xscale, guest.xoffset + guest.xscale);
			vx1 = std::max(guest.xoffset - guest.xscale, guest.xoffset + guest.xscale);
			vy0 = std::min(guest.yoffset - guest.yscale, guest.yoffset + guest.yscale);
			vy1 = std::max(guest.yoffset - guest.yscale, guest.yoffset + guest.yscale);
		}
		const auto    sc = calc_final_scissor(vp, regs.scan, extent, i);
		const int32_t lx = std::max<int32_t>(sc.left, to_int(std::floor(vx0)));
		const int32_t ly = std::max<int32_t>(sc.top, to_int(std::floor(vy0)));
		const int32_t hx = std::min<int32_t>(sc.right, to_int(std::ceil(vx1)));
		const int32_t hy = std::min<int32_t>(sc.bottom, to_int(std::ceil(vy1)));
		if (hx <= lx || hy <= ly) continue;
		x0 = std::min(x0, lx);
		y0 = std::min(y0, ly);
		x1 = std::max(x1, hx);
		y1 = std::max(y1, hy);
	}
	return x1 > x0 && y1 > y0;
}

// The commands of a draw's dynamic state, then its attachment feedback loop (KYTY_FIX_DRAW_STATE_FILTER: only those that
// change what `known` says the command buffer holds; a recorded command is remembered in `known`). The order and the
// values are those recorded before the split (dynamic state, then feedback loop). `sink`: the command buffer (today's
// calls, or the recording thread's with the record packets).
static void EmitGraphicsDynamicParams(DynState& known, vk::CommandBuffer sink, const DynamicParams& params,
                                      const vk::Viewport* viewports, const vk::Rect2D* scissors) {
	DynamicStateRecorder rec(known);
	rec.StartDraw();
	const uint32_t viewport_count = params.viewport_count;
	if (rec.Needs(DynState::Viewports, [&] {
		    return known.viewport_count == viewport_count &&
		           SameBits(viewports, known.viewports.data(), viewport_count);
	    })) {
		sink.setViewportWithCount(viewport_count, viewports);
		known.viewport_count = viewport_count;
		std::copy_n(viewports, viewport_count, known.viewports.begin());
		known.Set(DynState::Viewports);
	}
	if (rec.Needs(DynState::Scissors, [&] {
		    return known.scissor_count == viewport_count &&
		           SameBits(scissors, known.scissors.data(), viewport_count);
	    })) {
		sink.setScissorWithCount(viewport_count, scissors);
		known.scissor_count = viewport_count;
		std::copy_n(scissors, viewport_count, known.scissors.begin());
		known.Set(DynState::Scissors);
	}
	const float line_width = params.line_width;
	if (rec.Needs(DynState::LineWidth, [&] { return SameBits(known.line_width, line_width); })) {
		sink.setLineWidth(line_width);
		known.line_width = line_width;
		known.Set(DynState::LineWidth);
	}
	const auto& blend_constants = params.blend_constants;
	if (rec.Needs(DynState::BlendConstants,
	              [&] { return SameBits(known.blend_constants, blend_constants); })) {
		sink.setBlendConstants(blend_constants.data());
		known.blend_constants = blend_constants;
		known.Set(DynState::BlendConstants);
	}
	const vk::Bool32 depth_test  = params.depth_test;
	const vk::Bool32 depth_write = params.depth_write;
	if (rec.Needs(DynState::DepthTest, [&] { return known.depth_test == depth_test; })) {
		sink.setDepthTestEnable(depth_test);
		known.depth_test = depth_test;
		known.Set(DynState::DepthTest);
	}
	if (rec.Needs(DynState::DepthWrite, [&] { return known.depth_write == depth_write; })) {
		sink.setDepthWriteEnable(depth_write);
		known.depth_write = depth_write;
		known.Set(DynState::DepthWrite);
	}
	if (rec.Needs(DynState::DepthCompare,
	              [&] { return known.depth_compare == params.depth_compare; })) {
		sink.setDepthCompareOp(params.depth_compare);
		known.depth_compare = params.depth_compare;
		known.Set(DynState::DepthCompare);
	}
#if !defined(__APPLE__)
	const vk::Bool32 depth_bounds_test = params.depth_bounds_test;
	if (rec.Needs(DynState::DepthBoundsTest,
	              [&] { return known.depth_bounds_test == depth_bounds_test; })) {
		sink.setDepthBoundsTestEnable(depth_bounds_test);
		known.depth_bounds_test = depth_bounds_test;
		known.Set(DynState::DepthBoundsTest);
	}
	if (params.depth_bounds_enable) {
		const auto& bounds = params.depth_bounds;
		if (rec.Needs(DynState::DepthBounds, [&] { return SameBits(known.depth_bounds, bounds); })) {
			sink.setDepthBounds(bounds[0], bounds[1]);
			known.depth_bounds = bounds;
			known.Set(DynState::DepthBounds);
		}
	}
#endif

	const vk::Bool32 depth_bias_flag = params.depth_bias_flag;
	if (rec.Needs(DynState::DepthBiasEnable,
	              [&] { return known.depth_bias_enable == depth_bias_flag; })) {
		sink.setDepthBiasEnable(depth_bias_flag);
		known.depth_bias_enable = depth_bias_flag;
		known.Set(DynState::DepthBiasEnable);
	}
	if (params.depth_bias_enable) {
		const auto& bias = params.depth_bias;
		if (rec.Needs(DynState::DepthBias, [&] { return SameBits(known.depth_bias, bias); })) {
			sink.setDepthBias(bias[0], bias[1], bias[2]);
			known.depth_bias = bias;
			known.Set(DynState::DepthBias);
		}
	}

	const vk::Bool32 stencil_test = params.stencil_test;
	if (rec.Needs(DynState::StencilTest, [&] { return known.stencil_test == stencil_test; })) {
		sink.setStencilTestEnable(stencil_test);
		known.stencil_test = stencil_test;
		known.Set(DynState::StencilTest);
	}
	if (params.stencil_enable) {
		// A face is recorded in full (operations, masks, reference) when any of it changed.
		const auto set_stencil = [&](vk::StencilFaceFlagBits face, const vk::StencilOpState& state,
		                             DynState::Kind kind, vk::StencilOpState& recorded) {
			if (!rec.Needs(kind, [&] { return SameBits(recorded, state); })) {
				return;
			}
			sink.setStencilOp(face, state.failOp, state.passOp, state.depthFailOp, state.compareOp);
			sink.setStencilCompareMask(face, state.compareMask);
			sink.setStencilWriteMask(face, state.writeMask);
			sink.setStencilReference(face, state.reference);
			recorded = state;
			known.Set(kind);
		};
		set_stencil(vk::StencilFaceFlagBits::eFront, params.stencil[0], DynState::StencilFront,
		            known.stencil[0]);
		set_stencil(vk::StencilFaceFlagBits::eBack, params.stencil[1], DynState::StencilBack,
		            known.stencil[1]);
	}

#if defined(__APPLE__)
	// MoltenVK has no VK_EXT_color_write_enable; the pipeline is created without the
	// eColorWriteEnableEXT dynamic state and relies on the static colorWriteMask instead.
#else
	if (params.color_write_count != 0) {
		const uint32_t enable_count = params.color_write_count;
		const auto*    enable       = params.color_write.data();
		if (rec.Needs(DynState::ColorWrite, [&] {
			    return known.color_write_count == enable_count &&
			           SameBits(enable, known.color_write.data(), enable_count);
		    })) {
			sink.setColorWriteEnableEXT(enable_count, enable);
			known.color_write_count = enable_count;
			std::copy_n(enable, enable_count, known.color_write.begin());
			known.Set(DynState::ColorWrite);
		}
	} else {
		// A pipeline without color attachments does not declare color write enables dynamic:
		// binding it replaces the recorded ones, so the next draw with attachments sets them again.
		known.Forget(DynState::ColorWrite);
	}
#endif
	// The attachment feedback loop aspects a draw sets after its dynamic state.
	if (params.feedback_set) {
		const vk::ImageAspectFlags aspects = params.feedback;
		if (rec.Needs(DynState::FeedbackLoop, [&] { return known.feedback == aspects; })) {
			sink.setAttachmentFeedbackLoopEnableEXT(aspects);
			known.feedback = aspects;
			known.Set(DynState::FeedbackLoop);
		}
	}
}

// Binds the pipeline of a draw unless the command buffer holds it already
// (KYTY_FIX_DRAW_STATE_FILTER). Compute pipelines are another bind point and never get here.
static void EmitGraphicsPipelineBind(DynState& known, vk::CommandBuffer sink, vk::Pipeline pipeline) {
	DynamicStateRecorder rec(known);
	if (rec.Needs(DynState::PipelineBind, [&] { return known.pipeline == pipeline; })) {
		sink.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
		known.pipeline = pipeline;
		known.Set(DynState::PipelineBind);
	}
}

// ---- the record packets (recordPackets.h): a draw's dynamic state and pipeline bind as packets the recording
// thread makes with its own copy of what the command buffer holds. Based on chenxiao07/KytyPS5 v20261020b
// (src/graphics/host_gpu/renderer/renderDraw.cpp, LocalDrawRecording), GPL-2.0. Unlike there, the commands are decided by
// the same filter rules as today's (EmitGraphicsDynamicParams), so the recording thread makes exactly today's commands as
// long as its copy says what this thread's would: both copies are forgotten together (a new command buffer, a blit with
// its own pipeline: InvalidateGraphicsState sends the recording thread a "forget" packet at the same place of the stream)
// and a command buffer whose draws switch between the two copies forgets both first (GraphicsState::keeper).

// What the recording thread's commands left in the command buffer it records, for the draws whose packets it runs (the
// recording thread only: thread-local).
static DynState& WorkerGraphicsState(VkCommandBuffer buffer) {
	struct Copy {
		VkCommandBuffer buffer = nullptr;
		DynState        known;
	};
	static thread_local Copy copy;
	if (copy.buffer != buffer) {
		copy.buffer = buffer;
		copy.known  = DynState {};
	}
	return copy.known;
}

// The dynamic-state packet: this header, then the viewports, then the scissors (viewport_count of each).
struct DynPacketHeader {
	VkCommandBuffer buffer = nullptr;
	DynamicParams   params;
};
static_assert(std::is_trivially_copyable_v<DynPacketHeader>);
constexpr size_t DynPacketViewports = (sizeof(DynPacketHeader) + 7u) & ~size_t {7u};

[[nodiscard]] static size_t DynPacketBytes(uint32_t viewport_count) {
	return DynPacketViewports + viewport_count * (sizeof(vk::Viewport) + sizeof(vk::Rect2D));
}

static void PackDynPacket(uint8_t* out, VkCommandBuffer buffer, const DynamicParams& params, const vk::Viewport* viewports,
                          const vk::Rect2D* scissors) {
	DynPacketHeader header;
	header.buffer = buffer;
	header.params = params;
	std::memcpy(out, &header, sizeof(header));
	const size_t count = params.viewport_count;
	std::memcpy(out + DynPacketViewports, viewports, count * sizeof(vk::Viewport));
	std::memcpy(out + DynPacketViewports + count * sizeof(vk::Viewport), scissors, count * sizeof(vk::Rect2D));
}

static void UnpackDynPacket(const void* blob, DynPacketHeader& header, DrawViewports& viewports, DrawScissors& scissors) {
	const auto* bytes = static_cast<const uint8_t*>(blob);
	std::memcpy(&header, bytes, sizeof(header));
	const size_t count = header.params.viewport_count;
	EXIT_IF(count == 0 || count > viewports.size());
	std::memcpy(viewports.data(), bytes + DynPacketViewports, count * sizeof(vk::Viewport));
	std::memcpy(scissors.data(), bytes + DynPacketViewports + count * sizeof(vk::Viewport), count * sizeof(vk::Rect2D));
}

// The recording thread: the commands of the packet's draw.
static void RunDynPacket(const void* blob) {
	DynPacketHeader header;
	DrawViewports   viewports;
	DrawScissors    scissors;
	UnpackDynPacket(blob, header, viewports, scissors);
	vk::CommandBuffer buffer(header.buffer);
	EmitGraphicsDynamicParams(WorkerGraphicsState(header.buffer), buffer, header.params, viewports.data(), scissors.data());
}

struct PipelinePacket {
	VkCommandBuffer buffer   = nullptr;
	VkPipeline      pipeline = nullptr;
};
static_assert(std::is_trivially_copyable_v<PipelinePacket>);

static void RunPipelinePacket(const void* blob) {
	PipelinePacket packet;
	std::memcpy(&packet, blob, sizeof(packet));
	vk::CommandBuffer buffer(packet.buffer);
	EmitGraphicsPipelineBind(WorkerGraphicsState(packet.buffer), buffer, vk::Pipeline(packet.pipeline));
}

static void RunForgetPacket(const void* blob) {
	VkCommandBuffer buffer = nullptr;
	std::memcpy(&buffer, blob, sizeof(buffer));
	WorkerGraphicsState(buffer) = DynState {};
}

// A draw's dynamic state: a packet when the recording thread records the buffer, else today's commands.
static void CommitGraphicsDynamicParams(const CommandBuffer& buffer, vk::CommandBuffer vk_buffer, const DynamicParams& params,
                                        const DrawViewports& viewports, const DrawScissors& scissors) {
	auto& known = buffer.GetGraphicsState();
	if (RecordPackets::On() && CommandRecorder::RecordedByWorker(vk_buffer)) {
		// This thread's copy was in use in this command buffer: the recording thread's is stale (forget packet).
		if (known.keeper == 1u) buffer.InvalidateGraphicsState();
		auto* out = static_cast<uint8_t*>(CommandRecorder::ReserveCall(vk_buffer, DynPacketBytes(params.viewport_count)));
		if (out != nullptr) {
			PackDynPacket(out, static_cast<VkCommandBuffer>(vk_buffer), params, viewports.data(), scissors.data());
			CommandRecorder::CommitCall(&RunDynPacket);
			known.keeper = 2u;
			RecordPackets::Note(RecordPackets::Stat::DynPackets);
			return;
		}
	}
	// The recording thread's copy was in use: this one is stale.
	if (known.keeper == 2u) buffer.InvalidateGraphicsState();
	known.keeper = 1u;
	EmitGraphicsDynamicParams(known, vk_buffer, params, viewports.data(), scissors.data());
}

// The draw's pipeline bind, by the copy its dynamic state used (keeper).
static void BindGraphicsPipeline(const CommandBuffer& buffer, vk::CommandBuffer vk_buffer,
                                 vk::Pipeline pipeline) {
	auto& known = buffer.GetGraphicsState();
	if (known.keeper == 2u && CommandRecorder::RecordedByWorker(vk_buffer)) {
		if (auto* out = CommandRecorder::ReserveCall(vk_buffer, sizeof(PipelinePacket)); out != nullptr) {
			PipelinePacket packet;
			packet.buffer   = static_cast<VkCommandBuffer>(vk_buffer);
			packet.pipeline = static_cast<VkPipeline>(pipeline);
			std::memcpy(out, &packet, sizeof(packet));
			CommandRecorder::CommitCall(&RunPipelinePacket);
			RecordPackets::Note(RecordPackets::Stat::PipelinePackets);
			return;
		}
	}
	if (known.keeper == 2u) buffer.InvalidateGraphicsState();
	known.keeper = 1u;
	EmitGraphicsPipelineBind(known, vk_buffer, pipeline);
}

} // namespace

void NoteGraphicsStateInvalidated(vk::CommandBuffer buffer) noexcept {
	if (!buffer || !CommandRecorder::RecordedByWorker(buffer)) return;
	auto* out = CommandRecorder::ReserveCall(buffer, sizeof(VkCommandBuffer));
	if (out == nullptr) return;
	const auto handle = static_cast<VkCommandBuffer>(buffer);
	std::memcpy(out, &handle, sizeof(handle));
	CommandRecorder::CommitCall(&RunForgetPacket);
	RecordPackets::Note(RecordPackets::Stat::Invalidations);
}


// KYTY_DIAG_BARRIER_EVERY_CMD=1 (diagnostic, slow): a full memory barrier before every draw and
// dispatch, so no GPU command can overlap an earlier one (tests for a missing barrier).
static bool BarrierEveryCmdEnabled() {
	static constinit Common::OnceValue<bool> on_once;
	const bool on = on_once.Get([] {
		const bool enabled = Common::EnvRead::BarrierEveryCmd();
		if (enabled) std::fprintf(stderr, "DIAG_BARRIER_EVERY_CMD enabled\n");
		return enabled;
	});
	return on;
}
static void FullMemoryBarrier(vk::CommandBuffer vk_buffer) {
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	vk_buffer.pipelineBarrier2(dependency);
}

static bool DrawHasValidVertexShader(const HW::Shader& sh_ctx) {

	const auto& vs = sh_ctx.GetVs();
	return vs.es_regs.data_addr != 0;
}

static bool PixelShaderHasDepthOrCoverageSideEffects(const HW::ShaderRegisters& sh_regs) {
	const auto& db = sh_regs.db_shader_control;
	return db.shader_kill_enable || db.shader_z_export_enable || db.shader_mask_export_enable ||
	       db.shader_dual_export_enable || db.shader_execute_on_noop || sh_regs.AlphaToMask();
}

namespace {
// The draw state of a draw: `DrawRenderState state {};` zero-initialised and constructed about 12 KB
// for every draw (three vertex stages, eight color targets), almost all of which is written before
// anything reads it. With KYTY_FIX_DRAW_STATE_FILTER the thread's one DrawRenderState is reused
// and only what PrepareDrawRenderState reads without writing is reset:
//  - vertex_info: GetGraphicsPrograms resets every entry it uses (`info = {}` in
//    ShaderGetStaticVertexInputInfo, the NGG branch of PrepareProgram and
//    PrepareTessellationPrograms) before anything reads it, and a draw reads the first
//    programs.VertexStageCount() entries only;
//  - color_info[1..7]: ResolveRenderColorTarget starts with `r = {}`, and a draw reads the first
//    color_count entries only; entry 0 is reset, because the skip logs read it without targets;
//  - ps_input_info and programs: RefreshShaders assigns both `{}` first;
//  - depth_info is reset: ResolveRenderDepthTarget leaves it untouched without a depth attachment;
//  - drop_reason is reset: only a draw PrepareDrawRenderState leaves out sets it.
// Without the flag the state is a fresh `DrawRenderState {}` as before. The state is the draw slot's.
void ResetDrawState(DrawRenderState& state) {
	if (!DrawStateFilterConfig().enabled) {
		state = DrawRenderState {};
		return;
	}
	state.depth_info    = {};
	state.color_info[0] = {};
	state.color_count   = 0;
	state.ps_active     = true;
	state.indirect_draw = {};
	state.drop_reason   = nullptr;
}

class DrawStateHolder {
public:
	// `resolved`: the state the resolve thread made (reset there), used instead of the slot's.
	DrawStateHolder(DrawRenderState& slot_state, DrawRenderState* resolved): m_state(resolved != nullptr ? resolved : &slot_state) {
		if (resolved == nullptr) ResetDrawState(slot_state);
	}
	KYTY_CLASS_NO_COPY(DrawStateHolder);

	[[nodiscard]] DrawRenderState& Get() const noexcept { return *m_state; }

private:
	DrawRenderState* m_state = nullptr;
};
} // namespace

RenderState RenderExecutor::AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
                                                 uint32_t color_count, RenderDepthInfo& depth,
                                                 vk::ImageAspectFlags& feedback_aspects,
                                                 std::span<PreparedBindings* const> stages) {
	KYTY_PROFILER_BLOCK("Draw::AcquireRenderTargets");
	EXIT_IF(colors == nullptr || color_count > RENDER_COLOR_ATTACHMENTS_MAX);
	feedback_aspects = {};
	auto&       cache = m_context.GetTextureCache();
	RenderState state {};
	state.width                 = std::numeric_limits<uint32_t>::max();
	state.height                = std::numeric_limits<uint32_t>::max();
	state.num_layers            = std::numeric_limits<uint32_t>::max();
	state.num_color_attachments = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		auto& target = colors[i];
		EXIT_IF(!target.image_id);
		const auto owner = cache.m_slot_images.try_get(target.image_id);
		if (owner == nullptr || (!owner->registered && !owner->info.data.Empty()) ||
		    owner->binding.needs_rebind) {
			EXIT("color target changed after render-state discovery\n");
		}
		const auto image_view = cache.FindRenderTarget(target.image_id, target.desc);
		auto&      image      = cache.GetImage(target.image_id);
		EXIT_IF(image.backing.samples != target.desc.info.samples || image_view == nullptr);
		const auto& view   = target.desc.view_info;
		const auto  layout = image.binding.is_bound ? vk::ImageLayout::eGeneral
		                                            : vk::ImageLayout::eColorAttachmentOptimal;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access =
		    vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
		image.Transit(layout, image.binding.attachment_access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Handle());
		const auto extent       = target.Extent();
		state.width             = std::min(state.width, extent.width);
		state.height            = std::min(state.height, extent.height);
		state.num_layers        = std::min(state.num_layers, view.layer_count);
		state.num_color_attachments = std::max(state.num_color_attachments, target.target_slot + 1);
		auto& attachment            = state.color_attachments[target.target_slot];
		attachment.image_view   = image_view;
		attachment.image_layout = layout;
	}
	if (depth.image_id) {
		const auto owner = cache.m_slot_images.try_get(depth.image_id);
		if (owner == nullptr || !owner->registered || owner->binding.needs_rebind) {
			EXIT("depth target changed after render-state discovery\n");
		}
		const auto  image_view = cache.FindDepthTarget(depth.image_id, depth.desc);
		const auto& metadata   = depth.desc.info.metadata;
		if (metadata.kind == ImageMetadataKind::Htile && depth.depth_clear_enable &&
		    !cache.ClearMeta(metadata.range.address)) {
			EXIT("failed to acquire HTile metadata for a depth clear\n");
		}
		// KYTY_WORKAROUND_HTILE_FULL_RANGE=1 (logged APPROXIMATION): the host never writes HTILE contents,
		// but shaders read its per-tile min/max depth (e.g. to enqueue shadow-resolve tiles). Once per
		// frame and HTILE address, the guest HTILE is filled with "expanded, full depth range"
		// (ZMask=0xF, MinZ=0, MaxZ=0x3FFF), so such tests are conservative instead of empty.
		static constinit Common::OnceValue<bool> htile_full_range_once;
		const bool htile_full_range = htile_full_range_once.Get([] {
			// Default from Common::EnvDefaults; workaround: D8 (stencil /
			// HTILE), to be replaced by the proper implementation; =0 turns it off.
			const char* value = Common::EnvOrDefault("KYTY_WORKAROUND_HTILE_FULL_RANGE");
			const bool  on    = value != nullptr && value[0] == '1';
			if (on) std::fprintf(stderr, "WORKAROUND_HTILE_FULL_RANGE APPROXIMATION enabled\n");
			return on;
		});
		if (htile_full_range && metadata.kind == ImageMetadataKind::Htile && metadata.range.size >= 4u &&
		    (metadata.range.address & 3u) == 0) {
			static std::unordered_map<uint64_t, uint32_t> filled_frame;
			const auto frame = DebugPresentCount().load();
			auto [it, inserted] = filled_frame.try_emplace(metadata.range.address, frame);
			if (inserted || it->second != frame || depth.depth_clear_enable) {
				it->second = frame;
				const auto size = metadata.range.size & ~uint64_t {3};
				auto [htile, htile_offset] =
				    m_context.GetBufferCache().ObtainBuffer(metadata.range.address, size, true);
				if (htile != nullptr) {
					htile->Fill(htile_offset, size, 0xFFFC000Fu);
				}
				static std::atomic_uint logged {0};
				if (logged.fetch_add(1) < 4u) {
					std::fprintf(stderr, "WORKAROUND_HTILE_FULL_RANGE fill addr=0x%llx size=0x%llx\n",
					             static_cast<unsigned long long>(metadata.range.address),
					             static_cast<unsigned long long>(size));
				}
			}
		}
		const bool meta_clear =
		    metadata.kind == ImageMetadataKind::Htile &&
		    cache.IsMetaCleared(metadata.range.address, depth.desc.view_info.base_layer);
		// The load clears by aspect (upstream 57791645e; ours: depth_load_clear_enable = depth clear or a pending HTILE
		// clear). Upstream 5bf37f52a also clears compressed stencil on a pending HTILE clear, together with dropping the
		// ClearMeta above; not taken: our depth clear marks the HTILE cleared (ClearMeta), so that rule would clear the
		// stencil of every depth-only clear.
		depth.load_clear_aspects = {};
		if (depth.depth_clear_enable || meta_clear)
			depth.load_clear_aspects |= vk::ImageAspectFlagBits::eDepth;
		if (depth.stencil_clear_enable)
			depth.load_clear_aspects |= vk::ImageAspectFlagBits::eStencil;
		if (meta_clear &&
		    !cache.TouchMeta(metadata.range.address, depth.desc.view_info.base_layer, false)) {
			EXIT("failed to consume HTile clear state\n");
		}
		auto& image = cache.GetImage(depth.image_id);
		EXIT_IF(image_view == nullptr || image.backing.samples != depth.desc.info.samples);
		const auto draw_writes = depth.AttachmentWriteAspects();
		vk::ImageAspectFlags sampled_aspects;
		for (const auto* stage: stages) {
			for (const auto& binding: stage->images) {
				if (binding.image_id != depth.image_id ||
				    binding.desc.type != TextureCache::BindingType::Texture) continue;
				const auto native =
				    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
				EXIT_IF(native == image.views.end());
				sampled_aspects |= native->info.aspect;
				feedback_aspects |= DepthFeedbackAspects(draw_writes, depth.desc.view_info,
				                                         native->info);
			}
		}
		if (feedback_aspects && !m_context.GetGraphics().attachment_feedback_loop_enabled) {
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				std::printf("depth attachment feedback loop is not supported by the host\n");
			}
		}
		auto layout = depth_attachment_layout(depth);
		if (sampled_aspects & ~DepthReadableAspects(layout)) {
			layout = m_context.GetGraphics().attachment_feedback_loop_enabled
			             ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
			             : vk::ImageLayout::eGeneral;
		}
		// The attachment store writes even when guest depth/stencil tests do not.
		const auto access = vk::AccessFlagBits2::eDepthStencilAttachmentRead |
		                    vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access = access;
		const auto& view                = depth.desc.view_info;
		image.Transit(layout, access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Handle());
		state.width               = std::min(state.width, depth.desc.info.extent.width);
		state.height              = std::min(state.height, depth.desc.info.extent.height);
		state.num_layers          = std::min(state.num_layers, view.layer_count);
		const auto aspects        = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		auto&      attachment     = state.depth_stencil_attachment;
		attachment.image_view     = image_view;
		attachment.image_layout   = layout;
		attachment.clear_value[0] = std::bit_cast<uint32_t>(depth.depth_clear_value);
		attachment.clear_value[1] = depth.stencil_clear_value;
		attachment.has_depth      = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.depth_clear =
		    static_cast<bool>(depth.load_clear_aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.has_stencil    = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eStencil);
		attachment.stencil_clear =
		    static_cast<bool>(depth.load_clear_aspects & vk::ImageAspectFlagBits::eStencil);
	}
	if (color_count == 0 && !depth.image_id) {
		const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
		state.width        = limits.maxFramebufferWidth;
		state.height       = limits.maxFramebufferHeight;
	}
	if (state.num_layers == std::numeric_limits<uint32_t>::max()) {
		state.num_layers = 1;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.width == std::numeric_limits<uint32_t>::max() ||
	        state.height == std::numeric_limits<uint32_t>::max());
	return state;
}

static uint32_t DrawColorOutputMask(const HW::Context& ctx) {
	const auto& sh_regs     = ctx.GetShaderRegisters();
	const auto  write_mask  = ctx.GetRenderTargetMask() & sh_regs.m_cbShaderMask;
	// (Upstream f7226ca7d adds RT7 for HasObjectPrimitiveIdOutput; ours writes the ObjPrimID through
	// the ObjPrimID write, PipelineCache objprim_slot.)
	uint32_t    output_mask = 0;
	for (uint32_t index = 0; index < RENDER_COLOR_ATTACHMENTS_MAX; index++) {
		const auto slot = ShaderPixelExportTarget(sh_regs.m_cbShaderMask, index);
		if (slot >= RENDER_COLOR_ATTACHMENTS_MAX) {
			break;
		}
		if (sh_regs.target_output_mode[index] != 0 &&
		    render_target_mask_slot(write_mask, slot) != 0) {
			output_mask |= 1u << slot;
		}
	}
	return output_mask;
}

enum class CbColorMode : uint8_t {
	Disable            = 0,
	Normal             = 1,
	EliminateFastClear = 2,
	Resolve            = 3,
	FmaskDecompress    = 5,
	DccDecompress      = 6,
};

bool RenderExecutor::ConsumeMetadataColorOperation(CommandBuffer& buffer,
                                                   uint32_t render_target_slice_offset) {
	const auto& ctx  = buffer.GetRegisters();
	const auto  mode = ctx.GetColorControl().mode;
	if (mode == static_cast<uint8_t>(CbColorMode::EliminateFastClear)) {
		// FCE can precede any ordinary draw. Discover the target to materialize its native
		// clear metadata before the expanded image is sampled.
		RenderColorInfo target {};
		ResolveRenderColorTarget(buffer, target, render_target_slice_offset, 0, true);
		return true;
	}
	// Metadata/decompression shaders must not export colors into the expanded image.
	return mode == static_cast<uint8_t>(CbColorMode::FmaskDecompress) ||
	       mode == static_cast<uint8_t>(CbColorMode::DccDecompress);
}

static uint64_t VertexBufferDescriptorSize(int binding, const ShaderVertexInputInfo& info) {
	const auto& buffer = info.buffers[binding];
	if (buffer.stride != 0 || buffer.num_records == 0) {
		return static_cast<uint64_t>(buffer.stride) * buffer.num_records;
	}

	uint64_t size = 0;
	for (int i = 0; i < info.resources_num; i++) {
		if (info.resources_dst[i].buffer_index != binding) {
			continue;
		}
		const auto& resource = info.resources[i];
		// RDNA2 OOB_SELECT=2 only checks NumRecords != 0. A constant attribute still
		// fetches its entire format; NumRecords is not a byte count in this mode.
		const uint64_t extent = resource.OutOfBounds() == 2
		                            ? resource.Base48() - buffer.addr +
		                                  ShaderRecompiler::Format::GetFormatInfo(resource.Format()).byte_size
		                            : buffer.num_records;
		size = std::max(size, extent);
	}
	return size;
}

// Trivial on purpose: AcquireVertexBuffers keeps RES_MAX of them on the stack and writes all fields of the ones it
// uses; the rest stay unwritten.
struct VertexBufferRange {
	uint64_t base_address;
	uint64_t requested_end;
	uint64_t acquired_end;
	Buffer*  buffer;
	uint64_t buffer_offset;

	[[nodiscard]] uint64_t RequestedSize() const { return requested_end - base_address; }
};

// Fills `prepared` in place: it is three RES_MAX arrays, which a returned copy moved per draw.
// The non-empty guest vertex ranges of a draw, sorted and merged where they overlap or touch (AcquireVertexBuffers; the
// plan's builder, project 15 N4.2b). An invalid range: EXIT, or (exit_on_bad false) UINT32_MAX.
static uint32_t MergedVertexRanges(const ShaderVertexInputInfo& vs_input_info, std::array<uint64_t, ShaderVertexInputInfo::RES_MAX>& sizes,
                                   std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX>& merged_ranges, bool exit_on_bad) {
	// The arrays are not zeroed (~4.4 KB per draw): every entry read below is written earlier in this call (sizes[i] for
	// each i < buffers_num, the ranges below their counts, all fields of each).
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> ranges;
	uint32_t range_count = 0;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = VertexBufferDescriptorSize(i, vs_input_info);
		sizes[i]           = size;
		if (size == 0) {
			continue;
		}
		if (vertex.addr == 0 || size > UINT64_MAX - vertex.addr) {
			if (!exit_on_bad) return UINT32_MAX;
			EXIT("invalid vertex buffer range: addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
			     vertex.addr, size);
		}
		ranges[range_count++] = {vertex.addr, vertex.addr + size, 0, nullptr, 0};
	}

	std::sort(ranges.begin(), ranges.begin() + range_count,
	          [](const VertexBufferRange& left, const VertexBufferRange& right) {
		          return left.base_address < right.base_address;
	          });

	// Merge overlapping or touching ranges before acquiring host buffers.
	uint32_t merged_count = 0;
	for (uint32_t i = 0; i < range_count; i++) {
		const auto& range = ranges[i];
		if (merged_count != 0 &&
		    merged_ranges[merged_count - 1].requested_end >= range.base_address) {
			merged_ranges[merged_count - 1].requested_end =
			    std::max(merged_ranges[merged_count - 1].requested_end, range.requested_end);
			continue;
		}
		merged_ranges[merged_count++] = {range.base_address, range.requested_end, 0, nullptr, 0};
	}
	return merged_count;
}

static void AcquireVertexBuffers(CommandBuffer& buffer, const ShaderVertexInputInfo& vs_input_info,
                                 PreparedVertexBuffers& prepared) {
	KYTY_PROFILER_BLOCK("Draw::AcquireVertexBuffers");
	EXIT_IF(vs_input_info.buffers_num < 0 ||
	        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX);
	std::array<uint64_t, ShaderVertexInputInfo::RES_MAX>          sizes;
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> merged_ranges;
	const uint32_t merged_count = MergedVertexRanges(vs_input_info, sizes, merged_ranges, true);

	auto& cache = buffer.GetContext().GetBufferCache();
	for (uint32_t i = 0; i < merged_count; i++) {
		auto& range = merged_ranges[i];
		// PPSA20298
		const auto size =
		    Libs::LibKernel::Memory::ClampRangeSize(range.base_address, range.RequestedSize());
		range.acquired_end        = range.base_address + size;
		const auto [host, offset] = cache.ObtainBuffer(range.base_address, size, false);
		range.buffer              = host;
		range.buffer_offset       = offset;
	}

	// Rebuild slot bindings, offsetting non-empty slots into their acquired merged range.
	prepared.count         = static_cast<uint32_t>(vs_input_info.buffers_num);
	vk::Buffer null_buffer = nullptr;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = sizes[i];
		if (size == 0) {
			if (null_buffer == nullptr) {
				null_buffer = cache.GetBuffer(NULL_BUFFER_ID).Handle();
			}
			prepared.buffers[i] = null_buffer;
			prepared.offsets[i] = 0;
			prepared.sizes[i]   = 0;
			continue;
		}

		const auto range = std::find_if(merged_ranges.begin(), merged_ranges.begin() + merged_count,
		                                [&](const VertexBufferRange& value) {
			                                return vertex.addr >= value.base_address &&
			                                       vertex.addr < value.acquired_end;
		                                });
		if (range == merged_ranges.begin() + merged_count) {
			EXIT("vertex buffer address is outside the acquired range: addr=0x%016" PRIx64 "\n",
			     vertex.addr);
		}

		prepared.buffers[i] = range->buffer->Handle();
		prepared.offsets[i] = range->buffer_offset + vertex.addr - range->base_address;
		prepared.sizes[i]   = std::min(size, range->acquired_end - vertex.addr);
	}
}

static void SetDrawDebugPhase(CommandBuffer& buffer, uint64_t submit_id, const DrawCallInfo& draw,
                              uint32_t phase) {
	buffer.SetDebugInfo(static_cast<uint32_t>(draw.debug_op), submit_id, phase, draw.index_count, 0,
	                    draw.instance_count, draw.first_instance);
}

static bool GetDrawTopology(const HW::UserConfig& ucfg, vk::PrimitiveTopology& topology) {

	topology = vk::PrimitiveTopology::ePointList;

	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kNone: return false;
		case Prospero::PrimitiveType::kPointList:
			topology = vk::PrimitiveTopology::ePointList;
			break;
		case Prospero::PrimitiveType::kLineList: topology = vk::PrimitiveTopology::eLineList; break;
		case Prospero::PrimitiveType::kLineStrip:
			topology = vk::PrimitiveTopology::eLineStrip;
			break;
		case Prospero::PrimitiveType::kTriList:
			topology = vk::PrimitiveTopology::eTriangleList;
			break;
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kPolygon:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		case Prospero::PrimitiveType::kTriStrip:
			topology = vk::PrimitiveTopology::eTriangleStrip;
			break;
		case Prospero::PrimitiveType::kPatch:
			if (!Config::TessellationEnabled()) return false;
			[[fallthrough]];
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kRectListLegacy:
			topology = vk::PrimitiveTopology::ePatchList;
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		default: {
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				std::printf("Skipping draw with unknown primitive type: %u\n",
				            static_cast<uint32_t>(ucfg.GetPrimType()));
			}
			return false;
		}
	}

	return true;
}

// What the registers ask of primitive restart for indices of `element_size` bytes. Native: the marker the host uses; the
// 8-bit path widens it to 0xffff. Custom: a reset value the host cannot take.
enum class PrimitiveRestartKind { Unsupported, Off, Native, Custom };

static PrimitiveRestartKind ClassifyPrimitiveRestartRegs(const HW::UserConfig& ucfg, const HW::Context& ctx, uint32_t element_size);

static PrimitiveRestartKind ClassifyPrimitiveRestart(const CommandBuffer& buffer, uint32_t element_size) {
	return ClassifyPrimitiveRestartRegs(buffer.GetUserConfig(), buffer.GetRegisters(), element_size);
}

// Project 15 N4.1: the same from the registers alone.
static PrimitiveRestartKind ClassifyPrimitiveRestartRegs(const HW::UserConfig& ucfg, const HW::Context& ctx, uint32_t element_size) {
	const auto control = ucfg.GetPrimitiveResetControl();
	if ((control & ~0x3u) != 0) {
		return PrimitiveRestartKind::Unsupported;
	}
	if ((control & 0x1u) == 0) {
		return PrimitiveRestartKind::Off;
	}
	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip: break;
		default: return PrimitiveRestartKind::Off;
	}

	const auto index_mask  = UINT32_MAX >> ((4 - element_size) * 8);
	const auto reset_index = ctx.GetPrimitiveResetIndex();
	if ((control & 0x2u) != 0 && (reset_index & ~index_mask) != 0) {
		return PrimitiveRestartKind::Off;
	}
	return (reset_index & index_mask) == index_mask ? PrimitiveRestartKind::Native : PrimitiveRestartKind::Custom;
}

static bool ResolvePrimitiveRestart(const CommandBuffer& buffer,
                                    const DrawIndexBufferSource& source) {
	const auto element_size = source.guest_element_size;
	const auto kind         = ClassifyPrimitiveRestart(buffer, element_size);
	EXIT_NOT_IMPLEMENTED(kind == PrimitiveRestartKind::Unsupported);
	if (kind != PrimitiveRestartKind::Custom) {
		return kind == PrimitiveRestartKind::Native;
	}

	// A game can set a custom reset value without using it in the index buffer.
	// Keep restart off in that case; fail if we actually find the value.
	// Scan before preparing draw resources: readback can restart the command buffer.
	EXIT_NOT_IMPLEMENTED(source.address == 0);
	const auto  restart_index = buffer.GetRegisters().GetPrimitiveResetIndex() & (UINT32_MAX >> ((4 - element_size) * 8));
	const auto* indices       = reinterpret_cast<const uint8_t*>(source.address);
	for (uint64_t offset = 0; offset < source.size; offset += element_size) {
		uint32_t index = 0;
		std::memcpy(&index, indices + offset, element_size);
		EXIT_NOT_IMPLEMENTED(index == restart_index);
	}
	return false;
}

// Project 15 N7.2: a GPU indirect packet's checks that need only the registers and the packet (none of the draw's
// programs): the execution thread's DrawIndirectGpu and a helper's plan (`ahead`: the ranges clamped without a write).
// One function, so the two cannot differ.
struct IndirectEarly {
	DrawIndirectSource    indirect {};
	DrawIndexBufferSource index_source {};
	bool                  primitive_restart = false;
	bool                  no_draw           = false; // max_count == 0
};
template <typename Graphics>
static const char* IndirectEarlyChecks(const Graphics& graphics, const HW::UserConfig& ucfg, const HW::Context& regs,
                                       const DrawIndirectGpuArgs& args, bool ahead, IndirectEarly& out) {
	// Argument records larger than this, or index ranges, stay on the CPU path (cache buffers).
	constexpr uint64_t MaxArgumentBytes = 4ull << 20u;
	constexpr uint64_t MaxIndexBytes    = 64ull << 20u;
	const auto         clamp            = [ahead](uint64_t address, uint64_t size) {
        if (!ahead) return Libs::LibKernel::Memory::ClampRangeSize(address, size);
        uint64_t generation = 0;
        return Libs::LibKernel::Memory::ClampRangeSizeAhead(address, size, &generation);
	};
	out = {};
	if (!graphics.gpu_indirect_draw_enabled) {
		return "host device features";
	}
	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kPointList:
		case Prospero::PrimitiveType::kLineList:
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriList:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip:
		case Prospero::PrimitiveType::kPolygon: break;
		default: return "primitive type";
	}
	// Merged (NGG GS) stages draw through the mesh path, which sizes its workgroups from the counts.
	if ((regs.GetShaderStages() & 0x20u) != 0) {
		return "mesh vertex stage";
	}
	const uint32_t record_size = args.indexed ? 20u : 16u;
	static_assert(sizeof(vk::DrawIndexedIndirectCommand) == 20u && sizeof(vk::DrawIndirectCommand) == 16u);
	if (args.max_count == 0) {
		out.no_draw = true;
		return nullptr;
	}
	if ((args.args_addr & 3u) != 0 || (args.count_addr & 3u) != 0) {
		return "unaligned arguments";
	}
	if ((args.count_addr != 0 || args.max_count > 1u) && ((args.stride & 3u) != 0 || args.stride < record_size)) {
		return "argument stride";
	}
	if (args.max_count > graphics.GetPhysicalDeviceProperties().limits.maxDrawIndirectCount) {
		return "draw count limit";
	}
	out.indirect = DrawIndirectSource {
	    .args_addr  = args.args_addr,
	    .args_size  = uint64_t {args.stride} * (args.max_count - 1u) + record_size,
	    .count_addr = args.count_addr,
	    .max_count  = args.max_count,
	    .stride     = args.stride,
	};
	if (out.indirect.args_size > MaxArgumentBytes || clamp(out.indirect.args_addr, out.indirect.args_size) != out.indirect.args_size) {
		return "argument range";
	}
	if (args.indexed) {
		auto& index_source = out.index_source;
		switch (static_cast<Prospero::IndexType>(args.index_type_and_size)) {
			case Prospero::IndexType::kIndex16:
				index_source.type               = vk::IndexType::eUint16;
				index_source.guest_element_size = 2;
				break;
			case Prospero::IndexType::kIndex32:
				index_source.type               = vk::IndexType::eUint32;
				index_source.guest_element_size = 4;
				break;
			default: return "index type"; // 8-bit indices are widened on the CPU
		}
		if (args.index_base_addr == 0 || (args.index_base_addr & (index_source.guest_element_size - 1u)) != 0) {
			return "index base address";
		}
		// Without INDEX_BUFFER_SIZE the bound range would be unknown.
		if (args.index_buffer_size == 0) {
			return "index buffer size";
		}
		index_source.address = args.index_base_addr;
		index_source.size    = uint64_t {args.index_buffer_size} * index_source.guest_element_size;
		if (index_source.size > MaxIndexBytes) {
			return "index range";
		}
		// ResolvePrimitiveRestart without the index scan (the index count is in GPU memory): a custom restart index the
		// CPU path would have to check.
		const auto kind = ClassifyPrimitiveRestartRegs(ucfg, regs, index_source.guest_element_size);
		if (kind != PrimitiveRestartKind::Off && kind != PrimitiveRestartKind::Native) {
			return "primitive restart index";
		}
		out.primitive_restart = kind == PrimitiveRestartKind::Native;
		index_source.size     = clamp(index_source.address, index_source.size);
	}
	return nullptr;
}

// The export mapping of each color target the draw writes (RefreshShaders).
static std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX> TargetExportMapping(const HW::Context& ctx,
                                                                                                     uint32_t color_output_mask) {
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX> target_export_mapping {};
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		const auto& rt = ctx.GetRenderTarget(slot);
		if ((color_output_mask & (1u << slot)) != 0 && rt.base.addr != 0) {
			target_export_mapping[slot] =
			    TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type, rt.info.channel_order).export_mapping;
		}
	}
	return target_export_mapping;
}

static void RefreshShaders(CommandBuffer& buffer, const DrawCallInfo& draw,
                           uint32_t color_output_mask, DrawRenderState& state) {
	KYTY_PROFILER_BLOCK("Draw::RefreshShaders");
	// KYTY_TRACE_CLEAN_DRAWS: the draw's program lookup time (step K1).
	const bool c0       = CleanDraws::On();
	const auto c0_start = c0 ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();

	const auto& vertex_shader_info = sh_ctx.GetVs();
	const auto& pixel_shader_info  = sh_ctx.GetPs();
	const auto& shader_regs        = ctx.GetShaderRegisters();

	state.programs      = {};
	state.ps_input_info = {};
	const auto target_export_mapping = TargetExportMapping(ctx, color_output_mask);
	auto& pipeline_cache = buffer.GetContext().GetPipelineCache();
	// (Upstream starts its image heap here, ImageHeap::BeginDraw; not in use, RenderContext::ImageHeapEnabled.)
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "GetGraphicsPrograms");
	}
	state.programs = pipeline_cache.GetGraphicsPrograms(
	    vertex_shader_info, pixel_shader_info, shader_regs, ctx, buffer.GetUserConfig(),
	    target_export_mapping, state.ps_active, state.vertex_info, state.ps_input_info,
	    state.indirect_draw);
	if (c0) {
		CleanDraws::t_front_parts.programs_ns += static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - c0_start).count());
	}
}

// Missing graphics pipelines are created in the background (28.10; tested in Snake's house with a cold pipeline cache:
// 100 stops of the GPU thread for the driver, 38.8 s, became 0; the same stretch took 91 s instead of 133 s): a draw
// whose graphics pipeline does not exist yet is not issued (nothing of it is recorded) while a worker creates the
// pipeline (PipelineCache::GetGraphicsPipelineAsync, KYTY_PERF_TABLE_ASYNC's workers); later draws of the same key take
// it once it is ready. APPROXIMATION: the object is missing for the frames until then, instead of the whole game
// stopping 0.2-0.8 s (cutscenes jumped ahead to catch up with their sound). Only draws that write no memory (no written
// buffer, image or indirect range, no address writes in any stage): a skipped write could leave the game's own data
// stale; those create their pipeline on the GPU thread as before. Dispatches are never skipped.
static bool StageWritesMemory(const ShaderStageRuntime& stage) {
	const auto& program = *stage.program;
	if (program.has_address_writes) return true;
	if (std::ranges::any_of(program.info.buffers, [](const auto& b) { return b.written; })) return true;
	if (std::ranges::any_of(program.info.images, [](const auto& i) { return i.written; })) return true;
	return stage.resources != nullptr &&
	       std::ranges::any_of(stage.resources->indirect_ranges, [](const auto& r) { return r.written; });
}

static bool DrawWritesMemory(const DrawRenderState& state) {
	for (uint32_t i = 0; i < state.programs.VertexStageCount(); i++) {
		if (StageWritesMemory(state.vertex_info[i].stage)) return true;
	}
	return state.ps_active && StageWritesMemory(state.ps_input_info.stage);
}

// PERF async_draw_pipelines lines: draws skipped while their pipeline was being created, and draws that had to create
// it on the GPU thread because they write memory (every 5 s while either grows).
static void NoteAsyncDraw(bool skipped) {
	static std::atomic<uint64_t> skipped_draws {0};
	static std::atomic<uint64_t> writing_draws {0};
	static std::atomic<int64_t>  last_line {0};
	(skipped ? skipped_draws : writing_draws).fetch_add(1, std::memory_order_relaxed);
	const auto now  = std::chrono::steady_clock::now().time_since_epoch().count();
	auto       last = last_line.load(std::memory_order_relaxed);
	if (Common::EnvRead::PerfLines() &&
	    now - last >= std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::seconds(5)).count() &&
	    last_line.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
		std::fprintf(stderr, "PERF async_draw_pipelines skipped=%llu writing_draws_waited=%llu\n",
		             static_cast<unsigned long long>(skipped_draws.load(std::memory_order_relaxed)),
		             static_cast<unsigned long long>(writing_draws.load(std::memory_order_relaxed)));
	}
}

// A shader stage with effects outside the pipeline: buffer/image stores or atomics, address (BDA/global) writes, GDS.
static bool StageWritesMemory(const ShaderRecompiler::IR::CompiledShaderInfo& program) {
	if (program.has_address_writes || program.info.uses_gds) {
		return true;
	}
	return std::ranges::any_of(program.info.buffers, [](const auto& buffer) { return buffer.written || buffer.atomic; }) ||
	       std::ranges::any_of(program.info.images, [](const auto& image) { return image.written || image.atomic; });
}

// A draw that writes no color target, needs no depth/stencil attachment (no depth, bounds or stencil test, clear or
// copy) and whose pixel shader is off or without effects. On the PS5 it is still not nothing:
//  - an open occlusion query counts its samples: with nothing to test every covered sample passes (Ghost of Yotei,
//    Frame>Postprocessing>Bloom: each sun/moon flare is drawn twice, depth-tested and then with the depth test off,
//    and a CS divides the two query results into the flare's visibility);
//  - its vertex stages run, with their memory writes;
//  - DB_RENDER_CONTROL DEPTH/STENCIL_COMPRESS_DISABLE or RESUMMARIZE_ENABLE expand or resummarize the HTILE of the
//    bound depth surface, which writes the clear value of fast-cleared tiles into the depth memory. The host depth
//    images are never compressed, so only a fast clear the host has not applied yet is visible.
struct NoOutputDraw {
	bool counted        = false; // a guest occlusion query is open
	bool vertex_writes  = false; // a vertex-pipeline stage writes memory
	bool db_metadata_op = false; // HTILE expand / resummarize
	bool pending_clear  = false; // ... of a depth surface with a fast clear the host has not applied
};

static NoOutputDraw ClassifyNoOutputDraw(const CommandBuffer& buffer, const DrawRenderState& state) {
	NoOutputDraw result;
	auto&        context = buffer.GetContext();
	result.counted       = context.GetOcclusionQueries().Counting();
	for (uint32_t i = 0; i < state.programs.VertexStageCount(); i++) {
		const auto* program = state.vertex_info[i].stage.program;
		if (program != nullptr && StageWritesMemory(*program)) {
			result.vertex_writes = true;
		}
	}
	const auto& regs = buffer.GetRegisters();
	const auto& rc   = regs.GetRenderControl();
	const auto& ro   = regs.GetDepthRenderOverride();
	const auto& z    = regs.GetDepthRenderTarget();
	// DB_RENDER_OVERRIDE FORCE_Z/STENCIL_DIRTY rewrite the tiles in place too (a copy, with different read and write
	// addresses, never gets here: DepthStencilCopy).
	result.db_metadata_op = rc.depth_compress_disable || rc.stencil_compress_disable || rc.resummarize_enable ||
	                        ro.force_z_dirty || ro.force_stencil_dirty;
	if (result.db_metadata_op && z.z_info.htile_acceleration && z.htile_data_base_addr != 0) {
		result.pending_clear =
		    context.GetTextureCache().IsMetaCleared(z.htile_data_base_addr, z.depth_view.slice_start);
	}
	return result;
}

static const char* NoOutputDropReason(const NoOutputDraw& draw) {
	if (draw.counted) return "render state (no output): occlusion query draw, not counted (KYTY_FIX_OCCLUSION_QUERIES off)";
	// Running these (vertex stage writes; the expand binding the surface so its pending fast clear lands) left the
	// game on its loading screen: they stay dropped.
	if (draw.vertex_writes) return "render state (no output): vertex stages write memory (not run)";
	if (draw.pending_clear) return "render state (no output): HTILE expand of a fast-cleared depth surface (not run)";
	if (draw.db_metadata_op) return "render state (no effect): HTILE expand/resummarize, the host depth is never compressed";
	return "render state (no effect): no color/depth/stencil work, no shader effects, no occlusion query";
}

// A draw PrepareDrawRenderState left out, with its reason.
static void TraceRenderStateDrop(const char* fallback, const CommandBuffer& buffer, const DrawRenderState& state) {
	TraceDrawDrop(state.drop_reason != nullptr ? state.drop_reason : fallback, buffer);
}

// Project C, stage C1 step 1: a draw whose render target search reads the same registers as the previous draw's (the CB
// registers of the color slots its pixel shader writes, the target mask, the depth target, the render / depth / stencil
// controls, the stencil masks and clear values, the written slots and the slice offset) takes that draw's targets instead of
// searching again, when nothing else the search depends on changed in between: the image registry (no image registered or
// unregistered, so no rebind either), the GPU tick and the texture cache's GC tick (the access tick and LRU touch the search
// writes hold these already), no metadata / copy / resolve operation, and every search of the previous draw was a found-only
// answer (no full search: TextureCache::FullSearches). The full search would then find the same images and only touch
// them; the draw binds them to its plan as the search does.
namespace C1 {
struct TargetKey {
	HW::RenderTarget      rt[8];
	HW::DepthRenderTarget depth;
	HW::RenderControl     rc;
	HW::DepthControl      dc;
	HW::StencilControl    sc;
	HW::StencilMask       sm;
	float                 depth_clear   = 0.0f;
	uint32_t              stencil_clear = 0;
	uint32_t              rt_mask       = 0;
	uint32_t              mrt_mask      = 0;
	uint32_t              slice_offset  = 0;
	uint32_t              pad           = 0;
	uint64_t              registry      = 0;
	uint64_t              gc_tick       = 0;
	uint64_t              tick          = 0;
	uint64_t              meta_ops      = 0;
};

template <typename T>
void CopyBytes(T& to, const T& from) {
	std::memcpy(static_cast<void*>(&to), static_cast<const void*>(&from), sizeof(T));
}

// Byte copies (the padding too), on a zeroed key: two keys of equal registers compare equal.
void FillKey(TargetKey& k, const HW::Context& hw, uint32_t mrt_mask, uint32_t slice_offset, uint64_t gc_tick, uint64_t tick) {
	std::memset(static_cast<void*>(&k), 0, sizeof(k));
	for (uint32_t slot = 0; slot < 8u; slot++) {
		if ((mrt_mask & (1u << slot)) != 0) CopyBytes(k.rt[slot], hw.GetRenderTarget(slot));
	}
	CopyBytes(k.depth, hw.GetDepthRenderTarget());
	CopyBytes(k.rc, hw.GetRenderControl());
	CopyBytes(k.dc, hw.GetDepthControl());
	CopyBytes(k.sc, hw.GetStencilControl());
	CopyBytes(k.sm, hw.GetStencilMask());
	k.depth_clear   = hw.GetDepthClearValue();
	k.stencil_clear = hw.GetStencilClearValue();
	k.rt_mask       = hw.GetRenderTargetMask();
	k.mrt_mask      = mrt_mask;
	k.slice_offset  = slice_offset;
	k.registry      = DrawRepeat::g_image_registry_epoch.load(std::memory_order_acquire);
	k.gc_tick       = gc_tick;
	k.tick          = tick;
	k.meta_ops      = CleanDraws::g_meta_ops.load(std::memory_order_relaxed);
}

// The previous draw's targets (GPU thread).
struct TargetMemo {
	bool                                                    valid = false;
	TargetKey                                               key;
	uint32_t                                                color_count = 0;
	std::array<RenderColorInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
	RenderDepthInfo                                         depth {};
	uint64_t draws = 0, equal = 0, hits = 0, stored = 0;
};
TargetMemo g_targets;

template <typename T>
bool SameBytes(const T& a, const T& b) {
	return std::memcmp(static_cast<const void*>(&a), static_cast<const void*>(&b), sizeof(T)) == 0;
}

void Report(TargetMemo& m) {
	if ((m.draws & 0xffffu) != 0 || !Common::EnvRead::PerfLines()) return;
	std::fprintf(stderr, "PERF clean_draw_skip draws=%llu equal=%llu hits=%llu stored=%llu\n",
	             static_cast<unsigned long long>(m.draws), static_cast<unsigned long long>(m.equal),
	             static_cast<unsigned long long>(m.hits), static_cast<unsigned long long>(m.stored));
}
} // namespace C1

// Project 15 S2c (KYTY_CHECK_PEEK=2, diagnostic): the draw's targets described from the registers, its pipeline key
// from them and the pipeline found without a write, against today's (execution thread).
namespace PeekTargets {
struct Stats {
	uint64_t draws = 0, targets_bad = 0, key_bad = 0, compared = 0, bad = 0, created = 0, injected = 0, detected = 0;
	std::array<uint64_t, static_cast<size_t>(PipelineCache::PipelineFind::Count)> find {}, member_find {};
	uint32_t logged = 0;
};
Stats g_stats;
struct Scratch {
	bool            valid = false;
	uint32_t        count = 0;
	RenderColorInfo colors[RENDER_COLOR_ATTACHMENTS_MAX];
	RenderDepthInfo depth;
};
Scratch g_scratch;
bool    Check() {
	static const bool on = [] {
		const char* env = std::getenv("KYTY_CHECK_PEEK");
		return env != nullptr && env[0] == '2';
	}();
	return on;
}
uint64_t FaultEvery() {
	static const uint64_t every = [] {
		const char* env = std::getenv("KYTY_CHECK_PEEK_FAULT");
		return env != nullptr ? std::strtoull(env, nullptr, 10) : 0u;
	}();
	return every;
}
void Report() {
	auto& st = g_stats;
	if ((st.draws & ((1u << 18u) - 1u)) != 0) return;
	const auto join = [](const auto& a) {
		std::string out;
		for (size_t i = 0; i < a.size(); i++) out += std::string(i == 0 ? "" : "|") + std::to_string(a[i]);
		return out;
	};
	std::fprintf(stderr,
	             "PERF peek_pipeline draws=%llu targets_bad=%llu key_bad=%llu find[found|evicted|pending_done|pending|missing]=%s "
	             "member=%s compared=%llu bad=%llu created=%llu fault[injected|detected]=%llu|%llu\n",
	             static_cast<unsigned long long>(st.draws), static_cast<unsigned long long>(st.targets_bad),
	             static_cast<unsigned long long>(st.key_bad), join(st.find).c_str(), join(st.member_find).c_str(),
	             static_cast<unsigned long long>(st.compared), static_cast<unsigned long long>(st.bad),
	             static_cast<unsigned long long>(st.created), static_cast<unsigned long long>(st.injected),
	             static_cast<unsigned long long>(st.detected));
}
// The key fields of a described target against today's resolved one.
bool SameColor(const RenderColorInfo& a, const RenderColorInfo& b) {
	return a.target_slot == b.target_slot && a.desc.view_info.format == b.desc.view_info.format &&
	       a.desc.info.samples == b.desc.info.samples && a.desc.info.guest_format == b.desc.info.guest_format &&
	       std::memcmp(&a.export_mapping, &b.export_mapping, sizeof(a.export_mapping)) == 0;
}
} // namespace PeekTargets

bool RenderExecutor::PrepareDrawRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
                                            uint32_t            render_target_slice_offset,
	                                        DrawRenderState& state) {
	const ChunkLead::FloorTimer floor_targets(ChunkLead::FpTargets); // KYTY_TRACE_PEEK_CENSUS (N3.0)
	const auto& shader_regs       = buffer.GetRegisters().GetShaderRegisters();
	const auto  color_output_mask = DrawColorOutputMask(buffer.GetRegisters());
	state.ps_active = buffer.GetShaders().GetPs().ps_regs.data_addr != 0 &&
	                  (color_output_mask != 0 ||
	                   PixelShaderHasDepthOrCoverageSideEffects(shader_regs));
	std::optional<DispatchSubTimer> census_programs(std::in_place, DispatchSub::DrawPrograms); // KYTY_TRACE_DRAW_CENSUS
	if (!std::exchange(m_state_resolved, false)) RefreshShaders(buffer, draw, color_output_mask, state);
	census_programs.reset();
	uint32_t mrt_mask     = 0;
	uint32_t objprim_mask = 0;
	if (state.ps_active) {
		for (const auto& output: state.ps_input_info.stage.program->info.outputs) {
			if (output.kind == ShaderRecompiler::IR::StageOutputKind::Mrt ||
			    output.kind == ShaderRecompiler::IR::StageOutputKind::ObjectPrimitiveId) {
				mrt_mask |= 1u << output.location;
			} else if (output.kind == ShaderRecompiler::IR::StageOutputKind::ObjPrimId) {
				// The ObjPrimID write: a slot the pixel shader does not export (outside color_output_mask).
				objprim_mask |= 1u << output.location;
			}
		}
	}
	mrt_mask = (mrt_mask & color_output_mask) | objprim_mask;
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "ResolveRenderColorTarget");
	}
	std::optional<DispatchSubTimer> census_targets(std::in_place, DispatchSub::DrawTargets); // KYTY_TRACE_DRAW_CENSUS
	const bool c0_targets       = CleanDraws::On(); // KYTY_TRACE_CLEAN_DRAWS: the target search's time (step K1)
	const auto c0_targets_start = c0_targets ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
	// The previous draw's targets when its search would find them again.
	C1::TargetKey target_key;
	auto&         memo = C1::g_targets;
	C1::FillKey(target_key, buffer.GetRegisters(), mrt_mask, render_target_slice_offset,
	            m_context.GetTextureCache().GcTick(), m_context.GetCommandScheduler().CurrentTick());
	const bool target_equal = memo.valid && C1::SameBytes(target_key, memo.key);
	memo.draws++;
	memo.equal += target_equal ? 1u : 0u;
	if (target_equal) {
		state.color_count = memo.color_count;
		for (uint32_t i = 0; i < memo.color_count; i++) state.color_info[i] = memo.colors[i];
		state.depth_info = memo.depth;
		for (uint32_t i = 0; i < memo.color_count; i++) BindRenderTarget(state.color_info[i].image_id);
		if (state.depth_info.image_id) BindRenderTarget(state.depth_info.image_id);
		memo.hits++;
		if (c0_targets) CleanDraws::t_front_parts.target_hit = true; // KYTY_TRACE_CLEAN_DRAWS
		C1::Report(memo);
	} else {
		const uint64_t full_searches = TextureCache::FullSearches();
		for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
			if ((mrt_mask & (1u << slot)) != 0) {
				ResolveRenderColorTarget(buffer, state.color_info[state.color_count],
				                         render_target_slice_offset, slot);
				if (state.color_info[state.color_count].image_id) {
					state.color_count++;
				}
			}
		}
		if (draw.IsIndexed()) {
			LogDrawPhase(draw.Name(), "ResolveRenderDepthTarget");
		}
		ResolveRenderDepthTarget(buffer, state.depth_info);
		// Kept when every search was found-only and nothing was registered meanwhile (the key's epoch still holds).
		memo.valid = TextureCache::FullSearches() == full_searches &&
		             DrawRepeat::g_image_registry_epoch.load(std::memory_order_acquire) == target_key.registry &&
		             CleanDraws::g_meta_ops.load(std::memory_order_relaxed) == target_key.meta_ops;
		if (memo.valid) {
			memo.key         = target_key;
			memo.color_count = state.color_count;
			for (uint32_t i = 0; i < state.color_count; i++) memo.colors[i] = state.color_info[i];
			memo.depth = state.depth_info;
			memo.stored++;
		}
		C1::Report(memo);
	}
	census_targets.reset();
	if (c0_targets) {
		CleanDraws::t_front_parts.targets_ns += static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - c0_targets_start).count());
	}
	if (PeekTargets::Check()) {
		// KYTY_CHECK_PEEK=2 (S2c): the same targets from the registers alone, inside the translator section.
		auto& sc = PeekTargets::g_scratch;
		++t_translator;
		sc.count = 0;
		for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
			if ((mrt_mask & (1u << slot)) == 0) continue;
			if (DescribeRenderColorTarget(buffer.GetRegisters(), sc.colors[sc.count], render_target_slice_offset, slot, false)) sc.count++;
		}
		sc.depth = {};
		(void)DescribeRenderDepthTarget(buffer.GetGraphics(), buffer.GetRegisters(), sc.depth);
		--t_translator;
		sc.valid  = true;
		bool same = sc.count == state.color_count;
		for (uint32_t i = 0; same && i < sc.count; i++) same = PeekTargets::SameColor(sc.colors[i], state.color_info[i]);
		const bool depth_described = sc.depth.desc.view_info.format != vk::Format::eUndefined;
		const bool depth_today = state.depth_info.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(state.depth_info.image_id);
		same = same && depth_described == depth_today &&
		       (!depth_today || (sc.depth.desc.view_info.format == state.depth_info.desc.view_info.format &&
		                         sc.depth.desc.info.samples == state.depth_info.desc.info.samples));
		if (!same) {
			auto& st = PeekTargets::g_stats;
			st.targets_bad++;
			if (st.logged++ < 20u) {
				std::fprintf(stderr, "CHECK_PEEK_TARGETS bad colors=%u/%u depth=%d/%d mrt=0x%x\n", sc.count, state.color_count,
				             depth_described ? 1 : 0, depth_today ? 1 : 0, mrt_mask);
			}
		}
	}

	if (state.color_count == 0 && !state.depth_info.image_id && !state.ps_active) {
		// Nothing to write and no pixel shader effects: what the PS5 still does with such a draw (NoOutputDraw).
		const auto no_output = ClassifyNoOutputDraw(buffer, state);
		// KYTY_FIX_OCCLUSION_QUERIES: the draw runs without attachments and the open host query counts its samples.
		if (no_output.counted && m_context.GetOcclusionQueries().Enabled()) {
			return true;
		}
		state.drop_reason = NoOutputDropReason(no_output);
		LogFramebufferSkip(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   draw.index_count, 0);
		return false;
	}

	return true;
}

static PreparedIndexBuffer PrepareIndexBuffer(CommandBuffer&               buffer,
                                              const DrawIndexBufferSource& source) {
	KYTY_PROFILER_BLOCK("Draw::PrepareIndexBuffer");
	PreparedIndexBuffer prepared;
	if (source.size == 0) {
		return prepared;
	}
	prepared.type = source.type;
	if (source.host_data != nullptr) {
		auto& stream = buffer.GetContext().GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		prepared.offset = stream.Copy(source.host_data, source.size, 16);
		prepared.buffer = stream.Handle();
	} else {
		auto [buffer_ptr, offset] =
		    buffer.GetContext().GetBufferCache().ObtainBuffer(source.address, source.size, false);
		prepared.buffer = buffer_ptr->Handle();
		prepared.offset = offset;
	}
	return prepared;
}

static void CommitVertexBuffers(vk::CommandBuffer            vk_buffer,
                                const PreparedVertexBuffers& prepared) {
	for (uint32_t i = 0; i < prepared.count; i++) {
		EXIT_IF(prepared.buffers[i] == nullptr);
	}
	if (prepared.count != 0) {
		// Guest descriptor bounds must survive allocation merging in the cache.
		vk_buffer.bindVertexBuffers2(0, prepared.count, prepared.buffers.data(),
		                             prepared.offsets.data(), prepared.sizes.data(), nullptr);
	}
}

static void CommitIndexBuffer(vk::CommandBuffer vk_buffer, const PreparedIndexBuffer& prepared) {
	if (prepared.buffer == nullptr) {
		return;
	}
	vk_buffer.bindIndexBuffer(prepared.buffer, prepared.offset, prepared.type);
}

static void LogDrawStateIfNeeded(const CommandBuffer& buffer, const DrawCallInfo& draw,
	                             const DrawRenderState& state, uint32_t index_type_and_size,
                                 const void* index_addr) {
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	if (!draw.IsIndexed() && !Prospero::IsRectList(buffer.GetUserConfig().GetPrimType())) {
		return;
	}

	if (state.ps_active) {
		LogDrawTargetState(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   state.ps_input_info, draw.index_count, 0);
	}
	LogDrawInputState(buffer, state.color_info[0], state.vertex_info[0], index_type_and_size,
	                  draw.index_count, index_addr);
}

static void EmitDrawPrimitives(const DrawDynamicRegs& ucfg, vk::CommandBuffer vk_buffer,
                               const DrawCallInfo& draw, const DrawEmitInfo& emit) {
	switch (ucfg.prim_type) {
		case Prospero::PrimitiveType::kPointList:
		case Prospero::PrimitiveType::kLineList:
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriList:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip:
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kPolygon:
		case Prospero::PrimitiveType::kRectListLegacy:
		case Prospero::PrimitiveType::kPatch:
			if (draw.IsIndexed()) {
				vk_buffer.drawIndexed(draw.index_count, draw.instance_count, 0, emit.vertex_offset,
				                      emit.first_instance);
			} else {
				vk_buffer.draw(draw.index_count, draw.instance_count, emit.first_vertex,
				               emit.first_instance);
			}
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			EXIT_NOT_IMPLEMENTED((draw.index_count & 0x3u) != 0);
			for (uint32_t i = 0; i < draw.index_count; i += 4) {
				if (draw.IsIndexed()) {
					vk_buffer.drawIndexed(4, draw.instance_count, i, emit.vertex_offset,
					                      emit.first_instance);
				} else {
					vk_buffer.draw(4, draw.instance_count, i + emit.first_vertex,
					               emit.first_instance);
				}
			}
			break;
		default: EXIT("unknown primitive type: %u\n", static_cast<uint32_t>(ucfg.prim_type));
	}
}

// GPU indirect draws: the guest argument records (and count) drive the host draw directly.
static void EmitIndirectDraw(vk::CommandBuffer vk_buffer, const DrawCallInfo& draw,
                             const DrawIndirectSource& source, std::pair<Buffer*, uint64_t> args,
                             std::pair<Buffer*, uint64_t> count) {
	EXIT_IF(args.first == nullptr || (source.count_addr != 0 && count.first == nullptr));
	if ((args.second & 3u) != 0 || (count.first != nullptr && (count.second & 3u) != 0)) {
		static std::atomic_bool logged = false;
		if (!logged.exchange(true, std::memory_order_relaxed)) {
			std::fprintf(stderr,
			             "DIAG_GPU_INDIRECT skipped a draw: host argument offsets 0x%llx/0x%llx are not "
			             "dword aligned (args=0x%llx)\n",
			             static_cast<unsigned long long>(args.second),
			             static_cast<unsigned long long>(count.second),
			             static_cast<unsigned long long>(source.args_addr));
		}
		return;
	}
	if (count.first != nullptr) {
		if (draw.IsIndexed()) {
			vk_buffer.drawIndexedIndirectCount(args.first->Handle(), args.second,
			                                   count.first->Handle(), count.second,
			                                   source.max_count, source.stride);
		} else {
			vk_buffer.drawIndirectCount(args.first->Handle(), args.second, count.first->Handle(),
			                            count.second, source.max_count, source.stride);
		}
		return;
	}
	if (draw.IsIndexed()) {
		vk_buffer.drawIndexedIndirect(args.first->Handle(), args.second, source.max_count,
		                              source.stride);
	} else {
		vk_buffer.drawIndirect(args.first->Handle(), args.second, source.max_count, source.stride);
	}
}

// KYTY_TRACE_CS_IMAGE_DUMP=<hex,...> also lists vertex shaders: at their draws (3 per shader once
// 1000 frames were presented) every image bound to the vertex stage is dumped as the GPU sees it
// before the draw, to <KYTY_TRACE_FRAME_DIR>/vsimg_<hash>_<n>_b<binding>_<addr>_<WxH>_<fmt>.raw,
// with an IMAGE_DUMP line per binding. Diagnostic only.
static void DrawImageDumpTrace(RenderContext& context, vk::CommandBuffer vk_buffer,
                               const ShaderRecompiler::IR::CompiledShaderInfo& program, PreparedBindings& bindings,
                               const ShaderRecompiler::IR::ResourceSnapshot* resources) {
	static constinit Common::OnceObject<std::string> list_once;
	const std::string& list = list_once.Get([] { return Common::EnvRead::CsImageDump(); });
	const bool window = DebugStencilWindow();
	if (list.empty() || (DebugPresentCount().load() < 1000u && !window)) return;
	const char* dir = Common::EnvRead::FrameDir();
	if (dir == nullptr) return;
	char text[17];
	std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(program.shader_hash));
	if (list.find(text) == std::string::npos) return;
	static std::unordered_map<uint64_t, unsigned> counts;
	static std::unordered_map<uint64_t, unsigned> window_counts;
	unsigned n = 0;
	if (window) {
		static uint32_t last_window_frame = 0;
		const auto      frame             = DebugPresentCount().load();
		if (frame > last_window_frame + 3u) window_counts.clear();
		last_window_frame = frame;
		n = 100u + window_counts[program.shader_hash]++;
		if (n >= 102u) return;
	} else {
		n = counts[program.shader_hash]++;
		if (n >= 3u) return;
	}
	auto& cache = context.GetTextureCache();
	// Copies are not allowed inside a rendering instance.
	context.GetCommandScheduler().EndRendering();
	for (uint32_t i = 0; i < bindings.images.size(); i++) {
		auto& binding = bindings.images[i];
		if (!binding.image_id) continue;
		auto& image = cache.GetImage(binding.image_id);
		std::fprintf(stderr,
		             "IMAGE_DUMP frame=%u hash=0x%s n=%u binding=%u %s desc_addr=0x%llx desc=%ux%u bound_addr=0x%llx bound=%ux%u fmt=%s depth=%d stencil=%d depth_id=%d\n",
		             DebugPresentCount().load(), text, n, i, program.info.images[i].written ? "write" : "read",
		             static_cast<unsigned long long>(binding.desc.info.data.address), binding.desc.info.extent.width,
		             binding.desc.info.extent.height, static_cast<unsigned long long>(image.info.data.address),
		             image.backing.extent.width, image.backing.extent.height, vk::to_string(image.backing.format).c_str(),
		             image.info.IsDepth() ? 1 : 0, image.info.HasStencil() ? 1 : 0, image.depth_id ? 1 : 0);
		char name[200];
		std::snprintf(name, sizeof(name), "%s/vsimg_%s_%u_b%u_%llx", dir, text, n, i,
		              static_cast<unsigned long long>(binding.desc.info.data.address));
		DebugQueueFullDump(context.GetGraphics(), vk_buffer, image, name, false);
		// The guest-memory bytes of the same range (CPU view; tiled layout as stored), to compare
		// with the GPU copy: a stale texture shows different contents here.
		const auto mem_addr = binding.desc.info.data.address;
		const auto mem_size = std::min<uint64_t>(binding.desc.info.data.size, 4ull << 20u);
		// Read the backing without touching protected pages: a fault here would download, submit
		// the scheduler and leave the draw being recorded in a finished command buffer.
		std::vector<uint8_t> backing;
		if (mem_addr != 0 && mem_size != 0) {
			backing.resize(mem_size);
			if (!LibKernel::Memory::TryReadBacking(mem_addr, backing.data(), mem_size)) backing.clear();
		}
		if (!backing.empty()) {
			const auto* bytes = backing.data();
			std::snprintf(name, sizeof(name), "%s/vsmem_%s_%u_b%u_%llx.bin", dir, text, n, i,
			              static_cast<unsigned long long>(mem_addr));
			if (FILE* file = std::fopen(name, "wb")) {
				std::fwrite(bytes, 1, mem_size, file);
				std::fclose(file);
			}
			uint64_t nonzero = 0;
			for (uint64_t k = 0; k < mem_size; k++) nonzero += bytes[k] != 0 ? 1u : 0u;
			std::fprintf(stderr, "IMAGE_MEM frame=%u hash=0x%s n=%u binding=%u addr=0x%llx size=0x%llx nonzero_bytes=%llu cpu_dirty=%d\n",
			             DebugPresentCount().load(), text, n, i, static_cast<unsigned long long>(mem_addr),
			             static_cast<unsigned long long>(mem_size), static_cast<unsigned long long>(nonzero),
			             image.IsCpuDirty() ? 1 : 0);
		}
	}
	// The bound buffers (<= 1 MiB each) as the GPU sees them before the draw, with a BUFFER_DUMP line.
	for (uint32_t j = 0; resources != nullptr && j < program.info.buffers.size() && j < resources->buffers.size(); j++) {
		const auto v     = DecodeNativeDescriptor<ShaderBufferResource>(resources->buffers[j]);
		const auto bytes = std::min<uint64_t>(v.GetSize(), 1ull << 20u) & ~uint64_t {3};
		std::fprintf(stderr, "BUFFER_DUMP frame=%u hash=0x%s n=%u buf=%u %s addr=0x%llx stride=%u records=%u words=%08x %08x %08x %08x\n",
		             DebugPresentCount().load(), text, n, j, program.info.buffers[j].written ? "rw" : "ro",
		             static_cast<unsigned long long>(v.Base48()), v.Stride(), v.NumRecords(), v.fields[0], v.fields[1],
		             v.fields[2], v.fields[3]);
		if (v.Base48() == 0 || bytes == 0) continue;
		auto [source, source_offset] = context.GetBufferCache().ObtainBuffer(v.Base48(), bytes, false);
		if (source == nullptr) continue;
		char name[200];
		std::snprintf(name, sizeof(name), "%s/vsbuf_%s_%u_b%u_%llx.bin", dir, text, n, j,
		              static_cast<unsigned long long>(v.Base48()));
		DebugQueueBufferDump(context.GetGraphics(), vk_buffer, source->Handle(), source_offset, bytes, name);
	}
}

bool RenderExecutor::CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
                                       const DrawEmitInfo& emit, DrawRenderState& state) {
	if (!buffer.GetRegisters().GetRenderControl().copy_depth_to_color) {
		return false;
	}
	auto& ucfg = buffer.GetUserConfig();
	const auto& hw = buffer.GetRegisters();
	auto& depth = state.depth_info;
	auto& color = state.color_info[0];
	EXIT_NOT_IMPLEMENTED(!m_context.GetGraphics().maintenance8_enabled ||
	                     !Prospero::IsRectList(ucfg.GetPrimType()) ||
	                     draw.IsIndexed() || draw.index_count != 3 || draw.instance_count != 1 ||
	                     emit.first_vertex != 0 || emit.first_instance != 0 ||
	                     state.color_count != 1 || color.target_slot != 0 || !depth.image_id ||
	                     color.desc.info.samples != 1 ||
	                     depth.desc.info.extent.width != color.Extent().width ||
	                     depth.desc.info.extent.height != color.Extent().height ||
	                     depth.desc.view_info.layer_count != 1 ||
	                     color.desc.view_info.layer_count != 1 ||
	                     hw.GetBlendControl(0).enable || state.ps_input_info.ps_pixel_kill_enable ||
	                     state.ps_input_info.ps_depth_export_enable ||
	                     state.ps_input_info.ps_sample_mask_export_enable);
	vk::ImageAspectFlags feedback;
	ApplyPlanFlags(); // the plan flags: the copy acquires the targets (reads their flags)
	const auto rendering = AcquireRenderTargets(buffer, &color, 1, depth, feedback);
	if (depth.load_clear_aspects & vk::ImageAspectFlagBits::eDepth) { // ours: depth_load_clear_enable
		m_context.GetCommandScheduler().BeginRendering(rendering);
	}
	auto& cache = m_context.GetTextureCache();
	auto& source = cache.GetImage(depth.image_id);
	auto& destination = cache.GetImage(color.image_id);
	EXIT_NOT_IMPLEMENTED((source.backing.format != vk::Format::eD32Sfloat &&
	                      source.backing.format != vk::Format::eD32SfloatS8Uint) ||
	                     destination.backing.format != vk::Format::eR32Sfloat);
	const auto rect = calc_final_scissor(hw.GetScreenViewport(), hw.GetScanModeControl(),
	                                    {rendering.width, rendering.height}, 0);
	if (rect.right > rect.left && rect.bottom > rect.top) {
		const vk::ImageCopy region {
		    {vk::ImageAspectFlagBits::eDepth, depth.desc.view_info.base_level,
		     depth.desc.view_info.base_layer, 1}, {rect.left, rect.top, 0},
		    {vk::ImageAspectFlagBits::eColor, color.desc.view_info.base_level,
		     color.desc.view_info.base_layer, 1}, {rect.left, rect.top, 0},
		    {static_cast<uint32_t>(rect.right - rect.left),
		     static_cast<uint32_t>(rect.bottom - rect.top), 1}};
		m_context.GetCommandScheduler().EndRendering();
		const auto command = buffer.Handle();
		source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
		               {}, command);
		destination.Transit(vk::ImageLayout::eTransferDstOptimal,
		                    vk::AccessFlagBits2::eTransferWrite, {}, command);
		command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal,
		                  destination.backing.image, vk::ImageLayout::eTransferDstOptimal,
		                  1, &region);
	}
	return true;
}

// Everything of one draw.
struct DrawSlot {
	DrawRenderState                    state;
	DrawPlan                           plan;
	std::vector<uint16_t>              expanded_indices;
	RenderExecutor::GraphicsBindings   bindings;
	RenderExecutor::PlanImageFlags     flags;
};

RenderExecutor::PlanImageFlags& RenderExecutor::SlotFlags(DrawSlot& slot) {
	return slot.flags;
}

RenderExecutor::~RenderExecutor() {
	delete m_draw_slot;
}

DrawSlot* RenderExecutor::NextDrawSlot() {
	if (m_draw_slot == nullptr) m_draw_slot = new DrawSlot;
	return m_draw_slot;
}

namespace {
// KYTY_TRACE_LONG_PREP=<us> (diagnostic): a draw preparation longer than that prints its phases (us: vertex stages'
// PrepareBindings, the pixel stage's, PrepareGraphicsBindings, vertex/index/indirect buffers), the shaders and whether
// the job took the GPU (first 300 lines), so the long preparations the GPU thread waits for can be named.
uint32_t LongPrepThresholdUs() {
	static const uint32_t us = [] {
		const char* value = std::getenv("KYTY_TRACE_LONG_PREP");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	return us;
}
} // namespace

bool LongPrepPartsOn() {
	return LongPrepThresholdUs() != 0;
}

// KYTY_CHECK_CHUNKS=2 (project 15, the oracle v2 of the parallel translation): a chunk member (a draw or a direct dispatch
// of the project's passes) is prepared twice, the first time into bindings of its own (ChunkScratchBindings, as a worker
// would), the second time into the ones the commit uses; the two prints are compared. The print: the storage, GDS, LDS,
// vertex and index buffers binding by binding (the guest range always; when both preparations took a cached or fixed
// buffer, its Vulkan buffer, offset and range; when both took the stream ring, its bytes (a sample: the ring is
// write-combined memory); one of each is counted as mixed: which of them a preparation takes depends on what the
// preparations before it uploaded), the images (ids, views) and samplers, the SRT (the flat words, the bindless patches,
// the GPU-read words as the shader reaches them), the shader data, the bindless images. Not the pipeline (looked up once,
// before both preparations) and not the images' layouts (the commit sets them). A cached buffer that is no longer the one
// registered for its guest range when the print is taken (stale: a later acquisition of the same preparation replaced it)
// is a difference (the buffer registry epoch guard, PrepareDrawBindings, took those away: 2310 per walk before it, 0
// after). The bindless images are counted apart (bindless_info=): not a worker's result.
// KYTY_CHECK_CHUNKS_FAULT=N (with =2, diagnostic): every N-th member gets one error in its first preparation's data, the
// fields in turn; the oracle must report every one (PERF chunk_check fault[...]).
namespace {
enum ChunkField : uint32_t { CfBuffers, CfImages, CfSrt, CfData, CfBindless, CfVertex, CfCount };
const char* const ChunkFieldNames[CfCount] = {"buffers", "images", "srt", "data", "bindless", "vertex"};
inline uint64_t FpMix(uint64_t h, uint64_t v) {
	return (h ^ (v + 0x9E3779B97F4A7C15ull + (h << 6u) + (h >> 2u))) * 0x100000001B3ull;
}
uint64_t FpBytes(const uint8_t* data, size_t size) {
	uint64_t h = 0xCBF29CE484222325ull;
	for (size_t i = 0; i < size; i++) h = (h ^ data[i]) * 0x100000001B3ull;
	return h;
}
struct ChunkBufRec {
	uint32_t field = CfBuffers; // CfBuffers or CfVertex
	uint32_t kind  = 0;         // 0: a cached or fixed buffer (its Vulkan buffer, offset, range), 1: the stream ring (bytes)
	uint64_t address = 0, size = 0; // the guest range (0 for GDS and LDS)
	uint64_t buffer = 0, offset = 0, range = 0, bytes = 0;
	bool     stale   = false; // a cached buffer that is no longer the one registered for its guest range
	[[nodiscard]] bool Same(const ChunkBufRec& o) const {
		return field == o.field && kind == o.kind && address == o.address && size == o.size && buffer == o.buffer &&
		       offset == o.offset && range == o.range && bytes == o.bytes;
	}
};
struct ChunkPrint {
	std::array<uint64_t, CfCount> f {};
	std::vector<ChunkBufRec>      bufs;
	std::vector<uint64_t>         bindless; // the bindless images (generation << 32 | index), stage by stage
};
struct ChunkCheckState {
	ChunkPrint                    first, second;
	std::vector<PreparedBindings*> first_stages; // the first preparation's own bindings, stage by stage
	uint64_t                      draws = 0, dispatches = 0, bad = 0, mixed = 0, members = 0;
	uint64_t                      bindless_info = 0;
	std::array<uint64_t, CfCount> mismatches {}, injected {}, detected {}, injected_mixed {};
	uint32_t                      fault_turn  = 0;
	int                           fault_field = -1;       // the field of this member's injected error (-1: none)
	size_t                        fault_rec   = SIZE_MAX; // its binding record, for a binding
	uint32_t                      logged      = 0;
	std::chrono::steady_clock::time_point window = std::chrono::steady_clock::now();
};
ChunkCheckState g_chunk_check; // execution thread

uint64_t ChunkFaultEvery() {
	static const uint64_t every = [] {
		const char*    value = std::getenv("KYTY_CHECK_CHUNKS_FAULT");
		const uint64_t n     = value != nullptr ? std::strtoull(value, nullptr, 10) : 0u;
		if (n != 0) {
			std::fprintf(stderr, "CHECK_CHUNKS_FAULT=%llu: one error in the first preparation of every %llu-th chunk member\n",
			             static_cast<unsigned long long>(n), static_cast<unsigned long long>(n));
		}
		return n;
	}();
	return every;
}

void ChunkPrintOf(BufferCache& cache, ChunkPrint& p, std::span<PreparedBindings* const> stages, const DrawPlan* plan) {
	p.f = {};
	p.bufs.clear();
	p.bindless.clear();
	auto&      stream        = cache.GetUtilityBuffer(MemoryUsage::Stream);
	const auto stream_handle = stream.Handle();
	const auto mapped        = stream.Mapped();
	const auto add = [&](uint32_t field, uint64_t address, uint64_t size, vk::Buffer buffer, uint64_t offset, uint64_t range) {
		ChunkBufRec r;
		r.field   = field;
		r.address = address;
		r.size    = size;
		if (buffer && buffer == stream_handle) {
			r.kind  = 1;
			r.range = range;
			if (range != VK_WHOLE_SIZE && offset + range <= mapped.size()) {
				// The first and the last 64 bytes (reads of write-combined memory are slow).
				const uint64_t head = std::min<uint64_t>(range, 64u);
				r.bytes = FpMix(FpBytes(mapped.data() + offset, head), FpBytes(mapped.data() + offset + range - head, head));
			} else {
				r.bytes = ~offset;
			}
		} else {
			r.buffer = reinterpret_cast<uint64_t>(static_cast<VkBuffer>(buffer));
			r.offset = offset;
			r.range  = range;
			if (address != 0 && size != 0 && buffer) {
				const auto id = cache.LookupBuffer(address, size);
				r.stale       = id && cache.GetBuffer(id).Handle() != buffer;
			}
		}
		p.bufs.push_back(r);
	};
	auto& f = p.f;
	for (const auto* stage: stages) {
		const auto& st = *stage;
		EXIT_IF(st.buffers.size() < st.buffer_sources.size());
		for (size_t i = 0; i < st.buffer_sources.size(); i++) {
			const auto& b = st.buffers[i];
			add(CfBuffers, st.buffer_sources[i].address, st.buffer_sources[i].size, b.buffer, b.offset, b.range);
		}
		add(CfBuffers, 0, 0, st.gds.buffer, st.gds.offset, st.gds.range);
		add(CfBuffers, 0, 0, st.shared_memory.buffer, st.shared_memory.offset, st.shared_memory.range);
		for (const auto& image: st.images) {
			f[CfImages] = FpMix(f[CfImages], (uint64_t {image.image_id.generation} << 32u) | image.image_id.index);
			f[CfImages] = FpMix(f[CfImages], reinterpret_cast<uint64_t>(static_cast<VkImageView>(image.image_view)));
			for (const auto view: image.mip_views) f[CfImages] = FpMix(f[CfImages], reinterpret_cast<uint64_t>(static_cast<VkImageView>(view)));
		}
		for (const auto sampler: st.samplers) f[CfImages] = FpMix(f[CfImages], reinterpret_cast<uint64_t>(static_cast<VkSampler>(sampler)));
		if (st.runtime != nullptr && *st.runtime) {
			const auto& flat = st.runtime->resources->flattened_srt;
			f[CfSrt] = FpMix(f[CfSrt], FpBytes(reinterpret_cast<const uint8_t*>(flat.data()), flat.size() * sizeof(uint32_t)));
		}
		for (const auto& patch: st.bindless_patches) f[CfSrt] = FpMix(FpMix(FpMix(f[CfSrt], patch.offset), patch.region), patch.entries);
		f[CfSrt]  = FpMix(f[CfSrt], st.gpu_words_key);
		f[CfData] = FpMix(f[CfData], FpBytes(reinterpret_cast<const uint8_t*>(st.shader_data.data()), st.shader_data.size() * sizeof(uint32_t)));
		f[CfBindless] = FpMix(f[CfBindless], st.bindless_images.size());
		for (const auto id: st.bindless_images) {
			p.bindless.push_back((uint64_t {id.generation} << 32u) | id.index);
			f[CfBindless] = FpMix(f[CfBindless], p.bindless.back());
		}
	}
	if (plan != nullptr && !plan->mesh_active) {
		const auto& vin = plan->state->vertex_info[0];
		const auto& vb  = plan->vertex_bindings;
		f[CfVertex]     = FpMix(FpMix(f[CfVertex], static_cast<uint64_t>(vin.buffers_num)), vb.count);
		for (uint32_t i = 0; i < vb.count; i++) {
			const uint64_t address = static_cast<int>(i) < vin.buffers_num ? vin.buffers[i].addr : 0u;
			add(CfVertex, address, vb.sizes[i], vb.buffers[i], vb.offsets[i], vb.sizes[i]);
		}
		const auto& ib = plan->index_binding;
		f[CfVertex]    = FpMix(f[CfVertex], static_cast<uint64_t>(ib.type));
		if (ib.buffer) add(CfVertex, plan->index_source.address, plan->index_source.size, ib.buffer, ib.offset, plan->index_source.size);
	}
}

// The images fault: two image views (or two samplers) of the stage swapped; false when the stage has no two distinct ones.
bool ChunkSwapImages(PreparedBindings& st) {
	for (size_t j = 1; j < st.images.size(); j++) {
		if (st.images[j].image_view != st.images[0].image_view) {
			std::swap(st.images[0].image_view, st.images[j].image_view);
			return true;
		}
	}
	for (size_t j = 1; j < st.samplers.size(); j++) {
		if (st.samplers[j] != st.samplers[0]) {
			std::swap(st.samplers[0], st.samplers[j]);
			return true;
		}
	}
	return false;
}

// KYTY_CHECK_CHUNKS_FAULT: one error in the first preparation's data (its own bindings; the commit uses the second's):
// the next field in turn this member has data for.
void ChunkInjectFault(ChunkCheckState& s, std::span<PreparedBindings* const> stages, DrawPlan* plan) {
	size_t stage_records = 0;
	for (const auto* st: stages) stage_records += st->buffer_sources.size() + 2u;
	for (uint32_t tries = 0; tries < CfCount && s.fault_field < 0; tries++) {
		const uint32_t field = s.fault_turn++ % CfCount;
		size_t         rec   = 0;
		for (auto* stage: stages) {
			auto& st = *stage;
			if (field == CfBuffers && !st.buffer_sources.empty() && st.buffers[0].buffer) {
				st.buffers[0].offset += 256u;
				s.fault_rec = rec;
			} else if (field == CfImages && ChunkSwapImages(st)) {
			} else if (field == CfSrt && st.gpu_words_key != 0) {
				st.gpu_words_key ^= 1u; // the GPU-read words as the shader reaches them
			} else if (field == CfSrt && !st.bindless_patches.empty()) {
				st.bindless_patches[0].entries += 1u;
			} else if (field == CfData && !st.shader_data.empty()) {
				st.shader_data.back() ^= 0x10u;
			} else if (field == CfBindless && !st.bindless_images.empty()) {
				st.bindless_images[0].generation += 1u;
			} else {
				rec += st.buffer_sources.size() + 2u;
				continue;
			}
			s.fault_field = static_cast<int>(field);
			break;
		}
		if (s.fault_field < 0 && field == CfVertex && plan != nullptr && !plan->mesh_active && plan->vertex_bindings.count != 0) {
			plan->vertex_bindings.offsets[0] += 256u;
			s.fault_rec   = stage_records;
			s.fault_field = static_cast<int>(field);
		}
	}
}

void ChunkCompare(ChunkCheckState& s, bool dispatch, std::span<PreparedBindings* const> stages) {
	(dispatch ? s.dispatches : s.draws)++;
	std::array<bool, CfCount>   bad {};
	std::array<size_t, CfCount> bad_rec {};
	bad_rec.fill(SIZE_MAX);
	for (uint32_t i = 0; i < CfCount; i++) bad[i] = s.first.f[i] != s.second.f[i];
	const auto& a           = s.first.bufs;
	const auto& b           = s.second.bufs;
	bool        fault_mixed = false;
	if (a.size() != b.size()) {
		bad[CfBuffers] = true;
	} else {
		for (size_t i = 0; i < a.size(); i++) {
			const bool same_range = a[i].field == b[i].field && a[i].address == b[i].address && a[i].size == b[i].size;
			if (same_range && a[i].kind != b[i].kind) {
				s.mixed++;
				if (i == s.fault_rec) fault_mixed = true;
				continue;
			}
			// A stale binding (a replaced buffer) is a difference in either preparation (the buffer registry epoch guard).
			if (a[i].Same(b[i]) && !a[i].stale && !b[i].stale) continue;
			bad[a[i].field] = true;
			if (bad_rec[a[i].field] == SIZE_MAX && i != s.fault_rec) bad_rec[a[i].field] = i;
		}
	}
	if (s.fault_field >= 0) {
		const auto f = static_cast<uint32_t>(s.fault_field);
		s.injected[f]++;
		if (bad[f]) {
			s.detected[f]++;
		} else if (fault_mixed) {
			s.injected_mixed[f]++;
		}
		bad[f] = bad_rec[f] != SIZE_MAX; // the injected error is not a difference; another binding of its field still is
	}
	// The bindless images are not a worker's result (the tables stay at the commit, plan 9.11 ~20:45; the first
	// preparation's table state - in-draw marks, the once-per-draw refresh - is what the second sees): counted apart.
	if (bad[CfBindless]) {
		s.bindless_info++;
		bad[CfBindless] = false;
	}
	bool any = false;
	for (uint32_t i = 0; i < CfCount; i++) {
		if (!bad[i]) continue;
		s.mismatches[i]++;
		any = true;
		if (s.logged >= 40u) continue;
		s.logged++;
		const auto hash = [&](size_t k) {
			return static_cast<unsigned long long>(k < stages.size() && stages[k]->runtime != nullptr && stages[k]->runtime->program != nullptr
			                                           ? stages[k]->runtime->program->shader_hash
			                                           : 0u);
		};
		std::fprintf(stderr, "CHUNK_CHECK mismatch field=%s op=%s frame=%u hash0=%016llx hash1=%016llx marker=[%s]", ChunkFieldNames[i],
		             dispatch ? "dispatch" : "draw", DebugPresentCount().load(), hash(0), hash(1), DebugMarkerPath().c_str());
		if (const auto r = bad_rec[i]; r < a.size() && r < b.size()) {
			std::fprintf(stderr, " rec=%zu/%zu guest=%llx+%llx/%llx+%llx kind=%u/%u buffer=%llx/%llx offset=%llx/%llx range=%llx/%llx", r,
			             a.size(), static_cast<unsigned long long>(a[r].address), static_cast<unsigned long long>(a[r].size),
			             static_cast<unsigned long long>(b[r].address), static_cast<unsigned long long>(b[r].size), a[r].kind, b[r].kind,
			             static_cast<unsigned long long>(a[r].buffer), static_cast<unsigned long long>(b[r].buffer),
			             static_cast<unsigned long long>(a[r].offset), static_cast<unsigned long long>(b[r].offset),
			             static_cast<unsigned long long>(a[r].range), static_cast<unsigned long long>(b[r].range));
		} else if (i == CfBuffers && a.size() != b.size()) {
			std::fprintf(stderr, " records=%zu/%zu", a.size(), b.size());
		} else if (i == CfBindless) {
			const auto& x = s.first.bindless;
			const auto& y = s.second.bindless;
			size_t      k = 0;
			while (k < x.size() && k < y.size() && x[k] == y[k]) k++;
			std::fprintf(stderr, " images=%zu/%zu first_diff=%zu id=%llx/%llx", x.size(), y.size(), k,
			             static_cast<unsigned long long>(k < x.size() ? x[k] : ~0ull),
			             static_cast<unsigned long long>(k < y.size() ? y[k] : ~0ull));
		}
		std::fprintf(stderr, "\n");
	}
	if (any) s.bad++;
	const auto now = std::chrono::steady_clock::now();
	if (now - s.window < std::chrono::seconds(5)) return;
	std::string text;
	for (uint32_t i = 0; i < CfCount; i++) {
		text += std::string(i == 0 ? "" : " ") + ChunkFieldNames[i] + "=" + std::to_string(s.mismatches[i]);
	}
	if (ChunkFaultEvery() != 0) {
		// Per field: injected/detected/injected into a mixed binding (not detectable: the two preparations' buffers differ).
		text += "] fault[";
		for (uint32_t i = 0; i < CfCount; i++) {
			text += std::string(i == 0 ? "" : " ") + ChunkFieldNames[i] + "=" + std::to_string(s.injected[i]) + "/" +
			        std::to_string(s.detected[i]) + "/" + std::to_string(s.injected_mixed[i]);
		}
	}
	std::fprintf(stderr,
	             "PERF chunk_check draws=%llu dispatches=%llu bad=%llu mixed=%llu bindless_info=%llu mismatches[%s]\n",
	             static_cast<unsigned long long>(s.draws), static_cast<unsigned long long>(s.dispatches),
	             static_cast<unsigned long long>(s.bad), static_cast<unsigned long long>(s.mixed),
	             static_cast<unsigned long long>(s.bindless_info), text.c_str());
	s.window = now;
}
} // namespace

PreparedBindings& RenderExecutor::ChunkScratchBindings(uint32_t index) {
	static std::array<PreparedBindings, 5> scratch; // execution thread
	return scratch.at(index);
}

void RenderExecutor::ChunkCheckFirst(std::span<PreparedBindings* const> stages, DrawPlan* plan) {
	auto& s       = g_chunk_check;
	s.fault_field = -1;
	s.fault_rec   = SIZE_MAX;
	if (const uint64_t every = ChunkFaultEvery(); every != 0 && ++s.members % every == 0) ChunkInjectFault(s, stages, plan);
	ChunkPrintOf(m_context.GetBufferCache(), s.first, stages, plan);
	s.first_stages.assign(stages.begin(), stages.end());
}

void RenderExecutor::ChunkCheckSecond(std::span<PreparedBindings* const> stages, const DrawPlan* plan, bool dispatch) {
	auto& s = g_chunk_check;
	ChunkPrintOf(m_context.GetBufferCache(), s.second, stages, plan);
	ChunkCompare(s, dispatch, stages);
	// A table lists only the images whose layout changed since it last listed them (ListTableImagesByEvents): the first
	// preparation took those events, so the committed stage also gets the images the first one listed (the commit's
	// layout transitions are repeatable; it gets at least what a run without the oracle gets).
	for (size_t k = 0; k < stages.size() && k < s.first_stages.size(); k++) {
		auto&        committed = stages[k]->bindless_images;
		const auto&  first     = s.first_stages[k]->bindless_images;
		const size_t before    = committed.size();
		for (const auto id: first) {
			// Without the duplicate check above a few images (the search is linear; a repeated transition is harmless).
			if (before > 64u || first.size() > 64u || std::ranges::find(committed, id) == committed.end()) committed.push_back(id);
		}
	}
}

// Project 15 (the buffer registry epoch guard, 9.11; found by the oracle v2: merged=): a draw takes its buffer bindings
// one after another - the stages' storage buffers (RebindBuffers), then the vertex, index and indirect buffers - and a later
// acquisition can create a buffer that joins one an earlier binding named (CreateBuffer, JoinOverlap: the old buffer is
// unregistered, its bytes copied into the new one; a read through the old one gets the same bytes, a write would be lost).
// FindBuffers registers the stages' ranges first, the vertex and index ranges it does not. So when the buffer registry
// changed while the bindings were taken (DrawRepeat::g_buffer_registry_epoch, every registration and unregistration), the
// buffer part is taken again: the next pass creates nothing (every range is registered now), its bindings name the
// registered buffers. PrepareBindings is not repeated (a table's image listing is taken once, ListTableImagesByEvents).
void RenderExecutor::PrepareDrawBindings(DrawPlan& plan, CommandBuffer& buffer, std::span<RenderColorInfo> colors,
                                         std::chrono::steady_clock::time_point* buffers_mark) {
	const ChunkLead::FloorTimer floor_draw_bind(ChunkLead::FpDrawBind); // KYTY_TRACE_PEEK_CENSUS (way B W0 A8)
	const auto stages = std::span {plan.descriptor_stages.data(), plan.stage_count};
	PrepareGraphicsBuffersHead(stages);
	PrepareGraphicsImages(stages, colors);
	const auto& epoch = DrawRepeat::g_buffer_registry_epoch;
	uint64_t    seen  = epoch.load(std::memory_order_acquire);
	struct GuardStats {
		uint64_t repeated = 0, passes = 0, unstable = 0;
		std::chrono::steady_clock::time_point window = std::chrono::steady_clock::now();
	};
	static GuardStats stats; // execution thread
	for (uint32_t pass = 0;; pass++) {
		PrepareGraphicsBuffersTail(stages);
		if (pass == 0 && buffers_mark != nullptr) *buffers_mark = std::chrono::steady_clock::now();
		if (!plan.mesh_active) {
			if (pass == 0) LogDrawPhase(plan.draw.Name(), "PrepareVertexBuffers");
			plan.vertex_bindings = {};
			AcquireVertexBuffers(buffer, plan.state->vertex_info[0], plan.vertex_bindings);
			plan.index_binding = PrepareIndexBuffer(buffer, plan.index_source);
			if (plan.indirect) {
				// GPU-written records stay in their cached buffer (no readback); CPU-written ones upload.
				auto& cache        = m_context.GetBufferCache();
				plan.indirect_args = cache.ObtainBuffer(plan.indirect->args_addr, plan.indirect->args_size, false);
				if (plan.indirect->count_addr != 0) {
					plan.indirect_count = cache.ObtainBuffer(plan.indirect->count_addr, sizeof(uint32_t), false);
				}
			}
		}
		const uint64_t now_epoch = epoch.load(std::memory_order_acquire);
		if (now_epoch == seen) break;
		seen = now_epoch;
		if (pass == 0) stats.repeated++;
		stats.passes++;
		if (pass == 3) {
			// Still changing after four passes: today's bindings of the last pass are kept (counted; expected 0).
			if (stats.unstable++ == 0) {
				std::fprintf(stderr, "BUFFER_GUARD unstable frame=%u marker=[%s]\n", DebugPresentCount().load(), DebugMarkerPath().c_str());
			}
			break;
		}
	}
	static const bool perf = std::getenv("KYTY_TRACE_PERF") != nullptr;
	if (perf) {
		const auto now = std::chrono::steady_clock::now();
		if (now - stats.window >= std::chrono::seconds(5)) {
			std::fprintf(stderr, "PERF buffer_guard repeated=%llu passes=%llu unstable=%llu\n",
			             static_cast<unsigned long long>(stats.repeated), static_cast<unsigned long long>(stats.passes),
			             static_cast<unsigned long long>(stats.unstable));
			stats.window = now;
		}
	}
}

static bool PlanNoSetHint(); // N5.0 (below)

void RenderExecutor::ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer,
                                         const DrawCallInfo& draw, DrawRenderState& state,
                                         vk::PrimitiveTopology topology, const DrawEmitInfo& emit,
                                         const DrawIndexBufferSource& index_source,
	                                     bool primitive_restart_enable,
                                         const DrawIndirectSource* indirect) {
	const ChunkLead::FloorTimer floor_execute(ChunkLead::FpExecute); // KYTY_TRACE_PEEK_CENSUS (N3.0)
	auto& ucfg = buffer.GetUserConfig();
	// KYTY_DIAG_GPU_BURN_AT=draw (measurement only, map 6.11): the per-draw sensitivity - every draw's preparation.
	if (GpuBurnSelected() == GpuBurnSite::Draw) GpuBurnAt(GpuBurnSite::Draw);
	if (CopyDepthToColor(buffer, draw, emit, state)) {
		return;
	}
	const auto vertex_stages =
	    std::span {state.vertex_info.data(), state.programs.VertexStageCount()};
	const bool mesh_active = state.vertex_info[0].stage.program->stage == ShaderType::Mesh;
	uint32_t   mesh_groups = 0;
	if (mesh_active) {
		const auto& mesh = state.vertex_info[0].mesh;
		EXIT_NOT_IMPLEMENTED(mesh.fast_launch && (draw.IsIndexed() || primitive_restart_enable));
		static std::atomic_bool restart_warned = false;
		if (primitive_restart_enable && !restart_warned.exchange(true, std::memory_order_relaxed)) {
			std::printf("Warning: primitive restart is not implemented for mesh shaders; "
			            "continuing draw (primitive=%u indexed=%u)\n",
			            static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed());
		}
		if (mesh.primitives_per_group == 0) {
			EXIT("unsupported mesh draw: primitive=%u indexed=%u restart=%u\n",
			     static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed(), primitive_restart_enable);
		}
		const auto primitives = mesh.InputPrimitiveCount(draw.index_count);
		if (primitives == 0 || draw.instance_count == 0) {
			return;
		}
		mesh_groups        = (primitives - 1u) / mesh.primitives_per_group + 1u;
		const auto& limits = m_context.GetGraphics().mesh_shader_properties;
		if (mesh_groups > limits.maxMeshWorkGroupCount[0] ||
		    draw.instance_count > limits.maxMeshWorkGroupCount[1] ||
		    static_cast<uint64_t>(mesh_groups) * draw.instance_count >
		        limits.maxMeshWorkGroupTotalCount) {
			EXIT("mesh draw exceeds host workgroup limits: %ux%u\n", mesh_groups,
			     draw.instance_count);
		}
	}

	if (mesh_active && draw.IsIndexed()) {
		// Register the original guest indices for shader reads; PrepareGraphicsBindings
		// synchronizes registered BDA ranges before any draw commands are committed.
		(void)m_context.GetBufferCache().FindBuffer(
		    index_source.address, static_cast<uint64_t>(draw.index_count) *
		                              index_source.guest_element_size);
	}
	EXIT_IF(indirect != nullptr && mesh_active);
	if (indirect != nullptr) {
		// GPU indirect draws: register the argument and count ranges before the shader
		// bindings are prepared, so acquiring them below cannot merge buffers those bindings use.
		auto& cache = m_context.GetBufferCache();
		(void)cache.FindBuffer(indirect->args_addr, indirect->args_size);
		if (indirect->count_addr != 0) {
			(void)cache.FindBuffer(indirect->count_addr, sizeof(uint32_t));
		}
	}
	// S2 step B.4d: the pipeline is looked up before the binding preparation (it reads the live registers, which the
	// preparation does not; nothing of it depends on the bindings).
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "CreatePipeline");
	}
	std::optional<DispatchSubTimer> draw_census(std::in_place, DispatchSub::DrawPipeline); // KYTY_TRACE_DRAW_CENSUS
	// KYTY_TRACE_CLEAN_DRAWS: the pipeline lookup and qualification time, without the program lookups in it (step K1).
	const bool     c0_parts     = CleanDraws::On();
	const auto     c0_pipeline  = c0_parts ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
	const uint64_t c0_programs0 = CleanDraws::t_front_parts.programs_ns;
	// KYTY_CHECK_PEEK=2 (S2c): the key from the described targets and the registers, and the pipeline found without a write.
	std::pair<PipelineCache::Pipeline*, PipelineCache::PipelineFind> peek_find {nullptr, PipelineCache::PipelineFind::Count};
	size_t                                                           peek_pipelines = 0;
	if (PeekTargets::Check() && PeekTargets::g_scratch.valid) {
		auto& sc = PeekTargets::g_scratch;
		auto& st = PeekTargets::g_stats;
		auto& pc = m_context.GetPipelineCache();
		sc.valid = false;
		st.draws++;
		const auto* ps_info   = state.ps_active ? &state.ps_input_info : nullptr;
		++t_translator;
		const auto  key_peek  = pc.MakeGraphicsPipelineKeyFrom(std::span {sc.colors, sc.count}, sc.depth, vertex_stages, buffer.GetRegisters(),
		                                                       buffer.GetUserConfig(), ps_info, topology, primitive_restart_enable,
		                                                       state.programs, true);
		peek_find             = pc.FindGraphicsPipeline(key_peek);
		--t_translator;
		const auto key_today = pc.MakeGraphicsPipelineKeyFrom(std::span {state.color_info, state.color_count}, state.depth_info, vertex_stages,
		                                                      buffer.GetRegisters(), buffer.GetUserConfig(), ps_info, topology,
		                                                      primitive_restart_enable, state.programs, false);
		if (!(key_peek == key_today)) st.key_bad++;
		CheckAheadKey(key_today);
		st.find[static_cast<size_t>(peek_find.second)]++;
		if (peek_find.second == PipelineCache::PipelineFind::Found && !mesh_active) t_op_pipeline = true; // the operation's plan
		if (t_peek_member == 1) st.member_find[static_cast<size_t>(peek_find.second)]++;
		peek_pipelines = pc.GraphicsPipelineCount();
		PeekTargets::Report();
	}
	const auto peek_compare = [&](PipelineCache::Pipeline* today) {
		if (peek_find.second != PipelineCache::PipelineFind::Found) return;
		auto& st = PeekTargets::g_stats;
		st.compared++;
		auto*      predicted = peek_find.first;
		const bool fault     = PeekTargets::FaultEvery() != 0 && st.compared % PeekTargets::FaultEvery() == 0;
		if (fault) {
			predicted = nullptr;
			st.injected++;
		}
		const bool created = m_context.GetPipelineCache().GraphicsPipelineCount() != peek_pipelines;
		if (created) st.created++;
		if (predicted != today || created) (fault ? st.detected : st.bad)++;
	};
	// A draw of today's programs whose pipeline is not created yet is not issued (StageWritesMemory, above).
	PipelineCache::Pipeline* async_pipeline = nullptr;
	// N4.3/N5: a plan with programs adopted as predicted is used (its pipeline, sets, buffer entries); not under the checks
	// that compare it with today's way (the oracle, the buffer-entry check).
	const bool use_plan = !ChunkCensus::CheckOn() && !BufferCache::PlanBufferCheck() && PlanPresent() && PlanProgramsHold();
	if (use_plan) {
		g_ahead.used_draws++;
		async_pipeline = m_context.GetPipelineCache().TakePlannedPipeline(PlanPipeline());
		if (async_pipeline != nullptr) g_ahead.used_pipeline++;
	}
	if (async_pipeline != nullptr) {
		// (taken from the plan)
	} else if (DrawWritesMemory(state)) {
		NoteAsyncDraw(false);
	} else {
		bool failed    = false; // (a worker could not create it: today's creation on this thread)
		const ChunkLead::FloorTimer floor_pipeline(ChunkLead::FpPipeline); // KYTY_TRACE_PEEK_CENSUS (N3.0)
		async_pipeline = m_context.GetPipelineCache().GetGraphicsPipelineAsync(
		    std::span {state.color_info, state.color_count}, state.depth_info, vertex_stages, buffer,
		    state.ps_active ? &state.ps_input_info : nullptr, topology, primitive_restart_enable, state.programs, &failed);
		if (async_pipeline == nullptr && !failed) {
			peek_compare(nullptr);
			NoteAsyncDraw(true);
			m_async_skipped_draws++;
			return;
		}
	}
	auto& pipeline = async_pipeline != nullptr
	                     ? *async_pipeline
	                     : m_context.GetPipelineCache().GetGraphicsPipeline(
	                           std::span {state.color_info, state.color_count}, state.depth_info, vertex_stages, buffer,
	                           state.ps_active ? &state.ps_input_info : nullptr, topology, primitive_restart_enable,
	                           state.programs);
	peek_compare(&pipeline);
	if (c0_parts) {
		const auto total = static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - c0_pipeline).count());
		const uint64_t programs = CleanDraws::t_front_parts.programs_ns - c0_programs0;
		CleanDraws::t_front_parts.pipeline_ns += total > programs ? total - programs : 0u;
	}
	draw_census.emplace(DispatchSub::DrawPlanSetup);
	LogDrawPhase(draw.Name(), "PrepareBindings");
	m_context.GetBindlessTable().BeginPlan(); // S2 step B.4f
	DrawPlan& plan     = m_current_slot->plan;
	auto&     bindings = m_current_slot->bindings;
	plan.vertex_stages            = vertex_stages;
	plan.mesh_active              = mesh_active;
	plan.mesh_groups              = mesh_groups;
	plan.pipeline                 = &pipeline;
	plan.bindings                 = &bindings;
	plan.submit_id                = submit_id;
	plan.buffer                   = &buffer;
	plan.draw                     = draw;
	plan.state                    = &state;
	plan.emit                     = emit;
	plan.index_source             = index_source;
	if (indirect != nullptr) {
		plan.indirect = *indirect;
	} else {
		plan.indirect.reset();
	}
	plan.descriptor_stages = {};
	plan.stage_count       = 0;
	plan.vertex_bindings   = {};
	plan.index_binding     = {};
	plan.indirect_args     = {};
	plan.indirect_count    = {};
	if (CleanDraws::On()) C0::Plan(plan, buffer, state, vertex_stages, &pipeline); // KYTY_TRACE_CLEAN_DRAWS
	// KYTY_CHECK_CHUNKS=2 (project 15, the oracle v2): a chunk-member draw is prepared once more, into bindings of its own,
	// before its preparation below; the two preparations' prints must be equal (the commit uses the second).
	const bool chunk_check = ChunkCensus::CheckOn() && ChunkCensus::t_member && !mesh_active;
	if (chunk_check) {
		// N4.2a: with a plan (PeekAhead, programs predicted right), the first preparation takes the plan's table sets and
		// pipeline; without one, or with programs adopted otherwise (the plan rejected), today's.
		auto&      a        = g_ahead;
		const bool has_plan = PlanPresent();
		const bool use_plan = has_plan && PlanProgramsHold();
		if (!has_plan) {
			a.no_plan++;
		} else if (!use_plan) {
			a.rejected++;
		} else {
			a.plan_draws++;
			if (auto* planned = PlanPipeline(); planned != nullptr && planned != &pipeline) a.pipeline_bad++;
		}
		const auto prepare_first = [&](const ShaderStageRuntime& stage, PreparedBindings& out) {
			m_plan_set_version = use_plan ? PlanSetVersion(stage.program) : 0u;
			PrepareBindings(stage, out);
			m_plan_set_version = 0;
		};
		for (uint32_t i = 0; i < vertex_stages.size(); i++) {
			prepare_first(state.vertex_info[i].stage, ChunkScratchBindings(i));
			plan.descriptor_stages[plan.stage_count++] = &ChunkScratchBindings(i);
		}
		if (state.ps_active) {
			prepare_first(state.ps_input_info.stage, ChunkScratchBindings(3));
			plan.descriptor_stages[plan.stage_count++] = &ChunkScratchBindings(3);
		}
		PrepareDrawBindings(plan, buffer, std::span {state.color_info, state.color_count}, nullptr);
		ChunkCheckFirst(std::span {plan.descriptor_stages.data(), plan.stage_count}, &plan);
		plan.descriptor_stages = {};
		plan.stage_count       = 0;
		plan.vertex_bindings   = {};
		plan.index_binding     = {};
		plan.indirect_args     = {};
		plan.indirect_count    = {};
		m_context.GetBindlessTable().BeginPlan();
	}
	// N4.2b (KYTY_CHECK_PLAN_BUFFERS=2): the plan's buffer entries checked by the acquisitions below.
	struct PlanBuffersScope {
		bool on;
		~PlanBuffersScope() {
			if (on) BufferCache::SetPlanBuffers(nullptr);
		}
	} const plan_buffers_scope {BufferCache::PlanBufferCheck() || use_plan};
	if (plan_buffers_scope.on) PlanBuffersBegin();
	m_plan_set_version = 0;
	// The binding preparation.
	{
		const uint32_t long_us = LongPrepThresholdUs(); // KYTY_TRACE_LONG_PREP
		std::array<std::chrono::steady_clock::time_point, 5> marks {};
		if (long_us != 0) {
			t_long_prep_parts = {};
			marks[0]          = std::chrono::steady_clock::now();
		}
		{
			for (uint32_t i = 0; i < vertex_stages.size(); i++) {
				m_plan_set_version = use_plan && !PlanNoSetHint() ? PlanSetVersion(state.vertex_info[i].stage.program) : 0u; // N4.3
				PrepareBindings(state.vertex_info[i].stage, bindings.vertex[i]);
				m_plan_set_version = 0;
				plan.descriptor_stages[plan.stage_count++] = &bindings.vertex[i];
			}
			if (long_us != 0) marks[1] = std::chrono::steady_clock::now();
			if (state.ps_active) {
				if (!bindings.pixel) bindings.pixel.emplace();
				m_plan_set_version = use_plan && !PlanNoSetHint() ? PlanSetVersion(state.ps_input_info.stage.program) : 0u; // N4.3
				PrepareBindings(state.ps_input_info.stage, *bindings.pixel);
				m_plan_set_version = 0;
				plan.descriptor_stages[plan.stage_count++] = &*bindings.pixel;
			}
			if (long_us != 0) marks[2] = std::chrono::steady_clock::now();
			PrepareDrawBindings(plan, buffer, std::span {state.color_info, state.color_count}, long_us != 0 ? &marks[3] : nullptr);
		}
		if (long_us != 0) marks[4] = std::chrono::steady_clock::now();
		if (long_us != 0) {
			// Every draw's parts summed; about every 5 s their mean per draw (us) as one PERF prep_parts line.
			static std::array<uint64_t, 37> sum {};
			static uint64_t                 draws = 0;
			static auto                     window = std::chrono::steady_clock::now();
			for (size_t i = 0; i < 33; i++) sum[i] += t_long_prep_parts[i];
			for (size_t i = 0; i < 4; i++)
				sum[33 + i] += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(marks[i + 1] - marks[i]).count());
			draws++;
			if (marks[4] - window >= std::chrono::seconds(5)) {
				const double n = static_cast<double>(draws) * 1e3;
				std::fprintf(stderr, "PERF prep_parts draws=%llu vs=%.2f ps=%.2f graphics=%.2f vtx=%.2f |",
				             static_cast<unsigned long long>(draws), sum[33] / n, sum[34] / n, sum[35] / n, sum[36] / n);
				for (size_t i = 1; i < 33; i++) {
					if (i < 26 || i > 29) std::fprintf(stderr, " %zu=%.3f", i, sum[i] / n);
				}
				std::fprintf(stderr, "\n");
				sum = {}; draws = 0; window = marks[4];
			}
		}
		if (long_us != 0) {
			const auto us   = [&](size_t a, size_t b) {
                return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(marks[b] - marks[a]).count()) / 1e3;
			};
			static std::atomic<uint32_t> lines {0};
			static const uint32_t from = [] { // KYTY_TRACE_LONG_PREP_FROM: the first frame to report (default 1100)
				const char* value = std::getenv("KYTY_TRACE_LONG_PREP_FROM");
				return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 1100u;
			}();
			if (us(0, 4) >= static_cast<double>(long_us) && DebugPresentCount().load() >= from &&
			    lines.fetch_add(1, std::memory_order_relaxed) < 300u) {
				const auto& pp = t_long_prep_parts;
				const auto  pu = [&](size_t i) { return static_cast<double>(pp[i]) / 1e3; };
				std::fprintf(stderr, "LONG_PREP_PARTS head=%.1f set=%.1f images=%.1f samplers=%.1f store=%.1f bindless=%.1f data=%.1f find=%.1f bda=%.1f rebind_images=%.1f targets=%.1f rebind_buffers=%.1f "
				             "| bl head=%.1f requests=%.1f tables_frame=%.1f table_uses=%.1f heap=%.1f refresh=%.1f collect=%.1f tail=%.1f "
				             "| bda fallback=%.1f checks=%.1f publish=%.1f regions=%.1f epoch=%.1f n_regions=%llu buffers=%llu uploads=%llu bytes=%llu "
				             "| tbl read=%.1f translate=%.1f refresh=%.1f n_read=%llu n_translated=%llu n_refreshed=%llu "
				             "| why new=%llu covered=%llu fields=%llu dead=%llu hash=%llu cov=%llu cov_before=%llu\n",
				             pu(1), pu(2), pu(3), pu(4), pu(5), pu(6), pu(7), pu(8), pu(9), pu(10), pu(11), pu(12), pu(13), pu(14),
				             pu(15), pu(16), pu(17), pu(18), pu(19), pu(20), pu(21), pu(22), pu(23), pu(24), pu(25),
				             static_cast<unsigned long long>(pp[26]), static_cast<unsigned long long>(pp[27]),
				             static_cast<unsigned long long>(pp[28]), static_cast<unsigned long long>(pp[29]), pu(30), pu(31),
				             pu(32), static_cast<unsigned long long>(pp[33]), static_cast<unsigned long long>(pp[34]),
				             static_cast<unsigned long long>(pp[35]), static_cast<unsigned long long>(pp[36]),
				             static_cast<unsigned long long>(pp[37]), static_cast<unsigned long long>(pp[38]),
				             static_cast<unsigned long long>(pp[39]), static_cast<unsigned long long>(pp[40]),
				             static_cast<unsigned long long>(pp[41]), static_cast<unsigned long long>(pp[42]));
				std::fprintf(stderr, "LONG_PREP total=%.1f vs=%.1f ps=%.1f graphics=%.1f vtx=%.1f vs_hash=%016llx ps_hash=%016llx frame=%u marker=[%s]\n",
				             us(0, 4), us(0, 1), us(1, 2), us(2, 3), us(3, 4),
				             static_cast<unsigned long long>(vertex_stages.empty() ? 0 : vertex_stages[0].stage.program->shader_hash),
				             static_cast<unsigned long long>(state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0),
				             DebugPresentCount().load(), DebugMarkerPath().c_str());
			}
		}
		plan.bindless_heaps_end  = m_context.GetBindlessTable().DrawHeapCount(); // S2 step B.4f
		plan.bindless_tables_end = m_context.GetBindlessTable().DrawTableCount();
	}
	if (chunk_check) ChunkCheckSecond(std::span {plan.descriptor_stages.data(), plan.stage_count}, &plan, false);
	CommitPreparedDraw(plan);
}

// The second half of a draw: records what its preparation left in the plan (renderDrawPlan.h). With the render mutex held.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void RenderExecutor::CommitPreparedDraw(DrawPlan& plan) {
	const ChunkLead::FloorTimer floor_commit(ChunkLead::FpCommit); // KYTY_TRACE_PEEK_CENSUS (N3.0)
	const auto                submit_id                = plan.submit_id;
	CommandBuffer&            buffer                   = *plan.buffer;
	const DrawCallInfo&       draw                     = plan.draw;
	DrawRenderState&          state                    = *plan.state;
	const DrawEmitInfo&       emit                     = plan.emit;
	const auto&               index_source             = plan.index_source;
	const DrawIndirectSource* indirect                 = plan.indirect ? &*plan.indirect : nullptr;
	plan.regs.Capture(buffer);
	const auto  vertex_stages    = plan.vertex_stages;
	const bool  mesh_active      = plan.mesh_active;
	const auto  mesh_groups      = plan.mesh_groups;
	auto&       bindings         = *plan.bindings;
	const auto  stages           = std::span {plan.descriptor_stages.data(), plan.stage_count};
	auto&       vertex_bindings  = plan.vertex_bindings;
	auto&       index_binding    = plan.index_binding;
	auto&       indirect_args    = plan.indirect_args;
	auto&       indirect_count   = plan.indirect_count;
	auto&       pipeline         = *plan.pipeline;
	std::optional<C0::CommitScope> c0_commit; // KYTY_TRACE_CLEAN_DRAWS: the draw's kind and its commit's time
	if (CleanDraws::On()) c0_commit.emplace(C0::Commit(plan, stages, buffer.Handle()));
	std::optional<DispatchSubTimer> commit_census(std::in_place, DispatchSub::CAcquire); // KYTY_TRACE_DRAW_CENSUS
	std::optional<ChunkLead::FloorTimer> floor_acquire(std::in_place, ChunkLead::FpAcquire); // KYTY_TRACE_PEEK_CENSUS (W0 A12)
	ApplyPlanFlags(); // the plan flags: the draw's image flags, before the targets read them
	// S1.2: the render targets are acquired in order (GPU-written marks, tracking, layout transitions): part of the
	// commit, before its recording.
	vk::ImageAspectFlags feedback_aspects;
	// KYTY_FIX_IMAGE_MEMORY_OWNER: the colour targets' write rectangle for their acquisition (and a continued draw only
	// within what its targets already hold).
	const bool owner_on = TextureCache::OwnerMode() != 0u;
	if (owner_on) {
		int32_t    ox0 = 0, oy0 = 0, ox1 = 0, oy1 = 0;
		const bool any = OwnerDrawRect(vertex_stages.back(), state.color_info, state.color_count, plan.regs, ox0, oy0, ox1, oy1);
		TextureCache::OwnerSetDrawRect(any, ox0, oy0, ox1, oy1);
	}
	RenderState rendering = AcquireRenderTargets(buffer, state.color_info, state.color_count, state.depth_info, feedback_aspects, stages);
	if (owner_on) TextureCache::OwnerSetDrawRect(false, 0, 0, 0, 0);
	commit_census.reset();
	floor_acquire.reset();
	// Resource preparation (ExecutePreparedDraw) may synchronously finish and restart the scheduler. From this
	// point onward, every operation targets the current command buffer and cannot touch guest memory.
	auto vk_buffer = buffer.Handle();
	SetDrawDebugPhase(buffer, submit_id, draw, draw.IsIndexed() ? 0x100u : 0x200u);
	if (!mesh_active) {
		const DispatchSubTimer vtx_census(DispatchSub::CVtx); // KYTY_TRACE_DRAW_CENSUS
		CommitVertexBuffers(vk_buffer, vertex_bindings);
	}
	if (state.ps_active && !draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x300u);
	}
	// Image dumps run before the bindings are committed: the dump leaves an image in a transfer
	// layout, and CommitBindings transitions it back to the layout its descriptor names.
	for (uint32_t k = 0; k < vertex_stages.size(); k++) { // every vertex-side stage (LS, HS, DS of a tessellated draw)
		if (state.vertex_info[k].stage.program == nullptr) continue;
		DrawImageDumpTrace(m_context, vk_buffer, *state.vertex_info[k].stage.program, bindings.vertex[k],
		                   state.vertex_info[k].stage.resources);
	}
	if (state.ps_active && bindings.pixel && state.ps_input_info.stage.program != nullptr) {
		DrawImageDumpTrace(m_context, vk_buffer, *state.ps_input_info.stage.program, *bindings.pixel,
		                   state.ps_input_info.stage.resources);
	}
	{
		const DispatchSubTimer bind_census(DispatchSub::CBind); // KYTY_TRACE_DRAW_CENSUS
		const CommitTargets    commit_targets {state.color_info, state.color_count, &state.depth_info}; // K5
		m_commit_targets = &commit_targets;
		CommitBindings(buffer, vk::PipelineBindPoint::eGraphics, pipeline, stages);
		m_commit_targets    = nullptr;
	}
	// The ObjPrimID write: this draw's VGT_OBJECT_ID, after CommitBindings (which pushes the whole block). The
	// stage flags are the layout's push range (fragment + every vertex stage of the pipeline).
	if (state.ps_active && state.ps_input_info.HasObjprimOutput()) {
		vk::ShaderStageFlags objprim_stages = vk::ShaderStageFlagBits::eFragment;
		for (const auto& stage: vertex_stages) objprim_stages |= NativeShaderStage(stage.logical_stage);
		const uint32_t object_id = plan.regs.object_id;
		vk_buffer.pushConstants(pipeline.pipeline_layout, objprim_stages,
		                        uint32_t {state.ps_input_info.objprim_push_dword} * 4u, sizeof(object_id),
		                        &object_id);
	}
	// Upstream feed075a9: the user clip planes (PA_CL_UCP_*) of the stage GraphicsProgramInputs gave them to (the last
	// pre-rasterization stage), four dwords each from PushData::UserClipPlaneDword (before the ObjPrimID dword and the
	// shaders' data, see GraphicsProgramInputs). Also after CommitBindings; the same stage flags as above.
	if (const auto planes = vertex_stages.empty() ? 0u : vertex_stages.back().user_clip_planes; planes != 0) {
		vk::ShaderStageFlags ucp_stages = vk::ShaderStageFlagBits::eFragment;
		for (const auto& stage: vertex_stages) ucp_stages |= NativeShaderStage(stage.logical_stage);
		vk_buffer.pushConstants(pipeline.pipeline_layout, ucp_stages,
		                        ShaderRecompiler::IR::PushData::UserClipPlaneDword * sizeof(uint32_t),
		                        std::bit_width(planes) * 4u * sizeof(uint32_t), plan.regs.user_clip_planes.data());
	}
	if (mesh_active) {
		const uint32_t draw_data[] {
		    draw.index_count,
		    draw.IsIndexed() ? static_cast<uint32_t>(emit.vertex_offset) : emit.first_vertex,
		    emit.first_instance, index_source.guest_element_size,
		    static_cast<uint32_t>(index_source.address),
		    static_cast<uint32_t>(index_source.address >> 32u)};
		static_assert(std::size(draw_data) == ShaderRecompiler::IR::PushData::MeshDrawDwordCount);
		vk_buffer.pushConstants(pipeline.pipeline_layout,
		                        vk::ShaderStageFlagBits::eMeshEXT |
		                            vk::ShaderStageFlagBits::eFragment,
		                        0, sizeof(draw_data), draw_data);
	} else {
		const DispatchSubTimer index_census(DispatchSub::CIndex); // KYTY_TRACE_DRAW_CENSUS
		CommitIndexBuffer(vk_buffer, index_binding);
	}

	commit_census.emplace(DispatchSub::CDyn); // KYTY_TRACE_DRAW_CENSUS
	{
		// The values here; their commands here or, with the record packets, on the recording thread (one packet).
		DynamicParams dyn_params;
		DrawViewports dyn_viewports;
		DrawScissors  dyn_scissors;
		MakeGraphicsDynamicParams(buffer, vertex_stages.back(), state.depth_info, rendering, plan.regs, feedback_aspects,
		                          dyn_params, dyn_viewports, dyn_scissors);
		CommitGraphicsDynamicParams(buffer, vk_buffer, dyn_params, dyn_viewports, dyn_scissors);
	}
	commit_census.reset();

	LogDrawPhase(draw.Name(), "BeginRendering");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x400u);
	}
	if (BarrierEveryCmdEnabled()) {
		m_context.GetCommandScheduler().EndRendering();
		FullMemoryBarrier(m_context.GetCommandScheduler().Current().Handle());
	}
	TraceDrawStatePs(buffer, state, draw); // KYTY_TRACE_DRAW_STATE_PS
	// KYTY_TRACE_CS_IMAGE_DUMP also lists pixel shaders: inside the trace window, color target 0 right before their first
	// two draws (psbefore_, what the draw starts from; psafter_ below is what it wrote).
	if (state.ps_active && state.color_count > 0 && state.color_info[0].image_id && DebugStencilWindow()) {
		static const std::string list = Common::EnvRead::CsImageDump();
		char text[17];
		std::snprintf(text, sizeof(text), "%016llx",
		              static_cast<unsigned long long>(state.ps_input_info.stage.program->shader_hash));
		static uint32_t dumped = 0;
		const char*     dir    = Common::EnvRead::FrameDir();
		if (dir != nullptr && !list.empty() && list.find(text) != std::string::npos && dumped < 6u) {
			buffer.EndRendering();
			auto& image = buffer.GetContext().GetTextureCache().GetImage(state.color_info[0].image_id);
			DebugQueueFullDump(m_context.GetGraphics(), vk_buffer, image,
			                   std::string(dir) + "/psbefore_" + text + "_" + std::to_string(dumped) + "_" +
			                       fmt::format("{:x}", image.info.data.address),
			                   false);
			dumped++;
		}
	}
	m_context.GetCommandScheduler().BeginRendering(rendering);
	BindGraphicsPipeline(buffer, vk_buffer, pipeline.pipeline);
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x500u);
	}
	for (uint32_t i = 0; i < state.color_count; ++i) {
		const auto address = state.color_info[i].desc.info.data.address;
		if (state.ps_active && GpuCheckpoints::IsFlipAddress(address)) {
			std::vector<GpuCheckpoints::ChainRead> reads;
			const auto& ps_program   = *state.ps_input_info.stage.program;
			const auto& ps_resources = *state.ps_input_info.stage.resources;
			for (uint32_t j = 0; j < ps_program.info.images.size() && j < ps_resources.images.size(); ++j) {
				if (ps_program.info.images[j].written) continue;
				const auto r = DecodeNativeDescriptor<ShaderTextureResource>(ps_resources.images[j]);
				reads.push_back({r.Base40(), static_cast<uint32_t>(r.Format()),
				                 static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
				                 static_cast<uint32_t>(r.TileMode())});
			}
			GpuCheckpoints::TraceFlipWriter("ps", ps_program.shader_hash, address, reads);
		}
	}
	if (GpuCheckpoints::Enabled()) {
		for (const auto& stage: vertex_stages) GpuCheckpoints::Mark(vk_buffer, stage.stage.program->shader_hash);
		if (state.ps_active) GpuCheckpoints::Mark(vk_buffer, state.ps_input_info.stage.program->shader_hash);
	}
	// KYTY_DIAG_SKIP_PS_HASHES=<hex,...> (logged APPROXIMATION): draws with a listed pixel
	// shader are not issued (bisecting GPU faults); the first few name their vertex stages.
	static constinit Common::OnceObject<std::string> skip_ps_once;
	const std::string& skip_ps = skip_ps_once.Get([] {
		const char* value = std::getenv("KYTY_DIAG_SKIP_PS_HASHES");
		return value != nullptr ? std::string(value) : std::string();
	});
	bool skip_draw = false;
	if (!skip_ps.empty() && state.ps_active) {
		char text[17];
		std::snprintf(text, sizeof(text), "%016llx",
		              static_cast<unsigned long long>(state.ps_input_info.stage.program->shader_hash));
		skip_draw = skip_ps.find(text) != std::string::npos;
		if (skip_draw) {
			static std::atomic_uint logged {0};
			if (logged.fetch_add(1) < 12u) {
				std::fprintf(stderr, "DIAG_SKIP_PS APPROXIMATION ps=0x%s mesh=%d count=%u inst=%u vs=", text,
				             mesh_active ? 1 : 0, draw.index_count, draw.instance_count);
				for (const auto& stage: vertex_stages)
					std::fprintf(stderr, "0x%016llx(stage%u) ",
					             static_cast<unsigned long long>(stage.stage.program->shader_hash),
					             static_cast<uint32_t>(stage.stage.program->stage));
				std::fprintf(stderr, "\n");
			}
		}
	}
	// KYTY_DIAG_SKIP_VS_HASHES=<hex,...> (logged APPROXIMATION): draws with a listed vertex-side stage (VS, LS, HS,
	// DS, ES/GS) are not issued (bisecting GPU faults in tessellation and geometry pipelines).
	{
		static constinit Common::OnceObject<std::string> skip_vs_once;
		const std::string& skip_vs = skip_vs_once.Get([] {
			const char* value = std::getenv("KYTY_DIAG_SKIP_VS_HASHES");
			return value != nullptr ? std::string(value) : std::string();
		});
		if (!skip_draw && !skip_vs.empty()) {
			for (const auto& stage: vertex_stages) {
				char text[17];
				std::snprintf(text, sizeof(text), "%016llx",
				              static_cast<unsigned long long>(stage.stage.program->shader_hash));
				if (skip_vs.find(text) == std::string::npos) continue;
				skip_draw = true;
				static std::atomic_uint logged {0};
				if (logged.fetch_add(1) < 12u) {
					std::fprintf(stderr, "DIAG_SKIP_VS APPROXIMATION stage=0x%s count=%u inst=%u\n", text, draw.index_count,
					             draw.instance_count);
				}
				break;
			}
		}
	}
	// KYTY_DIAG_SKIP_DRAW_MARKERS=<text|text|...> (logged APPROXIMATION): draws whose debug marker
	// path contains one of the texts are not issued (bisecting which pass draws an artifact).
	// KYTY_TRACE_DRAW_MARKERS=1: inside the trace window, the draw count per marker path each frame.
	{
		static constinit Common::OnceObject<std::vector<std::string>> skip_markers_once;
		const std::vector<std::string>& skip_markers = skip_markers_once.Get([] {
			std::vector<std::string> list;
			const char* value = std::getenv("KYTY_DIAG_SKIP_DRAW_MARKERS");
			if (value == nullptr) return list;
			std::string text(value);
			size_t      start = 0;
			while (start <= text.size()) {
				const auto end = text.find('|', start);
				auto       item = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
				if (!item.empty()) list.push_back(item);
				if (end == std::string::npos) break;
				start = end + 1;
			}
			std::fprintf(stderr, "DIAG_SKIP_DRAW_MARKERS APPROXIMATION enabled (%zu texts)\n", list.size());
			return list;
		});
		static constinit Common::OnceValue<bool> trace_markers_once;
		const bool trace_markers = trace_markers_once.Get([&] { return std::getenv("KYTY_TRACE_DRAW_MARKERS") != nullptr; });
		if (!skip_markers.empty() || trace_markers) {
			const auto path = DebugMarkerPath();
			if (!skip_draw && !skip_markers.empty()) {
				for (const auto& item: skip_markers) {
					if (path.find(item) != std::string::npos) {
						skip_draw = true;
						static std::atomic_uint logged {0};
						if (logged.fetch_add(1) < 24u) {
							std::fprintf(stderr, "DIAG_SKIP_DRAW_MARKERS skipped marker=[%s] ps=0x%016llx count=%u inst=%u\n",
							             path.c_str(),
							             static_cast<unsigned long long>(state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0ull),
							             draw.index_count, draw.instance_count);
						}
						break;
					}
				}
			}
			if (trace_markers && DebugStencilWindow()) {
				static std::mutex                               mutex;
				static std::vector<std::pair<std::string, uint32_t>> counts;
				static uint32_t                                 counted_frame = 0;
				std::scoped_lock                                lock(mutex);
				const auto                                      frame = DebugPresentCount().load();
				if (frame != counted_frame) {
					if (!counts.empty()) {
						std::string line;
						for (const auto& [name, count]: counts) line += fmt::format(" [{}]={}", name, count);
						std::fprintf(stderr, "DRAW_MARKERS frame=%u%s\n", counted_frame, line.c_str());
					}
					counts.clear();
					counted_frame = frame;
				}
				auto it = std::find_if(counts.begin(), counts.end(), [&](const auto& e) { return e.first == path; });
				if (it != counts.end()) it->second++;
				else if (counts.size() < 200u) counts.emplace_back(path, 1u);
			}
		}
	}
	// A submit between the command-buffer capture and here (e.g. a guest-memory fault in a debug
	// hook) would record the draw into a finished buffer, where it never runs.
	if (buffer.Handle() != vk_buffer) {
		static std::atomic_uint stale_logged {0};
		if (stale_logged.fetch_add(1) < 32u) {
			std::fprintf(stderr, "STALE_CMD frame=%u ps=0x%016llx marker=[%s]\n", DebugPresentCount().load(),
			             static_cast<unsigned long long>(state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0ull),
			             DebugMarkerPath().c_str());
		}
	}
	// KYTY_FIX_OCCLUSION_QUERIES: while a guest occlusion query is open, a host query counts the samples of this
	// render pass instance (begun at its first guest draw; the emulator's own blits and copies are never counted).
	auto& occlusion = m_context.GetOcclusionQueries();
	if (!skip_draw && buffer.IsRendering()) {
		occlusion.BeforeDraw(buffer.Handle());
	}
	const auto draw_cmd = buffer.Handle();
	// KYTY_TRACE_GPU_TIME=1: GPU time of this draw, keyed by its vertex and pixel shaders.
	const bool draw_label = !skip_draw && GpuTimer::LabelsEnabled();
	if (draw_label) {
		GpuTimer::LabelBegin(draw_cmd, 'd', vertex_stages.empty() ? 0ull : vertex_stages[0].stage.program->shader_hash,
		                     state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0ull);
	}
	const uint32_t gpu_time = skip_draw ? ~0u : GpuTimer::BeginCommand(draw_cmd);
	if (skip_draw) {
	} else if (mesh_active) {
		draw_cmd.drawMeshTasksEXT(mesh_groups, draw.instance_count, 1);
	} else if (indirect != nullptr) {
		EmitIndirectDraw(draw_cmd, draw, *indirect, indirect_args, indirect_count);
	} else {
		EmitDrawPrimitives(plan.regs, draw_cmd, draw, emit);
	}
	if (!skip_draw) { // upstream 6aa3846fb: the performance overlay's draw counters
		Common::DebugCounters::Add(Common::DebugCounters::Counter::Draws);
		Common::DebugCounters::Add(Common::DebugCounters::Counter::DrawInstances, draw.instance_count);
	}
	if (draw_label) GpuTimer::LabelEnd(draw_cmd);
	if (!skip_draw) m_context.GetBufferCache().NoteCommandEmitted();
	if (gpu_time != ~0u) {
		GpuTimer::EndCommand(draw_cmd, gpu_time, 'd', vertex_stages.empty() ? 0ull : vertex_stages[0].stage.program->shader_hash,
		                     state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0ull);
	}
	// KYTY_TRACE_DEPTH_ADDR=<hex>: draws using that depth target (first 60, then a count per arm).
	static constinit Common::OnceValue<uint64_t> depth_trace_once;
	const uint64_t depth_trace = depth_trace_once.Get([] {
		const char* value = std::getenv("KYTY_TRACE_DEPTH_ADDR");
		return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
	});
	// Every distinct depth target / write combination, once (KYTY_TRACE_DEPTH_ADDR set).
	if (depth_trace != 0 && state.depth_info.image_id) {
		static std::mutex                                  mutex;
		static std::vector<std::pair<uint64_t, uint32_t>> seen;
		const auto address = state.depth_info.desc.info.data.address;
		const auto key     = (state.depth_info.depth_write_enable ? 1u : 0u) |
		                     (state.depth_info.depth_test_enable ? 2u : 0u) |
		                     (state.color_count << 2u) | (state.ps_active ? 64u : 0u);
		std::scoped_lock lock(mutex);
		if (std::ranges::find(seen, std::make_pair(address, key)) == seen.end() && seen.size() < 96u) {
			seen.emplace_back(address, key);
			std::fprintf(stderr,
			             "DEPTH_TARGET addr=0x%llx extent=%ux%u write=%d test=%d colors=%u ps=0x%016llx id=%u\n",
			             static_cast<unsigned long long>(address), state.depth_info.desc.info.extent.width,
			             state.depth_info.desc.info.extent.height, state.depth_info.depth_write_enable ? 1 : 0,
			             state.depth_info.depth_test_enable ? 1 : 0, state.color_count,
			             static_cast<unsigned long long>(
			                 state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0ull),
			             state.depth_info.image_id.index);
		}
	}
	if (depth_trace != 0 && state.depth_info.desc.info.data.address == depth_trace) {
		static std::atomic_uint logged {0};
		static std::atomic_uint since_arm {0};
		static std::atomic_uint arm_seen {0};
		static std::atomic_uint writes_since_arm {0};
		static std::atomic_uint clears_since_arm {0};
		const auto arm = DebugArm().load();
		if (arm_seen.exchange(arm) != arm) {
			std::fprintf(stderr, "DEPTH_DRAWS arm=%u draws_since_last=%u writes=%u clears=%u\n", arm,
			             since_arm.exchange(0), writes_since_arm.exchange(0), clears_since_arm.exchange(0));
		}
		since_arm.fetch_add(1);
		if (state.depth_info.depth_write_enable) writes_since_arm.fetch_add(1);
		if (state.depth_info.depth_clear_enable) clears_since_arm.fetch_add(1);
		{
			static std::atomic_uint last_present {0};
			static std::atomic_uint frame_draws {0};
			static std::atomic_uint frame_lines {0};
			const auto              present = DebugPresentCount().load();
			if (last_present.exchange(present) != present) {
				const auto draws = frame_draws.exchange(0);
				if (draws != 0 && frame_lines.fetch_add(1) < 3000u) {
					std::fprintf(stderr, "DEPTH_FRAME present=%u draws=%u\n", present, draws);
				}
			}
			frame_draws.fetch_add(1);
		}
		if (logged.fetch_add(1) < 600u) {
			std::fprintf(stderr,
			             "DEPTH_DRAW vs=0x%016llx ps=0x%016llx write=%d test=%d op=%d colors=%u count=%u inst=%u "
			             "extent=%ux%u stencil test=%d clear=%d pass=%d ref=0x%02x wmask=0x%02x dbsc_other=0x%08x "
			             "z_format=0x%x bias=%d/%d off=%g/%g scale=%g/%g clamp=%g float=%d negbits=%d cull=%d/%d "
			             "layers=%u+%u vs_out_cntl=0x%08x marker=[%s]\n",
			             static_cast<unsigned long long>(vertex_stages.front().stage.program->shader_hash),
			             static_cast<unsigned long long>(state.ps_active ? state.ps_input_info.stage.program->shader_hash
			                                                             : 0ull),
			             state.depth_info.depth_write_enable ? 1 : 0, state.depth_info.depth_test_enable ? 1 : 0,
			             static_cast<int>(state.depth_info.depth_compare_op), state.color_count, draw.index_count,
			             draw.instance_count, state.depth_info.desc.info.extent.width,
			             state.depth_info.desc.info.extent.height, state.depth_info.stencil_test_enable ? 1 : 0,
			             state.depth_info.stencil_clear_enable ? 1 : 0,
			             static_cast<int>(state.depth_info.stencil_front.passOp),
			             state.depth_info.stencil_front.reference, state.depth_info.stencil_front.writeMask,
			             buffer.GetRegisters().GetShaderRegisters().db_shader_control.other_bits,
			             buffer.GetRegisters().GetShaderRegisters().shader_z_format,
			             buffer.GetRegisters().GetModeControl().poly_offset_front_enable ? 1 : 0,
			             buffer.GetRegisters().GetModeControl().poly_offset_back_enable ? 1 : 0,
			             buffer.GetRegisters().GetPolyOffset().front_offset,
			             buffer.GetRegisters().GetPolyOffset().back_offset,
			             buffer.GetRegisters().GetPolyOffset().front_scale,
			             buffer.GetRegisters().GetPolyOffset().back_scale, buffer.GetRegisters().GetPolyOffset().clamp,
			             buffer.GetRegisters().GetPolyOffset().db_is_float_fmt ? 1 : 0,
			             static_cast<int>(buffer.GetRegisters().GetPolyOffset().neg_num_db_bits),
			             buffer.GetRegisters().GetModeControl().cull_front ? 1 : 0,
			             buffer.GetRegisters().GetModeControl().cull_back ? 1 : 0,
			             state.depth_info.desc.view_info.base_layer, state.depth_info.desc.view_info.layer_count,
			             vertex_stages.back().pa_cl_vs_out_cntl, DebugMarkerPath().c_str());
			const auto& regs = buffer.GetRegisters();
			const auto& vp   = regs.GetScreenViewport();
			const auto& v0   = vp.viewports[0];
			const auto& clip = regs.GetClipControl();
			std::fprintf(stderr,
			             "DEPTH_VP vte=0x%x xs=%g xo=%g ys=%g yo=%g zs=%g zo=%g zmin=%g zmax=%g vsc=(%d,%d)-(%d,%d) "
			             "screen=(%d,%d)-(%d,%d) window=(%d,%d)-(%d,%d) generic=(%d,%d)-(%d,%d) clip_disable=%d dx_clip=%d "
			             "zclip_min_off=%d zclip_max_off=%d gb=%g/%g/%g/%g\n",
			             vp.transform_control, v0.xscale, v0.xoffset, v0.yscale, v0.yoffset, v0.zscale, v0.zoffset, v0.zmin,
			             v0.zmax, v0.viewport_scissor_left, v0.viewport_scissor_top, v0.viewport_scissor_right,
			             v0.viewport_scissor_bottom, vp.screen_scissor_left, vp.screen_scissor_top, vp.screen_scissor_right,
			             vp.screen_scissor_bottom, vp.window_scissor_left, vp.window_scissor_top, vp.window_scissor_right,
			             vp.window_scissor_bottom, vp.generic_scissor_left, vp.generic_scissor_top, vp.generic_scissor_right,
			             vp.generic_scissor_bottom, clip.clip_disable ? 1 : 0, clip.dx_clip_space ? 1 : 0,
			             clip.min_z_clip_disable ? 1 : 0, clip.max_z_clip_disable ? 1 : 0, vp.guard_band_horz_clip,
			             vp.guard_band_vert_clip, vp.guard_band_horz_discard, vp.guard_band_vert_discard);
		}
	}
	// KYTY_TRACE_CS_IMAGE_DUMP also lists pixel shaders: inside the trace window, color target 0 right
	// after their first two draws (what the draw wrote).
	if (state.ps_active && state.color_count > 0 && state.color_info[0].image_id && DebugStencilWindow()) {
		static const std::string list = Common::EnvRead::CsImageDump();
		char text[17];
		std::snprintf(text, sizeof(text), "%016llx",
		              static_cast<unsigned long long>(state.ps_input_info.stage.program->shader_hash));
		static uint32_t dumped = 0;
		const char*     dir    = Common::EnvRead::FrameDir();
		if (dir != nullptr && !list.empty() && list.find(text) != std::string::npos && dumped < 6u) {
			buffer.EndRendering();
			auto& image = buffer.GetContext().GetTextureCache().GetImage(state.color_info[0].image_id);
			DebugQueueFullDump(m_context.GetGraphics(), vk_buffer, image,
			                   std::string(dir) + "/psafter_" + text + "_" + std::to_string(dumped) + "_" + fmt::format("{:x}", image.info.data.address), false);
			dumped++;
		}
	}
	// DRS_EXTENT: the main depth target's extent whenever it changes (dynamic resolution changes).
	if (state.depth_info.image_id && state.depth_info.desc.info.data.address == 0x5002870000ull) {
		static uint32_t last = 0;
		const auto      key  = state.depth_info.desc.info.extent.width * 65536u + state.depth_info.desc.info.extent.height;
		static std::atomic_uint logged {0};
		if (key != last && logged.fetch_add(1) < 2000u) {
			last = key;
			std::fprintf(stderr, "DRS_EXTENT frame=%u depth=%ux%u\n", DebugPresentCount().load(),
			             state.depth_info.desc.info.extent.width, state.depth_info.desc.info.extent.height);
		}
	}
	// Capture frame: the depth/stencil image right after the first draws that write stencil.
	if (DebugCaptureFrame().load() && state.depth_info.image_id &&
	    state.depth_info.desc.info.data.address == 0x5002870000ull) {
		static std::atomic_uint dumped {0};
		const char* dir = Common::EnvRead::FrameDir();
		const auto  n   = dumped.fetch_add(1);
		if (state.depth_info.stencil_clear_enable || state.depth_info.depth_clear_enable) {
			std::fprintf(stderr, "CAPTURE_DEPTH_CLEAR n=%u cs_index=%u depth_clear=%d stencil_clear=%d value=0x%x marker=[%s]\n",
			             n, DebugCaptureIndex().load(), state.depth_info.depth_clear_enable ? 1 : 0,
			             state.depth_info.stencil_clear_enable ? 1 : 0, state.depth_info.stencil_clear_value,
			             DebugMarkerPath().c_str());
		}
		if (dir != nullptr && n % 20u == 0u && n <= 400u) {
			buffer.EndRendering();
			auto& cache = buffer.GetContext().GetTextureCache();
			std::fprintf(stderr, "CAPTURE_STENCIL_DRAW n=%u cs_index=%u marker=[%s] ps=0x%016llx pass=%d ref=0x%02x has_stencil_view=%d\n", n,
			             DebugCaptureIndex().load(), DebugMarkerPath().c_str(),
			             static_cast<unsigned long long>(state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0ull),
			             static_cast<int>(state.depth_info.stencil_front.passOp), state.depth_info.stencil_front.reference,
			             (ImageViewOps::DepthAspectMask(state.depth_info.desc.view_info.format) &
			              vk::ImageAspectFlagBits::eStencil) ? 1 : 0);
			DebugQueueFullDump(m_context.GetGraphics(), vk_buffer, cache.GetImage(state.depth_info.image_id),
			                   std::string(dir) + "/stencil_after_draw" + std::to_string(n));
		}
	}
	// Capture frame: the depth image of a captured draw (below) as the next draw finds it, i.e. after that draw.
	static ImageId     capture_after_image {};
	static std::string capture_after_name;
	if (DebugCaptureFrame().load() && capture_after_image) {
		buffer.EndRendering();
		auto& cache = buffer.GetContext().GetTextureCache();
		if (cache.m_slot_images.try_get(capture_after_image) != nullptr) {
			DebugQueueFullDump(m_context.GetGraphics(), vk_buffer, cache.GetImage(capture_after_image),
			                   capture_after_name + "_after");
		}
		capture_after_image = {};
	}
	// Capture frame: vertex-stage textures of draws whose VS hash is listed in KYTY_CAPTURE_INPUTS.
	if (DebugCaptureFrame().load()) {
		static const std::string capture_inputs = Common::EnvRead::CaptureInputs();
		const char* dir = Common::EnvRead::FrameDir();
		for (const auto& stage: vertex_stages) {
			const auto& program = *stage.stage.program;
			char        hash_text[17];
			std::snprintf(hash_text, sizeof(hash_text), "%016llx",
			              static_cast<unsigned long long>(program.shader_hash));
			if (dir == nullptr || capture_inputs.empty() || capture_inputs.find(hash_text) == std::string::npos) {
				continue;
			}
			// Up to 6 draws of every listed shader (GPU thread), numbered across all of them.
			static std::atomic_uint          captured {0};
			static std::map<uint64_t, uint32_t> per_shader;
			if (per_shader[program.shader_hash]++ >= 6u) continue;
			const auto n = captured.fetch_add(1);
			buffer.EndRendering();
			auto&       cache     = buffer.GetContext().GetTextureCache();
			const auto& resources = *stage.stage.resources;
			if (state.depth_info.image_id) {
				// The depth image this draw writes, before it and (at the next draw) after it.
				capture_after_name  = std::string(dir) + "/vscap_" + std::to_string(n) + "_" + hash_text + "_depth";
				capture_after_image = state.depth_info.image_id;
				std::fprintf(stderr, "CAPTURE_VS_DEPTH n=%u hash=%s depth=0x%llx base_layer=%u layers=%u\n", n, hash_text,
				             static_cast<unsigned long long>(state.depth_info.desc.info.data.address),
				             state.depth_info.desc.view_info.base_layer, state.depth_info.desc.view_info.layer_count);
				DebugQueueFullDump(m_context.GetGraphics(), vk_buffer, cache.GetImage(state.depth_info.image_id),
				                   capture_after_name + "_before");
			}
			const auto  user_sgpr = [&](int32_t sgpr) -> long long {
				if (sgpr < static_cast<int32_t>(program.user_data_base)) return -1;
				const auto index = static_cast<uint32_t>(sgpr) - program.user_data_base;
				return index < resources.user_data.size() ? static_cast<long long>(resources.user_data[index]) : -2;
			};
			std::fprintf(stderr,
			             "CAPTURE_VS_DRAW n=%u hash=%s %s count=%u instances=%u emit_vertex_offset=%d "
			             "emit_first_vertex=%u emit_first_instance=%u fetch_embedded=%d vofs_sgpr=%d(%lld) "
			             "iofs_sgpr=%d(%lld) user_data_base=%u ud18=%lld ud19=%lld\n",
			             n, hash_text, draw.Name(), draw.index_count, draw.instance_count, emit.vertex_offset,
			             emit.first_vertex, emit.first_instance, stage.fetch_embedded ? 1 : 0,
			             program.info.vertex_offset_sgpr, user_sgpr(program.info.vertex_offset_sgpr),
			             program.info.instance_offset_sgpr, user_sgpr(program.info.instance_offset_sgpr),
			             program.user_data_base, user_sgpr(18), user_sgpr(19));
			for (uint32_t j = 0; j < program.info.buffers.size() && j < resources.buffers.size(); ++j) {
				const auto b = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[j]);
				std::fprintf(stderr, "CAPTURE_VS_BUFFER n=%u hash=%s buf=%u %s addr=0x%llx stride=%u records=%u fmt=%u\n",
				             n, hash_text, j, program.info.buffers[j].written ? "rw" : "ro",
				             static_cast<unsigned long long>(b.Base48()), b.Stride(), b.NumRecords(),
				             static_cast<uint32_t>(b.Format()));
				// The GPU-side copy the draw reads (copied into a host-visible buffer, written at present).
				const auto bytes = std::min<uint64_t>(b.GetSize(), 4ull << 20u);
				if (b.Base48() != 0 && bytes != 0) {
					auto [source, source_offset] =
					    m_context.GetBufferCache().ObtainBuffer(b.Base48(), bytes, false);
					char name[160];
					std::snprintf(name, sizeof(name), "%s/vsbuf_%u_%s_buf%u_%llx_stride%u.bin", dir, n, hash_text, j,
					              static_cast<unsigned long long>(b.Base48()), b.Stride());
					if (source != nullptr) {
						DebugQueueBufferDump(m_context.GetGraphics(), vk_buffer, source->Handle(), source_offset,
						                     bytes, name);
					}
				}
			}
			for (int a = 0; a < stage.resources_num; ++a) {
				const auto& v = stage.resources[a];
				std::fprintf(stderr, "CAPTURE_VS_ATTRIB n=%u hash=%s attr=%d addr=0x%llx stride=%u records=%u fmt=%u\n",
				             n, hash_text, a, static_cast<unsigned long long>(v.Base48()), v.Stride(), v.NumRecords(),
				             static_cast<uint32_t>(v.Format()));
			}
			for (uint32_t j = 0; j < program.info.images.size() && j < resources.images.size(); ++j) {
				const auto r       = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[j]);
				const auto address = r.Base40();
				const auto images  = cache.DebugImagesAt(address);
				std::fprintf(stderr,
				             "CAPTURE_VS_INPUT n=%u hash=%s img=%u addr=0x%llx type=%u fmt=%u %ux%ux%u tile=%u "
				             "null=%d cached=%zu marker=[%s]\n",
				             n, hash_text, j, static_cast<unsigned long long>(address),
				             static_cast<uint32_t>(r.Type()), static_cast<uint32_t>(r.Format()),
				             static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
				             static_cast<uint32_t>(r.Depth()) + 1u, static_cast<uint32_t>(r.TileMode()),
				             r.IsNull() ? 1 : 0, images.size(), DebugMarkerPath().c_str());
				for (size_t k = 0; k < images.size() && k < 3u; ++k) {
					char name[128];
					std::snprintf(name, sizeof(name), "/vscap_%u_%s_in%u_%zu_%llx", n, hash_text, j, k,
					              static_cast<unsigned long long>(address));
					DebugQueueFullDump(m_context.GetGraphics(), vk_buffer, *images[k], std::string(dir) + name);
				}
			}
		}
	}
	// Capture frame: pixel-shader textures of draws whose PS hash is listed in KYTY_CAPTURE_INPUTS.
	if (DebugCaptureFrame().load() && state.ps_active) {
		static const std::string capture_inputs = Common::EnvRead::CaptureInputs();
		const char* dir     = Common::EnvRead::FrameDir();
		const auto& program = *state.ps_input_info.stage.program;
		char        hash_text[17];
		std::snprintf(hash_text, sizeof(hash_text), "%016llx", static_cast<unsigned long long>(program.shader_hash));
		// The pixel-shader textures of a draw whose vertex shader is listed are taken too (a PS shared by many passes,
		// e.g. the sun-shadow alpha test, in the passes of one VS).
		char vs_hash_text[17];
		std::snprintf(vs_hash_text, sizeof(vs_hash_text), "%016llx",
		              static_cast<unsigned long long>(vertex_stages.front().stage.program->shader_hash));
		static std::atomic_uint captured {0};
		if (dir != nullptr && !capture_inputs.empty() &&
		    (capture_inputs.find(hash_text) != std::string::npos ||
		     capture_inputs.find(vs_hash_text) != std::string::npos) &&
		    captured.fetch_add(1) < 4u) {
			buffer.EndRendering();
			auto&       cache     = buffer.GetContext().GetTextureCache();
			const auto& resources = *state.ps_input_info.stage.resources;
			for (uint32_t j = 0; j < program.info.images.size() && j < resources.images.size(); ++j) {
				const auto r       = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[j]);
				const auto address = r.Base40();
				const auto images  = cache.DebugImagesAt(address);
				std::fprintf(stderr,
				             "CAPTURE_PS_INPUT hash=%s img=%u addr=0x%llx type=%u fmt=%u %ux%ux%u null=%d cached=%zu "
				             "marker=[%s]\n",
				             hash_text, j, static_cast<unsigned long long>(address), static_cast<uint32_t>(r.Type()),
				             static_cast<uint32_t>(r.Format()), static_cast<uint32_t>(r.Width5()) + 1u,
				             static_cast<uint32_t>(r.Height5()) + 1u, static_cast<uint32_t>(r.Depth()) + 1u,
				             r.IsNull() ? 1 : 0, images.size(), DebugMarkerPath().c_str());
				for (size_t k = 0; k < images.size() && k < 2u; ++k) {
					char name[128];
					std::snprintf(name, sizeof(name), "/pscap_%s_vs%s_in%u_%zu_%llx", hash_text, vs_hash_text, j, k,
					              static_cast<unsigned long long>(address));
					DebugQueueFullDump(m_context.GetGraphics(), vk_buffer, *images[k], std::string(dir) + name);
				}
			}
		}
	}
	if (DebugRtCollecting()) {
		const auto vs_hash = vertex_stages.front().stage.program->shader_hash;
		const auto ps_hash = state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0ull;
		for (uint32_t i = 0; i < state.color_count; ++i) {
			const auto& info = state.color_info[i].desc.info;
			DebugRtRecordDraw(info.data.address, info.extent.width, info.extent.height, false, vs_hash, ps_hash);
		}
		if (state.depth_info.image_id && state.depth_info.depth_write_enable) {
			const auto& info = state.depth_info.desc.info;
			DebugRtRecordDraw(info.data.address, info.extent.width, info.extent.height, true, vs_hash, ps_hash);
		}
		// KYTY_TRACE_MARKER_DRAWS=<text>[,<text>...]: every draw under a debug marker path containing
		// one of the texts (e.g. Water,Forward), with shaders, counts and targets (first 400).
		static const std::vector<std::string> marker_filters = [] {
			std::vector<std::string> result;
			if (const char* value = std::getenv("KYTY_TRACE_MARKER_DRAWS"); value != nullptr) {
				std::string text(value);
				size_t      start = 0;
				while (start <= text.size()) {
					const auto end = text.find(',', start);
					result.push_back(text.substr(start, end == std::string::npos ? std::string::npos : end - start));
					if (end == std::string::npos) break;
					start = end + 1;
				}
			}
			return result;
		}();
		if (!marker_filters.empty()) {
			const auto path = DebugMarkerPath();
			if (std::ranges::any_of(marker_filters, [&](const std::string& f) {
				    return !f.empty() && path.find(f) != std::string::npos;
			    })) {
				static std::atomic_uint logged {0};
				if (logged.fetch_add(1) < 400u) {
					std::fprintf(stderr,
					             "MARKER_DRAW marker=[%s] vs=0x%016llx ps=0x%016llx %s count=%u inst=%u indirect=%d rt_mask=0x%08x "
					             "colors=%u slot0=%u c0=0x%llx depth=0x%llx dtest=%d dwrite=%d\n",
					             path.c_str(), static_cast<unsigned long long>(vs_hash),
					             static_cast<unsigned long long>(ps_hash), draw.Name(), draw.index_count,
					             draw.instance_count, indirect != nullptr ? 1 : 0, plan.regs.rt_mask, state.color_count,
					             state.color_count != 0u ? state.color_info[0].target_slot : 0u,
					             static_cast<unsigned long long>(state.color_count != 0u
					                                                 ? state.color_info[0].desc.info.data.address
					                                                 : 0ull),
					             static_cast<unsigned long long>(state.depth_info.image_id
					                                                 ? state.depth_info.desc.info.data.address
					                                                 : 0ull),
					             state.depth_info.depth_test_enable ? 1 : 0, state.depth_info.depth_write_enable ? 1 : 0);
				}
			}
		}
	}

	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x600u);
	}
	vk::PipelineStageFlags shader_write_stages = {};
	for (const auto& stage: vertex_stages) {
		if (HasShaderBufferWrites(stage.stage)) {
			shader_write_stages |= ShaderPipelineStages(NativeShaderStage(stage.logical_stage));
		}
	}
	if (state.ps_active && HasShaderBufferWrites(state.ps_input_info.stage)) {
		shader_write_stages |= vk::PipelineStageFlagBits::eFragmentShader;
	}
	if (shader_write_stages) {
		m_context.GetCommandScheduler().EndRendering();
		ShaderWriteBarrier(vk_buffer, shader_write_stages);
	}
	LogDrawPhase(draw.Name(), "DrawComplete");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x700u);
	}
}

// A resolution ahead of execution (see ResolveTaskInto): the draw state with its shader programs or the dispatch's input
// info, the stages they point to, and whether the execution thread still uses them.
// Based on TheCruZ/KytyPS5-GTA daaf45201 (renderDraw.cpp), GPL-2.0.
struct RenderExecutor::ResolvedDraw {
	ResolvedDraw(): stages(NewResolvedStages()) {}
	~ResolvedDraw() { DeleteResolvedStages(stages); }
	KYTY_CLASS_NO_COPY(ResolvedDraw);

	ResolvedStages*        stages;
	DrawRenderState        state;
	ShaderComputeInputInfo compute;
	ShaderProgram          compute_program;
	const HW::Shader*      shaders    = nullptr; // the draw's register snapshot
	bool                   is_compute = false;
	bool                   consumed   = false; // an indirect packet's CPU expansion: only its first draw asks
	// Project 15 N2.2: the slot's phase (ResolvedPhase): free; pending (taken, its task not begun); running (a helper
	// resolves it); ready (resolved, `empty`: nothing to resolve); stolen (the execution thread took its operation before
	// the helper began; the helper frees it when it takes the task).
	std::atomic<uint8_t>   phase {0};
	bool                   empty = false;
	ResolvedDraw*          check = nullptr; // KYTY_CHECK_RESOLVE=2: the helper's slot of the same operation
	uint32_t               fault = 0;       // KYTY_CHECK_RESOLVE_FAULT: the error put into this (helper's) result
	// Project 15 N4.1 (KYTY_CHECK_PEEK=2): what the chunk's opening peeked for this draw ahead (PeekAhead).
	struct Ahead {
		bool                                                             key_valid = false, front = false, planned = false;
		uint32_t                                                         distance  = 0;
		PipelineCache::GraphicsPipelineKey                               key {};
		std::pair<PipelineCache::Pipeline*, PipelineCache::PipelineFind> find {nullptr, PipelineCache::PipelineFind::Count};
		std::vector<std::pair<const void*, PeekStageImagesResult>>      images;
		std::vector<PlanBuffer>                                          buffers; // N4.2b
	} ahead;
};

namespace {
enum ResolvedPhase : uint8_t { PhaseFree = 0, PhasePending = 1, PhaseRunning = 2, PhaseReady = 3, PhaseStolen = 4 };
} // namespace

// More resolutions than operations can wait between the resolve and execution threads (GuestGpu::ResolvedQueueSize), so
// the resolve thread only ever waits for room in that queue.
struct RenderExecutor::ResolvedDrawRing {
	static constexpr size_t Size = 320;

	std::array<ResolvedDraw, Size> draws;
	size_t                         next = 0;
};

RenderExecutor::ResolvedDraw& RenderExecutor::NextResolvedDraw() {
	if (m_resolved_ring == nullptr) {
		m_resolved_ring = {new ResolvedDrawRing {}, [](ResolvedDrawRing* ring) { delete ring; }};
	}
	auto& ring = *m_resolved_ring;
	auto& draw = ring.draws[ring.next];
	// The execution thread releases resolutions in order (a stolen slot: the helper, when it takes its task).
	for (Common::SpinWait wait; draw.phase.load(std::memory_order_acquire) != PhaseFree;) {
		if (!wait.Spin()) std::this_thread::yield();
	}
	ring.next     = (ring.next + 1) % ResolvedDrawRing::Size;
	draw.consumed = false;
	draw.ahead.key_valid = false; // N4.1
	draw.ahead.find      = {nullptr, PipelineCache::PipelineFind::Count}; // (n4-3-przeglad.md 2.1)
	draw.ahead.planned   = false;
	draw.ahead.images.clear();
	draw.ahead.buffers.clear();
	draw.empty    = false;
	draw.check    = nullptr;
	draw.fault    = 0;
	return draw;
}

RenderExecutor::ResolvedDraw& RenderExecutor::TakeResolvedSlot() {
	auto& draw = NextResolvedDraw();
	draw.phase.store(PhasePending, std::memory_order_relaxed); // taken from its allocation (published with its operation or task)
	return draw;
}

bool RenderExecutor::ResolveDrawInto(ResolvedDraw& draw, const HW::Context& context, const HW::UserConfig& user_config,
                                     const HW::Shader& shaders, const ShaderIndirectDrawInfo& indirect_draw) {
	std::optional<ChunkLead::PartTimer> slot_timer(std::in_place, ChunkLead::RpSlot); // KYTY_TRACE_CHUNK_LEAD (N2.0)
	auto& state = draw.state;
	ResetDrawState(state);
	state.indirect_draw = indirect_draw;
	state.programs      = {};
	state.ps_input_info = {};
	const auto& shader_regs       = context.GetShaderRegisters();
	const auto  color_output_mask = DrawColorOutputMask(context);
	state.ps_active               = shaders.GetPs().ps_regs.data_addr != 0 &&
	                  (color_output_mask != 0 || PixelShaderHasDepthOrCoverageSideEffects(shader_regs));
	const auto target_export_mapping = TargetExportMapping(context, color_output_mask);
	slot_timer.reset();
	m_context.GetPipelineCache().ResolveGraphicsPrograms(m_context, shaders.GetVs(), shaders.GetPs(), shader_regs, context,
	                                                     user_config, target_export_mapping, state.ps_active, state.vertex_info,
	                                                     state.ps_input_info, indirect_draw, state.programs, *draw.stages);
	if (ResolvedStagesEmpty(*draw.stages)) return false;
	draw.shaders    = &shaders;
	draw.is_compute = false;
	return true;
}

bool RenderExecutor::ResolveDispatchInto(ResolvedDraw& draw, const HW::Shader& shaders, uint32_t groups_x, uint32_t groups_y,
                                         uint32_t groups_z, uint32_t mode, bool indirect, bool async_compute) {
	m_context.GetPipelineCache().ResolveDispatchPrograms(m_context, shaders.GetCs(), groups_x, groups_y, groups_z, mode, indirect,
	                                                     async_compute, draw.compute, draw.compute_program, *draw.stages);
	if (ResolvedStagesEmpty(*draw.stages)) return false;
	draw.shaders    = &shaders;
	draw.is_compute = true;
	return true;
}

void RenderExecutor::UseResolvedDraw(ResolvedDraw* resolved) noexcept {
	m_resolved_draw   = resolved;
	m_slot_taken      = false;
	t_resolved_stages = resolved != nullptr ? resolved->stages : nullptr;
}

DrawRenderState* RenderExecutor::TakeResolvedState(const ShaderIndirectDrawInfo& indirect_draw) {
	const ChunkLead::FloorTimer floor_take(ChunkLead::FpTake); // KYTY_TRACE_PEEK_CENSUS (way B W0 A2)
	// N7.2: per draw, not per operation - a GPU indirect packet that took the slot and then went to the CPU expansion must
	// not leave its plan to the expansion's draws (they fail to take the used slot).
	m_slot_taken         = false;
	auto* const resolved = m_resolved_draw;
	if (resolved == nullptr || resolved->is_compute || std::exchange(resolved->consumed, true) ||
	    t_draw_regs.sh != resolved->shaders || indirect_draw.Key() != resolved->state.indirect_draw.Key() ||
	    !m_context.GetPipelineCache().AdoptResolvedPrograms(*resolved->stages)) {
		return nullptr;
	}
	m_state_resolved = true;
	m_slot_taken     = true; // N4.2: the draw runs on the slot's state (its plan may be used)
	return &resolved->state;
}

bool RenderExecutor::TakeResolvedDispatch(ShaderComputeInputInfo& input_info, ShaderProgram& program) {
	auto* const resolved = m_resolved_draw;
	if (resolved == nullptr || !resolved->is_compute || std::exchange(resolved->consumed, true) ||
	    !std::ranges::equal(input_info.workgroup_counts, resolved->compute.workgroup_counts) ||
	    !m_context.GetPipelineCache().AdoptResolvedPrograms(*resolved->stages)) {
		return false;
	}
	input_info             = resolved->compute;
	program                = resolved->compute_program;
	t_compute_program_path = 3u; // KYTY_TRACE_DISPATCH_CENSUS
	return true;
}

void RenderExecutor::ReleaseResolvedDraw(ResolvedDraw* resolved) noexcept {
	resolved->phase.store(PhaseFree, std::memory_order_release);
}

// Project 15 N2.2: the helper's resolution of a task (the draw's pre-check included) and the phases around it.
bool RenderExecutor::ResolveTaskInto(ResolvedDraw& slot, const ResolveTask& task) {
	bool ok = false;
	if (task.dispatch) {
		ok = ResolveDispatchInto(slot, *task.shaders, task.groups_x, task.groups_y, task.groups_z, task.mode, task.indirect,
		                         task.async_compute);
	} else {
		const auto mode = task.context->GetColorControl().mode;
		ok              = (mode == 0u || mode == 1u) && DrawHasValidVertexShader(*task.shaders) &&
		     ResolveDrawInto(slot, *task.context, *task.user_config, *task.shaders, task.indirect_draw);
	}
	if (ok && task.fault != 0u) { // KYTY_CHECK_RESOLVE_FAULT
		ResolveCheckFault(slot, task.fault);
		slot.fault = task.fault;
	}
	slot.empty = !ok;
	if (ok && !task.dispatch && task.build_plan) {
		// N5.3: the helper builds the draw's plan while the slot is still its own (before it is marked ready).
		PeekAhead(&slot, *task.context, *task.user_config, *task.shaders, task.indexed, task.draw_count, 1u, task.index_type,
		          task.slice_offset, task.draw_indirect, std::min(task.lead, 127u), task.index_address,
		          task.gpu_indirect ? &task.indirect_args : nullptr);
		// N7.1: the outcome counters for the report (the helper's scratch slot 3 + index names its row)
		if (slot.ahead.planned) t_helper_plans_made++;
		if (task.gpu_indirect) t_helper_packets++;
		const uint32_t row = ShaderRecompiler::IR::t_ahead_scratch_slot - 3u;
		if ((t_ahead_helper.planned & 255u) == 0 && row < 2u) {
			g_helper_plan_snap[row].Publish(t_ahead_helper, t_helper_plans_made, t_helper_packets);
		}
	}
	slot.phase.store(PhaseReady, std::memory_order_release);
	return ok;
}

bool RenderExecutor::BeginResolveTask(ResolvedDraw* slot) noexcept {
	uint8_t expected = PhasePending;
	if (slot->phase.compare_exchange_strong(expected, PhaseRunning, std::memory_order_acq_rel)) return true;
	// Stolen: the execution thread went on without it; this task was its last user.
	slot->phase.store(PhaseFree, std::memory_order_release);
	return false;
}

RenderExecutor::ResolvedClaim RenderExecutor::TryClaimResolvedDraw(ResolvedDraw* resolved, bool steal) noexcept {
	uint8_t phase = resolved->phase.load(std::memory_order_acquire);
	if (phase == PhasePending && steal &&
	    resolved->phase.compare_exchange_strong(phase, PhaseStolen, std::memory_order_acq_rel)) {
		return ResolvedClaim::Stolen;
	}
	if (phase == PhaseReady) return resolved->empty ? ResolvedClaim::Empty : ResolvedClaim::Ready;
	return ResolvedClaim::Running;
}

RenderExecutor::ResolvedDraw* RenderExecutor::ResolvedCheckSlot(ResolvedDraw* resolved) noexcept {
	return resolved->check;
}

void RenderExecutor::SetResolvedCheckSlot(ResolvedDraw* resolved, ResolvedDraw* check) noexcept {
	resolved->check = check;
}

int RenderExecutor::CompareResolved(const ResolvedDraw& reference, const ResolvedDraw& helper, const char** field) {
	*field = nullptr;
	if (reference.empty && helper.empty) return 0;
	if (reference.empty != helper.empty) {
		*field = "empty";
		return 1;
	}
	const int stages = ResolvedStagesCompare(*reference.stages, *helper.stages, field);
	if (stages != 3) return stages;
	const auto bad = [&](const char* name) {
		*field = name;
		return 4;
	};
	if (reference.is_compute != helper.is_compute) return bad("is_compute");
	if (reference.shaders != helper.shaders) return bad("shaders");
	if (reference.is_compute) {
		if (!std::ranges::equal(reference.compute.workgroup_counts, helper.compute.workgroup_counts)) return bad("compute_groups");
	} else {
		if (reference.state.ps_active != helper.state.ps_active) return bad("ps_active");
		if (reference.state.indirect_draw.Key() != helper.state.indirect_draw.Key()) return bad("indirect_draw");
	}
	return 3;
}

uint32_t RenderExecutor::ResolvedFault(const ResolvedDraw* slot) noexcept {
	return slot->fault;
}

void RenderExecutor::SetResolveCheckMode(bool on) {
	SetResolveCheck(on);
}

void RenderExecutor::ResolveCheckFault(ResolvedDraw& slot, uint32_t fault) {
	if (fault % 7u == 6u && !slot.is_compute) {
		slot.state.ps_active = !slot.state.ps_active;
		return;
	}
	if (fault % 7u == 5u && slot.is_compute) {
		slot.compute.workgroup_counts[0]++;
		return;
	}
	ResolvedStagesFault(*slot.stages, fault);
}

// KYTY_DIAG_GPU_BURN_NS=<ns> (measurement only, default 0; KYTY_LIVE_AB toggles 0/8000 or NAME=<ns>): the GPU thread spins
// that long at one place, KYTY_DIAG_GPU_BURN_AT:
// "indirect" at each GPU indirect draw (DrawIndirectGpu),
// "readback" each time the GPU thread serves a guest read of GPU-written data (a download that copied GPU bytes - host
// mapping, readback queue or copy and wait -, a wait for a pending eager readback the read needs, an exact walker read
// that publishes GPU bytes; bufferCache.cpp), "pm4" at each PM4 packet the GPU thread processes (graphicsRun.cpp):
// and "invalidate_caller" on a guest thread before its write fault reads GPU-modified bytes back (BufferCache::InvalidateMemory;
// that thread waits for the GPU thread there: how much frame time the guest threads' write-fault waits cost):
// how much frame time that work costs.
RenderExecutor::GpuBurnSite RenderExecutor::GpuBurnSelected() {
	static const GpuBurnSite burn_site = [] {
		const char* value = std::getenv("KYTY_DIAG_GPU_BURN_AT");
		const auto  s     = value == nullptr                        ? GpuBurnSite::None
		                    : std::strcmp(value, "indirect") == 0 ? GpuBurnSite::Indirect
		                    : std::strcmp(value, "dispatch") == 0 ? GpuBurnSite::Dispatch
		                    : std::strcmp(value, "dispatch_async") == 0 ? GpuBurnSite::DispatchAsync
		                    : std::strcmp(value, "dispatch_gfx") == 0   ? GpuBurnSite::DispatchGfx
		                    : std::strcmp(value, "readback") == 0       ? GpuBurnSite::Readback
		                    : std::strcmp(value, "pm4") == 0            ? GpuBurnSite::Pm4
		                    : std::strcmp(value, "invalidate_caller") == 0 ? GpuBurnSite::InvalidateCaller
		                    : std::strcmp(value, "direct_wait") == 0       ? GpuBurnSite::DirectWait
		                    : std::strcmp(value, "draw") == 0              ? GpuBurnSite::Draw
		                                                        : GpuBurnSite::None;
		if (value != nullptr) std::fprintf(stderr, "DIAG_GPU_BURN_AT %u\n", static_cast<uint32_t>(s));
		return s;
	}();
	return burn_site;
}

void RenderExecutor::GpuBurnAt(GpuBurnSite site) {
	static const uint32_t burn_fixed = [] {
		const char* value = std::getenv("KYTY_DIAG_GPU_BURN_NS");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	if (site != GpuBurnSelected()) return;
	// The site's calls per frame (both live A/B phases): DIAG_GPU_BURN calls=<n> frame=<present count> every 65536.
	static std::atomic<uint64_t> calls {0};
	if (const auto n = calls.fetch_add(1, std::memory_order_relaxed) + 1u; (n & 0xffffu) == 0u) {
		std::fprintf(stderr, "DIAG_GPU_BURN site=%u calls=%llu frame=%u\n", static_cast<uint32_t>(site),
		             static_cast<unsigned long long>(n), DebugPresentCount().load());
	}
	if (const uint32_t burn = Common::LiveAb::Gate(Common::LiveAb::Switch::GpuBurn, burn_fixed); burn != 0) {
		const auto until = std::chrono::steady_clock::now() + std::chrono::nanoseconds(burn);
		while (std::chrono::steady_clock::now() < until) _mm_pause();
	}
}

namespace {
// Upstream's GPU-read indirect records (DrawIndirectArgs) reached DrawIndex/DrawAuto: logged once; this renderer does
// not draw them (the caller's CPU expansion does).
void NoteUpstreamIndirectRecords() {
	static std::atomic_bool logged {false};
	if (!logged.exchange(true, std::memory_order_relaxed)) {
		std::fprintf(stderr, "UPSTREAM_GPU_INDIRECT_RECORDS: GPU-read records are not drawn by this renderer; the CPU "
		                     "expansion draws them\n");
	}
}
} // namespace

void RenderExecutor::DrawEntryStart(CommandBuffer& buffer) {
	m_context.GetCommandScheduler().NoteGpuCommand();
	if (!buffer.IsInvalid()) FlushComputeAccess(buffer.Handle());
}

bool RenderExecutor::ColorOperationInstead(CommandBuffer& buffer, uint32_t render_target_slice_offset, const char* drop_reason) {
	if (!ConsumeMetadataColorOperation(buffer, render_target_slice_offset) && !DepthStencilCopy(buffer) &&
	    !ResolveColorTargets(buffer, render_target_slice_offset)) {
		return false;
	}
	TraceDrawDrop(drop_reason, buffer);
	ResetBindings();
	return true;
}

// KYTY_DIAG_SKIP_DIRECT_DRAWS=1 (ceiling test only, map 5.11 "ruch przez miasto"): every direct draw (not GPU indirect) is
// dropped at its entry - no state, program, pipeline or binding work and no recording; the most the per-draw CPU cost
// can give. The image breaks; the game's own work (draw packets, dispatches per frame) must stay. Read once.
static bool SkipDirectDrawsCeiling() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_DIAG_SKIP_DIRECT_DRAWS");
		const bool  set   = value != nullptr && value[0] == '1';
		if (set) std::fprintf(stderr, "DIAG_SKIP_DIRECT_DRAWS ceiling: direct draws dropped""\n");
		return set;
	}();
	return on;
}

void RenderExecutor::CheckAheadKey(const PipelineCache::GraphicsPipelineKey& key_today) {
	if (m_resolved_draw == nullptr || !m_resolved_draw->ahead.key_valid) return;
	// N4.1: the key and pipeline the opening's peek found, when its programs were predicted right.
	auto& ah     = m_resolved_draw->ahead;
	auto& a      = g_ahead;
	ah.key_valid = false;
	if (!PipelineCache::AheadProgramsOk(*m_resolved_draw->stages)) {
		a.k_unchecked++;
		return;
	}
	a.k_compared++;
	const bool fault = PeekTargets::FaultEvery() != 0 && a.k_compared % PeekTargets::FaultEvery() == 0;
	if (fault) a.k_injected++;
	if (!(ah.key == key_today) || fault) (fault ? a.k_detected : a.k_bad)++;
	if (ah.find != m_context.GetPipelineCache().FindGraphicsPipeline(key_today)) a.k_find_diff++;
}

// KYTY_DIAG_PLAN_NO_SET_HINT=1 (N5.0 measurement only): the plans without the table-set hint.
static bool PlanNoSetHint() {
	static const bool on = [] {
		const char* env = std::getenv("KYTY_DIAG_PLAN_NO_SET_HINT");
		return env != nullptr && env[0] == '1';
	}();
	return on;
}

void RenderExecutor::PlanBuffersBegin() {
	BufferCache::SetPlanBuffers(PlanPresent() && PlanProgramsHold() ? &m_resolved_draw->ahead.buffers : nullptr);
}

bool RenderExecutor::PlanPresent() const {
	return m_resolved_draw != nullptr && m_slot_taken && m_resolved_draw->ahead.planned; // n4-2b-wejscia.md 1
}

uint64_t RenderExecutor::PlanSetVersion(const void* program) const {
	if (!PlanPresent()) return 0;
	for (const auto& [ahead_program, ahead]: m_resolved_draw->ahead.images) {
		if (ahead_program == program) return ahead.set_version;
	}
	return 0;
}

bool RenderExecutor::PlanProgramsHold() const {
	if (!PlanPresent()) return false;
	const auto& stages = *m_resolved_draw->stages;
	for (uint32_t i = 0; i < PipelineCache::StageCount(stages); i++) {
		if (PipelineCache::AheadProgram(stages, i) != PipelineCache::AdoptedProgram(stages, i)) return false;
	}
	return true;
}

PipelineCache::Pipeline* RenderExecutor::PlanPipeline() const {
	if (!PlanPresent()) return nullptr;
	const auto& find = m_resolved_draw->ahead.find;
	return find.second == PipelineCache::PipelineFind::Found ? find.first : nullptr;
}

bool RenderExecutor::TakeAheadImages(const void* program, PeekStageImagesResult& out, uint32_t& distance) {
	if (m_resolved_draw == nullptr) return false;
	for (auto& [ahead_program, ahead]: m_resolved_draw->ahead.images) {
		if (ahead_program != program) continue;
		out           = std::move(ahead);
		distance      = m_resolved_draw->ahead.distance;
		ahead_program = nullptr;
		return true;
	}
	return false;
}

void RenderExecutor::PeekAhead(ResolvedDraw* slot, const HW::Context& ctx, const HW::UserConfig& ucfg, const HW::Shader& shaders,
                               bool indexed, uint32_t count, uint32_t instances, uint32_t index_type, uint32_t slice_offset,
                               uint64_t indirect, uint32_t distance, uint64_t index_address,
                               const DrawIndirectGpuArgs* gpu_indirect) {
	const bool     on_helper = !GuestGpu::IsGpuThread();
	auto&          a         = on_helper ? t_ahead_helper : g_ahead;
	const uint32_t bucket = AheadBucket(distance);
	a.planned++;
	a.peeked[bucket]++;
	if (slot == nullptr || slot->empty || slot->is_compute) {
		a.gated[1]++;
		return;
	}
	slot->ahead.key_valid = false;
	slot->ahead.find      = {nullptr, PipelineCache::PipelineFind::Count}; // a mesh draw's plan has no pipeline
	slot->ahead.front     = false;
	slot->ahead.planned   = false;
	slot->ahead.distance  = distance;
	slot->ahead.images.clear();
	slot->ahead.buffers.clear();
	// The gates DrawIndex / DrawAuto pass before the targets are described (their order).
	if (slot->shaders != &shaders) {
		a.gated[2]++;
		return;
	}
	if (!PipelineCache::StagesAdoptable(*slot->stages)) {
		a.gated[3]++;
		return;
	}
	// N7.2: a GPU indirect packet passes the execution thread's early checks (ranges clamped without a write) or has no plan.
	IndirectEarly early;
	if (gpu_indirect != nullptr) {
		if (IndirectEarlyChecks(m_context.GetGraphics(), ucfg, ctx, *gpu_indirect, true, early) != nullptr || early.no_draw) {
			a.gated[8]++;
			return;
		}
	} else if (count == 0 || instances == 0 || indirect != 0) {
		a.gated[4]++;
		return;
	}
	if (ctx.GetColorControl().mode > static_cast<uint8_t>(CbColorMode::Normal) || DepthStencilCopyAsked(ctx)) {
		a.gated[5]++;
		return;
	}
	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	uint32_t              element  = 0;
	if (indexed) {
		switch (static_cast<Prospero::IndexType>(index_type)) {
			case Prospero::IndexType::kIndex16: element = 2; break;
			case Prospero::IndexType::kIndex32: element = 4; break;
			case Prospero::IndexType::kIndex8: element = 1; break;
			default: element = 0; break;
		}
	}
	if (!GetDrawTopology(ucfg, topology) || (indexed && element == 0)) {
		a.gated[6]++;
		return;
	}
	auto& pc = m_context.GetPipelineCache();
	// N5.0: without the checks that read them, the builder skips the check-only data.
	static const bool slim = std::getenv("KYTY_CHECK_PEEK") == nullptr && std::getenv("KYTY_CHECK_CHUNKS") == nullptr;
	t_peek_slim            = slim;
	++t_translator;
	if (!pc.PeekAdoptAhead(*slot->stages, distance)) {
		--t_translator;
		a.compile++;
		return;
	}
	slot->ahead.planned = true;
	PipelineCache::AheadSaved saved;
	PipelineCache::InstallAhead(*slot->stages, saved);
	auto& state = slot->state;
	bool  front = false;
	if (state.vertex_info[0].stage.program == nullptr || state.vertex_info[0].stage.program->stage == ShaderType::Mesh) {
		a.gated[7]++;
	} else if (gpu_indirect != nullptr &&
	           (state.programs.VertexStageCount() != 1u || state.vertex_info[0].stage.program->stage != ShaderType::Vertex)) {
		a.gated[9]++; // N7.2: the packet goes to the CPU expansion after taking the slot (no key: nothing else is planned)
	} else {
		// The targets as PrepareDrawRenderState finds them, from the predicted pixel program's outputs.
		const auto color_output_mask = DrawColorOutputMask(ctx);
		uint32_t   mrt_mask = 0, objprim_mask = 0;
		if (state.ps_active && state.ps_input_info.stage.program != nullptr) {
			for (const auto& output: state.ps_input_info.stage.program->info.outputs) {
				if (output.kind == ShaderRecompiler::IR::StageOutputKind::Mrt ||
				    output.kind == ShaderRecompiler::IR::StageOutputKind::ObjectPrimitiveId) {
					mrt_mask |= 1u << output.location;
				} else if (output.kind == ShaderRecompiler::IR::StageOutputKind::ObjPrimId) {
					objprim_mask |= 1u << output.location;
				}
			}
		}
		mrt_mask = (mrt_mask & color_output_mask) | objprim_mask;
		std::array<RenderColorInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
		uint32_t                                                  color_count = 0;
		t_describe_soft   = true; // N5.1: a fail here means no plan, never a stop
		t_describe_failed = false;
		for (uint32_t target = 0; target < RENDER_COLOR_ATTACHMENTS_MAX; target++) {
			if ((mrt_mask & (1u << target)) == 0) continue;
			if (DescribeRenderColorTarget(ctx, colors[color_count], slice_offset, target, false)) color_count++;
		}
		RenderDepthInfo depth {};
		(void)DescribeRenderDepthTarget(m_context.GetGraphics(), ctx, depth);
		// A custom restart index: today's scan finds no restart (or stops the emulator), so false.
		const bool restart = indexed && ClassifyPrimitiveRestartRegs(ucfg, ctx, element) == PrimitiveRestartKind::Native;
		const auto vertex_stages = std::span {state.vertex_info.data(), state.programs.VertexStageCount()};
		slot->ahead.key = pc.MakeGraphicsPipelineKeyFrom(std::span {colors.data(), color_count}, depth, vertex_stages, ctx, ucfg,
		                                                 state.ps_active ? &state.ps_input_info : nullptr, topology, restart,
		                                                 state.programs, true);
		t_describe_soft = false;
		const bool describe_failed = std::exchange(t_describe_failed, false); // never left set (n5-1-uwagi.md 0.1)
		if (describe_failed) {
			a.describe_failed++;
			PipelineCache::RestoreAhead(*slot->stages, saved);
			slot->ahead.planned = false;
			--t_translator;
			return;
		}
		slot->ahead.key_valid = true;
		slot->ahead.find      = pc.FindGraphicsPipeline(slot->ahead.key, GuestGpu::IsGpuThread());
		front                 = slot->ahead.find.second == PipelineCache::PipelineFind::Found;
		// The stages' images and samplers, with the predicted programs.
		for (uint32_t i = 0; i < PipelineCache::StageCount(*slot->stages); i++) {
			const auto* program   = PipelineCache::AheadProgram(*slot->stages, i);
			const auto* resources = PipelineCache::StageResources(*slot->stages, i);
			if (program == nullptr || resources == nullptr) continue;
			if (!on_helper) { // (a helper does not read the table sets: they are the execution thread's, n5-projekt.md 4)
				PeekStageImagesResult images;
				front = PeekStageImages(*program, *resources, false, images, false) && front;
				slot->ahead.images.emplace_back(program, std::move(images));
			}
			// N4.2b: the stage's buffers
			PeekStageBuffers(*program, *resources, PipelineCache::StageRecord(*slot->stages, i), distance, slot->ahead.buffers);
		}
		// N4.2b: the merged vertex ranges (clamped as AcquireVertexBuffers clamps them) and the 16/32-bit index buffer.
		auto& cache = m_context.GetBufferCache();
		// N5.1: off the execution thread under the lookup lock, tried (busy: no entries).
		const bool helper      = !GuestGpu::IsGpuThread();
		const bool lookup_held = !helper || cache.TryLockLookup();
		if (!lookup_held) a.lookup_busy++;
		std::array<uint64_t, ShaderVertexInputInfo::RES_MAX>          sizes;
		std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> merged;
		const auto& vs_input = state.vertex_info[0];
		const uint32_t merged_count =
		    vs_input.buffers_num >= 0 && vs_input.buffers_num <= ShaderVertexInputInfo::RES_MAX ? MergedVertexRanges(vs_input, sizes, merged, false)
		                                                                                         : UINT32_MAX;
		for (uint32_t m = 0; lookup_held && merged_count != UINT32_MAX && m < merged_count; m++) {
			uint64_t       generation = 0;
			const uint64_t size = Libs::LibKernel::Memory::ClampRangeSizeAhead(merged[m].base_address, merged[m].RequestedSize(), &generation);
			if (size == 0) continue;
			auto& entry    = slot->ahead.buffers.emplace_back();
			entry.vaddr    = merged[m].base_address;
			entry.size     = size;
			entry.kind     = 2;
			entry.distance = distance;
			entry.peek     = cache.PeekBuffer(entry.vaddr, size, false, false, {});
		}
		// (N7.2: a packet's index buffer is its whole bound range, clamped as the execution thread clamps it)
		const uint64_t index_vaddr = gpu_indirect != nullptr ? early.index_source.address : index_address;
		const uint64_t index_size  = gpu_indirect != nullptr ? early.index_source.size : static_cast<uint64_t>(count) * element;
		if (lookup_held && indexed && (element == 2u || element == 4u) && index_vaddr != 0 && index_size != 0) {
			auto& entry    = slot->ahead.buffers.emplace_back();
			entry.vaddr    = index_vaddr;
			entry.size     = index_size;
			entry.kind     = 3;
			entry.distance = distance;
			entry.peek     = cache.PeekBuffer(entry.vaddr, entry.size, false, false, {});
		}
		if (lookup_held && gpu_indirect != nullptr) {
			// N7.2: the packet's argument records and draw count (ExecutePreparedDraw's acquisitions)
			auto& args    = slot->ahead.buffers.emplace_back();
			args.vaddr    = early.indirect.args_addr;
			args.size     = early.indirect.args_size;
			args.kind     = 4;
			args.distance = distance;
			args.peek     = cache.PeekBuffer(args.vaddr, args.size, false, false, {});
			if (early.indirect.count_addr != 0) {
				auto& counter    = slot->ahead.buffers.emplace_back();
				counter.vaddr    = early.indirect.count_addr;
				counter.size     = sizeof(uint32_t);
				counter.kind     = 5;
				counter.distance = distance;
				counter.peek     = cache.PeekBuffer(counter.vaddr, counter.size, false, false, {});
			}
		}
		if (helper && lookup_held) cache.UnlockLookup();
	}
	PipelineCache::RestoreAhead(*slot->stages, saved);
	--t_translator;
	if (!GuestGpu::IsGpuThread()) {
		// N5.1: the writes a helper's peeks noted (expected 0), into the shared counters.
		for (size_t i = 0; i < t_translator_writes.size(); i++) {
			if (t_translator_writes[i] != 0) g_helper_translator_writes[i].fetch_add(t_translator_writes[i], std::memory_order_relaxed);
			t_translator_writes[i] = 0;
		}
	}
	slot->ahead.front = front;
	if (front) a.front[bucket]++;
}

bool RenderExecutor::DrawIndex(uint64_t submit_id, CommandBuffer& buffer,
                               const DrawIndexArgs& args) {
	if (args.indirect.address != 0) {
		// Upstream c513a4a13 / 949de19bb: GPU-read indexed indirect records. Not drawn on this path (our
		// DrawIndirectGpu and the CPU expansion are): the caller expands them on the CPU.
		NoteUpstreamIndirectRecords();
		return false;
	}
	const DrawCensusScope           draw_census; // KYTY_TRACE_DRAW_CENSUS
	std::optional<DispatchSubTimer> census_pre(std::in_place, DispatchSub::DrawPre); // KYTY_TRACE_DRAW_CENSUS
	DrawEntryStart(buffer);
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	if (SkipDirectDrawsCeiling() && DebugPresentCount().load() >= 2600u) { // in the scene only (from frame 2600)
		ResetBindings();
		return true;
	}
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndex), submit_id,
	                    args.index_count, 0, 1, args.instance_count,
	                    reinterpret_cast<uint64_t>(args.index_addr));

	Common::LockGuard lock(m_context.GetMutex());
	m_current_slot = NextDrawSlot();
	if (args.index_count == 0 || args.instance_count == 0) {
		TraceDrawDrop("zero index/instance count", buffer);
		return true;
	}

	if (ColorOperationInstead(buffer, args.render_target_slice_offset, "metadata/copy/resolve operation")) {
		return true;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		TraceDrawDrop("no vertex shader", buffer);
		return true;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndex():Shader:\n");
		uc_print("GraphicsRenderDrawIndex():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndex():Parameters:\n"
		     "\t index_type_and_size = 0x%08" PRIx32 "\n"
		     "\t index_count         = 0x%08" PRIx32 "\n"
		     "\t index_addr          = 0x%016" PRIx64 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t base_vertex         = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.index_type_and_size, args.index_count,
		     reinterpret_cast<uint64_t>(args.index_addr), args.instance_count,
		     static_cast<uint32_t>(args.base_vertex), args.first_instance);
	}

	uc_check(ucfg);

	hw_check(buffer);

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) {
		TraceDrawDrop("primitive type", buffer);
		return true;
	}

	DrawIndexBufferSource index_source {};
	index_source.address = reinterpret_cast<uint64_t>(args.index_addr);
	switch (static_cast<Prospero::IndexType>(args.index_type_and_size)) {
		case Prospero::IndexType::kIndex16:
			index_source.type               = vk::IndexType::eUint16;
			index_source.guest_element_size = 2;
			break;
		case Prospero::IndexType::kIndex32:
			index_source.type               = vk::IndexType::eUint32;
			index_source.guest_element_size = 4;
			break;
		case Prospero::IndexType::kIndex8:
			index_source.guest_element_size = 1;
			break;
		default: EXIT("unknown index_type_and_size: %u\n", args.index_type_and_size);
	}
	index_source.size = static_cast<uint64_t>(args.index_count) * index_source.guest_element_size;
	const bool primitive_restart = ResolvePrimitiveRestart(buffer, index_source);

	std::vector<uint16_t>& expanded_indices = m_current_slot->expanded_indices;
	if (index_source.guest_element_size == 1) {
		EXIT_NOT_IMPLEMENTED(args.index_addr == nullptr);
		const auto* src = static_cast<const uint8_t*>(args.index_addr);
		expanded_indices.resize(args.index_count);
		for (uint32_t i = 0; i < args.index_count; i++) {
			expanded_indices[i] = primitive_restart && src[i] == 0xffu ? 0xffffu : src[i];
		}
		index_source.host_data = expanded_indices.data();
		index_source.size      = expanded_indices.size() * sizeof(uint16_t);
	}

	const DrawCallInfo draw {CommandBufferDebugOp::DrawIndex, args.index_count,
	                        args.instance_count, args.first_instance};
	DrawStateHolder  state_holder(m_current_slot->state, TakeResolvedState({}));
	DrawRenderState& state = state_holder.Get();
	BeginDrawPlan(); // the plan flags
	census_pre.emplace(DispatchSub::DrawRenderState);
	bool render_state = false;
	{
		render_state = PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state);
	}
	if (!render_state) {
		TraceRenderStateDrop("render state (targets/shaders)", buffer, state);
		ResetBindings();
		return true;
	}
	census_pre.reset();

	LogDrawStateIfNeeded(buffer, draw, state, args.index_type_and_size,
	                     args.index_addr);

	const bool indirect = args.offset_source == DrawOffsetSource::IndirectArgs;
	const auto [vertex_offset, instance_offset] =
	    indirect ? std::pair<int32_t, uint32_t> {0, args.first_instance}
	             : ResolveDrawOffsets(ucfg.GetIndexOffset(), state.vertex_info[0]);

	DrawEmitInfo emit {};
	// Native fetches consume the patched SGPRs; rewritten
	// (embedded) fetches need the Vulkan offsets.
	const auto [native_vertex, native_instance] =
	    indirect ? IndirectDrawNativeOffsets(state.vertex_info[0])
	             : std::pair<bool, bool> {false, false};
	emit.vertex_offset  = native_vertex ? 0 : vertex_offset + args.base_vertex;
	emit.first_instance = native_instance ? 0 : instance_offset;
	emit.cpu_indirect   = indirect;
	if (!indirect) {
		TraceDrawBaseIndexOffset(ucfg.GetIndexOffset(), state.vertex_info[0]);
	}

	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source,
	                    primitive_restart);
	ResetBindings();
	return true;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool RenderExecutor::DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args) {
	if (args.indirect.address != 0) {
		// Upstream 86fed10b8: GPU-read non-indexed indirect records; as in DrawIndex, the caller expands them.
		NoteUpstreamIndirectRecords();
		return false;
	}
	const DrawCensusScope       draw_census; // KYTY_TRACE_DRAW_CENSUS
	DrawEntryStart(buffer);
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	if (SkipDirectDrawsCeiling() && DebugPresentCount().load() >= 2600u) { // in the scene only (from frame 2600)
		ResetBindings();
		return true;
	}
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndexAuto), submit_id,
	                    args.vertex_count, 0, args.first_vertex, args.instance_count,
	                    args.first_instance);

	Common::LockGuard lock(m_context.GetMutex());
	m_current_slot = NextDrawSlot();
	if (args.vertex_count == 0 || args.instance_count == 0) {
		TraceDrawDrop("zero vertex/instance count", buffer);
		return true;
	}

	if (ColorOperationInstead(buffer, args.render_target_slice_offset, "metadata/copy/resolve operation")) {
		return true;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		TraceDrawDrop("no vertex shader", buffer);
		return true;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndexAuto():Shader:\n");
		uc_print("GraphicsRenderDrawIndexAuto():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndexAuto():Parameters:\n"
		     "\t vertex_count        = 0x%08" PRIx32 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t first_vertex        = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.vertex_count, args.instance_count, args.first_vertex, args.first_instance);
	}

	uc_check(ucfg);

	hw_check(buffer);

	const DrawCallInfo draw {CommandBufferDebugOp::DrawIndexAuto,
	                         args.vertex_count, args.instance_count, args.first_instance};

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) {
		TraceDrawDrop("primitive type", buffer);
		ResetBindings();
		return true;
	}
	DrawStateHolder  state_holder(m_current_slot->state, TakeResolvedState({}));
	DrawRenderState& state = state_holder.Get();
	BeginDrawPlan(); // the plan flags
	bool render_state = false;
	{
		render_state = PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state);
	}
	if (!render_state) {
		TraceRenderStateDrop("render state (targets/shaders)", buffer, state);
		ResetBindings();
		return true;
	}

	const bool rect_list = Prospero::IsRectList(ucfg.GetPrimType());
	if (rect_list && state.vertex_info[0].buffers_num == 0 &&
	    state.vertex_info[0].stage.program->param_export_mask == 0 &&
	    state.ps_input_info.input_num != 0) {
		if (graphics_debug_dump_enabled()) {
			LOGF("DrawIndexAuto: skipping rect-list draw with no VS param exports and PS inputs: "
			     "ps_inputs=%u ps=0x%016" PRIx64 " es=0x%016" PRIx64 " gs=0x%016" PRIx64 "\n",
			     state.ps_input_info.input_num, sh_ctx.GetPs().ps_regs.data_addr,
			     sh_ctx.GetVs().es_regs.data_addr, sh_ctx.GetVs().gs_regs.data_addr);
		}
		TraceDrawDrop("rect list without VS params", buffer);
		ResetBindings();
		return true;
	}

	LogDrawStateIfNeeded(buffer, draw, state, 0, nullptr);
	const bool indirect = args.offset_source == DrawOffsetSource::IndirectArgs;
	const auto [vertex_offset, instance_offset] =
	    indirect ? std::pair<int32_t, uint32_t> {0, args.first_instance}
	             : ResolveDrawOffsets(ucfg.GetIndexOffset(), state.vertex_info[0]);
	DrawEmitInfo emit {};
	const auto [native_vertex, native_instance] =
	    indirect ? IndirectDrawNativeOffsets(state.vertex_info[0])
	             : std::pair<bool, bool> {false, false};
	emit.first_vertex = native_vertex ? 0 :
	    static_cast<uint32_t>(vertex_offset + static_cast<int32_t>(args.first_vertex));
	emit.first_instance = native_instance ? 0 : instance_offset;
	emit.cpu_indirect   = indirect;
	if (!indirect) {
		TraceDrawBaseIndexOffset(ucfg.GetIndexOffset(), state.vertex_info[0]);
	}

	DrawIndexBufferSource index_source {};
	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source, false);
	ResetBindings();
	return true;
}

// GPU indirect draws (see CommandProcessor::DrawIndirectGpu): the
// draw reads its argument records and count from the cached GPU buffers of the guest ranges with
// vkCmdDraw[Indexed]Indirect[Count], so the command processor never reads GPU-written arguments
// back. Anything that needs the counts on the CPU (8-bit or custom-restart indices, mesh and
// tessellation stages, quad and rect lists) is refused before a draw command is recorded.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
const char* RenderExecutor::DrawIndirectGpu(uint64_t submit_id, CommandBuffer& buffer,
                                            const DrawIndirectGpuArgs& args) {
	// The checks that need only the registers and the packet (none of the draw's programs): IndirectEarlyChecks (N7.2: the
	// same function builds a helper's plan).
	IndirectEarly early;
	GpuBurnAt(GpuBurnSite::Indirect); // KYTY_DIAG_GPU_BURN_AT=indirect (measurement only)
	KYTY_PROFILER_FUNCTION();
	if (!buffer.IsInvalid()) FlushComputeAccess(buffer.Handle());
	EXIT_IF(buffer.IsInvalid());
	if (const char* reason = IndirectEarlyChecks(m_context.GetGraphics(), buffer.GetUserConfig(), buffer.GetRegisters(), args, false, early);
	    reason != nullptr) {
		return reason;
	}
	if (early.no_draw) return nullptr;
	const DrawIndirectSource&    indirect          = early.indirect;
	const DrawIndexBufferSource& index_source      = early.index_source;
	const bool                   primitive_restart = early.primitive_restart;
	auto&       ucfg     = buffer.GetUserConfig();
	auto&       sh_ctx   = buffer.GetShaders();

	m_context.GetCommandScheduler().PopPendingOperations();
	const auto debug_op =
	    args.indexed ? CommandBufferDebugOp::DrawIndex : CommandBufferDebugOp::DrawIndexAuto;
	buffer.SetDebugInfo(static_cast<uint32_t>(debug_op), submit_id, 0, 0, args.max_count, 0,
	                    args.args_addr);

	Common::LockGuard lock(m_context.GetMutex());
	m_current_slot = NextDrawSlot();
	if (ColorOperationInstead(buffer, 0, "metadata/copy/resolve operation (gpu indirect)")) {
		return nullptr;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		TraceDrawDrop("no vertex shader (gpu indirect)", buffer);
		return nullptr;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndirectGpu():Shader:\n");
		uc_print("GraphicsRenderDrawIndirectGpu():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndirectGpu():Parameters:\n"
		     "\t indexed             = %s\n"
		     "\t args_addr           = 0x%016" PRIx64 "\n"
		     "\t count_addr          = 0x%016" PRIx64 "\n"
		     "\t max_count           = 0x%08" PRIx32 "\n"
		     "\t stride              = 0x%08" PRIx32 "\n"
		     "\t index_addr          = 0x%016" PRIx64 "\n"
		     "\t index_buffer_size   = 0x%08" PRIx32 "\n"
		     "\t indirect_sgprs      = 0x%08" PRIx32 "\n",
		     args.indexed ? "true" : "false", args.args_addr, args.count_addr, args.max_count,
		     args.stride, args.index_base_addr, args.index_buffer_size, args.indirect_draw.Key());
	}

	uc_check(ucfg);

	hw_check(buffer);

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) {
		TraceDrawDrop("primitive type (gpu indirect)", buffer);
		ResetBindings();
		return nullptr;
	}

	// The counts live in GPU memory; DrawCallInfo carries none (debug output shows 0).
	const DrawCallInfo draw {debug_op, 0, 0, 0};
	DrawStateHolder    state_holder(m_current_slot->state, TakeResolvedState(args.indirect_draw));
	DrawRenderState&   state = state_holder.Get();
	state.indirect_draw = args.indirect_draw;
	BeginDrawPlan(); // the plan flags
	bool render_state = false;
	{
		render_state = PrepareDrawRenderState(buffer, draw, 0, state);
	}
	if (!render_state) {
		TraceRenderStateDrop("render state (targets/shaders, gpu indirect)", buffer, state);
		ResetBindings();
		return nullptr;
	}
	if (state.programs.VertexStageCount() != 1u ||
	    state.vertex_info[0].stage.program->stage != ShaderType::Vertex) {
		ResetBindings();
		return "vertex stage";
	}
	// The variant reads the patched SGPRs from draw parameters; host-evaluated descriptors cannot.
	if (state.vertex_info[0].indirect_draw_resource_sgprs) {
		ResetBindings();
		return "SGPR feeds host-evaluated resources";
	}

	LogDrawStateIfNeeded(buffer, draw, state, args.index_type_and_size,
	                     reinterpret_cast<const void*>(args.index_base_addr));

	const DrawEmitInfo emit {};
	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source,
	                    primitive_restart, &indirect);
	ResetBindings();
	return nullptr;
}

bool RenderExecutor::ResolveColorTargets(CommandBuffer& buffer, uint32_t render_target_slice_offset) {
	const auto& hw = buffer.GetRegisters();
	if (hw.GetColorControl().mode != 3) {
		return false;
	}

	const auto& src_rt = hw.GetRenderTarget(0);
	const auto& dst_rt = hw.GetRenderTarget(1);
	if (src_rt.base.addr == 0 || dst_rt.base.addr == 0) {
		return false;
	}

	RenderColorInfo src {};
	RenderColorInfo dst {};
	ResolveRenderColorTarget(buffer, src, render_target_slice_offset, 0, true, true);
	ResolveRenderColorTarget(buffer, dst, render_target_slice_offset, 1, true, true);
	if (!src.image_id || !dst.image_id) {
		return false;
	}
	if (src.desc.info.data.address == dst.desc.info.data.address &&
	    src.guest_mip_level == dst.guest_mip_level &&
	    src.guest_array_layer == dst.guest_array_layer) {
		return true;
	}

	auto& cache = m_context.GetTextureCache();
	cache.MarkGpuWritten(dst.image_id);
	auto& source      = cache.GetImage(src.image_id);
	auto& destination = cache.GetImage(dst.image_id);
	const auto past_chain = [](const Image& image, uint32_t level) {
		return level >= image.backing.mip_levels && image.backing.mip_levels < image.info.resources.levels;
	};
	if (past_chain(source, src.guest_mip_level) || past_chain(destination, dst.guest_mip_level)) {
		// A guest level past the host image's 1x1 level has no host subresource (ImageInfo::HostLevels): not resolved.
		return true;
	}
	destination.Resolve(source, {src.guest_mip_level, 1, src.guest_array_layer, 1},
	                    {dst.guest_mip_level, 1, dst.guest_array_layer, 1});
	return true;
}

} // namespace Libs::Graphics
