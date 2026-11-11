// Texture relocation: moving the clean textures that keep nearly
// empty VMA blocks alive. After the game streams textures in and out, device-local blocks stay allocated for a few
// bindless-pinned textures each (4.11: 9 blocks >= 50% empty held 1.98 GB of holes for 326 MiB of residents). VMA frees
// a block only when its last allocation goes, so the residents of the emptiest blocks get a new VkImage in a fuller block,
// a GPU copy of every level and layer, and the new backing inside the same Image (same ImageId: every table and heap
// keeps its slots, only the views in them are rewritten). The old images are destroyed at once (not parked), so their
// blocks empty and VMA gives them back to the driver.
#include "graphics/host_gpu/renderer/cache/textureCache.h"

#include "common/alignment.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/drawRepeat.h"
#include "graphics/host_gpu/renderer/pipeline/bindlessTable.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <vulkan/vulkan_format_traits.hpp>

namespace Libs::Graphics {

namespace {

// Thresholds (design §4-5, measured 4.-5.11): a batch when the empty space inside the device-local blocks reaches 1 GiB,
// at most once per 10 s, out of blocks at least half empty, whole blocks up to 128 MiB of residents.
// KYTY_DIAG_TEXMOVE_FAST=1 (diagnostic, 11.11, default off, removed with the square-shadow fix): many more batches
// (300 MiB, 3 s, 30% - the 4.11 check runs' settings), so a fault that rare batches cause shows up often.
constexpr uint64_t BatchBytes = uint64_t {128} << 20u;
struct TexMoveThresholds {
	uint64_t min_empty   = uint64_t {1024} << 20u;
	uint64_t interval_ms = 10000;
	uint64_t block_pct   = 50;
};
const TexMoveThresholds& Thresholds() {
	static const TexMoveThresholds th = [] {
		TexMoveThresholds t;
		const char*       value = std::getenv("KYTY_DIAG_TEXMOVE_FAST");
		if (value != nullptr && value[0] == '1') {
			t = {uint64_t {300} << 20u, 3000, 30};
			std::fprintf(stderr, "TEXMOVE fast (KYTY_DIAG_TEXMOVE_FAST=1): min_empty_MiB=300 interval_s=3 block_empty_pct=30\n");
		}
		return t;
	}();
	return th;
}

// A texture moved less than a minute ago stays (no ping-pong between blocks).
constexpr uint64_t MoveAgainMs = 60000;

uint64_t NowMs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

// The wall clock (HH:MM:SS) for the log lines a player's screenshot time is matched against.
std::string ClockText() {
	const std::time_t t = std::time(nullptr);
	std::tm           local {};
#ifdef _WIN32
	localtime_s(&local, &t);
#else
	localtime_r(&t, &local);
#endif
	char text[16];
	std::strftime(text, sizeof(text), "%H:%M:%S", &local);
	return text;
}

double MsSince(std::chrono::steady_clock::time_point start) {
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

vk::Extent3D LevelExtent(const VulkanImage& image, uint32_t level) {
	return {std::max(image.extent.width >> level, 1u), std::max(image.extent.height >> level, 1u),
	        image.image_type == vk::ImageType::e3D ? std::max(image.extent.depth >> level, 1u) : 1u};
}

// Whether the texture may move now: a registered, clean color texture (its contents are its guest bytes or an upload of
// them: not GPU- or buffer-written), no target, storage, depth, stencil, metadata or video-out image, nothing pending on
// it (eviction, image read, probe, rebind), not bound by a draw, not referenced by a resident image heap, one layout for
// all its subresources (a whole-image barrier and copy), single-sampled, contents defined. Null when it may move, else
// the first reason it may not (the TEXMOVE line names the first one of a batch).
const char* Unmovable(const Image& image, uint64_t now_ms) {
	const auto& b = image.backing;
	if (!image.registered || image.info.data.Empty() || b.image == nullptr || b.allocation == nullptr) return "unregistered";
	if (image.IsGpuModified() || image.IsBufferModified()) return "gpu_written";
	if (image.usage.render_target || image.usage.depth_target || image.usage.storage || image.usage.video_out) return "target";
	if (image.info.IsDepth() || image.info.HasStencil() || image.info.HasMetadata() || image.depth_id ||
	    !image.stencil_ids.empty() || vk::blockSize(b.format) == 0 || b.samples != 1)
		return "kind";
	if (image.eviction_pending || image.read_armed || image.probe_pending != 0) return "pending";
	if (image.heap_refs.load(std::memory_order_relaxed) != 0 || image.plan_uses != 0 || image.binding.is_bound ||
	    image.binding.is_target || image.binding.needs_rebind)
		return "bound";
	if (!b.subresource_states.empty() || b.state.layout == vk::ImageLayout::eUndefined) return "layout";
	if (image.texmove_ms != 0 && now_ms - image.texmove_ms < MoveAgainMs) return "moved_recently";
	return nullptr;
}

vk::ImageMemoryBarrier2 ImageBarrier(vk::Image image, vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access,
                                     vk::ImageLayout old_layout, vk::PipelineStageFlags2 dst_stage,
                                     vk::AccessFlags2 dst_access, vk::ImageLayout new_layout) {
	vk::ImageMemoryBarrier2 barrier {};
	barrier.srcStageMask        = src_stage;
	barrier.srcAccessMask       = src_access;
	barrier.dstStageMask        = dst_stage;
	barrier.dstAccessMask       = dst_access;
	barrier.oldLayout           = old_layout;
	barrier.newLayout           = new_layout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image               = image;
	barrier.subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
	return barrier;
}

void PipelineBarrier(vk::CommandBuffer command, const std::vector<vk::ImageMemoryBarrier2>& barriers) {
	if (barriers.empty()) return;
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
	dependency.pImageMemoryBarriers    = barriers.data();
	command.pipelineBarrier2(dependency);
}

// ---- The stale-view guard (11.11, issue #11: square shadows after cutscenes, since relocation came in 0.1.4). A holder
// the swap does not update (a cached view, a set, a prepared binding, a queued write) would keep using a moved texture's
// old view or image; destroyed at once, the handle can be reused by a new view and the draw samples another image. The
// old views and images stay alive for RetireMs (a stale user still samples the right bytes, so the guard changes no
// image), and every point where a view or image enters GPU work looks them up here. Port of the 4.11 detector
// (422b6eee3, then only in check mode), on by default so players' logs carry it.
constexpr uint64_t RetireMs = 10000;

bool GuardOn() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_TEXMOVE_GUARD");
		return value == nullptr || value[0] != '0';
	}();
	return on;
}

struct RetiredInfo {
	uint64_t     image_id = 0; // ImageId generation << 32 | index
	uint64_t     address  = 0; // guest address of the texture
	vk::Format   format   = vk::Format::eUndefined;
	vk::Extent3D extent {};
	uint32_t     levels = 0, layers = 0;
	uint64_t     frame = 0; // presented frame of the batch that moved it
	bool         view  = false;
};
struct RetiredBatch {
	uint64_t                                  deadline_ms = 0, frame = 0;
	std::vector<vk::ImageView>                views;
	std::vector<std::unique_ptr<VulkanImage>> images;
};
std::shared_mutex                         g_retired_mutex;
std::unordered_map<uint64_t, RetiredInfo> g_retired;         // old view and image handles
std::vector<RetiredBatch>                 g_retired_batches; // GPU thread
std::atomic<uint64_t>                     g_guard_frame {0};
std::atomic<uint64_t>                     g_stale_uses {0}, g_stale_checked {0}; // checked: filter hits looked up
// A filter over the retired handles (one bit per hash, 64 Ki bits): the hot path reads one word and takes the lock only
// on a hit (11.11: a lock and a shared counter per handle cost ~5% fps while batches moved every few seconds). Bits are
// set under the unique lock; an expiry rebuilds the words from the handles that stay (a word never loses the bit of a
// handle still retired, so a stale use is never missed).
constexpr uint32_t                                    FilterBits = 1u << 16u;
std::array<std::atomic<uint64_t>, FilterBits / 64u>   g_filter {};
uint32_t FilterIndex(uint64_t handle) {
	return static_cast<uint32_t>((handle * 0x9E3779B97F4A7C15ull) >> 48u);
}
void FilterSet(uint64_t handle) {
	const uint32_t i = FilterIndex(handle);
	g_filter[i >> 6u].fetch_or(uint64_t {1} << (i & 63u), std::memory_order_relaxed);
}
void FilterRebuild() { // under the unique lock
	std::array<uint64_t, FilterBits / 64u> words {};
	for (const auto& [handle, info]: g_retired) {
		const uint32_t i = FilterIndex(handle);
		words[i >> 6u] |= uint64_t {1} << (i & 63u);
	}
	for (size_t w = 0; w < words.size(); w++) g_filter[w].store(words[w], std::memory_order_relaxed);
}
// The image entries of every descriptor update template (their data holds VkDescriptorImageInfo at offset + n * stride).
std::mutex                                                                  g_template_mutex;
std::unordered_map<uint64_t, std::vector<VkDescriptorUpdateTemplateEntry>> g_templates;

bool ImageDescriptor(VkDescriptorType type) {
	return type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE || type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ||
	       type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER || type == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
}
void CheckWrites(uint32_t count, const VkWriteDescriptorSet* writes, const char* path) {
	if (!g_texmove_watch.load(std::memory_order_relaxed) || writes == nullptr) return;
	for (uint32_t i = 0; i < count; i++) {
		if (!ImageDescriptor(writes[i].descriptorType) || writes[i].pImageInfo == nullptr) continue;
		for (uint32_t j = 0; j < writes[i].descriptorCount; j++) TexMoveCheck(writes[i].pImageInfo[j].imageView, path);
	}
}
void CheckTemplate(VkDescriptorUpdateTemplate update, const void* data, const char* path) {
	if (!g_texmove_watch.load(std::memory_order_relaxed) || data == nullptr) return;
	std::lock_guard lock(g_template_mutex);
	const auto      found = g_templates.find(reinterpret_cast<uint64_t>(update));
	if (found == g_templates.end()) return;
	for (const auto& entry: found->second) {
		for (uint32_t j = 0; j < entry.descriptorCount; j++) {
			VkDescriptorImageInfo info {};
			std::memcpy(&info, static_cast<const uint8_t*>(data) + entry.offset + size_t {j} * entry.stride, sizeof(info));
			TexMoveCheck(info.imageView, path);
		}
	}
}
void CheckAttachment(const VkRenderingAttachmentInfo* attachment, const char* path) {
	if (attachment == nullptr) return;
	TexMoveCheck(attachment->imageView, path);
	TexMoveCheck(attachment->resolveImageView, path);
}

// The dispatcher's functions before the wrappers (CommandRecorder's hooks).
#define TEXMOVE_GUARDED_FUNCTIONS(X)                                                                                       \
	X(vkUpdateDescriptorSets)                                                                                          \
	X(vkUpdateDescriptorSetWithTemplate)                                                                               \
	X(vkCreateDescriptorUpdateTemplate)                                                                                \
	X(vkCmdPushDescriptorSetKHR)                                                                                       \
	X(vkCmdPushDescriptorSetWithTemplateKHR)                                                                           \
	X(vkCmdPipelineBarrier)                                                                                            \
	X(vkCmdPipelineBarrier2)                                                                                           \
	X(vkCmdBeginRendering)                                                                                             \
	X(vkCmdCopyImage)                                                                                                  \
	X(vkCmdCopyBufferToImage)                                                                                          \
	X(vkCmdCopyImageToBuffer)                                                                                          \
	X(vkCmdResolveImage)                                                                                               \
	X(vkCmdBlitImage)                                                                                                  \
	X(vkCmdClearColorImage)                                                                                            \
	X(vkCmdClearDepthStencilImage)                                                                                     \
	X(vkCreateImageView)
#define TEXMOVE_PREVIOUS(name) PFN_##name g_prev_##name = nullptr;
TEXMOVE_GUARDED_FUNCTIONS(TEXMOVE_PREVIOUS)
#undef TEXMOVE_PREVIOUS

VKAPI_ATTR void VKAPI_CALL Wrap_vkUpdateDescriptorSets(VkDevice device, uint32_t count, const VkWriteDescriptorSet* writes,
                                                       uint32_t copy_count, const VkCopyDescriptorSet* copies) {
	CheckWrites(count, writes, "vkUpdateDescriptorSets");
	g_prev_vkUpdateDescriptorSets(device, count, writes, copy_count, copies);
}
VKAPI_ATTR void VKAPI_CALL Wrap_vkUpdateDescriptorSetWithTemplate(VkDevice device, VkDescriptorSet set,
                                                                  VkDescriptorUpdateTemplate update, const void* data) {
	CheckTemplate(update, data, "vkUpdateDescriptorSetWithTemplate");
	g_prev_vkUpdateDescriptorSetWithTemplate(device, set, update, data);
}
VKAPI_ATTR VkResult VKAPI_CALL Wrap_vkCreateDescriptorUpdateTemplate(VkDevice device,
                                                                     const VkDescriptorUpdateTemplateCreateInfo* info,
                                                                     const VkAllocationCallbacks* callbacks,
                                                                     VkDescriptorUpdateTemplate* update) {
	const VkResult result = g_prev_vkCreateDescriptorUpdateTemplate(device, info, callbacks, update);
	if (result == VK_SUCCESS && info != nullptr && update != nullptr) {
		std::vector<VkDescriptorUpdateTemplateEntry> images;
		for (uint32_t i = 0; i < info->descriptorUpdateEntryCount; i++) {
			if (ImageDescriptor(info->pDescriptorUpdateEntries[i].descriptorType))
				images.push_back(info->pDescriptorUpdateEntries[i]);
		}
		std::lock_guard lock(g_template_mutex);
		g_templates[reinterpret_cast<uint64_t>(*update)] = std::move(images);
	}
	return result;
}
VKAPI_ATTR void VKAPI_CALL Wrap_vkCmdPushDescriptorSetKHR(VkCommandBuffer buffer, VkPipelineBindPoint point,
                                                          VkPipelineLayout layout, uint32_t set, uint32_t count,
                                                          const VkWriteDescriptorSet* writes) {
	CheckWrites(count, writes, "vkCmdPushDescriptorSetKHR");
	g_prev_vkCmdPushDescriptorSetKHR(buffer, point, layout, set, count, writes);
}
VKAPI_ATTR void VKAPI_CALL Wrap_vkCmdPushDescriptorSetWithTemplateKHR(VkCommandBuffer buffer,
                                                                      VkDescriptorUpdateTemplate update,
                                                                      VkPipelineLayout layout, uint32_t set,
                                                                      const void* data) {
	CheckTemplate(update, data, "vkCmdPushDescriptorSetWithTemplateKHR");
	g_prev_vkCmdPushDescriptorSetWithTemplateKHR(buffer, update, layout, set, data);
}
VKAPI_ATTR void VKAPI_CALL Wrap_vkCmdPipelineBarrier(VkCommandBuffer buffer, VkPipelineStageFlags src,
                                                     VkPipelineStageFlags dst, VkDependencyFlags flags,
                                                     uint32_t memory_count, const VkMemoryBarrier* memory,
                                                     uint32_t buffer_count, const VkBufferMemoryBarrier* buffers,
                                                     uint32_t image_count, const VkImageMemoryBarrier* images) {
	for (uint32_t i = 0; images != nullptr && i < image_count; i++) TexMoveCheck(images[i].image, "vkCmdPipelineBarrier");
	g_prev_vkCmdPipelineBarrier(buffer, src, dst, flags, memory_count, memory, buffer_count, buffers, image_count, images);
}
VKAPI_ATTR void VKAPI_CALL Wrap_vkCmdPipelineBarrier2(VkCommandBuffer buffer, const VkDependencyInfo* info) {
	for (uint32_t i = 0; info != nullptr && info->pImageMemoryBarriers != nullptr && i < info->imageMemoryBarrierCount; i++)
		TexMoveCheck(info->pImageMemoryBarriers[i].image, "vkCmdPipelineBarrier2");
	g_prev_vkCmdPipelineBarrier2(buffer, info);
}
VKAPI_ATTR void VKAPI_CALL Wrap_vkCmdBeginRendering(VkCommandBuffer buffer, const VkRenderingInfo* info) {
	if (info != nullptr && g_texmove_watch.load(std::memory_order_relaxed)) {
		for (uint32_t i = 0; info->pColorAttachments != nullptr && i < info->colorAttachmentCount; i++)
			CheckAttachment(&info->pColorAttachments[i], "vkCmdBeginRendering");
		CheckAttachment(info->pDepthAttachment, "vkCmdBeginRendering");
		CheckAttachment(info->pStencilAttachment, "vkCmdBeginRendering");
	}
	g_prev_vkCmdBeginRendering(buffer, info);
}
VKAPI_ATTR void VKAPI_CALL Wrap_vkCmdCopyImage(VkCommandBuffer buffer, VkImage source, VkImageLayout source_layout,
                                               VkImage target, VkImageLayout target_layout, uint32_t count,
                                               const VkImageCopy* regions) {
	TexMoveCheck(source, "vkCmdCopyImage");
	TexMoveCheck(target, "vkCmdCopyImage");
	g_prev_vkCmdCopyImage(buffer, source, source_layout, target, target_layout, count, regions);
}
VKAPI_ATTR void VKAPI_CALL Wrap_vkCmdCopyBufferToImage(VkCommandBuffer buffer, VkBuffer source, VkImage target,
                                                       VkImageLayout layout, uint32_t count, const VkBufferImageCopy* regions) {
	TexMoveCheck(target, "vkCmdCopyBufferToImage");
	g_prev_vkCmdCopyBufferToImage(buffer, source, target, layout, count, regions);
}
VKAPI_ATTR void VKAPI_CALL Wrap_vkCmdCopyImageToBuffer(VkCommandBuffer buffer, VkImage source, VkImageLayout layout,
                                                       VkBuffer target, uint32_t count, const VkBufferImageCopy* regions) {
	TexMoveCheck(source, "vkCmdCopyImageToBuffer");
	g_prev_vkCmdCopyImageToBuffer(buffer, source, layout, target, count, regions);
}
VKAPI_ATTR void VKAPI_CALL Wrap_vkCmdResolveImage(VkCommandBuffer buffer, VkImage source, VkImageLayout source_layout,
                                                  VkImage target, VkImageLayout target_layout, uint32_t count,
                                                  const VkImageResolve* regions) {
	TexMoveCheck(source, "vkCmdResolveImage");
	TexMoveCheck(target, "vkCmdResolveImage");
	g_prev_vkCmdResolveImage(buffer, source, source_layout, target, target_layout, count, regions);
}
VKAPI_ATTR void VKAPI_CALL Wrap_vkCmdBlitImage(VkCommandBuffer buffer, VkImage source, VkImageLayout source_layout,
                                               VkImage target, VkImageLayout target_layout, uint32_t count,
                                               const VkImageBlit* regions, VkFilter filter) {
	TexMoveCheck(source, "vkCmdBlitImage");
	TexMoveCheck(target, "vkCmdBlitImage");
	g_prev_vkCmdBlitImage(buffer, source, source_layout, target, target_layout, count, regions, filter);
}
VKAPI_ATTR void VKAPI_CALL Wrap_vkCmdClearColorImage(VkCommandBuffer buffer, VkImage image, VkImageLayout layout,
                                                     const VkClearColorValue* color, uint32_t count,
                                                     const VkImageSubresourceRange* ranges) {
	TexMoveCheck(image, "vkCmdClearColorImage");
	g_prev_vkCmdClearColorImage(buffer, image, layout, color, count, ranges);
}
VKAPI_ATTR void VKAPI_CALL Wrap_vkCmdClearDepthStencilImage(VkCommandBuffer buffer, VkImage image, VkImageLayout layout,
                                                            const VkClearDepthStencilValue* value, uint32_t count,
                                                            const VkImageSubresourceRange* ranges) {
	TexMoveCheck(image, "vkCmdClearDepthStencilImage");
	g_prev_vkCmdClearDepthStencilImage(buffer, image, layout, value, count, ranges);
}
VKAPI_ATTR VkResult VKAPI_CALL Wrap_vkCreateImageView(VkDevice device, const VkImageViewCreateInfo* info,
                                                      const VkAllocationCallbacks* callbacks, VkImageView* view) {
	if (info != nullptr) TexMoveCheck(info->image, "vkCreateImageView");
	return g_prev_vkCreateImageView(device, info, callbacks, view);
}

// The retire period is over (or `all`): the old views and images go; what the driver got back is counted.
void ExpireRetired(GraphicContext& graphics, uint64_t now, bool all) {
	for (size_t i = 0; i < g_retired_batches.size();) {
		auto& batch = g_retired_batches[i];
		if (!all && now < batch.deadline_ms) {
			i++;
			continue;
		}
		{
			std::unique_lock lock(g_retired_mutex);
			for (const auto view: batch.views) g_retired.erase(reinterpret_cast<uint64_t>(static_cast<VkImageView>(view)));
			for (const auto& image: batch.images)
				g_retired.erase(reinterpret_cast<uint64_t>(static_cast<VkImage>(image->image)));
			FilterRebuild();
			g_texmove_watch.store(!g_retired.empty(), std::memory_order_relaxed);
		}
		for (const auto view: batch.views) graphics.device.destroyImageView(view, nullptr);
		uint64_t frees_before = 0, bytes_before = 0, frees_after = 0, bytes_after = 0;
		GraphicContext::DeviceMemoryFrees(frees_before, bytes_before);
		for (auto& image: batch.images) {
			if (image != nullptr && image->image != nullptr) graphics.DestroyImageNow(*image);
		}
		GraphicContext::DeviceMemoryFrees(frees_after, bytes_after);
		std::fprintf(stderr,
		             "TEXMOVE_RETIRE frame=%llu time=%s batch_frame=%llu views=%zu images=%zu returned=%llu returned_MiB=%llu "
		             "stale_uses_total=%llu checked_total=%llu\n",
		             static_cast<unsigned long long>(g_guard_frame.load()), ClockText().c_str(),
		             static_cast<unsigned long long>(batch.frame), batch.views.size(), batch.images.size(),
		             static_cast<unsigned long long>(frees_after - frees_before),
		             static_cast<unsigned long long>((bytes_after - bytes_before) >> 20u),
		             static_cast<unsigned long long>(g_stale_uses.load()),
		             static_cast<unsigned long long>(g_stale_checked.load()));
		g_retired_batches.erase(g_retired_batches.begin() + static_cast<std::ptrdiff_t>(i));
	}
}

} // namespace

void TexMoveStaleCheck(uint64_t handle, const char* path) {
	const uint32_t index = FilterIndex(handle);
	if ((g_filter[index >> 6u].load(std::memory_order_relaxed) & (uint64_t {1} << (index & 63u))) == 0) return;
	g_stale_checked.fetch_add(1, std::memory_order_relaxed);
	RetiredInfo info;
	{
		std::shared_lock lock(g_retired_mutex);
		const auto       found = g_retired.find(handle);
		if (found == g_retired.end()) return;
		info = found->second;
	}
	const uint64_t n = g_stale_uses.fetch_add(1);
	if (n >= 64u) return; // the first 64 uses; the total is on the TEXMOVE_RETIRE lines
	std::fprintf(stderr,
	             "TEXMOVE_STALE_USE frame=%llu time=%s path=%s %s=0x%llx imageid=%u/%u guest=0x%llx fmt=%s extent=%ux%ux%u "
	             "levels=%u layers=%u moved_at_frame=%llu\n",
	             static_cast<unsigned long long>(g_guard_frame.load()), ClockText().c_str(), path, info.view ? "view" : "image",
	             static_cast<unsigned long long>(handle), static_cast<uint32_t>(info.image_id & 0xffffffffu),
	             static_cast<uint32_t>(info.image_id >> 32u), static_cast<unsigned long long>(info.address),
	             vk::to_string(info.format).c_str(), info.extent.width, info.extent.height, info.extent.depth, info.levels,
	             info.layers, static_cast<unsigned long long>(info.frame));
}

void TexMoveInstallGuard() {
	if (!GuardOn()) {
		std::fprintf(stderr, "TEXMOVE stale-view guard off (KYTY_TEXMOVE_GUARD=0)\n");
		return;
	}
	auto&       d = VULKAN_HPP_DEFAULT_DISPATCHER;
	std::string missing;
#define TEXMOVE_WRAP(name)                                                                                                 \
	if (d.name != nullptr) {                                                                                           \
		g_prev_##name = d.name;                                                                                    \
		d.name        = Wrap_##name;                                                                               \
	} else {                                                                                                           \
		missing += " " #name;                                                                                      \
	}
	TEXMOVE_GUARDED_FUNCTIONS(TEXMOVE_WRAP)
#undef TEXMOVE_WRAP
	std::fprintf(stderr, "TEXMOVE stale-view guard installed (missing:%s)\n", missing.empty() ? " none" : missing.c_str());
}

void TextureCache::RunTextureMove(BindlessTable& table) {
	if (!m_scheduler.Active()) return;
	const auto&    th  = Thresholds();
	const uint64_t now = NowMs();
	if (GuardOn()) { // the stale-view guard: the frame its lines name, and the old views and images whose time is up
		g_guard_frame.store(m_graphics.presented_frames.load(std::memory_order_relaxed), std::memory_order_relaxed);
		ExpireRetired(m_graphics, now, false);
	}
	static uint64_t last = now; // the first batch an interval after the first frame boundary
	// Back-off: an attempt that moved nothing and ended exactly like the one before (same candidate blocks, same
	// allocation counts, same reasons) doubles the wait, up to 60 s; anything else resets it to the interval.
	static uint64_t wait_ms = th.interval_ms, last_signature = 0;
	if (now - last < wait_ms) return;
	uint64_t usage = 0, budget = 0, block_bytes = 0, allocated = 0;
	m_graphics.GetDeviceLocalMemoryStats(usage, budget, block_bytes, allocated);
	const uint64_t empty = block_bytes > allocated ? block_bytes - allocated : 0;
	if (empty < th.min_empty) return;
	last = now; // one attempt per interval, whatever it finds

	const auto start = std::chrono::steady_clock::now();
	struct Move {
		ImageId                                    id;
		uint64_t                                   serial = 0;
		VmaAllocation                              old_allocation = nullptr;
		VulkanImageState                           state;
		std::unique_ptr<VulkanImage>               other; // the new image; after the swap, the old one
	};
	struct Drain {
		VkDeviceMemory    memory = VK_NULL_HANDLE;
		uint32_t          type   = 0;
		uint64_t          used   = 0;
		std::vector<Move> moves;
		bool              ok = true;
	};
	struct Counters {
		uint64_t candidates = 0, foreign = 0, busy = 0, over_batch = 0, no_room = 0, dest_in_drain = 0, newblock = 0;
		uint64_t plugs = 0, plug_bytes = 0, images = 0, moved = 0, emptied = 0, returned = 0, returned_bytes = 0;
		uint64_t slots = 0, memos = 0;
		const char* busy_why = "-";
	} n;
	double                  ms_record = 0, ms_wait = 0, ms_swap = 0;
	std::vector<Drain>      drains;
	double                  ms_census = 0;
	uint64_t                signature = 0xcbf29ce484222325ull; // 0 changes: the attempt never reached the census
	const auto report = [&] {
		if (n.images == 0 && signature != 0 && signature == last_signature) {
			wait_ms = std::min<uint64_t>(wait_ms * 2u, std::max<uint64_t>(th.interval_ms, 60000u));
		} else {
			wait_ms = th.interval_ms;
		}
		last_signature = signature;
		const auto u = [](uint64_t v) { return static_cast<unsigned long long>(v); };
		std::fprintf(stderr,
		             "TEXMOVE frame=%llu time=%s trigger_empty_MiB=%llu candidate_blocks=%llu skip(foreign=%llu busy=%llu busy_why=%s "
		             "over_batch=%llu no_room=%llu) dest_in_drain=%llu plugs=%llu/%lluMiB newblock_fallback=%llu drained=%zu "
		             "images=%llu moved_MiB=%.1f emptied=%llu returned=%llu returned_MiB=%llu slots_rewritten=%llu "
		             "memo_invalidations=%llu ms_record=%.2f ms_wait=%.2f ms_swap=%.2f ms_total=%.2f ms_census=%.2f "
		             "next_s=%.0f\n",
		             u(m_graphics.presented_frames.load(std::memory_order_relaxed)), ClockText().c_str(), u(empty >> 20u), u(n.candidates),
		             u(n.foreign), u(n.busy), n.busy_why, u(n.over_batch), u(n.no_room), u(n.dest_in_drain), u(n.plugs),
		             u(n.plug_bytes >> 20u), u(n.newblock), drains.size(), u(n.images), static_cast<double>(n.moved) / 1048576.0,
		             u(n.emptied), u(n.returned), u(n.returned_bytes >> 20u), u(n.slots), u(n.memos), ms_record, ms_wait,
		             ms_swap, MsSince(start), ms_census, static_cast<double>(wait_ms) / 1000.0);
	};

	// 1. Census and destinations, recorded copies (under m_lock).
	{
		std::scoped_lock lock {m_lock};
		if (table.DrawHeapCount() != 0 || table.DrawTableCount() != 0) { // a prepared draw not committed yet
			n.busy++;
			signature = 0; // a prepared draw: no back-off
			report();
			return;
		}
		// The census, kept cheap (most attempts move nothing): every block's size and fill, the allocations only of the
		// candidate blocks, and the images only for those allocations. Parked images are not trimmed here any more (a
		// drain that waits on the GPU): a block holding one is skipped until they go (2 s).
		const auto candidate = [&](const GraphicContext::BlockCensus& block) {
			return block.count != 0 && block.free * 100u >= block.size * th.block_pct;
		};
		auto census = m_graphics.CollectBlocks(static_cast<uint32_t>(th.block_pct));
		std::unordered_map<VmaAllocation, ImageId> owners;
		{
			std::unordered_set<VmaAllocation> wanted;
			for (const auto& block: census) {
				if (candidate(block)) wanted.insert(block.allocations.begin(), block.allocations.end());
			}
			if (!wanted.empty()) {
				m_slot_images.ForEach([&](ImageId id, const Image& image) {
					if (image.backing.allocation != nullptr && wanted.contains(image.backing.allocation))
						owners.emplace(image.backing.allocation, id);
				});
			}
		}
		std::unordered_set<VmaAllocation> parked;
		for (const auto allocation: m_graphics.RecycledAllocations()) parked.insert(allocation);
		// Emptiest first; whole blocks up to the batch size.
		std::ranges::sort(census, [](const auto& a, const auto& b) { return a.size - a.free < b.size - b.free; });
		ms_census = MsSince(start);
		// Which candidate blocks could empty (every allocation a movable texture); the outcome of each goes into the
		// attempt's signature (back-off: the same blocks failing the same way do not need a census every interval).
		const auto mix = [&](uint64_t value) { signature = (signature ^ value) * 0x100000001b3ull; };
		std::vector<Drain> eligible;
		for (const auto& block: census) {
			if (!candidate(block)) continue;
			n.candidates++;
			Drain       drain {block.memory, block.type, block.size - block.free, {}, true};
			const char* why = nullptr;
			for (const auto allocation: block.allocations) {
				const auto owner = owners.find(allocation);
				if (parked.contains(allocation)) {
					why = "parked";
				} else if (owner == owners.end()) { // a buffer, a placeholder, an image the cache does not hold
					why = "foreign";
					n.foreign++;
				} else if (why = Unmovable(m_slot_images[owner->second], now); why != nullptr) {
					if (n.busy++ == 0) n.busy_why = why;
				}
				if (why != nullptr) break;
				const auto& image = m_slot_images[owner->second];
				drain.moves.push_back({owner->second, image.serial, allocation, image.backing.state, nullptr});
			}
			mix(reinterpret_cast<uint64_t>(block.memory));
			mix(block.count);
			mix(reinterpret_cast<uint64_t>(why));
			if (why == nullptr) eligible.push_back(std::move(drain));
		}
		// Room before any allocation (5.11: attempts that ended with no_room still allocated plugs every interval): the
		// moves of each eligible block, largest first, must fit the free ranges of the other blocks of its memory type that
		// stay (not empty, not being emptied), best fit, with the allocation's alignment as slack; the one new block a batch
		// of two or more may take counts as a free range of the largest block size.
		if (!eligible.empty()) {
			const auto                                  ranges = m_graphics.CollectBlocks(101u, true); // no lists, ranges
			std::unordered_map<uint32_t, std::multiset<uint64_t>> room;
			std::unordered_map<uint32_t, uint64_t>                largest;
			for (const auto& block: ranges) {
				largest[block.type] = std::max(largest[block.type], block.size);
				if (block.count == 0 ||
				    std::ranges::any_of(eligible, [&](const Drain& drain) { return drain.memory == block.memory; }))
					continue;
				room[block.type].insert(block.free_ranges.begin(), block.free_ranges.end());
			}
			uint64_t batch_used = 0;
			bool     new_block  = false;
			for (auto& drain: eligible) {
				if (batch_used + drain.used > BatchBytes) {
					n.over_batch++;
					mix(1);
					continue;
				}
				std::vector<uint64_t> needs;
				for (const auto& move: drain.moves) {
					const uint64_t size  = m_graphics.GetAllocationSize(move.old_allocation);
					const uint64_t align = size >= (64ull << 10u) ? (64ull << 10u) : 4096u;
					needs.push_back(Common::AlignUp(size, align) + align - 1u);
				}
				std::ranges::sort(needs, std::greater<>());
				const auto fits = [&](std::multiset<uint64_t> pool) {
					for (const auto need: needs) {
						const auto range = pool.lower_bound(need);
						if (range == pool.end()) return std::optional<std::multiset<uint64_t>> {};
						const uint64_t rest = *range - need;
						pool.erase(range);
						if (rest != 0) pool.insert(rest);
					}
					return std::optional<std::multiset<uint64_t>> {std::move(pool)};
				};
				auto& pool  = room[drain.type];
				auto  after = fits(pool);
				if (!after && !new_block && !drains.empty()) { // the batch empties two or more: one new block allowed
					auto with_block = pool;
					with_block.insert(largest[drain.type]);
					if ((after = fits(std::move(with_block)))) new_block = true;
				}
				if (!after) {
					n.no_room++;
					mix(2);
					continue;
				}
				pool = std::move(*after);
				batch_used += drain.used;
				drains.push_back(std::move(drain));
			}
		}
		// The blocks no destination may take: those being emptied, and the empty block VMA keeps (a texture moved into it
		// only moves the hole: VMA frees an emptied block only while another empty one exists; the first run's batches
		// emptied 3 blocks and returned none). Their free ranges are plugged first; a destination that still lands in one
		// stays there as a plug too, and the next try goes elsewhere.
		std::unordered_map<uint32_t, std::vector<VkDeviceMemory>>   avoid;
		std::unordered_map<uint32_t, std::pair<uint64_t, uint64_t>> plug_sizes; // per type: largest free, smallest move
		for (const auto& drain: drains) {
			avoid[drain.type].push_back(drain.memory);
			auto& sizes = plug_sizes.try_emplace(drain.type, 0, ~uint64_t {0}).first->second;
			for (const auto& move: drain.moves)
				sizes.second = std::min(sizes.second, m_graphics.GetAllocationSize(move.old_allocation));
		}
		for (const auto& block: census) {
			if (!avoid.contains(block.type)) continue;
			auto& list = avoid[block.type];
			if (block.count == 0) list.push_back(block.memory);
			if (std::ranges::find(list, block.memory) != list.end())
				plug_sizes[block.type].first = std::max(plug_sizes[block.type].first, block.free);
		}
		std::vector<VmaAllocation> plugs;
		for (const auto& [type, sizes]: plug_sizes) {
			for (const auto plug: m_graphics.PlugBlocks(type, avoid[type], sizes.first, sizes.second)) {
				n.plugs++;
				n.plug_bytes += m_graphics.GetAllocationSize(plug);
				plugs.push_back(plug);
			}
		}
		std::vector<std::unique_ptr<VulkanImage>> image_plugs;
		// The new images: in a fuller block of the same memory type. One new block per batch is allowed when the batch
		// empties two or more (gain = emptied - 1).
		bool fallback = drains.size() >= 2;
		for (auto& drain: drains) {
			for (auto& move: drain.moves) {
				const auto&    b = m_slot_images[move.id].backing;
				VkDeviceMemory memory = VK_NULL_HANDLE;
				GraphicContext::ImageRelocation relocation {};
				(void)m_graphics.AllocationPlace(b.allocation, memory, relocation.memory_type);
				vk::ImageCreateInfo create {};
				create.flags         = b.flags;
				create.imageType     = b.image_type;
				create.format        = b.format;
				create.extent        = b.extent;
				create.mipLevels     = b.mip_levels;
				create.arrayLayers   = b.layers;
				create.samples       = vk::SampleCountFlagBits::e1;
				create.tiling        = vk::ImageTiling::eOptimal;
				create.usage         = b.usage;
				create.sharingMode   = vk::SharingMode::eExclusive;
				create.initialLayout = vk::ImageLayout::eUndefined;
				const auto& avoided = avoid[relocation.memory_type];
				for (;;) { // ends: every try that lands in an avoided block keeps some of its free space
					move.other   = std::make_unique<VulkanImage>();
					bool created = m_graphics.CreateImage(create, *move.other, &relocation);
					if (!created && fallback) {
						fallback                   = false;
						relocation.allow_new_block = true;
						n.newblock++;
						created = m_graphics.CreateImage(create, *move.other, &relocation);
					}
					if (!created) {
						n.no_room++;
						move.other.reset();
						drain.ok = false;
						break;
					}
					uint32_t type = 0;
					(void)m_graphics.AllocationPlace(move.other->allocation, memory, type);
					if (std::ranges::find(avoided, memory) == avoided.end()) break;
					n.dest_in_drain++;
					image_plugs.push_back(std::move(move.other));
				}
				if (!drain.ok) break;
			}
			if (!drain.ok) { // the block stays: its new images go
				for (auto& move: drain.moves) {
					if (move.other != nullptr && move.other->image != nullptr) m_graphics.DestroyImageNow(*move.other);
				}
				drain.moves.clear();
			}
		}
		// Freeing a plug can let VMA free its spare empty block (a free while another empty block exists): counted too.
		uint64_t frees_before = 0, bytes_before = 0, frees_after = 0, bytes_after = 0;
		GraphicContext::DeviceMemoryFrees(frees_before, bytes_before);
		m_graphics.FreePlugs(plugs);
		for (auto& plug: image_plugs) m_graphics.DestroyImageNow(*plug);
		GraphicContext::DeviceMemoryFrees(frees_after, bytes_after);
		n.returned += frees_after - frees_before;
		n.returned_bytes += bytes_after - bytes_before;
		std::erase_if(drains, [](const Drain& drain) { return !drain.ok; });
		if (drains.empty()) {
			ms_record = MsSince(start);
			report();
			return;
		}
		// Copies: every level and layer, raw barriers (Image::Transit would publish a layout change to the bindless layout
		// walk; the old image returns to its layout, the new one ends in it).
		std::vector<vk::ImageMemoryBarrier2> before, after;
		for (auto& drain: drains) {
			for (auto& move: drain.moves) {
				const auto& image = m_slot_images[move.id];
				before.push_back(ImageBarrier(image.backing.image, move.state.pl_stage, move.state.access_mask, move.state.layout,
				                              vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead,
				                              vk::ImageLayout::eTransferSrcOptimal));
				before.push_back(ImageBarrier(move.other->image, vk::PipelineStageFlagBits2::eNone, {}, vk::ImageLayout::eUndefined,
				                              vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
				                              vk::ImageLayout::eTransferDstOptimal));
				const auto new_layout = vk::ImageLayout::eTransferDstOptimal;
				for (const auto& [handle, layout]: {std::pair {image.backing.image, vk::ImageLayout::eTransferSrcOptimal},
				                                    std::pair {move.other->image, new_layout}}) {
					after.push_back(ImageBarrier(handle, vk::PipelineStageFlagBits2::eTransfer,
					                             vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite, layout,
					                             vk::PipelineStageFlagBits2::eAllCommands,
					                             vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
					                             move.state.layout));
				}
			}
		}
		m_scheduler.EndRendering();
		auto command = m_scheduler.Current().Handle();
		PipelineBarrier(command, before);
		for (auto& drain: drains) {
			for (auto& move: drain.moves) {
				const auto&                image = m_slot_images[move.id];
				std::vector<vk::ImageCopy> copies;
				for (uint32_t level = 0; level < image.backing.mip_levels; level++) {
					vk::ImageCopy copy {};
					copy.srcSubresource = {vk::ImageAspectFlagBits::eColor, level, 0, image.backing.layers};
					copy.dstSubresource = copy.srcSubresource;
					copy.extent         = LevelExtent(image.backing, level);
					copies.push_back(copy);
				}
				command.copyImage(image.backing.image, vk::ImageLayout::eTransferSrcOptimal, move.other->image,
				                  vk::ImageLayout::eTransferDstOptimal, copies);
				n.images++;
				n.moved += m_graphics.GetAllocationSize(move.old_allocation);
			}
		}
		PipelineBarrier(command, after);
		ms_record = MsSince(start);
	}

	// 2. One wait for the GPU (it also runs the deferred operations, so m_lock is not held).
	const auto wait_start = std::chrono::steady_clock::now();
	m_scheduler.Finish();
	ms_wait = MsSince(wait_start);

	// 3. The swap (the binder lock first, as PrepareBindings takes them).
	const auto                                   swap_start = std::chrono::steady_clock::now();
	std::unordered_map<VkImageView, vk::ImageView> view_map;
	std::vector<vk::ImageView>                   old_views;
	std::vector<std::unique_ptr<VulkanImage>>    destroy;   // old images of swapped textures
	std::vector<std::unique_ptr<VulkanImage>>    abandoned; // new images of abandoned blocks (never published)
	std::vector<std::pair<uint64_t, RetiredInfo>> retired;  // the guard: the old view and image handles
	const bool                                   guard = GuardOn();
	const uint64_t frame = m_graphics.presented_frames.load(std::memory_order_relaxed);
	vk::ImageLayout first_layout = vk::ImageLayout::eUndefined; // KYTY_TEXMOVE_SELFTEST: the first old image's layout
	{
		std::lock_guard  binder_lock(table.BinderMutex());
		std::scoped_lock lock {m_lock};
		const bool       busy = table.DrawHeapCount() != 0 || table.DrawTableCount() != 0;
		for (auto& drain: drains) {
			for (auto& move: drain.moves) {
				const auto* image = m_slot_images.try_get(move.id);
				// Freed, replaced, used as more than a texture or rewritten meanwhile: the block stays.
				const bool same = image != nullptr && image->serial == move.serial &&
				                  image->backing.allocation == move.old_allocation && Unmovable(*image, now) == nullptr &&
				                  image->backing.state.layout == move.state.layout;
				if (busy || !same) {
					n.busy++;
					drain.ok = false;
					break;
				}
			}
			if (!drain.ok) {
				for (auto& move: drain.moves) abandoned.push_back(std::move(move.other));
				continue;
			}
			for (auto& move: drain.moves) {
				auto& image = m_slot_images[move.id];
				// The same create info, contents and layout: only the handles change; `other` keeps the old ones.
				std::swap(image.backing.image, move.other->image);
				std::swap(image.backing.allocation, move.other->allocation);
				// The views again from their stored (already normalized) infos, on the new image.
				auto previous = std::move(image.views);
				image.views.clear();
				const RetiredInfo info {(uint64_t {move.id.generation} << 32u) | move.id.index,
				                        image.info.data.address,
				                        image.backing.format,
				                        image.backing.extent,
				                        image.backing.mip_levels,
				                        image.backing.layers,
				                        frame,
				                        true};
				for (const auto& cached: previous) {
					view_map[static_cast<VkImageView>(cached.view)] = image.FindView(cached.info);
					old_views.push_back(cached.view);
					if (guard) retired.emplace_back(reinterpret_cast<uint64_t>(static_cast<VkImageView>(cached.view)), info);
				}
				if (guard) {
					retired.emplace_back(reinterpret_cast<uint64_t>(static_cast<VkImage>(move.other->image)), info);
					retired.back().second.view = false;
				}
				if (destroy.empty()) first_layout = move.state.layout;
				image.texmove_ms = now;
				image.HeapChanged();
				// The memos that hold views (table sets) or check a slot by its image's pages take their full path once.
				DrawRepeat::NoteImageRegistryChange(image.info.data.address, image.info.data.size);
				n.memos++;
				SetVulkanObjectNameF(m_graphics.device, image.backing.image, "Kyty.Image[moved guest=0x{:016x}]",
				                     image.info.data.address);
				destroy.push_back(std::move(move.other));
			}
		}
		n.slots = table.RewriteViews(view_map);
	}
	for (auto& image: abandoned) {
		if (image != nullptr && image->image != nullptr) m_graphics.DestroyImageNow(*image);
	}
	// The images the wait's deferred deletions parked go first, outside the count (their dedicated allocations would be
	// freed within 2 s anyway; run 2 counted 6 frees / 383 MiB for one block that way).
	m_graphics.TrimRecycledImages();
	if (guard && !destroy.empty()) { // the old views and images stay for RetireMs; what goes back is on TEXMOVE_RETIRE
		RetiredBatch batch {now + RetireMs, frame, std::move(old_views), std::move(destroy)};
		{
			std::unique_lock lock(g_retired_mutex);
			for (const auto& [handle, info]: retired) {
				g_retired[handle] = info;
				FilterSet(handle);
			}
			g_texmove_watch.store(!g_retired.empty(), std::memory_order_relaxed);
		}
		// KYTY_TEXMOVE_SELFTEST=1 (positive control of the guard): one legal use of a retired image (a barrier that
		// keeps its layout) in the open command buffer must come out as TEXMOVE_STALE_USE path=vkCmdPipelineBarrier2.
		static const bool selftest      = std::getenv("KYTY_TEXMOVE_SELFTEST") != nullptr;
		static bool       selftest_done = false;
		if (selftest && !selftest_done && !batch.images.empty() && first_layout != vk::ImageLayout::eUndefined) {
			selftest_done = true;
			std::fprintf(stderr, "TEXMOVE_SELFTEST barrier on retired image 0x%llx\n",
			             static_cast<unsigned long long>(
			                 reinterpret_cast<uint64_t>(static_cast<VkImage>(batch.images[0]->image))));
			m_scheduler.EndRendering();
			PipelineBarrier(m_scheduler.Current().Handle(),
			                {ImageBarrier(batch.images[0]->image, vk::PipelineStageFlagBits2::eAllCommands,
			                              vk::AccessFlagBits2::eMemoryRead, first_layout,
			                              vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead,
			                              first_layout)});
		}
		g_retired_batches.push_back(std::move(batch));
		ms_swap = MsSince(swap_start);
		report();
		return;
	}
	for (const auto view: old_views) m_graphics.device.destroyImageView(view, nullptr);
	uint64_t frees_before = 0, bytes_before = 0, frees_after = 0, bytes_after = 0;
	GraphicContext::DeviceMemoryFrees(frees_before, bytes_before);
	for (auto& image: destroy) {
		if (image != nullptr && image->image != nullptr) m_graphics.DestroyImageNow(*image);
	}
	GraphicContext::DeviceMemoryFrees(frees_after, bytes_after);
	n.returned += frees_after - frees_before;
	n.returned_bytes += bytes_after - bytes_before;
	// The drained blocks: gone, or left empty (VMA keeps one empty block per memory type), or still holding allocations.
	std::unordered_map<VkDeviceMemory, size_t> present;
	for (const auto& block: m_graphics.CollectBlocks(101u)) present.emplace(block.memory, block.count);
	for (const auto& drain: drains) {
		if (!drain.ok) continue;
		const auto found = present.find(drain.memory);
		if (found == present.end() || found->second == 0) n.emptied++;
	}
	ms_swap = MsSince(swap_start);
	report();
}

} // namespace Libs::Graphics
