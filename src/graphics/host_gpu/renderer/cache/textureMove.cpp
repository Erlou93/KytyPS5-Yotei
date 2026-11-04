// KYTY_TEXMOVE (map §3v, notes\PROJEKT-PRZENOSZENIE-TEKSTUR-2026-11-04.md): moving the clean textures that keep nearly
// empty VMA blocks alive. After the game streams textures in and out, device-local blocks stay allocated for a few
// bindless-pinned textures each (4.11: 9 blocks >= 50% empty held 1.98 GB of holes for 326 MiB of residents). VMA frees
// a block only when its last allocation goes, so the residents of the emptiest blocks get a new VkImage in a fuller block,
// a GPU copy of every level and layer, and the new backing inside the same Image (same ImageId: every table and heap
// keeps its slots, only the views in them are rewritten). The old images are destroyed at once (not parked), so their
// blocks empty and VMA gives them back to the driver.
#include "graphics/host_gpu/renderer/cache/textureCache.h"

#include "common/alignment.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/drawRepeat.h"
#include "graphics/host_gpu/renderer/pipeline/bindlessTable.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
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

// Thresholds (design §4-5): a batch when the empty space inside the device-local blocks reaches KYTY_TEXMOVE_MIN_EMPTY_MB,
// at most once per KYTY_TEXMOVE_INTERVAL_S, out of blocks at least KYTY_TEXMOVE_BLOCK_EMPTY_PCT empty, whole blocks up to
// KYTY_TEXMOVE_BATCH_MB of residents.
struct TexMoveConfig {
	uint64_t min_empty   = 0;
	uint64_t interval_ms = 0;
	uint64_t block_pct   = 0;
	uint64_t batch_bytes = 0;
	uint64_t retire_ms   = 0; // KYTY_TEXMOVE=2: how long the old views and images stay (stale-use detector)
};
const TexMoveConfig& Config() {
	static const TexMoveConfig config = [] {
		const auto env = [](const char* name, uint64_t fallback) {
			const char* value = std::getenv(name);
			return value != nullptr ? std::strtoull(value, nullptr, 10) : fallback;
		};
		TexMoveConfig c;
		c.min_empty   = env("KYTY_TEXMOVE_MIN_EMPTY_MB", 1024) << 20u;
		c.interval_ms = env("KYTY_TEXMOVE_INTERVAL_S", 10) * 1000u;
		c.block_pct   = env("KYTY_TEXMOVE_BLOCK_EMPTY_PCT", 50);
		c.batch_bytes = env("KYTY_TEXMOVE_BATCH_MB", 128) << 20u;
		c.retire_ms   = env("KYTY_TEXMOVE_RETIRE_S", 10) * 1000u;
		std::fprintf(stderr, "TEXMOVE config min_empty_MiB=%llu interval_s=%llu block_empty_pct=%llu batch_MiB=%llu retire_s=%llu\n",
		             static_cast<unsigned long long>(c.min_empty >> 20u),
		             static_cast<unsigned long long>(c.interval_ms / 1000u), static_cast<unsigned long long>(c.block_pct),
		             static_cast<unsigned long long>(c.batch_bytes >> 20u), static_cast<unsigned long long>(c.retire_ms / 1000u));
		return c;
	}();
	return config;
}

// A texture moved less than a minute ago stays (no ping-pong between blocks).
constexpr uint64_t MoveAgainMs = 60000;

uint64_t NowMs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

double MsSince(std::chrono::steady_clock::time_point start) {
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

vk::Extent3D LevelExtent(const VulkanImage& image, uint32_t level) {
	return {std::max(image.extent.width >> level, 1u), std::max(image.extent.height >> level, 1u),
	        image.image_type == vk::ImageType::e3D ? std::max(image.extent.depth >> level, 1u) : 1u};
}

// The image's levels (all layers each) packed one after another from `offset` for vkCmdCopyImageToBuffer (offsets aligned
// to the texel block and to 4 bytes); each level's {offset, bytes} goes to `levels`.
void PackedCopies(const VulkanImage& image, uint64_t offset, std::vector<vk::BufferImageCopy>& copies,
                  std::vector<std::pair<uint64_t, uint64_t>>& levels) {
	const uint64_t block  = vk::blockSize(image.format);
	const auto     extent = vk::blockExtent(image.format);
	for (uint32_t level = 0; level < image.mip_levels; level++) {
		const auto e = LevelExtent(image, level);
		offset       = Common::AlignUp(offset, block * 4u);
		vk::BufferImageCopy copy {};
		copy.bufferOffset     = offset;
		copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, level, 0, image.layers};
		copy.imageExtent      = e;
		copies.push_back(copy);
		const uint64_t bytes = uint64_t {(e.width + extent[0] - 1u) / extent[0]} * ((e.height + extent[1] - 1u) / extent[1]) *
		                       ((e.depth + extent[2] - 1u) / extent[2]) * image.layers * block;
		levels.emplace_back(offset, bytes);
		offset += bytes;
	}
}

// Whether the texture may move now: a registered, clean color texture (its contents are its guest bytes or an upload of
// them: not GPU- or buffer-written), no target, storage, depth, stencil, metadata or video-out image, nothing pending on
// it (eviction, image read, probe, rebind), not bound by a draw, not referenced by a resident image heap, one layout for
// all its subresources (a whole-image barrier and copy), single-sampled, contents defined. Null when it may move, else
// the first reason it may not (the TEXMOVE line names the first one of a batch).
// `ignore` (the ceiling census only, never a move): IgnoreGpuWritten drops the GPU/buffer-written check, IgnoreTargets also
// the render-target and storage use (depth and video-out stay).
constexpr uint32_t IgnoreGpuWritten = 1u, IgnoreTargets = 2u;
const char* Unmovable(const Image& image, uint64_t now_ms, uint32_t ignore = 0) {
	const auto& b = image.backing;
	if (!image.registered || image.info.data.Empty() || b.image == nullptr || b.allocation == nullptr) return "unregistered";
	if ((ignore & IgnoreGpuWritten) == 0 && (image.IsGpuModified() || image.IsBufferModified())) return "gpu_written";
	if (image.usage.depth_target || image.usage.video_out ||
	    ((ignore & IgnoreTargets) == 0 && (image.usage.render_target || image.usage.storage)))
		return "target";
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

// ---- KYTY_TEXMOVE=2, the stale-use detector (4.11: Rafał saw the ground texture vanish once at a hitch with =1). A holder
// the swap does not update (a cached view, a set, a prepared binding) would keep using a moved texture's old view or
// image; destroyed at once, the handle can be reused by a new view and the draw samples another image. In check mode
// the old views and images stay alive for the retire period (a stale user still samples the right bytes) and every
// point where a view or image enters GPU work looks them up here.
struct RetiredInfo {
	uint64_t     image_id = 0; // ImageId generation << 32 | index
	vk::Format   format   = vk::Format::eUndefined;
	vk::Extent3D extent {};
	uint64_t     frame = 0; // presented frame of the batch that moved it
	bool         view  = false;
};
struct RetiredBatch {
	uint64_t                                  deadline_ms = 0, frame = 0;
	std::vector<vk::ImageView>                views;
	std::vector<std::unique_ptr<VulkanImage>> images;
	std::vector<VkDeviceMemory>               drained;
};
std::shared_mutex                         g_retired_mutex;
std::unordered_map<uint64_t, RetiredInfo> g_retired;         // old view and image handles
std::vector<RetiredBatch>                 g_retired_batches; // GPU thread
std::atomic<uint64_t>                     g_stale_frame {0};
std::atomic<uint64_t>                     g_stale_checked {0}; // handles looked up while watching (the detector sees traffic)
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
#define TEXMOVE_DETECTED_FUNCTIONS(X)                                                                                      \
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
TEXMOVE_DETECTED_FUNCTIONS(TEXMOVE_PREVIOUS)
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
	if (result == VK_SUCCESS && info != nullptr) {
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
	if (info != nullptr) {
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

// The retire period is over (or `all`): the old views and images go, the returned blocks are counted from the driver's
// frees, and the drained blocks of each batch are checked (gone, or the one empty block VMA keeps).
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
			g_texmove_watch.store(!g_retired.empty(), std::memory_order_relaxed);
		}
		for (const auto view: batch.views) graphics.device.destroyImageView(view, nullptr);
		uint64_t frees_before = 0, bytes_before = 0, frees_after = 0, bytes_after = 0;
		GraphicContext::DeviceMemoryFrees(frees_before, bytes_before);
		for (auto& image: batch.images) graphics.DestroyImageNow(*image);
		GraphicContext::DeviceMemoryFrees(frees_after, bytes_after);
		uint64_t emptied = 0, reused = 0; // reused: a block other allocations took during the retire period
		std::unordered_map<VkDeviceMemory, size_t> present;
		for (const auto& block: graphics.CollectBlocks(101u)) present.emplace(block.memory, block.count);
		for (const auto memory: batch.drained) {
			const auto found = present.find(memory);
			(found == present.end() || found->second == 0 ? emptied : reused)++;
		}
		auto& totals = GetTexMoveTotals();
		totals.emptied.fetch_add(emptied);
		totals.returned.fetch_add(frees_after - frees_before);
		totals.returned_bytes.fetch_add(bytes_after - bytes_before);
		std::fprintf(stderr,
		             "TEXMOVE_RETIRE frame=%llu batch_frame=%llu views=%zu images=%zu returned=%llu returned_MiB=%llu "
		             "emptied=%llu reused_meanwhile=%llu stale_uses_total=%llu checked_total=%llu\n",
		             static_cast<unsigned long long>(g_stale_frame.load()), static_cast<unsigned long long>(batch.frame),
		             batch.views.size(), batch.images.size(), static_cast<unsigned long long>(frees_after - frees_before),
		             static_cast<unsigned long long>((bytes_after - bytes_before) >> 20u),
		             static_cast<unsigned long long>(emptied), static_cast<unsigned long long>(reused),
		             static_cast<unsigned long long>(totals.stale_uses.load()),
		             static_cast<unsigned long long>(g_stale_checked.load()));
		g_retired_batches.erase(g_retired_batches.begin() + static_cast<std::ptrdiff_t>(i));
	}
}

} // namespace

void TexMoveStaleCheck(uint64_t handle, const char* path) {
	g_stale_checked.fetch_add(1, std::memory_order_relaxed);
	RetiredInfo info;
	{
		std::shared_lock lock(g_retired_mutex);
		const auto       found = g_retired.find(handle);
		if (found == g_retired.end()) return;
		info = found->second;
	}
	const uint64_t n = GetTexMoveTotals().stale_uses.fetch_add(1);
	if (n >= 64u) return;
	std::fprintf(stderr,
	             "TEXMOVE_STALE_USE frame=%llu path=%s %s=0x%llx imageid=%u/%u fmt=%s extent=%ux%ux%u moved_at_frame=%llu\n",
	             static_cast<unsigned long long>(g_stale_frame.load()), path, info.view ? "view" : "image",
	             static_cast<unsigned long long>(handle), static_cast<uint32_t>(info.image_id & 0xffffffffu),
	             static_cast<uint32_t>(info.image_id >> 32u), vk::to_string(info.format).c_str(), info.extent.width,
	             info.extent.height, info.extent.depth, static_cast<unsigned long long>(info.frame));
}

void TexMoveInstallDetector() {
	if (TexMoveMode() < 2) return;
	auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
	std::string missing;
#define TEXMOVE_WRAP(name)                                                                                                 \
	if (d.name != nullptr) {                                                                                           \
		g_prev_##name = d.name;                                                                                    \
		d.name        = Wrap_##name;                                                                               \
	} else {                                                                                                           \
		missing += " " #name;                                                                                      \
	}
	TEXMOVE_DETECTED_FUNCTIONS(TEXMOVE_WRAP)
#undef TEXMOVE_WRAP
	std::fprintf(stderr, "TEXMOVE stale-use detector installed (missing:%s)\n", missing.empty() ? " none" : missing.c_str());
}

void TextureCache::RunTextureMove(BindlessTable& table) {
	const uint32_t mode = TexMoveMode();
	if (mode == 0 || !m_scheduler.Active()) return;
	const auto&    config = Config();
	const uint64_t now    = NowMs();
	if (mode >= 2) { // the stale-use detector: the frame its lines name, and the old views and images whose time is up
		g_stale_frame.store(m_graphics.presented_frames.load(std::memory_order_relaxed), std::memory_order_relaxed);
		ExpireRetired(m_graphics, now, false);
	}
	static uint64_t last = now; // the first batch an interval after the first frame boundary
	// Back-off: an attempt that moved nothing and ended exactly like the one before (same candidate blocks, same
	// allocation counts, same reasons) doubles the wait, up to 60 s; anything else resets it to the interval.
	static uint64_t wait_ms = config.interval_ms, last_signature = 0;
	if (now - last < wait_ms) return;
	uint64_t usage = 0, budget = 0, block_bytes = 0, allocated = 0;
	m_graphics.GetDeviceLocalMemoryStats(usage, budget, block_bytes, allocated);
	const uint64_t empty = block_bytes > allocated ? block_bytes - allocated : 0;
	if (empty < config.min_empty) return;
	last = now; // one attempt per interval, whatever it finds

	const auto start = std::chrono::steady_clock::now();
	struct Move {
		ImageId                                    id;
		uint64_t                                   serial = 0;
		VmaAllocation                              old_allocation = nullptr;
		VulkanImageState                           state;
		std::unique_ptr<VulkanImage>               other; // the new image; after the swap, the old one
		std::vector<std::pair<uint64_t, uint64_t>> check_old, check_new; // =2: {offset, bytes} per level
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
		uint64_t slots = 0, memos = 0, mismatches = 0, stale = 0, still_present = 0;
		const char* busy_why = "-";
	} n;
	double                  ms_record = 0, ms_wait = 0, ms_swap = 0;
	std::vector<Drain>      drains;
	std::unique_ptr<Buffer> check;
	std::string             ceiling_text; // the ceiling census (measurement only, map §3v 5.11)
	double                  ms_census = 0;
	uint64_t                signature = 0xcbf29ce484222325ull; // 0 changes: the attempt never reached the census
	const auto report = [&] {
		if (n.images == 0 && signature != 0 && signature == last_signature) {
			wait_ms = std::min<uint64_t>(wait_ms * 2u, std::max<uint64_t>(config.interval_ms, 60000u));
		} else {
			wait_ms = config.interval_ms;
		}
		last_signature = signature;
		auto& totals = GetTexMoveTotals();
		totals.batches.fetch_add(1);
		totals.images.fetch_add(n.images);
		totals.moved_bytes.fetch_add(n.moved);
		totals.emptied.fetch_add(n.emptied);
		totals.returned.fetch_add(n.returned);
		totals.returned_bytes.fetch_add(n.returned_bytes);
		totals.mismatches.fetch_add(n.mismatches);
		totals.stale_slots.fetch_add(n.stale);
		totals.still_present.fetch_add(n.still_present);
		const auto u = [](uint64_t v) { return static_cast<unsigned long long>(v); };
		std::fprintf(stderr,
		             "TEXMOVE frame=%llu trigger_empty_MiB=%llu candidate_blocks=%llu skip(foreign=%llu busy=%llu busy_why=%s "
		             "over_batch=%llu no_room=%llu) dest_in_drain=%llu plugs=%llu/%lluMiB newblock_fallback=%llu drained=%zu "
		             "images=%llu moved_MiB=%.1f emptied=%llu returned=%llu returned_MiB=%llu slots_rewritten=%llu "
		             "memo_invalidations=%llu ms_record=%.2f ms_wait=%.2f ms_swap=%.2f ms_total=%.2f ms_census=%.2f "
		             "next_s=%.0f%s\n",
		             u(m_graphics.presented_frames.load(std::memory_order_relaxed)), u(empty >> 20u), u(n.candidates),
		             u(n.foreign), u(n.busy), n.busy_why, u(n.over_batch), u(n.no_room), u(n.dest_in_drain), u(n.plugs),
		             u(n.plug_bytes >> 20u), u(n.newblock), drains.size(), u(n.images), static_cast<double>(n.moved) / 1048576.0,
		             u(n.emptied), u(n.returned), u(n.returned_bytes >> 20u), u(n.slots), u(n.memos), ms_record, ms_wait,
		             ms_swap, MsSince(start), ms_census, static_cast<double>(wait_ms) / 1000.0, ceiling_text.c_str());
		if (mode >= 2) {
			std::fprintf(stderr,
			             "TEXMOVE_CHECK_SUMMARY batch=%llu images=%llu mismatches=%llu stale_slots=%llu "
			             "total_mismatches=%llu total_stale_slots=%llu total_still_present=%llu stale_uses=%llu "
			             "(blocks are counted at TEXMOVE_RETIRE)\n",
			             u(totals.batches.load()), u(n.images), u(n.mismatches), u(n.stale), u(totals.mismatches.load()),
			             u(totals.stale_slots.load()), u(totals.still_present.load()), u(totals.stale_uses.load()));
		}
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
			return block.count != 0 && block.free * 100u >= block.size * config.block_pct;
		};
		auto census = m_graphics.CollectBlocks(static_cast<uint32_t>(config.block_pct));
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
		// The ceiling census (measurement only, map §3v 5.11: most attempts gave up on gpu_written): the candidate blocks
		// that would empty if GPU-written textures could move (every other check kept), and if their render-target and
		// storage use did not stop them either; the first reason each other block still keeps; and what the GPU-written
		// residents of the candidate blocks are (top by bytes: format, extent, layers/levels, how written, usage flags,
		// frames since last use, GPU writes of any image since its own last one). KYTY_TEXMOVE_CEIL=1 only (5.11: the
		// extension is parked; the census costs time on every attempt).
		static const bool ceiling_on = std::getenv("KYTY_TEXMOVE_CEIL") != nullptr;
		if (ceiling_on) {
			struct Top {
				uint64_t count = 0, bytes = 0, unused = ~uint64_t {0}, writes = ~uint64_t {0};
			};
			std::map<std::string, Top> top;
			uint64_t                   gpu_count = 0, gpu_bytes = 0;
			for (const uint32_t ignore: {IgnoreGpuWritten, IgnoreGpuWritten | IgnoreTargets}) {
				uint64_t                        blocks = 0, block_bytes = 0, resident = 0;
				std::map<std::string, uint64_t> blockers;
				for (const auto& block: census) {
					if (!candidate(block)) continue;
					const char* why = nullptr;
					for (const auto allocation: block.allocations) {
						const auto owner = owners.find(allocation);
						why              = owner == owners.end() ? "foreign" : Unmovable(m_slot_images[owner->second], now, ignore);
						if (why != nullptr) break;
					}
					if (why != nullptr) {
						blockers[why]++;
						continue;
					}
					blocks++;
					block_bytes += block.size;
					resident += block.size - block.free;
				}
				char part[160];
				std::snprintf(part, sizeof(part), " | %s blocks=%llu MiB=%llu resident_MiB=%llu blockers_left=",
				              ignore == IgnoreGpuWritten ? "ceil_gpu_written" : "ceil_gpu_targets",
				              static_cast<unsigned long long>(blocks), static_cast<unsigned long long>(block_bytes >> 20u),
				              static_cast<unsigned long long>(resident >> 20u));
				ceiling_text += part;
				for (const auto& [why, count]: blockers) ceiling_text += why + ":" + std::to_string(count) + ",";
				if (blockers.empty()) ceiling_text += "-";
			}
			for (const auto& block: census) {
				if (!candidate(block)) continue;
				for (const auto allocation: block.allocations) {
					const auto owner = owners.find(allocation);
					if (owner == owners.end()) continue;
					const auto& image = m_slot_images[owner->second];
					if (!image.registered || !(image.IsGpuModified() || image.IsBufferModified())) continue;
					const char* how = image.IsBufferModified() && !image.IsGpuModified() ? "buf"
					                  : image.usage.render_target                       ? "rt"
					                  : image.usage.storage                             ? "st"
					                  : image.usage.depth_target                        ? "ds"
					                                                                    : "copy";
					char key[160];
					std::snprintf(key, sizeof(key), "%s/%ux%ux%u/l%u/m%u/%s/u%x%s", vk::to_string(image.backing.format).c_str(),
					              image.backing.extent.width, image.backing.extent.height, image.backing.extent.depth,
					              image.backing.layers, image.backing.mip_levels, how,
					              static_cast<uint32_t>(image.backing.usage), image.bindless_pinned ? "/P" : "");
					const uint64_t bytes  = m_graphics.GetAllocationSize(allocation);
					const uint64_t used   = m_lru_cache.GetTick(image.lru_id);
					auto&          entry  = top[key];
					entry.count++;
					entry.bytes += bytes;
					entry.unused = std::min(entry.unused, m_gc_tick > used ? m_gc_tick - used : 0);
					entry.writes = std::min(entry.writes, Image::s_content_serial - std::min(Image::s_content_serial, image.content_serial));
					gpu_count++;
					gpu_bytes += bytes;
				}
			}
			std::vector<std::pair<std::string, Top>> sorted(top.begin(), top.end());
			std::ranges::sort(sorted, [](const auto& a, const auto& b) { return a.second.bytes > b.second.bytes; });
			std::string line = "TEXMOVE_CEIL_TOP frame=" +
			                   std::to_string(m_graphics.presented_frames.load(std::memory_order_relaxed)) +
			                   " gpu_written_residents=" + std::to_string(gpu_count) + "/" +
			                   std::to_string(gpu_bytes >> 20u) + "MiB kinds=" + std::to_string(sorted.size());
			for (size_t i = 0; i < sorted.size() && i < 8u; i++) {
				const auto& [key, entry] = sorted[i];
				line += " [" + key + " x" + std::to_string(entry.count) + " " + std::to_string(entry.bytes >> 20u) +
				        "MiB unused>=" + std::to_string(entry.unused) + " writes_since>=" + std::to_string(entry.writes) + "]";
			}
			std::fprintf(stderr, "%s\n", line.c_str());
		}
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
				drain.moves.push_back({owner->second, image.serial, allocation, image.backing.state, nullptr, {}, {}});
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
				if (batch_used + drain.used > config.batch_bytes) {
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
		std::vector<vk::ImageMemoryBarrier2> before, middle, after;
		uint64_t                             check_bytes = 0;
		for (auto& drain: drains) {
			for (auto& move: drain.moves) {
				const auto& image = m_slot_images[move.id];
				before.push_back(ImageBarrier(image.backing.image, move.state.pl_stage, move.state.access_mask, move.state.layout,
				                              vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead,
				                              vk::ImageLayout::eTransferSrcOptimal));
				before.push_back(ImageBarrier(move.other->image, vk::PipelineStageFlagBits2::eNone, {}, vk::ImageLayout::eUndefined,
				                              vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
				                              vk::ImageLayout::eTransferDstOptimal));
				if (mode >= 2) {
					middle.push_back(ImageBarrier(move.other->image, vk::PipelineStageFlagBits2::eTransfer,
					                              vk::AccessFlagBits2::eTransferWrite, vk::ImageLayout::eTransferDstOptimal,
					                              vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead,
					                              vk::ImageLayout::eTransferSrcOptimal));
				}
				const auto new_layout = mode >= 2 ? vk::ImageLayout::eTransferSrcOptimal : vk::ImageLayout::eTransferDstOptimal;
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
		if (mode >= 2) { // both images into one host buffer, compared before the swap
			PipelineBarrier(command, middle);
			std::vector<std::vector<vk::BufferImageCopy>> old_copies, new_copies;
			for (auto& drain: drains) {
				for (auto& move: drain.moves) {
					const auto& image = m_slot_images[move.id];
					PackedCopies(image.backing, check_bytes, old_copies.emplace_back(), move.check_old);
					check_bytes = move.check_old.back().first + move.check_old.back().second;
					PackedCopies(*move.other, check_bytes, new_copies.emplace_back(), move.check_new);
					check_bytes = move.check_new.back().first + move.check_new.back().second;
				}
			}
			check = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0, AllFlags, check_bytes);
			size_t index = 0;
			for (auto& drain: drains) {
				for (auto& move: drain.moves) {
					const auto& image = m_slot_images[move.id];
					command.copyImageToBuffer(image.backing.image, vk::ImageLayout::eTransferSrcOptimal, check->Handle(),
					                          old_copies[index]);
					command.copyImageToBuffer(move.other->image, vk::ImageLayout::eTransferSrcOptimal, check->Handle(),
					                          new_copies[index]);
					index++;
				}
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
	vk::ImageLayout                              first_layout = vk::ImageLayout::eUndefined; // the first old image's layout
	std::vector<std::unique_ptr<VulkanImage>>    abandoned; // new images of abandoned blocks (never published)
	std::vector<std::pair<uint64_t, RetiredInfo>> retired;  // =2: the old view and image handles
	const uint64_t frame = m_graphics.presented_frames.load(std::memory_order_relaxed);
	{
		std::lock_guard  binder_lock(table.BinderMutex());
		std::scoped_lock lock {m_lock};
		const bool       busy = table.DrawHeapCount() != 0 || table.DrawTableCount() != 0;
		if (check != nullptr && !check->IsCoherent()) check->Invalidate(0, check->Size());
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
				if (check == nullptr) continue;
				const auto* bytes = check->Mapped().data();
				for (size_t level = 0; level < move.check_old.size(); level++) {
					const auto [old_offset, size] = move.check_old[level];
					const auto new_offset         = move.check_new[level].first;
					if (std::memcmp(bytes + old_offset, bytes + new_offset, size) == 0) continue;
					uint64_t at = 0;
					while (at < size && bytes[old_offset + at] == bytes[new_offset + at]) at++;
					n.mismatches++;
					drain.ok = false;
					std::fprintf(stderr, "TEXMOVE_CHECK mismatch addr=0x%llx format=%s %ux%ux%u layers=%u level=%zu/%u byte=%llu/%llu\n",
					             static_cast<unsigned long long>(image->info.data.address),
					             vk::to_string(image->backing.format).c_str(), image->backing.extent.width,
					             image->backing.extent.height, image->backing.extent.depth, image->backing.layers, level,
					             image->backing.mip_levels, static_cast<unsigned long long>(at),
					             static_cast<unsigned long long>(size));
					break;
				}
				if (!drain.ok) break;
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
				const RetiredInfo info {(uint64_t {move.id.generation} << 32u) | move.id.index, image.backing.format,
				                        image.backing.extent, frame, true};
				for (const auto& cached: previous) {
					view_map[static_cast<VkImageView>(cached.view)] = image.FindView(cached.info);
					old_views.push_back(cached.view);
					if (mode >= 2 && (image.views.empty() || !(image.views.back().info == cached.info))) n.mismatches++;
					if (mode >= 2) retired.emplace_back(reinterpret_cast<uint64_t>(static_cast<VkImageView>(cached.view)), info);
				}
				if (mode >= 2) {
					retired.emplace_back(reinterpret_cast<uint64_t>(static_cast<VkImage>(move.other->image)), info);
					retired.back().second.view = false;
				}
				image.texmove_ms = now;
				image.HeapChanged();
				// The memos that hold views (table sets) or check a slot by its image's pages take their full path once.
				DrawRepeat::NoteImageRegistryChange(image.info.data.address, image.info.data.size);
				n.memos++;
				SetVulkanObjectNameF(m_graphics.device, image.backing.image, "Kyty.Image[moved guest=0x{:016x}]",
				                     image.info.data.address);
				if (destroy.empty()) first_layout = move.state.layout; // KYTY_TEXMOVE_SELFTEST
				destroy.push_back(std::move(move.other));
			}
		}
		n.slots = table.RewriteViews(view_map);
		if (mode >= 2) {
			std::unordered_set<VkImageView> old_set;
			for (const auto view: old_views) old_set.insert(static_cast<VkImageView>(view));
			n.stale = table.CountSlotViews(old_set);
		}
	}
	for (auto& image: abandoned) {
		if (image != nullptr && image->image != nullptr) m_graphics.DestroyImageNow(*image);
	}
	// The images the wait's deferred deletions parked go first, outside the count (their dedicated allocations would be
	// freed within 2 s anyway; run 2 counted 6 frees / 383 MiB for one block that way).
	m_graphics.TrimRecycledImages();
	if (mode >= 2) { // the old views and images stay for the retire period; their blocks are counted when they go
		RetiredBatch batch {now + config.retire_ms, frame, std::move(old_views), std::move(destroy), {}};
		for (const auto& drain: drains) {
			if (drain.ok) batch.drained.push_back(drain.memory);
		}
		// The check of mode 1's census, now: a drained block holds nothing but the retired old images.
		std::unordered_set<VmaAllocation> old_allocations;
		for (const auto& image: batch.images) old_allocations.insert(image->allocation);
		for (const auto& block: m_graphics.CollectBlocks()) {
			if (std::ranges::find(batch.drained, block.memory) == batch.drained.end()) continue;
			if (std::ranges::any_of(block.allocations, [&](VmaAllocation a) { return !old_allocations.contains(a); }))
				n.still_present++;
		}
		{
			std::unique_lock lock(g_retired_mutex);
			for (const auto& [handle, info]: retired) g_retired[handle] = info;
			g_texmove_watch.store(!g_retired.empty(), std::memory_order_relaxed);
		}
		// KYTY_TEXMOVE_SELFTEST=1 (positive control of the detector): one legal use of a retired image (a barrier that
		// keeps its layout) in the open command buffer must come out as TEXMOVE_STALE_USE path=vkCmdPipelineBarrier2.
		static const bool selftest = std::getenv("KYTY_TEXMOVE_SELFTEST") != nullptr;
		static bool       selftest_done = false;
		if (selftest && !selftest_done && !batch.images.empty() && first_layout != vk::ImageLayout::eUndefined) {
			selftest_done = true;
			std::fprintf(stderr, "TEXMOVE_SELFTEST barrier on retired image 0x%llx\n",
			             static_cast<unsigned long long>(reinterpret_cast<uint64_t>(static_cast<VkImage>(batch.images[0]->image))));
			m_scheduler.EndRendering();
			PipelineBarrier(m_scheduler.Current().Handle(),
			                {ImageBarrier(batch.images[0]->image, vk::PipelineStageFlagBits2::eAllCommands,
			                              vk::AccessFlagBits2::eMemoryRead, first_layout,
			                              vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead,
			                              first_layout)});
		}
		g_retired_batches.push_back(std::move(batch));
		ms_swap = MsSince(swap_start);
		check.reset();
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
		if (found == present.end() || found->second == 0) {
			n.emptied++;
		} else {
			n.still_present++;
		}
	}
	ms_swap = MsSince(swap_start);
	check.reset();
	report();
}

} // namespace Libs::Graphics
