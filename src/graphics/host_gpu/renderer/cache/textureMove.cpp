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
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
		std::fprintf(stderr, "TEXMOVE config min_empty_MiB=%llu interval_s=%llu block_empty_pct=%llu batch_MiB=%llu\n",
		             static_cast<unsigned long long>(c.min_empty >> 20u),
		             static_cast<unsigned long long>(c.interval_ms / 1000u), static_cast<unsigned long long>(c.block_pct),
		             static_cast<unsigned long long>(c.batch_bytes >> 20u));
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
// all its subresources (a whole-image barrier and copy), single-sampled, contents defined.
bool Movable(const Image& image, uint64_t now_ms) {
	const auto& b = image.backing;
	return image.registered && !image.info.data.Empty() && b.image != nullptr && b.allocation != nullptr &&
	       !image.IsGpuModified() && !image.IsBufferModified() && !image.usage.render_target && !image.usage.depth_target &&
	       !image.usage.storage && !image.usage.video_out && !image.info.IsDepth() && !image.info.HasStencil() &&
	       !image.info.HasMetadata() && !image.depth_id && image.stencil_ids.empty() && !image.eviction_pending &&
	       !image.read_armed && image.probe_pending == 0 && image.heap_refs.load(std::memory_order_relaxed) == 0 &&
	       image.plan_uses == 0 && !image.binding.is_bound && !image.binding.is_target && !image.binding.needs_rebind &&
	       b.samples == 1 && b.subresource_states.empty() && b.state.layout != vk::ImageLayout::eUndefined &&
	       vk::blockSize(b.format) != 0 && (image.texmove_ms == 0 || now_ms - image.texmove_ms >= MoveAgainMs);
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

} // namespace

void TextureCache::RunTextureMove(BindlessTable& table) {
	const uint32_t mode = TexMoveMode();
	if (mode == 0 || !m_scheduler.Active()) return;
	const auto&     config = Config();
	const uint64_t  now    = NowMs();
	static uint64_t last   = now; // the first batch an interval after the first frame boundary
	if (now - last < config.interval_ms) return;
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
		uint64_t          used   = 0;
		std::vector<Move> moves;
		bool              ok = true;
	};
	struct Counters {
		uint64_t candidates = 0, foreign = 0, busy = 0, no_room = 0, dest_in_drain = 0, newblock = 0, images = 0;
		uint64_t moved = 0, emptied = 0, returned = 0, returned_bytes = 0, slots = 0, memos = 0, mismatches = 0;
		uint64_t stale = 0, still_present = 0;
	} n;
	double                  ms_record = 0, ms_wait = 0, ms_swap = 0;
	std::vector<Drain>      drains;
	std::unique_ptr<Buffer> check;
	const auto report = [&] {
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
		             "TEXMOVE frame=%llu trigger_empty_MiB=%llu candidate_blocks=%llu skip(foreign=%llu busy=%llu no_room=%llu "
		             "dest_in_drain=%llu) newblock_fallback=%llu drained=%zu images=%llu moved_MiB=%.1f emptied=%llu "
		             "returned=%llu returned_MiB=%llu slots_rewritten=%llu memo_invalidations=%llu ms_record=%.2f "
		             "ms_wait=%.2f ms_swap=%.2f ms_total=%.2f\n",
		             u(m_graphics.presented_frames.load(std::memory_order_relaxed)), u(empty >> 20u), u(n.candidates),
		             u(n.foreign), u(n.busy), u(n.no_room), u(n.dest_in_drain), u(n.newblock), drains.size(), u(n.images),
		             static_cast<double>(n.moved) / 1048576.0, u(n.emptied), u(n.returned), u(n.returned_bytes >> 20u),
		             u(n.slots), u(n.memos), ms_record, ms_wait, ms_swap, MsSince(start));
		if (mode >= 2) {
			std::fprintf(stderr,
			             "TEXMOVE_CHECK_SUMMARY batch=%llu images=%llu mismatches=%llu stale_slots=%llu "
			             "blocks_still_present=%llu total_mismatches=%llu total_stale_slots=%llu total_still_present=%llu\n",
			             u(totals.batches.load()), u(n.images), u(n.mismatches), u(n.stale), u(n.still_present),
			             u(totals.mismatches.load()), u(totals.stale_slots.load()), u(totals.still_present.load()));
		}
	};

	// 1. Census and destinations, recorded copies (under m_lock).
	{
		std::scoped_lock lock {m_lock};
		// Parked images hold allocations in the blocks too; none may be handed out while blocks are being emptied.
		m_graphics.TrimRecycledImages();
		if (table.DrawHeapCount() != 0 || table.DrawTableCount() != 0) { // a prepared draw not committed yet
			n.busy++;
			report();
			return;
		}
		auto census = m_graphics.CollectBlocks();
		std::unordered_map<VmaAllocation, ImageId> owners;
		m_slot_images.ForEach([&](ImageId id, const Image& image) {
			if (image.backing.allocation != nullptr) owners.emplace(image.backing.allocation, id);
		});
		// Emptiest first; whole blocks up to the batch size.
		std::ranges::sort(census, [](const auto& a, const auto& b) { return a.size - a.free < b.size - b.free; });
		uint64_t batch_used = 0;
		for (const auto& block: census) {
			if (block.allocations.empty() || block.free * 100u < block.size * config.block_pct) continue;
			n.candidates++;
			Drain drain {block.memory, block.size - block.free, {}, true};
			for (const auto allocation: block.allocations) {
				const auto owner = owners.find(allocation);
				if (owner == owners.end()) { // a buffer, a placeholder, an image the cache does not hold
					n.foreign++;
					drain.ok = false;
					break;
				}
				const auto& image = m_slot_images[owner->second];
				if (!Movable(image, now)) {
					n.busy++;
					drain.ok = false;
					break;
				}
				drain.moves.push_back({owner->second, image.serial, allocation, image.backing.state, nullptr, {}, {}});
			}
			if (!drain.ok || batch_used + drain.used > config.batch_bytes) continue;
			batch_used += drain.used;
			drains.push_back(std::move(drain));
		}
		std::unordered_set<VkDeviceMemory> drained;
		for (const auto& drain: drains) drained.insert(drain.memory);
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
				move.other = std::make_unique<VulkanImage>();
				bool created = m_graphics.CreateImage(create, *move.other, &relocation);
				if (!created && fallback) {
					fallback                   = false;
					relocation.allow_new_block = true;
					n.newblock++;
					move.other = std::make_unique<VulkanImage>();
					created    = m_graphics.CreateImage(create, *move.other, &relocation);
				}
				if (!created) {
					n.no_room++;
					move.other.reset();
					drain.ok = false;
					break;
				}
				uint32_t type = 0;
				(void)m_graphics.AllocationPlace(move.other->allocation, memory, type);
				if (drained.contains(memory)) {
					n.dest_in_drain++;
					drain.ok = false;
					break;
				}
			}
			if (!drain.ok) { // the block stays: its new images go
				for (auto& move: drain.moves) {
					if (move.other != nullptr && move.other->image != nullptr) m_graphics.DestroyImageNow(*move.other);
				}
				drain.moves.clear();
			}
		}
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
	std::vector<std::unique_ptr<VulkanImage>>    destroy; // old images of swapped textures, new ones of abandoned blocks
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
				                  image->backing.allocation == move.old_allocation && Movable(*image, now) &&
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
				for (auto& move: drain.moves) destroy.push_back(std::move(move.other));
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
				for (const auto& cached: previous) {
					view_map[static_cast<VkImageView>(cached.view)] = image.FindView(cached.info);
					old_views.push_back(cached.view);
					if (mode >= 2 && (image.views.empty() || !(image.views.back().info == cached.info))) n.mismatches++;
				}
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
		if (mode >= 2) {
			std::unordered_set<VkImageView> old_set;
			for (const auto view: old_views) old_set.insert(static_cast<VkImageView>(view));
			n.stale = table.CountSlotViews(old_set);
		}
	}
	for (const auto view: old_views) m_graphics.device.destroyImageView(view, nullptr);
	uint64_t frees_before = 0, bytes_before = 0, frees_after = 0, bytes_after = 0;
	GraphicContext::DeviceMemoryFrees(frees_before, bytes_before);
	for (auto& image: destroy) {
		if (image != nullptr && image->image != nullptr) m_graphics.DestroyImageNow(*image);
	}
	m_graphics.TrimRecycledImages();
	GraphicContext::DeviceMemoryFrees(frees_after, bytes_after);
	n.returned       = frees_after - frees_before;
	n.returned_bytes = bytes_after - bytes_before;
	// The drained blocks: gone, or left empty (VMA keeps one empty block per memory type), or still holding allocations.
	std::unordered_map<VkDeviceMemory, size_t> present;
	for (const auto& block: m_graphics.CollectBlocks()) present.emplace(block.memory, block.allocations.size());
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
