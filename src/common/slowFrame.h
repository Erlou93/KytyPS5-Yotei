#pragma once
// KYTY_TRACE_SLOW_FRAMES=<ms> (diagnostic, default off; hitch study 10.11): the execution thread's time in a few kinds of
// first-use work, summed per presented frame; a frame of at least <ms> (KYTY_TRACE_FRAMETIME's present-to-present time)
// prints SLOW_FRAME with each part (ms and calls), and every 200 frames SLOW_FRAME_BASE prints the same parts' mean over
// the frames below the threshold. Parts may nest (a download's fence wait is inside Download and FenceWait both).
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Common::SlowFrame {

enum Part : uint32_t {
	Download,    // BufferCache::DownloadBufferMemory (GPU-written bytes read back for the CPU)
	FenceWait,   // MasterSemaphore::Wait (the GPU reaching a tick)
	SyncBuffer,  // BufferCache::SynchronizeBuffer (CPU-written bytes uploaded)
	Variant,     // a new shader permutation: precompile wait, translation, compile (ProgramCache)
	DepthTarget, // TextureCache::FindDepthTarget
	NewImage,    // TextureCache::InsertImage
	NewBuffer,   // BufferCache::CreateBuffer
	MemAlloc,    // vkAllocateMemory
	MemFree,     // vkFreeMemory
	ReadsCheck,  // ReadsUnchangedFast (an adopted program's logged reads)
	Protect,     // PageManager page protection changes
	ExecIdle,    // the execution thread waiting for an operation (WaitForOperations)
	Claim,       // waiting for a helper's resolved operation (ClaimResolved)
	Commands,    // the game's commands served between operations (RunCommands)
	Draws,       // draw operations executed (calls only)
	Dispatches,  // dispatch operations executed (calls only)
	DrawOp,      // RenderExecutor::DrawIndex (whole draw)
	DrawIndirect, // RenderExecutor::DrawIndirectGpu
	DispatchOp,  // RenderExecutor::DispatchDirect
	PrepToday,   // PrepareBindingsToday
	FindBuf,     // FindBuffers
	RebindBuf,   // RebindBuffers
	RebindImg,   // RebindImages
	Commit,      // CommitBindings
	Bindless,    // PrepareBindlessHeaps
	Targets,     // AcquireRenderTargets
	Bda,         // RenderContext::PrepareBda
	UpdImage,    // TextureCache::UpdateImage (an image's contents brought up to date)
	UploadImage, // TextureCache::UploadImage (guest bytes to the image)
	FindImage,   // TextureCache::FindImage
	RefreshImage, // TextureCache::RefreshImage
	ResolveTex,  // RebindImages: ResolveTexture (the full image search)
	FindView,    // RebindImages: FindTextureReplay (the views)
	DccClear,    // RebindImages: ApplyDccFastClear
	TableReady,  // RebindImages: TableTexturesReady
	Count
};
inline constexpr const char* Names[Count] = {"download", "fence", "sync_buf", "variant", "depth", "new_img",
                                             "new_buf",  "alloc", "free",     "reads",   "protect", "idle", "claim", "commands", "draws", "dispatches", "draw_op", "draw_ind", "disp_op", "prep_today", "find_buf", "rebind_buf", "rebind_img", "commit", "bindless", "targets", "bda", "upd_img", "upload_img", "find_img", "refresh_img", "resolve_tex", "find_view", "dcc_clear", "table_ready"};

inline uint32_t Threshold() {
	static const uint32_t ms = [] {
		const char* value = std::getenv("KYTY_TRACE_SLOW_FRAMES");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	return ms;
}
// Set by the execution thread at its start: only its time is counted.
inline thread_local bool t_exec = false;

struct Acc {
	std::atomic<uint64_t> ns[Count] {};
	std::atomic<uint32_t> calls[Count] {};
};
inline Acc& Current() {
	static Acc acc;
	return acc;
}

inline uint64_t NowNs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

class Scope {
public:
	explicit Scope(Part part): m_part(part), m_on(t_exec && Threshold() != 0u) {
		if (m_on) m_start = NowNs();
	}
	~Scope() {
		if (!m_on) return;
		auto& acc = Current();
		acc.ns[m_part].fetch_add(NowNs() - m_start, std::memory_order_relaxed);
		acc.calls[m_part].fetch_add(1, std::memory_order_relaxed);
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	Part     m_part;
	bool     m_on;
	uint64_t m_start = 0;
};

inline void CountOne(Part part) {
	if (t_exec && Threshold() != 0u) Current().calls[part].fetch_add(1, std::memory_order_relaxed);
}

// Called once per presented frame (the presentation thread): takes the frame's sums and prints a slow frame.
inline void Report(uint32_t frame, double ms) {
	const uint32_t threshold = Threshold();
	if (threshold == 0u) return;
	auto&    acc = Current();
	uint64_t ns[Count];
	uint32_t calls[Count];
	for (uint32_t i = 0; i < Count; i++) {
		ns[i]    = acc.ns[i].exchange(0, std::memory_order_relaxed);
		calls[i] = acc.calls[i].exchange(0, std::memory_order_relaxed);
	}
	static uint64_t base_ns[Count] {};
	static uint64_t base_calls[Count] {};
	static uint32_t base_frames = 0;
	static double   base_ms     = 0.0;
	if (ms >= threshold) {
		char line[1024];
		int  n = std::snprintf(line, sizeof(line), "SLOW_FRAME frame=%u ms=%.1f", frame, ms);
		for (uint32_t i = 0; i < Count && n > 0 && n < static_cast<int>(sizeof(line)); i++) {
			if (calls[i] == 0) continue;
			n += std::snprintf(line + n, sizeof(line) - n, " %s=%.1f/%u", Names[i], static_cast<double>(ns[i]) / 1e6, calls[i]);
		}
		std::fprintf(stderr, "%s\n", line);
		return;
	}
	for (uint32_t i = 0; i < Count; i++) {
		base_ns[i] += ns[i];
		base_calls[i] += calls[i];
	}
	base_ms += ms;
	if (++base_frames >= 200u) {
		char line[1024];
		int  n = std::snprintf(line, sizeof(line), "SLOW_FRAME_BASE frame=%u frames=%u mean_ms=%.1f", frame, base_frames,
		                       base_ms / base_frames);
		for (uint32_t i = 0; i < Count && n > 0 && n < static_cast<int>(sizeof(line)); i++) {
			n += std::snprintf(line + n, sizeof(line) - n, " %s=%.2f/%.1f", Names[i],
			                   static_cast<double>(base_ns[i]) / 1e6 / base_frames,
			                   static_cast<double>(base_calls[i]) / base_frames);
			base_ns[i]    = 0;
			base_calls[i] = 0;
		}
		std::fprintf(stderr, "%s\n", line);
		base_frames = 0;
		base_ms     = 0.0;
	}
}

} // namespace Common::SlowFrame
