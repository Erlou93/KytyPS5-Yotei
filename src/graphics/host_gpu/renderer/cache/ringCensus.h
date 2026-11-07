#pragma once

#include "common/onceValue.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

// KYTY_TRACE_RING_CENSUS (diagnostic, default off; when off every hook is one load of a constant-initialized value and a
// branch not taken). The census of the write-fault cycle on GPU-written pages (the E0 census
// of the D4 write-range study):
//   1. a draw or dispatch binds a buffer range for writing: NoteObtainedWrite marks the whole range GPU-modified (no
//      access for the CPU) - the census records the writer of the range (shader hash, binding, the recompiler's facts,
//      tick, time);
//   2. a guest thread writes into such a page: write fault, a ReadMemory request on the execution thread - the census
//      records which writer left the page GPU-modified, which way the request went (nothing to download, dead ring bytes,
//      append release, a download), how long after the GPU write the guest wrote, and whether the GPU had changed any byte
//      of the page at all (a compare of the GPU copy with guest memory; a page the shader never wrote is a page that
//      marking only the bytes the shader writes would have left to the CPU).
// Values: 1 = with the byte compares (BAR reads and 4 KiB copies on the execution thread: perturbs timing, not an fps run);
// 2 = attribution only (no compares). Every ~5 s: RING_CENSUS lines (totals, top writers, top 64 KiB chunks), printed
// whenever the switch is on (KYTY_TRACE_PERF not needed). (a log parser reads them)
// Both sides of the cycle run on the execution thread (NoteObtainedWrite while a draw or dispatch is prepared, ReadMemory
// requests through SendCommandSync), so the census keeps its tables without locks; a write noted on another thread is only
// counted (offthread=).
namespace Libs::Graphics::RingCensus {

[[nodiscard]] inline uint32_t Mode() {
	static constinit Common::OnceValue<uint32_t> mode_once;
	return mode_once.Get([] {
		const char*    value = std::getenv("KYTY_TRACE_RING_CENSUS");
		const uint32_t mode  = value != nullptr && (value[0] == '1' || value[0] == '2') ? static_cast<uint32_t>(value[0] - '0') : 0u;
		if (mode != 0u) std::fprintf(stderr, "TRACE_RING_CENSUS mode=%u (%s)\n", mode, mode == 1u ? "with byte compares" : "attribution only");
		return mode;
	});
}
[[nodiscard]] inline bool Enabled() {
	return Mode() != 0u;
}

// What the shader recompiler knows of the written buffer binding being obtained on this thread: RebindBuffers sets it
// around NativeStorageBuffer (none: a written device-address range, a fill, a copy, an image or query write).
struct Binding {
	uint32_t resource        = 0;  // index in the program's buffer resources
	uint32_t descriptor      = 0;  // the binding's descriptor index
	uint32_t max_byte_extent = 0;
	// fixed_writes: every write at an address fixed in the shader, the written bytes [first, second) from the base.
	const std::vector<std::pair<uint32_t, uint32_t>>* fixed_ranges = nullptr;
	bool append = false, read = false, atomic = false, formatted = false;
};
// Who called NoteObtainedWrite.
enum class Kind : uint8_t { Storage, BdaRange, Other, Count };

inline thread_local const Binding* t_binding = nullptr;
inline thread_local Kind           t_kind    = Kind::Other;

class BindingScope {
public:
	explicit BindingScope(const Binding* binding): m_previous(t_binding), m_previous_kind(t_kind) {
		t_binding = binding;
		t_kind    = Kind::Storage;
	}
	~BindingScope() {
		t_binding = m_previous;
		t_kind    = m_previous_kind;
	}
	BindingScope(const BindingScope&)            = delete;
	BindingScope& operator=(const BindingScope&) = delete;

private:
	const Binding* m_previous;
	Kind           m_previous_kind;
};
// ObtainBdaRange's written range (only when the census is on).
class BdaRangeScope {
public:
	BdaRangeScope(): m_on(Enabled()) {
		if (!m_on) return;
		m_previous = t_kind;
		t_kind     = Kind::BdaRange;
	}
	~BdaRangeScope() {
		if (m_on) t_kind = m_previous;
	}
	BdaRangeScope(const BdaRangeScope&)            = delete;
	BdaRangeScope& operator=(const BdaRangeScope&) = delete;

private:
	bool m_on       = false;
	Kind m_previous = Kind::Other;
};

// The way a write request on a page went (BufferCache::ReadMemory with is_write).
enum class Path : uint8_t {
	NotGpu,   // the page was no longer GPU-modified when the request ran (another request published or released it)
	Eager,    // a pending eager readback published it (then not GPU-modified)
	Append,   // an append pool's bytes no window wrote were released (nothing downloaded)
	DeadRing, // the data ring's GPU bytes were older than the previous frame: dropped, nothing downloaded
	WindowDl, // the read window was downloaded
	PageDl,   // the request's own range was downloaded
	NoCopy,   // a download found no GPU-dirty byte to copy
	Count
};

// One write request (a stack object in the request; inactive unless the census is on and the request is a write).
struct Request {
	bool     active       = false;
	bool     gpu_modified = false; // the range was GPU-modified when the request began
	bool     snapshot     = false; // the page's guest bytes from the request's start are kept (bufferCacheRingCensus.cpp)
	bool     downloaded   = false; // a download or an eager copy published GPU bytes
	bool     compared     = false; // the compare ran (once, before the page is released)
	Path     path         = Path::NotGpu;
	uint8_t  cmp          = 0; // the compare's outcome (bufferCacheRingCensus.cpp, Cmp)
	uint32_t changed      = 0; // bytes of the page the GPU changed (8-byte words that differ, times 8)
	uint32_t line_changed = 0; // ... of them in the faulting byte's 64-byte line
	uint64_t vaddr        = 0;
	uint64_t size         = 0;
	uint64_t start_tick   = 0; // the current tick when the request began
	uint64_t known_tick   = 0; // the finished GPU tick when the request began
	int64_t  start_us     = 0;
};

} // namespace Libs::Graphics::RingCensus
