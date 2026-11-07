// KYTY_TRACE_RING_CENSUS (diagnostic, default off): the census of the write-fault cycle on GPU-written pages. What it
// records and why: ringCensus.h. This file holds the BufferCache members of the census and its tables; bufferCache.cpp
// only calls them (NoteObtainedWrite / NoteObtainedWriteRepeat, ObtainBdaRange, the ReadMemory request).

#include "graphics/host_gpu/renderer/cache/ringCensus.h"

#include "common/assert.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/debugSample.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <immintrin.h>

namespace Libs::Graphics {

namespace {

using RingCensus::Kind;
using RingCensus::Path;

constexpr uint64_t CensusPage = TRACKER_PAGE_SIZE; // the tracker's protection unit: one write fault opens one such page
static_assert(CensusPage == 4096u);

// The compare's outcome (Request::cmp).
enum Cmp : uint8_t {
	CmpNone,     // not compared: the range was not GPU-modified when the request began
	CmpZero,     // the GPU had changed no byte of the page (download: the published bytes equal guest memory)
	CmpChanged,  // the GPU had changed bytes of the page
	CmpNoMap,    // not compared: the buffer has no host mapping (BAR)
	CmpPending,  // not compared: the newest write of the page (or the buffer's join copies) not finished, never waited for
	CmpNoBuffer, // not compared: no cached buffer holds the page
	CmpReadFail, // not compared: guest memory of the page could not be read as one plain mapping
	CmpOff,      // not compared: KYTY_TRACE_RING_CENSUS=2
	CmpCount
};
constexpr std::array<const char*, CmpCount> CmpNames {"none", "zero", "changed", "nomap", "pending", "nobuf", "readfail", "off"};
constexpr std::array<const char*, static_cast<size_t>(Path::Count)> PathNames {"not_gpu", "eager",   "append", "dead",
                                                                                "window_dl", "page_dl", "nocopy"};
constexpr std::array<const char*, static_cast<size_t>(Kind::Count) + 1u> KindNames {"storage", "bda", "other", "none"};

int64_t NowUs() {
	return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// A GPU write of a byte range (NoteObtainedWrite / NoteObtainedWriteRepeat): the newest writer of each byte.
struct Writer {
	uint64_t end   = 0; // the record covers [key, end)
	uint64_t base  = 0; // the whole range the write marked (the binding)
	uint64_t size  = 0;
	uint64_t hash  = 0; // the shader of the draw or dispatch whose bindings were obtained (0 for Kind::Other)
	uint64_t tick  = 0; // the command buffer the write was recorded in
	int64_t  us    = 0; // steady-clock microseconds when it was noted
	uint32_t frame = 0;
	uint32_t descriptor = UINT32_MAX, resource = UINT32_MAX, max_byte_extent = 0;
	uint32_t fixed = UINT32_MAX; // index into Census::fixed_lists (fixed_writes), UINT32_MAX: not fixed
	Kind     kind  = Kind::Other;
	uint8_t  flags = 0; // 1 append, 2 read, 4 atomic, 8 formatted, 16 noted by the write memo (a repeat)
};

// [start, end) -> newest writer, non-overlapping (a new write replaces what it covers).
class WriterMap {
public:
	void Insert(uint64_t start, uint64_t end, const Writer& writer) {
		if (start >= end) return;
		auto it = m_map.lower_bound(start);
		if (it != m_map.begin()) {
			auto prev = std::prev(it);
			if (prev->second.end > start) {
				if (prev->second.end > end) m_map.emplace_hint(it, end, prev->second); // its tail beyond the new range
				prev->second.end = start;
			}
		}
		it = m_map.lower_bound(start);
		while (it != m_map.end() && it->first < end) {
			if (it->second.end > end) {
				const Writer tail = it->second;
				it                = m_map.erase(it);
				m_map.emplace_hint(it, end, tail);
				break;
			}
			it = m_map.erase(it);
		}
		Writer copy = writer;
		copy.end    = end;
		m_map.emplace(start, copy);
	}
	// The writer of the byte, else the newest writer of any byte of [lo, hi).
	[[nodiscard]] const Writer* Find(uint64_t address, uint64_t lo, uint64_t hi, bool* exact) const {
		*exact  = false;
		auto it = m_map.upper_bound(address);
		if (it != m_map.begin()) {
			const auto prev = std::prev(it);
			if (prev->second.end > address) {
				*exact = true;
				return &prev->second;
			}
		}
		const Writer* newest = nullptr;
		it                   = m_map.upper_bound(lo);
		if (it != m_map.begin()) it = std::prev(it);
		for (; it != m_map.end() && it->first < hi; ++it) {
			if (it->second.end <= lo) continue;
			if (newest == nullptr || it->second.tick > newest->tick) newest = &it->second;
		}
		return newest;
	}
	[[nodiscard]] size_t Size() const { return m_map.size(); }

private:
	std::map<uint64_t, Writer> m_map;
};

struct WriterKey {
	uint64_t hash       = 0;
	uint32_t descriptor = UINT32_MAX;
	uint8_t  kind       = static_cast<uint8_t>(Kind::Count); // Kind::Count: no writer known
	bool     operator==(const WriterKey&) const = default;
};
struct WriterKeyHash {
	size_t operator()(const WriterKey& key) const noexcept {
		return static_cast<size_t>(key.hash * 0x9E3779B97F4A7C15ull ^ (uint64_t {key.descriptor} << 8u) ^ key.kind);
	}
};

struct Bucket {
	uint64_t                                             requests = 0;
	std::array<uint64_t, static_cast<size_t>(Path::Count)> paths {};
	std::array<uint64_t, CmpCount>                        cmps {};
	uint64_t dl_changed = 0, dl_same = 0;       // downloaded: the GPU had changed the page / had not (same bytes)
	uint64_t fixed = 0, fixed_outside = 0;      // the writer had fixed_writes; of them, the page lies outside every range
	uint64_t head = 0, middle = 0, tail = 0;    // the page in the written range: first eighth / middle / last eighth
	uint64_t done = 0;                          // the writer's command buffer had finished when the request began
	std::array<uint64_t, 4> frames {};          // presents between the write and the request: 0, 1, 2, 3+
	uint64_t changed_bytes = 0;                 // of the compared pages
	uint64_t line_changed  = 0;                 // compared pages whose faulting 64-byte line the GPU had changed
	std::vector<uint32_t> age_us;               // a sample: the first 1024 of the window
	uint64_t size_min = UINT64_MAX, size_max = 0;
	uint64_t sample_base = 0, sample_size = 0, sample_page = 0;
	// The cached buffer holding the sample page when the request ran (id 0: none).
	uint64_t buffer_base = 0, buffer_size = 0, buffer_id = 0;
	uint32_t resource = UINT32_MAX, max_byte_extent = 0;
	uint8_t  flags = 0;
};

struct Chunk {
	uint64_t  requests = 0, zero = 0, changed = 0;
	WriterKey candidate {}; // the majority writer (Boyer-Moore vote)
	uint64_t  votes = 0;
};

struct Census {
	WriterMap                                             writers;
	std::map<std::vector<std::pair<uint32_t, uint32_t>>, uint32_t> fixed_index;
	std::vector<std::vector<std::pair<uint32_t, uint32_t>>>        fixed_lists;
	std::unordered_map<uint64_t, std::string>             markers; // shader hash -> debug marker path at its first write
	// The window (reset at each report).
	Bucket                                                totals;
	std::unordered_map<WriterKey, Bucket, WriterKeyHash>  by_writer;
	std::unordered_map<uint64_t, Chunk>                   chunks; // 64 KiB chunk -> requests
	uint64_t writes = 0, repeats = 0, write_bytes = 0, exact = 0, page_only = 0, unattributed = 0, census_ns = 0;
	int64_t  window_us    = 0;
	uint32_t window_frame = 0;
};
Census& GetCensus() {
	static Census census;
	return census;
}
std::atomic<uint64_t> g_offthread_writes {0};

// The page's guest bytes from the request's start (RingCensusBegin), the GPU copy and the bytes now (RingCensusCompare).
thread_local std::array<uint8_t, CensusPage> t_snapshot {};
thread_local std::array<uint8_t, CensusPage> t_other {};
thread_local uint32_t                        t_depth = 0; // requests begun and not ended (a nested one keeps no snapshot)

// A read of write-combined (BAR) memory with streaming loads (the BAR download's way, bufferCache.cpp
// CopyFromWriteCombinedImpl): one 4 KiB page at most.
void ReadWriteCombined(uint8_t* dst, const uint8_t* src, uint64_t size) {
	_mm_mfence();
	const auto head = std::min<uint64_t>(size, (32u - (reinterpret_cast<uintptr_t>(src) & 31u)) & 31u);
	if (head != 0) {
		std::memcpy(dst, src, head);
		dst += head;
		src += head;
		size -= head;
	}
	while (size >= 32) {
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst),
		                    _mm256_stream_load_si256(reinterpret_cast<__m256i*>(const_cast<uint8_t*>(src))));
		src += 32;
		dst += 32;
		size -= 32;
	}
	if (size != 0) std::memcpy(dst, src, size);
}

// Bytes [lo, hi) of the page (offsets): 8-byte words that differ, counted in bytes; and those in the faulting line.
void Diff(const uint8_t* a, const uint8_t* b, uint64_t lo, uint64_t hi, uint64_t line, RingCensus::Request& r) {
	uint32_t changed = 0, line_changed = 0;
	for (uint64_t i = lo & ~uint64_t {7}; i < hi; i += 8u) {
		if (std::memcmp(a + i, b + i, 8u) == 0) continue;
		changed += 8u;
		if (i >= line && i < line + 64u) line_changed += 8u;
	}
	r.changed      = changed;
	r.line_changed = line_changed;
	r.cmp          = changed != 0u ? CmpChanged : CmpZero;
}

void Account(Bucket& b, const RingCensus::Request& r, const Writer* w, uint64_t page, uint32_t frame_now) {
	b.requests++;
	b.paths[static_cast<size_t>(r.path)]++;
	b.cmps[r.cmp]++;
	if (r.downloaded && r.cmp == CmpChanged) b.dl_changed++;
	if (r.downloaded && r.cmp == CmpZero) b.dl_same++;
	if (r.cmp == CmpChanged || r.cmp == CmpZero) {
		b.changed_bytes += r.changed;
		if (r.line_changed != 0u) b.line_changed++;
	}
	if (w == nullptr) return;
	if (w->fixed != UINT32_MAX) {
		b.fixed++;
		bool inside = false;
		for (const auto& [first, second]: GetCensus().fixed_lists[w->fixed]) {
			const uint64_t lo = w->base + first, hi = w->base + second;
			if (lo < page + CensusPage && page < hi) inside = true;
		}
		if (!inside) b.fixed_outside++;
	}
	const uint64_t offset = page > w->base ? page - w->base : 0u;
	const uint64_t eighth = w->size / 8u;
	if (w->size <= CensusPage || offset < eighth) {
		b.head++;
	} else if (offset + CensusPage > w->size - eighth) {
		b.tail++;
	} else {
		b.middle++;
	}
	if (r.known_tick >= w->tick) b.done++;
	const uint32_t frames = frame_now - w->frame;
	b.frames[std::min<uint32_t>(frames, 3u)]++;
	if (b.age_us.size() < 1024u) b.age_us.push_back(static_cast<uint32_t>(std::clamp<int64_t>(r.start_us - w->us, 0, UINT32_MAX)));
	b.size_min = std::min(b.size_min, w->size);
	b.size_max = std::max(b.size_max, w->size);
	if (b.sample_page == 0u) {
		b.sample_base = w->base;
		b.sample_size = w->size;
		b.sample_page = page;
	}
	b.resource        = w->resource;
	b.max_byte_extent = w->max_byte_extent;
	b.flags           = w->flags & 15u;
}

std::string BucketText(const Bucket& b) {
	std::string text = "req=" + std::to_string(b.requests) + " paths[";
	for (size_t i = 0; i < PathNames.size(); i++) {
		text += (i != 0 ? " " : "") + std::string(PathNames[i]) + "=" + std::to_string(b.paths[i]);
	}
	text += "] cmp[";
	for (size_t i = 0; i < CmpNames.size(); i++) {
		text += (i != 0 ? " " : "") + std::string(CmpNames[i]) + "=" + std::to_string(b.cmps[i]);
	}
	text += "] dl[changed=" + std::to_string(b.dl_changed) + " same=" + std::to_string(b.dl_same) + "]";
	text += " fixed=" + std::to_string(b.fixed) + "/" + std::to_string(b.fixed_outside);
	text += " pos[head=" + std::to_string(b.head) + " mid=" + std::to_string(b.middle) + " tail=" + std::to_string(b.tail) + "]";
	text += " done=" + std::to_string(b.done);
	text += " frames[0=" + std::to_string(b.frames[0]) + " 1=" + std::to_string(b.frames[1]) + " 2=" +
	        std::to_string(b.frames[2]) + " 3+=" + std::to_string(b.frames[3]) + "]";
	auto ages = b.age_us;
	std::sort(ages.begin(), ages.end());
	const auto at = [&](size_t percent) { return ages.empty() ? 0u : ages[std::min(ages.size() - 1u, ages.size() * percent / 100u)]; };
	text += " age_us[p10=" + std::to_string(at(10)) + " p50=" + std::to_string(at(50)) + " p90=" + std::to_string(at(90)) + "]";
	text += " changed_bytes=" + std::to_string(b.changed_bytes) + " line_changed=" + std::to_string(b.line_changed);
	return text;
}

std::string FlagText(uint8_t flags) {
	std::string text;
	if ((flags & 1u) != 0u) text += "append,";
	if ((flags & 2u) != 0u) text += "read,";
	if ((flags & 4u) != 0u) text += "atomic,";
	if ((flags & 8u) != 0u) text += "formatted,";
	if (text.empty()) return "-";
	text.pop_back();
	return text;
}

void ReportWindow(Census& c, int64_t now_us) {
	const uint32_t frame  = DebugPresentCount().load(std::memory_order_relaxed);
	const auto&    t      = c.totals;
	const auto     frames = frame - c.window_frame;
	std::fprintf(stderr,
	             "RING_CENSUS window frame=%u frames=%u ms=%lld %s exact=%llu page_only=%llu unattributed=%llu writes=%llu "
	             "repeats=%llu write_MiB=%llu offthread=%llu writer_ranges=%zu census_ms=%llu\n",
	             frame, frames, static_cast<long long>((now_us - c.window_us) / 1000), BucketText(t).c_str(),
	             static_cast<unsigned long long>(c.exact), static_cast<unsigned long long>(c.page_only),
	             static_cast<unsigned long long>(c.unattributed), static_cast<unsigned long long>(c.writes),
	             static_cast<unsigned long long>(c.repeats), static_cast<unsigned long long>(c.write_bytes >> 20u),
	             static_cast<unsigned long long>(g_offthread_writes.exchange(0, std::memory_order_relaxed)), c.writers.Size(),
	             static_cast<unsigned long long>(c.census_ns / 1000000u));
	std::vector<std::pair<WriterKey, const Bucket*>> writers;
	writers.reserve(c.by_writer.size());
	for (const auto& [key, bucket]: c.by_writer) writers.emplace_back(key, &bucket);
	std::sort(writers.begin(), writers.end(), [](const auto& a, const auto& b) { return a.second->requests > b.second->requests; });
	for (size_t i = 0; i < std::min<size_t>(writers.size(), 16u); i++) {
		const auto& [key, b] = writers[i];
		const auto  marker   = c.markers.find(key.hash);
		std::fprintf(stderr,
		             "RING_CENSUS writer frame=%u rank=%zu hash=%016llx kind=%s res=%d desc=%d flags=%s ext=0x%x "
		             "size=0x%llx-0x%llx %s base=0x%llx bsize=0x%llx page=0x%llx buffer=%llu:0x%llx+0x%llx marker=[%s]\n",
		             frame, i + 1u, static_cast<unsigned long long>(key.hash), KindNames[std::min<size_t>(key.kind, 3u)],
		             b->resource == UINT32_MAX ? -1 : static_cast<int>(b->resource),
		             key.descriptor == UINT32_MAX ? -1 : static_cast<int>(key.descriptor), FlagText(b->flags).c_str(),
		             b->max_byte_extent, static_cast<unsigned long long>(b->size_min == UINT64_MAX ? 0u : b->size_min),
		             static_cast<unsigned long long>(b->size_max), BucketText(*b).c_str(),
		             static_cast<unsigned long long>(b->sample_base), static_cast<unsigned long long>(b->sample_size),
		             static_cast<unsigned long long>(b->sample_page), static_cast<unsigned long long>(b->buffer_id),
		             static_cast<unsigned long long>(b->buffer_base), static_cast<unsigned long long>(b->buffer_size),
		             marker != c.markers.end() ? marker->second.c_str() : "-");
	}
	std::vector<std::pair<uint64_t, const Chunk*>> chunks;
	chunks.reserve(c.chunks.size());
	for (const auto& [key, chunk]: c.chunks) chunks.emplace_back(key, &chunk);
	std::sort(chunks.begin(), chunks.end(), [](const auto& a, const auto& b) { return a.second->requests > b.second->requests; });
	for (size_t i = 0; i < std::min<size_t>(chunks.size(), 16u); i++) {
		const auto& [key, chunk] = chunks[i];
		std::fprintf(stderr, "RING_CENSUS chunk frame=%u rank=%zu addr=0x%llx req=%llu zero=%llu changed=%llu writer=%016llx/%s/%d\n",
		             frame, i + 1u, static_cast<unsigned long long>(key << 16u), static_cast<unsigned long long>(chunk->requests),
		             static_cast<unsigned long long>(chunk->zero), static_cast<unsigned long long>(chunk->changed),
		             static_cast<unsigned long long>(chunk->candidate.hash), KindNames[std::min<size_t>(chunk->candidate.kind, 3u)],
		             chunk->candidate.descriptor == UINT32_MAX ? -1 : static_cast<int>(chunk->candidate.descriptor));
	}
	c.totals    = {};
	c.by_writer.clear();
	c.chunks.clear();
	c.writes = c.repeats = c.write_bytes = c.exact = c.page_only = c.unattributed = c.census_ns = 0;
	c.window_us    = now_us;
	c.window_frame = frame;
}

void MaybeReport(Census& c) {
	const int64_t now = NowUs();
	if (c.window_us == 0) {
		c.window_us    = now;
		c.window_frame = DebugPresentCount().load(std::memory_order_relaxed);
		return;
	}
	if (now - c.window_us >= 5000000) ReportWindow(c, now);
}

} // namespace

void BufferCache::RingCensusWrite(uint64_t vaddr, uint64_t size, bool repeat) {
	if (!GuestGpu::IsGpuThread()) {
		g_offthread_writes.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	const auto t0 = std::chrono::steady_clock::now();
	auto&      c  = GetCensus();
	Writer     w;
	const auto* binding = RingCensus::t_binding;
	w.kind              = binding != nullptr ? Kind::Storage : RingCensus::t_kind == Kind::BdaRange ? Kind::BdaRange : Kind::Other;
	if (binding != nullptr) {
		w.descriptor      = binding->descriptor;
		w.resource        = binding->resource;
		w.max_byte_extent = binding->max_byte_extent;
		w.flags = static_cast<uint8_t>((binding->append ? 1u : 0u) | (binding->read ? 2u : 0u) | (binding->atomic ? 4u : 0u) |
		                               (binding->formatted ? 8u : 0u));
		if (binding->fixed_ranges != nullptr) {
			const auto [it, added] = c.fixed_index.try_emplace(*binding->fixed_ranges, static_cast<uint32_t>(c.fixed_lists.size()));
			if (added) c.fixed_lists.push_back(*binding->fixed_ranges);
			w.fixed = it->second;
		}
	}
	if (repeat) w.flags = static_cast<uint8_t>(w.flags | 16u);
	// Fills, copies, image and query writes run outside a draw's binding: the last bound shader is not their writer.
	w.hash  = w.kind != Kind::Other ? DebugWriterHash().load(std::memory_order_relaxed) : 0u;
	w.base  = vaddr;
	w.size  = size;
	w.tick  = m_scheduler.CurrentTick();
	w.us    = NowUs();
	w.frame = DebugPresentCount().load(std::memory_order_relaxed);
	c.writers.Insert(vaddr, vaddr + size, w);
	if (w.hash != 0u && !c.markers.contains(w.hash)) c.markers.emplace(w.hash, DebugMarkerPath());
	(repeat ? c.repeats : c.writes)++;
	c.write_bytes += size;
	MaybeReport(c);
	c.census_ns += static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
}

void BufferCache::RingCensusBegin(RingCensus::Request& r, uint64_t vaddr, uint64_t size) {
	const auto t0 = std::chrono::steady_clock::now();
	auto&      master = m_scheduler.GetMasterSemaphore();
	master.Refresh(); // for `done=` (if census_ms is high, the first cost to drop: a stale tick only shifts `done=`)
	r.active       = true;
	r.vaddr        = vaddr;
	r.size         = size;
	r.start_tick   = m_scheduler.CurrentTick();
	r.known_tick   = master.KnownGpuTick();
	r.start_us     = NowUs();
	r.gpu_modified = m_memory_tracker.IsRegionGpuModified(vaddr, size);
	// The page is no-access while GPU-modified and the faulting thread waits for this request: its guest bytes stay as
	// they are until a publish of this request (or the release).
	if (r.gpu_modified && RingCensus::Mode() == 1u && t_depth == 0u) {
		r.snapshot = LibKernel::Memory::TryReadBacking(vaddr & ~(CensusPage - 1u), t_snapshot.data(), CensusPage);
	}
	t_depth++;
	GetCensus().census_ns += static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
}

void BufferCache::RingCensusCompare(RingCensus::Request& r) {
	if (r.compared) return;
	r.compared = true;
	if (!r.gpu_modified) {
		r.cmp = CmpNone;
		return;
	}
	if (RingCensus::Mode() != 1u) {
		r.cmp = CmpOff;
		return;
	}
	const auto     t0   = std::chrono::steady_clock::now();
	const uint64_t page = r.vaddr & ~(CensusPage - 1u);
	const uint64_t line = (r.vaddr - page) & ~uint64_t {63};
	const auto     done = [&] {
		GetCensus().census_ns += static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
	};
	if (r.downloaded) {
		// The publish wrote the GPU bytes into guest memory: what it changed of the page is what the GPU had changed.
		if (!r.snapshot || !LibKernel::Memory::TryReadBacking(page, t_other.data(), CensusPage)) {
			r.cmp = CmpReadFail;
		} else {
			Diff(t_snapshot.data(), t_other.data(), 0, CensusPage, line, r);
		}
		done();
		return;
	}
	// Nothing published: the GPU copy against guest memory, read from the host mapping when the bytes are final (never
	// waited for; the conditions of TryBarDownload).
	const auto* owner = m_page_table.Find(page >> PageTable::kPageBits);
	if (owner == nullptr || !*owner) {
		r.cmp = CmpNoBuffer;
		done();
		return;
	}
	auto&          buffer = m_slot_buffers[*owner];
	const uint64_t lo     = std::max(page, buffer.CpuAddress());
	const uint64_t hi     = std::min(page + CensusPage, buffer.CpuAddress() + buffer.Size());
	if (lo >= hi || buffer.IsDirect() || buffer.is_deleted) {
		r.cmp = CmpNoBuffer;
		done();
		return;
	}
	if (buffer.Mapped().empty()) {
		r.cmp = CmpNoMap;
		done();
		return;
	}
	uint64_t tick = QueryWriteTick(lo, hi);
	if (tick == 0u) {
		const auto granule = GranuleWriteTick(WriteTickKey(lo));
		if (granule) tick = *granule;
	}
	const auto ready = m_buffer_ready_tick.find(reinterpret_cast<uint64_t>(static_cast<VkBuffer>(buffer.Handle())));
	if (ready != m_buffer_ready_tick.end()) tick = std::max(tick, ready->second);
	const bool pending_write = std::ranges::any_of(m_bar_pending_writes, [&](const auto& w) {
		return w.first < hi && lo < w.first + w.second;
	});
	if (tick == 0u || tick >= m_scheduler.CurrentTick() || !m_scheduler.GetMasterSemaphore().IsFree(tick) || pending_write) {
		r.cmp = CmpPending;
		done();
		return;
	}
	if (!r.snapshot && (t_depth > 1u || !LibKernel::Memory::TryReadBacking(page, t_snapshot.data(), CensusPage))) {
		r.cmp = CmpReadFail;
		done();
		return;
	}
	if (!buffer.IsCoherent()) buffer.Invalidate(buffer.Offset(lo), hi - lo);
	ReadWriteCombined(t_other.data() + (lo - page), buffer.Mapped().data() + buffer.Offset(lo), hi - lo);
	Diff(t_snapshot.data(), t_other.data(), lo - page, hi - page, line, r);
	done();
}

void BufferCache::RingCensusEnd(RingCensus::Request& r) {
	const auto t0 = std::chrono::steady_clock::now();
	if (t_depth != 0u) t_depth--;
	if (!r.compared) { // every write path compares before it releases the page; nothing is read after it
		r.compared = true;
		r.cmp      = r.gpu_modified ? CmpOff : CmpNone;
	}
	auto&          c     = GetCensus();
	const uint64_t page  = r.vaddr & ~(CensusPage - 1u);
	bool           exact = false;
	const Writer*  w     = c.writers.Find(r.vaddr, page, page + CensusPage, &exact);
	const uint32_t frame = DebugPresentCount().load(std::memory_order_relaxed);
	(w == nullptr ? c.unattributed : exact ? c.exact : c.page_only)++;
	WriterKey key;
	if (w != nullptr) {
		key.hash       = w->hash;
		key.descriptor = w->descriptor;
		key.kind       = static_cast<uint8_t>(w->kind);
	}
	Account(c.totals, r, w, page, frame);
	auto& bucket = c.by_writer[key];
	Account(bucket, r, w, page, frame);
	if (bucket.buffer_id == 0u) { // the cached buffer holding the page now (a lookup, nothing created)
		const auto* owner = std::as_const(m_page_table).Find(page >> PageTable::kPageBits);
		if (owner != nullptr && *owner) {
			const auto& buffer = m_slot_buffers[*owner];
			bucket.buffer_id   = owner->index;
			bucket.buffer_base = buffer.CpuAddress();
			bucket.buffer_size = buffer.Size();
		}
	}
	auto& chunk = c.chunks[page >> 16u];
	chunk.requests++;
	if (r.cmp == CmpZero) chunk.zero++;
	if (r.cmp == CmpChanged) chunk.changed++;
	if (chunk.votes == 0u) {
		chunk.candidate = key;
		chunk.votes     = 1;
	} else if (chunk.candidate == key) {
		chunk.votes++;
	} else {
		chunk.votes--;
	}
	c.census_ns += static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
	MaybeReport(c);
}

} // namespace Libs::Graphics
