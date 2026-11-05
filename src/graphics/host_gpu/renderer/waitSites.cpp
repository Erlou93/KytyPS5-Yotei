#include "graphics/host_gpu/renderer/waitSites.h"

#include "graphics/host_gpu/renderer/perfCounters.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h> // IWYU pragma: keep (RtlCaptureStackBackTrace, GetModuleHandleW)

namespace Libs::Graphics::WaitSites {

namespace {

constexpr uint32_t Depth = 8; // return addresses kept above the scheduler

struct Site {
	uint64_t                     count = 0;
	uint64_t                     nanos = 0;
	bool                         gpu   = false;
	std::array<uint64_t, Depth>  rva {};
};

std::mutex                              g_mutex;
std::unordered_map<uint64_t, Site>      g_sites;
std::array<uint64_t, 2>                 g_count {}; // [0] GPU thread, [1] other threads
std::array<uint64_t, 2>                 g_nanos {};
std::chrono::steady_clock::time_point   g_window = std::chrono::steady_clock::now();

void Report() { // under g_mutex
	const auto now = std::chrono::steady_clock::now();
	const auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_window).count();
	g_window       = now;
	std::fprintf(stderr, "WAIT_SITES window_ms=%lld gpu_thread=%llu/%llums other=%llu/%llums sites=%zu\n",
	             static_cast<long long>(ms), static_cast<unsigned long long>(g_count[0]),
	             static_cast<unsigned long long>(g_nanos[0] / 1000000u), static_cast<unsigned long long>(g_count[1]),
	             static_cast<unsigned long long>(g_nanos[1] / 1000000u), g_sites.size());
	std::vector<std::pair<uint64_t, Site>> sorted(g_sites.begin(), g_sites.end());
	std::ranges::sort(sorted, [](const auto& a, const auto& b) { return a.second.nanos > b.second.nanos; });
	for (size_t i = 0; i < sorted.size() && i < 24u; i++) {
		const auto& site = sorted[i].second;
		std::fprintf(stderr, "WAIT_SITE %s n=%llu ms=%.1f stack=", site.gpu ? "gpu" : "other",
		             static_cast<unsigned long long>(site.count), static_cast<double>(site.nanos) / 1e6);
		for (const auto rva: site.rva) {
			if (rva == 0) break;
			std::fprintf(stderr, "%llx,", static_cast<unsigned long long>(rva));
		}
		std::fprintf(stderr, "\n");
	}
	g_sites.clear();
	g_count = {};
	g_nanos = {};
}

} // namespace

bool Enabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_TRACE_WAIT_SITES");
		const bool  set   = value != nullptr && value[0] == '1';
		if (set) std::fprintf(stderr, "WAIT_SITES enabled (KYTY_TRACE_WAIT_SITES)\n");
		return set;
	}();
	return on;
}

void Record(uint64_t nanos) {
	static const uint64_t base = reinterpret_cast<uint64_t>(GetModuleHandleW(nullptr));
	std::array<void*, Depth> frames {};
	// Skip Record; the first frames are the scheduler function (the Scope is inlined there) and its callers.
	const auto     captured = RtlCaptureStackBackTrace(1, Depth, frames.data(), nullptr);
	const bool     gpu      = PerfIsGpuThread();
	Site           site;
	uint64_t       key = gpu ? 1u : 2u;
	for (uint32_t i = 0; i < captured; i++) {
		const auto address = reinterpret_cast<uint64_t>(frames[i]);
		site.rva[i]        = address >= base ? address - base : address;
		key                = (key ^ site.rva[i]) * 0x100000001b3ull;
	}
	std::lock_guard lock(g_mutex);
	auto& entry = g_sites[key];
	if (entry.count == 0) {
		entry.rva = site.rva;
		entry.gpu = gpu;
	}
	entry.count++;
	entry.nanos += nanos;
	g_count[gpu ? 0 : 1]++;
	g_nanos[gpu ? 0 : 1] += nanos;
	if (std::chrono::steady_clock::now() - g_window >= std::chrono::seconds(5)) Report();
}

} // namespace Libs::Graphics::WaitSites
