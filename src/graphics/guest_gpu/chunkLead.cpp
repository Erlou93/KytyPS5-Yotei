#include "graphics/guest_gpu/chunkLead.h"

#include "graphics/guest_gpu/command_processor/passCensus.h"
#include "graphics/host_gpu/renderer/debugSample.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace Libs::Graphics::ChunkLead {

namespace {

constexpr uint32_t ChunkCap = 128;
constexpr int      Passes   = 3;
constexpr int      LeadBuckets = 8; // 0 | 1-15 | 16-63 | 64-127 | 128-255 | 256-511 | 512-1023 | 1024+
constexpr int      AgeBuckets  = 7; // <50us | 50-200 | 200-500 | 500-1000 | 1-2ms | 2-5ms | 5ms+
constexpr int      Causes      = 6; // resolve | drain | parser | blocked | no_work | commands
const char* const  CauseNames[Causes] = {"resolve", "drain", "parser", "blocked", "no_work", "commands"};
constexpr int      CpStates    = 6; // working | snapshot | drain | ring_full | blocked | no_work
const char* const  CpStateNames[CpStates] = {"working", "snapshot", "drain", "ring_full", "blocked", "no_work"};

int LeadBucket(size_t n) {
	return n == 0 ? 0 : n < 16 ? 1 : n < 64 ? 2 : n < 128 ? 3 : n < 256 ? 4 : n < 512 ? 5 : n < 1024 ? 6 : 7;
}
int AgeBucket(double us) {
	return us < 50 ? 0 : us < 200 ? 1 : us < 500 ? 2 : us < 1000 ? 3 : us < 2000 ? 4 : us < 5000 ? 5 : 6;
}
int CpStateIndex(uint8_t s) {
	switch (s) {
		case CpSnapshot: return 1;
		case CpDrain: return 2;
		case CpRingFull: return 3;
		case CpBlocked: return 4;
		case CpNoWork: return 5;
		default: return 0;
	}
}

const std::vector<std::string>& TargetList() {
	static const std::vector<std::string> list = [] {
		std::vector<std::string> out;
		const char*              env  = std::getenv("KYTY_TRACE_CHUNK_PASSES");
		const std::string        text = env != nullptr ? env : "Frame>Deferred,Frame>Cascaded shadow map,Lighting>PS Lights";
		size_t                   start = 0;
		while (start <= text.size()) {
			const auto end = std::min(text.find(',', start), text.size());
			if (end > start) out.push_back(text.substr(start, end - start));
			start = end + 1;
		}
		return out;
	}();
	return list;
}

// The tsc rate (ticks per ns), refreshed every report window by each reporting thread.
struct Clock {
	uint64_t                              tsc0 = 0;
	std::chrono::steady_clock::time_point t0   = std::chrono::steady_clock::now();
	double                                ticks_per_ns = 0.0;
	double Rate() {
		if (ticks_per_ns > 0.0) return ticks_per_ns;
		const auto ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
		return ns > 1e6 ? static_cast<double>(Tsc() - tsc0) / ns : 3.0; // ~3 GHz until a millisecond passed
	}
	void Refresh(uint64_t tsc, std::chrono::steady_clock::time_point now) {
		const auto ns = std::chrono::duration<double, std::nano>(now - t0).count();
		if (ns > 0 && tsc > tsc0) ticks_per_ns = static_cast<double>(tsc - tsc0) / ns;
		tsc0 = tsc;
		t0   = now;
	}
};

struct PassStats {
	uint64_t                             chunks = 0, ops = 0;
	std::array<uint64_t, LeadBuckets>    raw {}, res {};
	std::array<uint64_t, AgeBuckets>     age {};
	std::array<uint64_t, 3>              ahead {};
	std::array<uint64_t, CpStates>       low_state {};
	// Idle: [0] inside the pass, [1] its entry; per cause: events and ns.
	std::array<std::array<uint64_t, Causes>, 2> idle_events {}, idle_ns {};
};

// Execution thread only.
struct ExecState {
	std::vector<int8_t>         target; // per marker pass id: -2 unknown, -1 none, 0..2
	int                         open_pass = -1;
	uint32_t                    open_n    = 0;
	uint64_t                    open_seq = 0, last_seq = 0;
	size_t                      open_raw = 0, open_res = 0;
	double                      open_age_us = 0;
	int                         open_cp = 0;
	uint64_t                    chunk_index = 0;
	std::array<uint64_t, 4>     emitted_at_open {};
	int                         prev_pass = -1;
	// The idle wait waiting for the next operation.
	bool                        idle_pending = false;
	uint64_t                    idle_ns      = 0;
	int                         idle_cause   = 0;
	std::array<PassStats, Passes>                      pass {};
	std::array<uint64_t, Causes>                       other_events {}, other_ns {};
	std::array<uint64_t, LeadBuckets>                  res_all {};
	uint64_t                                           ops_seen = 0;
	Clock                                              clock;
	std::chrono::steady_clock::time_point              window = std::chrono::steady_clock::now();
	uint32_t                                           window_frame = 0;
};
ExecState& Exec() {
	static ExecState s;
	return s;
}

std::array<std::array<std::atomic<uint64_t>, 3>, 2> g_snap_taken {}, g_snap_waits {}, g_snap_wait_ns {};

std::string Hist(const uint64_t* v, int n, uint64_t total) {
	std::string out;
	for (int i = 0; i < n; i++) {
		char buf[24];
		std::snprintf(buf, sizeof(buf), "%s%.1f", i == 0 ? "" : "|", total != 0 ? 100.0 * static_cast<double>(v[i]) / static_cast<double>(total) : 0.0);
		out += buf;
	}
	return out;
}

void CloseChunk(ExecState& s) {
	if (s.open_pass < 0) return;
	auto&          p = s.pass[static_cast<size_t>(s.open_pass)];
	const uint64_t n = s.open_n;
	p.chunks++;
	p.ops += n;
	p.raw[static_cast<size_t>(LeadBucket(s.open_raw))] += n;
	p.res[static_cast<size_t>(LeadBucket(s.open_res))] += n;
	p.age[static_cast<size_t>(AgeBucket(s.open_age_us))] += n;
	if (s.open_raw < 128) p.low_state[static_cast<size_t>(s.open_cp)] += n;
	// Whether the whole chunk was emitted when the execution thread opened the chunk k before it.
	for (uint64_t k = 1; k <= 3 && k <= s.chunk_index; k++) {
		if (s.emitted_at_open[(s.chunk_index - k) % 4] >= s.last_seq) p.ahead[k - 1] += n;
	}
	s.chunk_index++;
	s.open_pass = -1;
	s.open_n    = 0;
}

void Report(ExecState& s) {
	const auto now = std::chrono::steady_clock::now();
	if (now - s.window < std::chrono::seconds(5)) return;
	s.clock.Refresh(Tsc(), now);
	const uint32_t frame  = DebugPresentCount().load(std::memory_order_relaxed);
	const uint32_t frames = frame > s.window_frame ? frame - s.window_frame : 1u;
	const double   f      = frames;
	const auto&    names  = TargetList();
	for (int i = 0; i < Passes && i < static_cast<int>(names.size()); i++) {
		auto& p = s.pass[static_cast<size_t>(i)];
		std::string low;
		for (int k = 0; k < CpStates; k++) low += std::string(k == 0 ? "" : " ") + CpStateNames[k] + "=" + std::to_string(p.low_state[static_cast<size_t>(k)]);
		std::fprintf(stderr,
		             "PERF chunk_lead pass=\"%s\" frames=%u chunks/frame=%.1f ops/frame=%.1f raw%%[0|1-15|16-63|64-127|128-255|256-511|512-1023|1024+]=%s "
		             "res%%=%s age%%[<50us|50-200|200-500|500-1000|1-2ms|2-5ms|5ms+]=%s ahead%%[1|2|3]=%s low_ops[%s]\n",
		             names[static_cast<size_t>(i)].c_str(), frames, p.chunks / f, p.ops / f, Hist(p.raw.data(), LeadBuckets, p.ops).c_str(),
		             Hist(p.res.data(), LeadBuckets, p.ops).c_str(), Hist(p.age.data(), AgeBuckets, p.ops).c_str(),
		             Hist(p.ahead.data(), 3, p.ops).c_str(), low.c_str());
		for (int where = 0; where < 2; where++) {
			uint64_t    ev = 0, ns = 0;
			std::string causes;
			for (int c = 0; c < Causes; c++) {
				ev += p.idle_events[static_cast<size_t>(where)][static_cast<size_t>(c)];
				ns += p.idle_ns[static_cast<size_t>(where)][static_cast<size_t>(c)];
				char buf[64];
				std::snprintf(buf, sizeof(buf), "%s%s=%.1f/%.3f", c == 0 ? "" : " ", CauseNames[c],
				              p.idle_events[static_cast<size_t>(where)][static_cast<size_t>(c)] / f,
				              p.idle_ns[static_cast<size_t>(where)][static_cast<size_t>(c)] / 1e6 / f);
				causes += buf;
			}
			std::fprintf(stderr, "PERF chunk_idle pass=\"%s\" %s events/frame=%.1f ms/frame=%.3f cause(events/ms per frame)[%s]\n",
			             names[static_cast<size_t>(i)].c_str(), where == 0 ? "inside" : "entry", ev / f, ns / 1e6 / f, causes.c_str());
		}
		p = PassStats {};
	}
	{
		uint64_t    ev = 0, ns = 0;
		std::string causes;
		for (int c = 0; c < Causes; c++) {
			ev += s.other_events[static_cast<size_t>(c)];
			ns += s.other_ns[static_cast<size_t>(c)];
			char buf[64];
			std::snprintf(buf, sizeof(buf), "%s%s=%.1f/%.3f", c == 0 ? "" : " ", CauseNames[c], s.other_events[static_cast<size_t>(c)] / f,
			              s.other_ns[static_cast<size_t>(c)] / 1e6 / f);
			causes += buf;
		}
		std::fprintf(stderr, "PERF chunk_idle pass=\"other\" events/frame=%.1f ms/frame=%.3f cause(events/ms per frame)[%s]\n", ev / f,
		             ns / 1e6 / f, causes.c_str());
		s.other_events = {};
		s.other_ns     = {};
	}
	uint64_t all = 0;
	for (const auto v: s.res_all) all += v;
	std::fprintf(stderr, "PERF resolved_depth samples=%llu all%%[0|1-15|16-63|64-127|128-255|256-511|512-1023|1024+]=%s\n",
	             static_cast<unsigned long long>(all), Hist(s.res_all.data(), LeadBuckets, all).c_str());
	s.res_all = {};
	const char* const kinds[3] = {"ctx", "ucfg", "sh"};
	for (int q = 0; q < 2; q++) {
		std::string text;
		for (int k = 0; k < 3; k++) {
			char buf[96];
			std::snprintf(buf, sizeof(buf), " %s taken/frame=%.0f waits/frame=%.1f wait_ms/frame=%.3f", kinds[k],
			              g_snap_taken[static_cast<size_t>(q)][static_cast<size_t>(k)].exchange(0, std::memory_order_relaxed) / f,
			              g_snap_waits[static_cast<size_t>(q)][static_cast<size_t>(k)].exchange(0, std::memory_order_relaxed) / f,
			              g_snap_wait_ns[static_cast<size_t>(q)][static_cast<size_t>(k)].exchange(0, std::memory_order_relaxed) / 1e6 / f);
			text += buf;
		}
		std::fprintf(stderr, "PERF snapshot_rings queue=%s%s\n", q == 0 ? "gfx" : "compute", text.c_str());
	}
	s.window       = now;
	s.window_frame = frame;
}

// Resolve thread only.
struct ResolveState {
	struct Pass {
		uint64_t draw_ticks = 0, draws = 0, dispatch_ticks = 0, dispatches = 0, resolved = 0;
		std::array<uint64_t, RpCount> parts {}; // N2.0
		uint64_t                      buffers = 0;
	};
	std::vector<Pass>                     pass;
	uint64_t                              ops = 0;
	Clock                                 clock;
	std::chrono::steady_clock::time_point window       = std::chrono::steady_clock::now();
	uint32_t                              window_frame = 0;
};

} // namespace

uint64_t Tsc() {
	return __rdtsc();
}

bool On() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_TRACE_CHUNK_LEAD");
		const bool  o     = value != nullptr && value[0] == '1';
		if (o) std::fprintf(stderr, "TRACE_CHUNK_LEAD enabled (project 15 N1: the chunks' lead)\n");
		return o;
	}();
	return on;
}

int TargetPass(uint16_t pass) {
	auto& s = Exec();
	if (pass >= s.target.size()) s.target.resize(static_cast<size_t>(pass) + 1u, int8_t {-2});
	if (s.target[pass] == -2) {
		const auto name = PassCensus::PassName(pass);
		int        hit  = -1;
		const auto& list = TargetList();
		for (size_t i = 0; i < list.size() && i < Passes; i++) {
			if (name.find(list[i]) != std::string::npos) hit = static_cast<int>(i);
		}
		s.target[pass] = static_cast<int8_t>(hit);
	}
	return s.target[pass];
}

void NoteIdle(uint64_t ns, int cause, bool commands) {
	auto& s = Exec();
	if (!s.idle_pending) {
		s.idle_pending = true;
		s.idle_ns      = 0;
		s.idle_cause   = commands ? 5 : cause;
	} else if (commands) {
		s.idle_cause = 5;
	}
	s.idle_ns += ns;
}

void NoteOperation(int pass, uint64_t sequence, size_t raw, size_t resolved, uint64_t emit_tsc) {
	auto& s = Exec();
	if (s.idle_pending) {
		const auto c = static_cast<size_t>(s.idle_cause);
		if (pass >= 0) {
			const size_t where = s.prev_pass == pass ? 0u : 1u;
			s.pass[static_cast<size_t>(pass)].idle_events[where][c]++;
			s.pass[static_cast<size_t>(pass)].idle_ns[where][c] += s.idle_ns;
		} else {
			s.other_events[c]++;
			s.other_ns[c] += s.idle_ns;
		}
		s.idle_pending = false;
	}
	if ((s.ops_seen++ & 15u) == 0) s.res_all[static_cast<size_t>(LeadBucket(resolved))]++;
	if (pass >= 0 && s.open_pass == pass && s.open_n < ChunkCap) {
		s.open_n++;
		s.last_seq = sequence;
	} else {
		CloseChunk(s);
		if (pass >= 0) {
			s.open_pass   = pass;
			s.open_n      = 1;
			s.open_seq    = sequence;
			s.last_seq    = sequence;
			s.open_raw    = raw;
			s.open_res    = resolved;
			const auto now_tsc = Tsc();
			s.open_age_us = emit_tsc != 0 && now_tsc > emit_tsc ? static_cast<double>(now_tsc - emit_tsc) / s.clock.Rate() / 1e3 : 0.0;
			s.open_cp     = CpStateIndex(g_cp_state.load(std::memory_order_relaxed));
			s.emitted_at_open[s.chunk_index % 4] = sequence + raw;
		}
	}
	s.prev_pass = pass;
	Report(s);
}

void NoteResolve(uint16_t pass, bool dispatch, uint64_t ticks, bool resolved) {
	static ResolveState s; // resolve thread
	if (pass >= s.pass.size()) s.pass.resize(static_cast<size_t>(pass) + 1u);
	auto& p = s.pass[pass];
	if (dispatch) {
		p.dispatch_ticks += ticks;
		p.dispatches++;
	} else {
		p.draw_ticks += ticks;
		p.draws++;
	}
	if (resolved) p.resolved++;
	for (uint32_t i = 0; i < RpCount; i++) p.parts[i] += t_parts[i];
	p.buffers += t_part_buffers;
	if ((++s.ops & 1023u) != 0) return;
	const auto now = std::chrono::steady_clock::now();
	if (now - s.window < std::chrono::seconds(5)) return;
	s.clock.Refresh(Tsc(), now);
	const double   rate   = s.clock.Rate(); // ticks per ns
	const uint32_t frame  = DebugPresentCount().load(std::memory_order_relaxed);
	const uint32_t frames = frame > s.window_frame ? frame - s.window_frame : 1u;
	const double   f      = frames;
	ResolveState::Pass total;
	std::vector<std::pair<uint64_t, uint16_t>> order;
	for (size_t i = 0; i < s.pass.size(); i++) {
		const auto& q = s.pass[i];
		total.draw_ticks += q.draw_ticks;
		total.draws += q.draws;
		total.dispatch_ticks += q.dispatch_ticks;
		total.dispatches += q.dispatches;
		total.resolved += q.resolved;
		if (q.draws + q.dispatches != 0) order.emplace_back(q.draw_ticks + q.dispatch_ticks, static_cast<uint16_t>(i));
	}
	std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
	std::string top;
	for (size_t i = 0; i < order.size() && i < 8; i++) {
		const auto& q   = s.pass[order[i].second];
		const auto  ops = q.draws + q.dispatches;
		char        buf[400];
		const auto  per_op = [&](uint64_t ticks) { return ops != 0 ? static_cast<double>(ticks) / rate / 1e3 / static_cast<double>(ops) : 0.0; };
		std::snprintf(buf, sizeof(buf),
		              " | \"%s\" ms/frame=%.2f ops/frame=%.0f us/op=%.2f parts_us[slot|inputs|key|mat|sources(clamp)|perm]=%.2f|%.2f|%.2f|%.2f|%.2f(%.2f)|%.2f buffers/op=%.1f",
		              PassCensus::PassName(order[i].second).c_str(), static_cast<double>(order[i].first) / rate / 1e6 / f, ops / f,
		              per_op(order[i].first), per_op(q.parts[RpSlot]), per_op(q.parts[RpInputs]), per_op(q.parts[RpKey]),
		              per_op(q.parts[RpMaterialize]), per_op(q.parts[RpSources]), per_op(q.parts[RpClamp]), per_op(q.parts[RpPermutation]),
		              ops != 0 ? static_cast<double>(q.buffers) / static_cast<double>(ops) : 0.0);
		top += buf;
	}
	const auto ops = total.draws + total.dispatches;
	std::fprintf(stderr,
	             "PERF resolve_work frames=%u work_ms/frame=%.2f draws/frame=%.0f us/draw=%.2f dispatches/frame=%.0f us/dispatch=%.2f resolved%%=%.1f%s\n",
	             frames, static_cast<double>(total.draw_ticks + total.dispatch_ticks) / rate / 1e6 / f, total.draws / f,
	             total.draws != 0 ? static_cast<double>(total.draw_ticks) / rate / 1e3 / static_cast<double>(total.draws) : 0.0,
	             total.dispatches / f,
	             total.dispatches != 0 ? static_cast<double>(total.dispatch_ticks) / rate / 1e3 / static_cast<double>(total.dispatches) : 0.0,
	             ops != 0 ? 100.0 * static_cast<double>(total.resolved) / static_cast<double>(ops) : 0.0, top.c_str());
	for (auto& q: s.pass) q = {};
	s.window       = now;
	s.window_frame = frame;
}

void NoteSnapshotTaken(bool compute, int kind) {
	g_snap_taken[compute ? 1u : 0u][static_cast<size_t>(kind)].fetch_add(1, std::memory_order_relaxed);
}

void NoteSnapshotWait(bool compute, int kind, uint64_t ns) {
	g_snap_waits[compute ? 1u : 0u][static_cast<size_t>(kind)].fetch_add(1, std::memory_order_relaxed);
	g_snap_wait_ns[compute ? 1u : 0u][static_cast<size_t>(kind)].fetch_add(ns, std::memory_order_relaxed);
}

} // namespace Libs::Graphics::ChunkLead
