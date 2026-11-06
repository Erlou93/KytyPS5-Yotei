#include "graphics/guest_gpu/chunkCensus.h"

#include "graphics/guest_gpu/command_processor/passCensus.h"
#include "graphics/host_gpu/renderer/debugSample.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace Libs::Graphics::ChunkCensus {

namespace {

constexpr uint32_t ChunkCap = 128;

enum Close : uint32_t { CloseCap, CloseOtherPass, CloseOtherQueue, CloseCallback, CloseIndirect, CloseHole, CloseCount };
const char* const CloseNames[CloseCount] = {"cap", "other_pass", "other_queue", "callback", "indirect", "hole"};
enum Hole : uint32_t { HoleWait, HolePipeline, HoleShader, HoleDownload, HoleImage, HoleCount };
const char* const HoleNames[HoleCount] = {"gpu_wait", "pipeline", "shader", "download", "image_create"};

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

// Execution thread only.
struct State {
	std::vector<int8_t> target; // per pass id: -1 unknown, 0 no, 1 yes
	uint32_t            open       = 0; // operations in the open chunk
	uint64_t            open_ns    = 0;
	uint64_t            chunks     = 0, chunked_ops = 0, chunked_ns = 0;
	uint64_t            target_ops = 0, target_ns = 0, refused_ops = 0, refused_ns = 0;
	std::array<uint64_t, 5>          sizes {}; // 1, 2-15, 16-63, 64-127, 128
	std::array<uint64_t, CloseCount> closes {};
	std::array<uint64_t, HoleCount>  holes {};
	std::chrono::steady_clock::time_point window_start = std::chrono::steady_clock::now();
	uint32_t                              window_frame = 0;
};

State& St() {
	static State s;
	return s;
}

bool IsTarget(State& s, uint16_t pass) {
	if (pass >= s.target.size()) s.target.resize(static_cast<size_t>(pass) + 1u, int8_t {-1});
	if (s.target[pass] < 0) {
		const auto name = PassCensus::PassName(pass);
		bool       hit  = false;
		for (const auto& t: TargetList()) {
			if (name.find(t) != std::string::npos) hit = true;
		}
		s.target[pass] = hit ? 1 : 0;
	}
	return s.target[pass] == 1;
}

void CloseChunk(State& s, Close why) {
	if (s.open == 0) return;
	s.chunks++;
	s.chunked_ops += s.open;
	s.chunked_ns += s.open_ns;
	const size_t bucket = s.open == 1 ? 0 : s.open < 16 ? 1 : s.open < 64 ? 2 : s.open < ChunkCap ? 3 : 4;
	s.sizes[bucket]++;
	s.closes[why]++;
	s.open    = 0;
	s.open_ns = 0;
}

void Report(State& s) {
	const auto now = std::chrono::steady_clock::now();
	if (now - s.window_start < std::chrono::seconds(5)) return;
	const uint32_t frame  = DebugPresentCount().load(std::memory_order_relaxed);
	const uint32_t frames = frame > s.window_frame ? frame - s.window_frame : 1u;
	const double   f      = frames;
	std::fprintf(stderr,
	             "CHUNK_CENSUS frames=%u chunks/frame=%.0f ops/chunk=%.1f target_ops/frame=%.0f chunked_ops/frame=%.0f "
	             "target_ms/frame=%.2f chunked_ms/frame=%.2f coverage=%.1f%% refused_ops/frame=%.0f refused_ms/frame=%.2f "
	             "sizes(1/2-15/16-63/64-127/128)=%llu/%llu/%llu/%llu/%llu\n",
	             frames, s.chunks / f, s.chunks != 0 ? static_cast<double>(s.chunked_ops) / s.chunks : 0.0, s.target_ops / f,
	             s.chunked_ops / f, s.target_ns / 1e6 / f, s.chunked_ns / 1e6 / f,
	             s.target_ns != 0 ? 100.0 * s.chunked_ns / s.target_ns : 0.0, s.refused_ops / f, s.refused_ns / 1e6 / f,
	             static_cast<unsigned long long>(s.sizes[0]), static_cast<unsigned long long>(s.sizes[1]),
	             static_cast<unsigned long long>(s.sizes[2]), static_cast<unsigned long long>(s.sizes[3]),
	             static_cast<unsigned long long>(s.sizes[4]));
	std::string closes;
	for (uint32_t i = 0; i < CloseCount; i++) {
		closes += std::string(i == 0 ? "" : " ") + CloseNames[i] + "=" + std::to_string(s.closes[i]);
	}
	std::string holes;
	for (uint32_t i = 0; i < HoleCount; i++) {
		holes += std::string(i == 0 ? "" : " ") + HoleNames[i] + "=" + std::to_string(s.holes[i]);
	}
	std::fprintf(stderr, "CHUNK_CENSUS closes[%s] refused_by[%s]\n", closes.c_str(), holes.c_str());
	const auto target = std::move(s.target);
	s                 = State {};
	s.target          = target;
	s.window_start    = now;
	s.window_frame    = frame;
}

} // namespace

bool Enabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_TRACE_CHUNK_CENSUS");
		const bool  o     = value != nullptr && value[0] == '1' && PassCensus::Enabled();
		if (o) std::fprintf(stderr, "TRACE_CHUNK_CENSUS enabled\n");
		return o;
	}();
	return on;
}

void Note(const OpInfo& op) {
	auto&      s      = St();
	const bool target = op.gfx_queue && IsTarget(s, op.pass);
	if (!op.gfx_queue) {
		// Other queues' operations are ordered between the chunk's operations: the chunk ends.
		CloseChunk(s, CloseOtherQueue);
	} else if (!target) {
		CloseChunk(s, op.kind == 0 ? CloseCallback : CloseOtherPass);
	} else if (op.kind == 0) {
		CloseChunk(s, CloseCallback);
	} else {
		s.target_ops++;
		s.target_ns += op.ns;
		int hole = -1;
		if (op.gpu_waits != 0) hole = HoleWait;
		else if (op.pipelines != 0) hole = HolePipeline;
		else if (op.shaders != 0) hole = HoleShader;
		else if (op.download_bytes != 0) hole = HoleDownload;
		else if (op.image_creates != 0) hole = HoleImage;
		if (op.kind == 3) {
			CloseChunk(s, CloseIndirect);
			s.refused_ops++;
			s.refused_ns += op.ns;
		} else if (hole >= 0) {
			CloseChunk(s, CloseHole);
			s.holes[static_cast<uint32_t>(hole)]++;
			s.refused_ops++;
			s.refused_ns += op.ns;
		} else {
			if (s.open == ChunkCap) CloseChunk(s, CloseCap);
			s.open++;
			s.open_ns += op.ns;
		}
	}
	Report(s);
}

} // namespace Libs::Graphics::ChunkCensus
