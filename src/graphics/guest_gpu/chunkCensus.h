#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_CHUNKCENSUS_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_CHUNKCENSUS_H_

#include <cstdint>

// Parallel-translation project (AGENTS 15), step S1a - diagnostic only, default off, removed in S6.
// KYTY_TRACE_CHUNK_CENSUS=1 (needs KYTY_TRACE_PASS_CENSUS=1 for the pass ids): on the execution thread, the operations of
// the graphics queue inside the project's passes (KYTY_TRACE_CHUNK_PASSES=<substring,...>, default the three ceiling passes)
// are grouped as the parallel translation would group them: contiguous draws/dispatches of one pass set, at most 128, closed
// by anything that must stay in order. CHUNK_CENSUS lines every ~5 s: chunks per frame, their sizes, the share of the passes'
// execution time they cover, why chunks closed and why single operations could not join one.
namespace Libs::Graphics::ChunkCensus {

struct OpInfo {
	bool     gfx_queue = false;
	uint16_t pass      = 0;
	int      kind      = 0; // 1 draw, 2 dispatch, 3 indirect draw, 0 callback / state callback
	uint64_t ns        = 0; // execution time less card waits
	// Events inside the operation's execution that would make it a hole (counter deltas).
	uint64_t gpu_waits = 0, pipelines = 0, shaders = 0, download_bytes = 0, image_creates = 0;
};

// Hole events on the calling thread (the execution thread reads the deltas around each operation).
inline thread_local uint64_t t_downloads = 0, t_image_creates = 0, t_pipelines = 0, t_shaders = 0;

[[nodiscard]] bool Enabled();
// Execution thread, after each operation.
void Note(const OpInfo& op);

// Whether an operation is a chunk member: a non-indirect draw/dispatch of the graphics queue inside the project's passes
// (KYTY_TRACE_CHUNK_PASSES, default the three passes). Execution thread.
[[nodiscard]] bool IsMember(bool gfx_queue, uint16_t pass, int kind);

// KYTY_CHECK_CHUNKS=2 (project 15 S1.2, the oracle): every chunk-member draw is prepared twice and the two plans' fingerprints
// are compared field by field (renderDraw.cpp ChunkCheckDraw); PERF chunk_check lines. Read once.
[[nodiscard]] bool CheckOn();
// The operation the execution thread runs now is a chunk member (set before each operation while CheckOn()).
inline thread_local bool t_member = false;

} // namespace Libs::Graphics::ChunkCensus

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_CHUNKCENSUS_H_
