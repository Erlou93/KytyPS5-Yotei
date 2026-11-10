#ifndef GRAPHICS_HOST_GPU_TRANSLATOR_GUARD_H_
#define GRAPHICS_HOST_GPU_TRANSLATOR_GUARD_H_

#include <array>
#include <atomic>
#include <cstdint>

namespace Libs::Graphics {

// Project 15 S2a translator guard (KYTY_CHECK_PEEK=2, diagnostic): the depth of a read-only peek on this thread and the
// writes noted inside one (expected 0; PERF peek prints them). Sites: 0 buffer registry, 1 touch, 2 upload, 3 stream
// copy, 4 write note, 5 texture-cache write section.
inline thread_local int                     t_translator = 0;
inline thread_local std::array<uint64_t, 8> t_translator_writes {};
// N5.1: the writes resolve helpers' peeks noted (their own thread-local counts gathered after each plan; expected 0).
inline std::array<std::atomic<uint64_t>, 8> g_helper_translator_writes {};

inline void TranslatorNoteSite(int site) {
	t_translator_writes[static_cast<size_t>(site) & 7u]++;
}

// The peek's split by operation (KYTY_CHECK_PEEK, graphicsRun.cpp): whether the running operation is a chunk-member draw
// (-1 outside, 0 another operation, 1 a member draw), its in-order peeks, and member draws by in-order peeks (0, 1, 2, 3+).
inline thread_local int                     t_peek_member  = -1;
inline thread_local uint32_t                t_peek_markers = 0;
inline thread_local std::array<uint64_t, 4> t_peek_marker_hist {};

// The operation's plan (n3-s2b-przeglad.md section 4): its programs predicted (S2d) and its pipeline found (S2c, not mesh);
// member draws with a plan, by in-order markers (buffers in order + image stages without a full answer).
inline thread_local bool                    t_op_programs = false, t_op_pipeline = false;
inline thread_local uint64_t                t_member_draws = 0, t_plan_draws = 0;
inline thread_local std::array<uint64_t, 4> t_plan_marker_hist {};
// Project 15 S2b (n3-s2b-one-byte.md): the stencil tail of today's texture resolve ran (a stencil plane reader or an
// association under the address), set by SyncStencilPlaneReader and FindResolvedTexture, read by the image replay.
inline thread_local bool t_stencil_tail = false;
// Slow paths of today's image binding (S2b: a peeked-ready slot that took one went stale on the way).
inline thread_local uint64_t t_peek_slow = 0;

// Project 15 N4.1 (KYTY_CHECK_PEEK=2): the peeks a chunk's opening makes for the draws ahead, and their checks when each
// draw runs (execution thread). Distance buckets from the opening draw: 0 | 1-15 | 16-63 | 64-127.
struct AheadStats {
	uint64_t                opened = 0, planned = 0, compile = 0;
	std::array<uint64_t, 8> gated {}; // 0 unused, 1 slot, 2 shaders, 3 not adoptable, 4 counts/indirect, 5 color op/depth copy,
	                                  // 6 topology/index type, 7 mesh or no vertex program
	std::array<uint64_t, 4> peeked {}, front {};
	uint64_t p_compared = 0, p_bad = 0, p_share_diff = 0, p_lost = 0, p_injected = 0, p_detected = 0;
	uint64_t k_compared = 0, k_bad = 0, k_unchecked = 0, k_find_diff = 0, k_injected = 0, k_detected = 0;
	uint64_t i_stages = 0, i_compared = 0, i_bad = 0, i_moved = 0, i_stale = 0, i_lost = 0, i_injected = 0, i_detected = 0;
	std::array<uint64_t, 4> p_by_distance {}, i_moved_by_distance {}, i_compared_by_distance {};
	// n4-2-doprecyzowanie.md 2: the image registry moved / today's path took a slow path for an answered slot, whether or
	// not the answer differed
	std::array<uint64_t, 4> i_moved_any {}, i_stale_any {};
	// N4.2a (KYTY_CHECK_CHUNKS=2): the oracle's first preparation from the plan
	uint64_t plan_draws = 0, no_plan = 0, rejected = 0, pipeline_bad = 0, set_hint = 0, set_hint_missed = 0;
	// N4.3/N5: draws committed from a plan; the pipeline taken from it; buffer acquisitions it
	// answered (touch only) and the ones that went today's way (no entry, another class, the proof failed)
	uint64_t used_draws = 0, used_pipeline = 0, buffers_answered = 0, buffers_today = 0;
	uint64_t describe_failed = 0; // N5.1: the soft description or key failed (no plan)
	uint64_t lookup_busy     = 0; // N5.1: a helper's buffer peeks found the lookup lock busy (no entries)
};
inline AheadStats g_ahead;
inline thread_local AheadStats t_ahead_helper; // N5.3: a resolve helper's plans count here (not into the execution thread's)
// N7.1: a helper's plan outcomes, published from t_ahead_helper every 256 plans for the report (one row per helper, each
// written by its own helper only: relaxed stores). 0 planned, 1 made, 2 compile, 3 describe_failed, 4 lookup_busy,
// 5-11 gated[1..7].
struct HelperPlanSnap {
	static constexpr size_t              Count = 12;
	std::array<std::atomic<uint64_t>, 12> v {};
	void Publish(const AheadStats& a, uint64_t made) {
		const uint64_t values[Count] = {a.planned,  made,        a.compile,   a.describe_failed, a.lookup_busy, a.gated[1],
		                                a.gated[2], a.gated[3], a.gated[4], a.gated[5],        a.gated[6],    a.gated[7]};
		for (size_t i = 0; i < Count; i++) v[i].store(values[i], std::memory_order_relaxed);
	}
};
inline HelperPlanSnap g_helper_plan_snap[2];
inline thread_local uint64_t t_helper_plans_made = 0;
inline uint32_t   AheadBucket(uint32_t distance) {
    return distance == 0u ? 0u : distance < 16u ? 1u : distance < 64u ? 2u : 3u;
}
// N5.1: the plan's builder describes targets and builds the pipeline key softly: where today's path would stop the emulator
// (EXIT), the description fails instead (no plan; today's path meets the same registers when the draw runs).
inline thread_local bool t_describe_soft = false, t_describe_failed = false;
// N5.0: the plan's builder without the data only the checks use (no slot-by-slot image peek, no record-keyed ranges).
inline thread_local bool t_peek_slim = false;
// The operations the execution thread began (S2a's lag check measures its age in them).
inline thread_local uint64_t t_op_index = 0;

inline void PeekOperationBegin(bool member) {
	t_op_index++;
	t_peek_member  = member ? 1 : 0;
	t_peek_markers = 0;
	t_op_programs  = false;
	t_op_pipeline  = false;
}

inline void PeekOperationEnd() {
	if (t_peek_member == 1) {
		const uint32_t bucket = t_peek_markers < 3u ? t_peek_markers : 3u;
		t_peek_marker_hist[bucket]++;
		t_member_draws++;
		if (t_op_programs && t_op_pipeline) {
			t_plan_draws++;
			t_plan_marker_hist[bucket]++;
		}
	}
	t_peek_member = -1;
}

} // namespace Libs::Graphics

#endif
