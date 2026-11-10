#ifndef GRAPHICS_HOST_GPU_TRANSLATOR_GUARD_H_
#define GRAPHICS_HOST_GPU_TRANSLATOR_GUARD_H_

#include <array>
#include <cstdint>

namespace Libs::Graphics {

// Project 15 S2a translator guard (KYTY_CHECK_PEEK=2, diagnostic): the depth of a read-only peek on this thread and the
// writes noted inside one (expected 0; PERF peek prints them). Sites: 0 buffer registry, 1 touch, 2 upload, 3 stream
// copy, 4 write note, 5 texture-cache write section.
inline thread_local int                     t_translator = 0;
inline thread_local std::array<uint64_t, 8> t_translator_writes {};

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

inline void PeekOperationBegin(bool member) {
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
