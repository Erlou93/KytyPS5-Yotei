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

inline void PeekOperationBegin(bool member) {
	t_peek_member  = member ? 1 : 0;
	t_peek_markers = 0;
}

inline void PeekOperationEnd() {
	if (t_peek_member == 1) t_peek_marker_hist[t_peek_markers < 3u ? t_peek_markers : 3u]++;
	t_peek_member = -1;
}

} // namespace Libs::Graphics

#endif
