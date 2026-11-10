#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWCOVERAGE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWCOVERAGE_H_

#include <cstdint>

// Project 15 way B W0 (KYTY_TRACE_PEEK_CENSUS, measurement only): per-thread counts of in-order work a draw did, read by
// the execution thread before and after each draw (Table B of the coverage: whether a helper could write the draw's block).
// image barriers recorded (a layout or access change), render passes begun (the targets changed), scalar-read overlap scans.
namespace Libs::Graphics::DrawCoverage {
inline thread_local uint64_t t_barriers = 0, t_render_begins = 0, t_scans = 0;
} // namespace Libs::Graphics::DrawCoverage

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWCOVERAGE_H_
