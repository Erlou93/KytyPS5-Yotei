#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWCOVERAGE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWCOVERAGE_H_

#include <cstdint>
#include <cstdlib>

// KYTY_TRACE_TARGET_REPEAT=1 (diagnostic, 11.11): image layout transitions (Image::Transit) timed per phase of a draw -
// 0 elsewhere, 1 the targets (AcquireRenderTargets), 2 the bindings' commit (CommitBindings) - calls, ns, barriers.
namespace Libs::Graphics::TargetTrace {
inline bool On() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_TRACE_TARGET_REPEAT");
		return value != nullptr && value[0] == '1';
	}();
	return on;
}
inline thread_local int      t_phase = 0;
inline thread_local uint64_t t_calls[3] {}, t_ns[3] {}, t_barriers[3] {};
} // namespace Libs::Graphics::TargetTrace

// Project 15 way B W0 (KYTY_TRACE_PEEK_CENSUS, measurement only): per-thread counts of in-order work a draw did, read by
// the execution thread before and after each draw (Table B of the coverage: whether a helper could write the draw's block).
// image barriers recorded (a layout or access change), render passes begun (the targets changed), scalar-read overlap scans.
namespace Libs::Graphics::DrawCoverage {
inline thread_local uint64_t t_barriers = 0, t_render_begins = 0, t_scans = 0;
} // namespace Libs::Graphics::DrawCoverage

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWCOVERAGE_H_
