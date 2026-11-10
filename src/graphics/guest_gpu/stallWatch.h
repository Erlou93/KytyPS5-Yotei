#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_STALLWATCH_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_STALLWATCH_H_

// KYTY_DIAG_STALL_WATCH=1 (diagnostic only, hitch project 6.11): see stallWatch.cpp.
#include <atomic>
#include <cstdint>

namespace Libs::Graphics::StallWatch {

void StartIfEnabled();

// KYTY_DIAG_STALL_WATCH_IDLE_MS=<ms> (hitch study 10.11): when the execution thread has waited for an operation this
// long, the watchdog captures every thread (STALL_STACK, stalled_ms = the idle time): what the game does meanwhile.
// The execution thread stores the steady-clock ns when its wait begins, 0 when it ends.
inline std::atomic<int64_t> g_exec_idle_since {0};

} // namespace Libs::Graphics::StallWatch

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_STALLWATCH_H_
