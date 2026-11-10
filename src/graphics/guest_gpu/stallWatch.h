#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_STALLWATCH_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_STALLWATCH_H_

// KYTY_DIAG_STALL_WATCH=1 (diagnostic only, hitch project 6.11): see stallWatch.cpp.
#include <atomic>
#include <chrono>
#include <cstdint>

namespace Libs::Graphics::StallWatch {

void StartIfEnabled();

// KYTY_DIAG_STALL_WATCH_IDLE_MS=<ms> (hitch study 10.11): when the execution thread has waited for an operation this
// long, the watchdog captures every thread (STALL_STACK, stalled_ms = the idle time): what the game does meanwhile.
// The execution thread stores the steady-clock ns when its wait begins, 0 when it ends.
inline std::atomic<int64_t> g_exec_idle_since {0};
// KYTY_DIAG_STALL_WATCH_DRIVER_MS=<ms> (hitch study 10.11): the same capture while the execution thread is inside a Vulkan
// create/allocate call (vkCreateImageView, vkAllocateMemory) this long: who else is in the driver.
inline std::atomic<int64_t> g_exec_driver_since {0};

// RAII marker for the execution thread's driver calls.
struct DriverCallMark {
	bool on;
	explicit DriverCallMark(bool exec_thread): on(exec_thread) {
		if (on) g_exec_driver_since.store(static_cast<int64_t>(std::chrono::steady_clock::now().time_since_epoch().count()), std::memory_order_release);
	}
	~DriverCallMark() {
		if (on) g_exec_driver_since.store(0, std::memory_order_release);
	}
};

} // namespace Libs::Graphics::StallWatch

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_STALLWATCH_H_
