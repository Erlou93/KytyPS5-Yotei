#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_STALLWATCH_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_STALLWATCH_H_

// KYTY_DIAG_STALL_WATCH=1 (diagnostic only, hitch project 6.11): see stallWatch.cpp.
namespace Libs::Graphics::StallWatch {

void StartIfEnabled();

} // namespace Libs::Graphics::StallWatch

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_STALLWATCH_H_
