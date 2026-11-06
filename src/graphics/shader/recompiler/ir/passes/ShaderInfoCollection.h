#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERINFOCOLLECTION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERINFOCOLLECTION_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

// Collects the shader interface, resource usage and instruction features after resource tracking.
void CollectShaderInfo(Program& program, ShaderStageInputInfo input_info);

// KYTY_PERF_READONLY_GLC_LOADS (ReadOnlyGlcLoadsMode, ShaderIR.h): in a program that writes no
// device memory, the GLC/DLC loads outside loops lose device-scope coherence
// (MemoryInfo::coherent); =2 marks them readonly_glc_verify instead, so the emitter loads them both
// ways. Runs after resource tracking, before CollectShaderInfo. Returns the records changed.
uint32_t ApplyReadOnlyGlcLoads(Program& program, int mode);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERINFOCOLLECTION_H_ */
