#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_INDIRECTLOADFAST_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_INDIRECTLOADFAST_H_

#include <cstdint>

namespace Libs::Graphics {

class Buffer;
class CommandScheduler;

// KYTY_PERF_INDIRECT_LOAD_FAST (default 1; 0 off, 2 verify, 3 live) with KYTY_PERF_INDIRECT_LOAD_FAST_HASHES=<hex,...>|all
// (default: the five "Deferred Ambient>Eval non-RT Probes" compute shaders): a BUFFER_LOAD_FORMAT_D16_XYZW through a
// GPU-selected V# (the per-shape function, an OpSwitch over all 78 buffer formats,
// ~260 ALU and two page-table lookups per call) first tests the V# for the one layout these shaders read,
// k16_16_16_16Float with DST_SEL X,Y,Z,W; then it reads the element's two dwords inline (the same bounds check, one
// page-table lookup when both dwords are in one 16 KiB page) and returns them with only the signaling-NaN quieting
// that the half -> float -> half round trip of the generic path applies. Any other V# calls the function as before.
// =2 also calls the function on the fast path and returns its result, counting checks / mismatches / other V#s in
// fault-buffer diag words DiagWord + 0/1/2 (PERF indirect_load_fast verify line).
// =3 compiles both paths and takes the fast one while bit 0 of the live shader flags is set (SpirvLive below): by
// default always; KYTY_LIVE_AB=KYTY_PERF_INDIRECT_LOAD_FAST toggles it per phase.
namespace IndirectLoadFast {

// After the watchdog's miss slots (BufferCache::FAULT_DIAG_MISS_FIRST + 4 * FAULT_DIAG_MISS_SLOTS = 20544; words 12.. are
// KYTY_FIX_CLAMP_NAN's, 12-13 free since KYTY_PERF_TABLE_BUFFERS left); checked in faultManager.cpp.
inline constexpr uint32_t DiagWord = 20544;

[[nodiscard]] int  Mode();
[[nodiscard]] bool Listed(uint64_t shader_hash);

// KYTY_DIAG_INDIRECT_FORMAT_CENSUS=1 (diagnostic, default off; changes translated shader code): every formatted load
// through a GPU-selected V# adds 1 per wave (its elected lane's V#) to diag word CensusWord + 4 * FORMAT + channels - 1;
// faultManager.cpp prints the totals (PERF indirect_format_census).
inline constexpr uint32_t CensusWord  = 20608;
inline constexpr uint32_t CensusWords = 512;
[[nodiscard]] bool CensusEnabled();
[[nodiscard]] bool CensusFor(uint64_t shader_hash); // KYTY_DIAG_INDIRECT_FORMAT_CENSUS_HASHES (default all)

} // namespace IndirectLoadFast

// KYTY_PERF_WIDE_BUFFER_LOAD (test switch, default 0 = today; 1 on, 2 check, 3 live) with
// KYTY_PERF_WIDE_BUFFER_LOAD_HASHES=<hex,...>|all (default: c2a548aaae05d492, the skinning and cloth compute shader of
// the characters, 3.2 ms of GPU time per frame in the town walk at 23.6 dispatches of one 1024-thread workgroup): a raw
// BUFFER_LOAD_DWORDX4 of a direct buffer in a compute shader is today four dword reads, each with its own address, wrap
// and bounds selects (~17 SPIR-V instructions per dword; Volatile for GLC, so the driver keeps four separate reads).
// With the switch it reads its 16 bytes as one uvec4 element of a buffers_u128 alias of the same binding when its first
// dword's byte address is a multiple of 16 and all four dwords are in the binding; any other X4 load takes the four
// dword reads as before (spirvEmitterMemory.cpp, LoadBufferWide). The skinning loops of c2a548aa read the bone
// matrices this way: 12 X4 loads per four influences, 48 -> 12 reads per lane.
// =2 reads both ways and returns today's values, counting in fault-buffer diag words DiagWord + 0/1/2 the X4 loads that
// would take the wide read and those whose element indices / values differ from today's (PERF wide_buffer_load verify;
// indices must stay 0; values can differ only where another invocation wrote the bytes between the two reads).
// =3 compiles both paths and takes the wide one while bit 1 of the live shader flags is set (SpirvLive below): by
// default always; KYTY_LIVE_AB=KYTY_PERF_WIDE_BUFFER_LOAD toggles it per phase. =2 and =3 need the fault buffer in the
// shader (every shader that loads through BDA has it, c2a548aa does); without it a listed shader keeps today's reads.
namespace WideBufferLoad {

// After IndirectLoadFast's words (checked in faultManager.cpp).
inline constexpr uint32_t DiagWord = IndirectLoadFast::DiagWord + 8u;

[[nodiscard]] int  Mode();
[[nodiscard]] bool Listed(uint64_t shader_hash);

} // namespace WideBufferLoad

// Live shader flags: one word of the fault buffer, after the diagnostic tail (BufferCache::FAULT_DIAG_BASE +
// FAULT_DIAG_WORDS; the fault buffer has LiveWordsBytes more for it), which shaders compiled with a switch's live mode
// read before taking the switch's path. The GPU thread rewrites it at frame boundaries when a live A/B phase changes
// it (vkCmdUpdateBuffer between full barriers), so either value runs whole frames. Both paths of such a switch give
// the same results, so a dispatch recorded before the change that runs after it is still exact.
namespace SpirvLive {

inline constexpr uint32_t LiveWordsBytes      = 256;
inline constexpr uint32_t FlagIndirectLoadFast = 1u << 0u;
inline constexpr uint32_t FlagWideBufferLoad   = 1u << 1u;

// Whether any switch is in its live mode (the fault buffer then has the flags word and Update writes it).
[[nodiscard]] bool Enabled();
// The flags for the current live A/B phase.
[[nodiscard]] uint32_t Flags();

// GPU thread, at a frame boundary: writes Flags() into `fault_buffer` when it differs from the last written value.
void Update(CommandScheduler& scheduler, Buffer& fault_buffer, uint32_t word_index);

} // namespace SpirvLive

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_INDIRECTLOADFAST_H_ */
