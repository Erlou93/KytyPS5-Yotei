#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_CHUNKLEAD_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_CHUNKLEAD_H_

#include <atomic>
#include <cstddef>
#include <cstdint>

// KYTY_TRACE_CHUNK_LEAD=1 (project 15 N1, measurement only; design: the analysis session's n1-projekt-licznikow.md): how
// far ahead of the execution thread the work of a chunk is when the execution thread opens it.
// - PERF chunk_lead per pass of the project: the raw lead (operations emitted, not executed), the resolved lead (the
//   resolved queue), the age of the opening operation, how many chunks ahead were wholly emitted, the command processor's
//   state when the lead was small; weighted by the chunk's operations.
// - PERF chunk_idle: the execution thread's idle time attributed to the operation after it (inside a pass, its entry,
//   other), by cause (resolve behind, drain, parser working, game queues blocked, no submissions, commands served).
// - PERF resolve_work (resolve thread): its work per frame, per draw and dispatch, by pass.
// - PERF snapshot_rings: register snapshots taken and waited for, per kind (context, user config, shaders) and queue.
namespace Libs::Graphics::ChunkLead {

[[nodiscard]] bool On();

// The command processor thread's state (written only while On()).
enum CpState : uint8_t { CpWorking = 0, CpSnapshot = 1, CpDrain = 2, CpRingFull = 8, CpBlocked = 9, CpNoWork = 10 };
inline std::atomic<uint8_t> g_cp_state {CpWorking};
// The resolve thread holds an operation (written only while On()).
inline std::atomic<bool> g_resolving {false};

// Execution thread, every operation: its project pass (0..2, -1 not a chunk member), sequence, the leads and its emit tsc.
void NoteOperation(int pass, uint64_t sequence, size_t raw, size_t resolved, uint64_t emit_tsc);
// Execution thread, after an idle wait: its length, the cause sampled when it began (0 resolve, 1 drain, 2 parser,
// 3 blocked, 4 no work) and whether game-thread commands were served during it.
void NoteIdle(uint64_t ns, int cause, bool commands);
// Execution thread: which project pass (0..2) a marker pass is, -1 none.
[[nodiscard]] int TargetPass(uint16_t pass);

// Resolve thread, every draw or dispatch it resolved: its marker pass, whether a dispatch, the tsc ticks, whether resolved.
void NoteResolve(uint16_t pass, bool dispatch, uint64_t ticks, bool resolved);

// Command processor thread: a snapshot slot taken / waited for (kind 0 context, 1 user config, 2 shaders).
void NoteSnapshotTaken(bool compute, int kind);
void NoteSnapshotWait(bool compute, int kind, uint64_t ns);

[[nodiscard]] uint64_t Tsc();

// N2.0 (resolve parts, resolve thread): ticks per part of one resolution, reset per operation, summed per pass by
// NoteResolve. Slot: the result slot and the draw state's reset; inputs: GraphicsProgramInputs; key: the program key and
// its lookup; materialize: MaterializeResourcesAhead; sources: StageF::BuildResolveRecord, clamp: its
// ClampRangeSizeAhead calls (the game memory manager's lock); permutation: FindPermutationAhead.
enum ResolvePart : uint32_t { RpSlot, RpInputs, RpKey, RpMaterialize, RpSources, RpClamp, RpPermutation, RpCount };
inline thread_local uint64_t t_parts[RpCount] {};
inline thread_local uint32_t t_part_buffers = 0;
class PartTimer {
public:
	explicit PartTimer(ResolvePart part): m_part(part), m_t0(On() ? Tsc() : 0) {}
	~PartTimer() {
		if (m_t0 != 0) t_parts[m_part] += Tsc() - m_t0;
	}
	PartTimer(const PartTimer&)            = delete;
	PartTimer& operator=(const PartTimer&) = delete;

private:
	ResolvePart m_part;
	uint64_t    m_t0;
};

} // namespace Libs::Graphics::ChunkLead

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_CHUNKLEAD_H_
