#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_CHUNKLEAD_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_CHUNKLEAD_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

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
// How many resolving threads hold an operation (written only while On(); project 15 N2.2: the helper too).
inline std::atomic<int> g_resolving {0};

// Execution thread, every operation: its project pass (0..2, -1 not a chunk member), sequence, the leads and its emit tsc.
void NoteOperation(int pass, uint64_t sequence, size_t raw, size_t resolved, uint64_t emit_tsc);
// Execution thread, after an idle wait: its length, the cause sampled when it began (0 resolve, 1 drain, 2 parser,
// 3 blocked, 4 no work) and whether game-thread commands were served during it.
void NoteIdle(uint64_t ns, int cause, bool commands);
// Execution thread: which project pass (0..2) a marker pass is, -1 none.
[[nodiscard]] int TargetPass(uint16_t pass);

// Resolve thread, every draw or dispatch it resolved: its marker pass, whether a dispatch, the tsc ticks, whether resolved.
void NoteResolve(uint16_t pass, bool dispatch, uint64_t ticks, bool resolved);

// Execution thread (N2.0b): one check of a resolution's reads at its adoption (ResolvedReadsHold), for the operation
// NoteOperation saw last: its ticks, log entries and bytes.
void NoteAdopt(uint64_t ticks, size_t entries, size_t bytes);

// Resolve thread start: the N2.0b materialization parts (ResourceMaterialization.h g_mat_trace) and, with
// KYTY_TRACE_PERF=1 KYTY_TRACE_PERF_SAMPLE=resolve, the sampler of the resolve thread (PERF_SAMPLE_RESOLVE lines).
void ResolveThreadStart(bool sampler, const char* name);
// Resolve thread, before each operation: its materialization counters reset.
void ResolveOperationStart();

// Project 15 N3.0 (KYTY_TRACE_PEEK_CENSUS=1, measurement only; design: the analysis session's n3-projekt.md section 2.2):
// the commit floor - per part, the tsc ticks of chunk-member draws (the three passes, the graphics queue): the whole
// operation, ExecutePreparedDraw, CommitPreparedDraw, the render targets (PrepareDrawRenderState), the adoption's read
// check, the bindless tables, PrepareBda, the pipeline lookup. Parts nest (the commit is inside ExecutePreparedDraw);
// PERF commit_floor prints each part's mean per member draw. Execution thread.
[[nodiscard]] bool     PeekCensusOn();
[[nodiscard]] uint64_t Tsc();
// Way B W0 (10.11): A2 FpTake (TakeResolvedState, contains FpAdopt), A5 FpStageBind (PrepareBindings, contains FpBindless and
// FpUpload), A7 FpUpload, A8 FpDrawBind (PrepareDrawBindings, contains FpBda and FpImages), A10 FpImages, A12 FpAcquire,
// A13 FpScanTransit; a dispatch's phases D1-D7 (FloorPhase) and D5w FpWritten (ObtainBuffer for a written range).
enum FloorPart : uint32_t {
	FpTotal, FpExecute, FpCommit, FpTargets, FpAdopt, FpBindless, FpBda, FpPipeline, FpBuild,
	FpTake, FpStageBind, FpUpload, FpDrawBind, FpImages, FpAcquire, FpScanTransit,
	FpD1, FpD2, FpD3, FpD4, FpD5, FpD6, FpD7, FpWritten,
	// W3 P0 (10.11): the draw-buffer pool - FindBuffers, the record-keyed V# ranges, the SRT words, the buffer loop of
	// RebindBuffers, vertex/index/argument acquisitions, the plan-answer branch of ObtainBuffer (time; t_floor_answers count)
	FpFindBuffers, FpVRanges, FpSrtWords, FpBufLoop, FpVtxIdx, FpPlanAnswer,
	FpCount
};
inline thread_local uint64_t t_floor[FpCount] {};
inline thread_local bool     t_floor_on = false; // set per operation: the census is on and the operation a member draw
class FloorTimer {
public:
	explicit FloorTimer(FloorPart part): m_part(part), m_t0(t_floor_on ? Tsc() : 0) {}
	~FloorTimer() {
		if (m_t0 != 0) t_floor[m_part] += Tsc() - m_t0;
	}
	FloorTimer(const FloorTimer&)            = delete;
	FloorTimer& operator=(const FloorTimer&) = delete;

private:
	FloorPart m_part;
	uint64_t  m_t0;
};
// Way B W0: consecutive phases of one function; the time since the last mark goes to the part being left.
class FloorPhase {
public:
	explicit FloorPhase(FloorPart first): m_part(first), m_last(t_floor_on ? Tsc() : 0) {}
	void Next(FloorPart part) {
		if (m_last == 0) return;
		const uint64_t now = Tsc();
		t_floor[m_part] += now - m_last;
		m_last = now;
		m_part = part;
	}
	~FloorPhase() {
		if (m_last != 0) t_floor[m_part] += Tsc() - m_last;
	}
	FloorPhase(const FloorPhase&)            = delete;
	FloorPhase& operator=(const FloorPhase&) = delete;

private:
	FloorPart m_part;
	uint64_t  m_last;
};
// Way B W0 Table B: a draw's first unmet condition (0 none; 1 no plan, 2 pipeline not from it, 3 a buffer acquisition went
// today's way, 4 an image set or slot was not ready, 5 an image barrier, 6 a render pass begun, 7 other in-order work -
// buffer work or an image created, 8 a scalar-read overlap scan), per draw bucket, with the runs of draws meeting all.
void NoteCover(int bucket, int reason, int reason_without_buffers);
// Way B W0: the draw bucket of the running operation (-1: none or the census off) and a read acquisition with a plan:
// the entry's kind (0 buffer, 1 range, 2 vertex, 3 index, 4 args, 5 count, 6 no entry) and the reason (0 answered).
inline thread_local int t_floor_bucket = -1;
[[nodiscard]] inline bool B3On() {
	static const bool on = std::getenv("KYTY_TRACE_W0_B3") != nullptr; // (W3 P0: the B3 split costs per acquisition)
	return on;
}
void NoteB3(uint32_t kind, int reason);

// After an operation (bucket 0..2 the three passes' draws, then the buckets below): its parts summed per bucket;
// PERF commit_floor every ~5 s, and PERF commit_floor_frame (ms per frame of every bucket and part).
// t_floor_wait: the card waits (ns, masterSemaphore) inside the operation, set before NoteFloor.
constexpr int BucketOtherDraws = 3, BucketDispatches = 4, BucketOther = 5, Buckets = 6;
inline thread_local uint64_t t_floor_wait = 0;
inline thread_local uint64_t t_floor_answers = 0; // W3 P0: plan answers in the operation
// The execution thread between operations (same switch, way B measurement 10.11): idle with nothing resolved (cause 0:
// operations queued - the resolve dispatcher behind; 1: none - the command processor has produced nothing), the commands
// it serves while idle (inside the idle time), the claim of a slot from a helper, and the card waits by marker pass.
void NoteExecIdle(int cause, uint64_t ticks);
inline thread_local uint64_t t_idle_commands = 0; // tsc ticks, added by WaitForOperations around RunCommands
void NoteExecClaim(uint64_t ticks);
void NoteExecCommands(uint64_t ticks, uint64_t wait_ns); // the game's commands served between operations
void NoteCardWait(uint16_t pass_id, uint64_t ns);
void NoteFloor(int pass);

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
