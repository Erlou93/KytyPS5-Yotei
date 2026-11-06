# Fixes

What was broken, why, how it was fixed and what the fix costs. Newest first. Each entry names the commit that holds the
fix, so the code and its comment can be read next to it.

## Cache-bypassing loads of read-only shaders read through L1 (2026-11-03, `KYTY_PERF_READONLY_GLC_LOADS`, off)

**Symptom.** None visible. A tessellated draw in the village (about two per frame) took 0.50 ms of GPU time per
frame; its first stage has a dozen loads with GLC set.

**Cause.** A GLC/DLC vector load (bypass the GPU's L0/L1 caches) is emitted as a device-coherent access: the
`buffers_coherent` alias with Volatile, or a Coherent physical pointer. NVIDIA serves such loads from L2 and skips L1,
so a shader that reads the same data many times pays the L2 latency every time. The coherence buys nothing in a
program that writes no device memory: only another invocation of the same draw or dispatch could change the data while
the command runs, and a program with no buffer, address, image or GDS stores or atomics and no cache invalidation is
not such a writer. Writes of earlier commands are made visible by the emulator's barriers, which do not depend on the
decoration. The PS5's GLC exists to see data another wave wrote moments ago; in a read-only program no wave of the
command writes.

**Fix.** `ApplyReadOnlyGlcLoads` (`ir/passes/ShaderInfoCollection.cpp`) runs before the shader info is collected. In a
program that writes no device memory it takes the coherence off the GLC loads, which then read through the plain
bindings. A GLC load in a block on a loop keeps it: a loop that reads the same address may be waiting for a writer in
another command. `=2` makes every such load both ways in the same invocation, uses the coherent result and counts the
loads whose values differ (`PERF readonly_glc_loads verify`). The idea comes from another KytyPS5 branch (CarRamRod-963,
"shader: drop device coherence from GLC loads of read-only programs"), where a full-screen compose pass went from 6.3 to
0.26 ms.

**Verify.** `=2` at the gate at night, 300 s standing and 180 s walking through the village: 3.88 billion loads
compared, 0 differences. 0.24 billion 16-bit loads were not compared: their storage alias cannot be plain and coherent at
once in one program.

**Cost.** A gain on the GPU only. Standing at the gate (RTX 5080, driver 617.42): the draw above 0.50 → 0.03 ms per
frame, the 11 changed programs together 0.92 → 0.43 ms, GPU busy 35.7 → 35.2 ms per frame (three pairs of separate
runs, every run with the switch below every run without). The frame rate does not change measurably: the frame is
limited by the execution thread (see PERFORMANCE.md). The switch changes the SPIR-V, so it cannot be toggled in a live
A/B. Default off until it has been tested on more of the game.

## GPU pipeline threads on whole cores of any processor (2026-11-02, `cfb35beab`)

**Symptom.** None visible on the development PC. The emulator pins its four GPU pipeline threads (execution, command
recording, resolve, command processor) to their own physical cores, but the cores were a fixed list made for one
processor (logical CPUs 2, 6, 8, 4 of a Ryzen 7 9800X3D). On another processor the list could put the threads on
Intel efficient cores, on the second CCD of a two-CCD Ryzen, or on SMT siblings of each other.

**Cause.** A list for one CPU. How much the pinning matters was measured on 2026-11-02: the same run at the gate with
the pinning taken away every other 30 s ran 13.8 fps against 16.3 fps pinned (+18%, 6 of 6 pairs).

**Fix.** `Common::ChooseGpuStageCpus` (`src/common/gpuStageCores.h`) picks from the processor's CPU sets: the fastest
efficiency class, the last-level cache with most such cores (on a tie the larger cache, the 3D V-Cache CCD), core 0
left out when enough cores remain, the first logical processor of each core; fewer than four such cores: no pinning.
On a 9800X3D it picks the same cores as before. `KYTY_GPU_THREAD_CPUS` still overrides it; a unit test covers a
9800X3D, two-CCD Ryzens, Intel hybrid CPUs and small processors.

**Cost.** None; one query of the CPU sets at start.

## A 4-byte read at the start of every frame waited for the whole GPU queue (2026-11-02, `b2b55c928`)

**Symptom.** A timeline of single frames showed one compute dispatch at the start of every frame that took 6-9.5 ms,
almost all of it waiting for the GPU, while the bytes it needed had been written by the GPU long before.

**Cause.** The dispatch's shader resource walker reads 4 bytes the GPU wrote. A CPU read of GPU-written memory takes a
read window: every GPU-written range of the buffer within 4 MiB around it, so that neighbouring reads need no new
synchronization. Here the window held 1.36 MB of GPU-written data, more than the host-mapped read serves (1 MiB), so
the read took the slow path: submit the open command buffer and wait for every command buffer before it (five).

**Fix.** When the host mapping cannot serve the window at once but can serve the pages of the read itself, the read
takes only those pages and waits only for their writes (`BufferCache::ReadRange`). The rest of the window stays marked
as GPU-written and is synchronized when it is read. The bytes read are the GPU's newest either way.

**Cost.** A gain: in one run with the change switched every 30 s, 15.39 against 15.13 fps standing at the gate
(+1.8%, 6 of 6 pairs) and 15.51 against 14.40 fps walking through the village (+7.7%, 3 pairs), with the same number
of draws per frame. A second wait at the start of the frame remains: there the 4 bytes are written by the command
buffer being recorded, so the CPU really has to wait for the GPU.

## Pixel helper lanes kept in the EXEC words (2026-11-02, `ee0b4072b`)

**Symptom.** After rebasing onto Kyty main, distant terrain in Ghost of Yōtei lost its detail: grass and flower fields
seen from a hill turned into smooth patches with dark blotches, the snow on Mount Yōtei showed square blocks, and fog
looked pixelated. Up close everything was fine. The build before the rebase drew the same place sharply.

**Cause.** Upstream `5429888d4` started the EXEC words of every pixel shader from `LiveMask`, the lanes that are not
helper invocations. Helper lanes exist only so a quad can compute derivatives for implicit-LOD texture samples. Once
they were missing from EXEC, the masks a shader derives from EXEC and VCC (branches, `s_and_saveexec`, compares) left
them out as well, so inside branches the helpers stopped computing the texture coordinates. The quad derivatives were
then wrong and the sampler picked far too coarse a mip. Small, distant triangles have the most partly covered quads,
which is why the far valley, the mountain snow mask and the fog were hit hardest.

**How it was found.** Every SPIR-V module of the pre-rebase build and of the rebased build was dumped on the same save
(`KYTY_TRACE_SPIRV_ALL_DIR`) and compared with `spirv-diff`. Every changed pixel shader had gained a `HelperInvocation`
built-in and a subgroup ballot at entry. A test switch that started EXEC from all lanes again brought the detail back in
the game; four other suspects (16-bit packing, `v_fract`, the `_cl` LOD clamp, the A16 address layout) and the image heap
tables had been ruled out first.

**Fix.** The EXEC words start from every lane the wave runs, helper lanes included, as before the rebase
(`frontend/translate/Translate.cpp`). `LiveMask` stays in the code, unused.

**Cost.** None measured: the hill ran at the same frame rate with and without the fix (24 fps).

## Older fixes

The fixes below were made before the fork was rebased onto Kyty main; their code is part of the rebase commit
`f2e5a0034` (one change), so the comments in the named files are the place to read them.

## Black particles from paired half-waves (2026-10-30)

**Symptom.** Black particle-like spots in a few places of the frame, appearing on a few percent of frames.

**Cause.** A PS5 wave has 64 lanes; on the host a wave of a compute shader runs as two 32-lane halves ("paired
wave"). Lanes of one guest wave see each other's LDS writes without a barrier, because the hardware runs the wave in
lockstep; a blur compute shader relies on that: it writes its texel to LDS and reads the neighbours ±1-4 lanes away
with no `S_BARRIER`. The two host halves do not run together, so lanes 28-31 read lanes 32-35 of the other half
before those were written. Found by switching the pairing on for subsets of the shaders until one shader alone
reproduced it.

**Fix.** In a paired compute shader with one wave per workgroup, a workgroup barrier goes after every LDS write or
atomic and before an LDS write that follows an LDS read (`backend/spirv/spirvEmitterMemory.cpp`, `PairLdsSync`).
Verified by running every dispatch of the shader twice and comparing: 0 differences in 4,000 (3,993 before).

**Cost.** Not measurable (a few small dispatches).

## Distant objects vanishing after cutscenes and while the camera moves (2026-10-21 and 2026-10-31)

**Symptom.** After the first cutscene, distant houses, trees, fog and water vanished and flickered every other frame;
inside cutscenes the world could disappear around the characters. Later, objects near and far also vanished for a
moment while the camera turned or tilted up.

**Cause.** The game runs its own occlusion test on the CPU, against the GPU-built hi-Z depth pyramid it reads from
memory (two copies, alternate frames). On a PS5 the CPU sees the GPU's writes in shared memory; in the emulator the
pyramid stayed in host VRAM. First, the depth copy of the pyramid the texture cache makes every frame was never
written back to guest memory, so the CPU tested against a pyramid frozen around the cutscene (one of the two
per-frame lists lost ~25-30% of its objects). After that was fixed, the write-back still ran only at the frame
boundary, so the CPU read the pyramid one frame late and objects vanished while the camera moved.

**Fix.** (1) The depth copy is written back like the image it replaces, and the readback of GPU-written linear images
is on by default (2026-10-21). (2) Read coherence (2026-10-31, `cache/textureCache.cpp`, "Image read coherence"): a
guest page of a GPU-written image loses CPU access when the write enters the GPU stream; the first CPU read faults,
the game's thread waits for the GPU and gets the bytes. Checked on ~1.9 million pages at the gate and in the open world
against the image at the frame boundary: 0 differences.

**Cost.** −3.3% at the gate, 0% in the open world for the read coherence; the first half was a gain (11.2 → 14.0 fps
at the gate), because the CPU test works and the game sends fewer objects to the GPU.
