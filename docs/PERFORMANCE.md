# Performance: where the time goes, what was tried

Ghost of Yōtei on this fork runs at roughly 13-17 fps in towns and 21-25 fps in the open world (Performance mode,
Ryzen 7 9800X3D, RTX 5070 Ti; dates below are 2026). This page is the state of the performance work: what limits the
frame, how a change is measured, and every larger idea that was tried, including the ones that did not work, so nobody
has to try them again blind. Numbers are from single test places; small differences (< 2%) are within noise unless a
run says otherwise.

## 1. Where the frame time goes

### The chain of one frame (2026-11-02)

Each 10 µs of three frames standing at the gate and three frames walking through the village was given one state
(the GPU's command buffer timestamps and every wait of every emulator thread on one clock; the states sum to the
frame time, ~71 ms):

| State | Standing | Walking |
|---|---|---|
| GPU idle, the execution thread working | 46% | 48% |
| GPU busy, the execution thread working (overlap) | 35% | 29% |
| The execution thread waiting for the GPU | 15% | 16% |
| The execution thread waiting for the resolve thread | 4% | 7% |
| The game, the command processor, the resolve thread (the rest of the chain) | ~1% | ~1% |

- **The execution thread is the critical path**: it works for ~58 of ~71 ms. The GPU runs ~35 ms per frame and waits
  for the next command buffer the rest of the time (~100 command buffers per frame).
- Its work per frame: ~4,300-4,700 draws (typical 2.7-4.1 µs each), ~3,000-3,600 dispatches (typical 1.7-2.1 µs),
  ~400-600 GPU indirect draws, ~150-190 synchronous commands.
- About **1,200-1,500 operations per frame are 4-5× slower** than typical ones (10-100 µs, together 17-26 ms per
  frame). Hardware counters (instructions and cycles sampled per operation) show they execute **~4× more
  instructions** (a draw 25 thousand → 96 thousand) at a lower IPC (2.5 → 1.4): draws whose resources changed against
  the previous draw take the long paths everywhere (image and buffer lookups, binding changes, image transitions,
  checking that the guest memory the shaders read is unchanged). It is work in the emulator's code, not a slow memory
  system or lock contention (waits for a held lock: ~190 per frame, ~1 ms).
- **One GPU→CPU dependency at the start of every frame costs 8-9 ms**: a dispatch's resource walk needs 4 bytes the
  GPU computes at the start of the same frame, so the CPU waits for the GPU to finish the previous frame's tail first.
  The value is the same every frame; only speculation (use the last value, check it later) could remove that wait.
- The execution thread changes guest page protection ~1,100 times per frame (~2.5 ms).

### Sensitivity: what one microsecond per draw costs (2026-10-27)

Extra work of 2 µs added to every draw (live A/B) cost **8-10% fps at the gate and in the inn** (each µs per draw
≈ 4-5% fps, the frame grows almost 1:1) but only ~2% in the meadow, where the GPU is the limit (busy 53 ms per frame,
two compute shaders 12 ms of it). Fewer draws is the strongest lever: the game's own detail levels (forced through
mods) gave +13% (4,456 draws), +17% (3,747), +59% (2,370) and +79% (1,070) against 5,180 draws per frame in the
village; below ~2,000 draws the gain flattens.

## 2. How a change is measured

- **Ceiling first.** Before building anything, a temporary switch removes the work completely (the picture may break)
  and one run shows the most that path can give. A path is built only when the ceiling is worth it (we used 15%).
- **Check that the game does the same work.** A ceiling that skips a wait for GPU-written data lets the game read
  stale data, and the game then draws less; that looks like a gain and is not. Every ceiling run counts draws per
  frame in both phases (they must be equal).
- **Live A/B in one run.** The switch flips every 30 s in the same run at the same place, so both phases share the
  scene, the caches and the CPU's state; 6+ pairs, report the mean and every pair.
- **Verify mode (`=2`)** for a real change: old and new path computed side by side, 0 differences, before the new
  path becomes the default (and the old one is removed).
- Tools in the emulator: `KYTY_TRACE_FRAMETIME=1` (one line per frame), `KYTY_TRACE_PERF=1` (counters every 5 s,
  including `PERF semaphore` = CPU blocked on the GPU, `PERF locks`), `KYTY_TRACE_SYNC_TIMELINE=1` (every wait for the
  GPU with its thread, site and reason), `KYTY_TRACE_GPU_TIME=2` (command buffer GPU time; with
  `KYTY_TRACE_GPU_TIMELINE=<frame>` a per-buffer timeline), the F2 performance panel. A CPU profile of a build without
  LTO (`/Z7`) reads well in Visual Studio's or Windows' profilers.

## 3. What was tried

| Idea | Result | Kept? |
|---|---|---|
| GPU stage threads pinned to their own physical cores (TheCruZ's design) | **+18%** at the gate (pinned 16.3 vs unpinned 13.8 fps, 6 of 6 pairs) | yes, now chosen per processor |
| GLC loads of shaders that write no device memory read through L1 (`KYTY_PERF_READONLY_GLC_LOADS`, 2026-11-03) | GPU −0.5 ms per frame at the gate (one tessellated draw 0.50 → 0.03 ms); no fps change, the execution thread limits the frame; `=2` 0 differences in 3.9 billion loads | off, offered |
| Read window fitted to the host-mapped read (2026-11-02) | +1.8% standing, +7.7% walking, same draws | yes |
| Removing the image half of the old read-ahead "scout" thread | +3.9% at the gate | yes (removed) |
| Cheaper draws: a short path for draws equal to the previous one ("draw-path rebuild", 2026-10-28) | ceiling **~0%** (9.6 vs 9.8 fps): the work it removed ran on another thread in parallel | no, reverted |
| Skipping clean draws' preparation entirely (ceiling) | +8.8% at the gate | no (below the threshold) |
| A second preparation worker | +0.1% | no |
| No GPU wait before CPU reads of GPU-written data (ceiling) | +21%, **invalid**: the game drew 7-10% less on stale data; the valid part was +3.9% | no |
| Smaller read window (4 KiB instead of ±4 MiB) | −5.1%: the waits moved to the game's threads | no |
| The GPU thread not waiting for synchronous downloads (ceiling, 2026-11-02) | +7.8%, **invalid**: 23% fewer draws, then a crash | no |
| Exact image-alias handling (`KYTY_FIX_IMAGE_MEMORY_OWNER`, fixes the Quality-mode flicker) | costs 10-15%; its GPU work's ceiling +16.5%; fast versions: retile copies per row of blocks +2% (kept), retile directly on images ≤ +9%, skipping copies before full overwrites ≤ +6% | off by default; the workaround stays |
| Cloth simulation skipped (measurement only, wrong picture) | +10.6%; idea: run it on a second GPU queue as the PS5 does | open, risky |
| Locking the internal resolution at 1080p | +21% in cutscenes, nothing on the route | as a mod |
| Shorter detail distance (the game's LOD scale 0.4) | +16%, +29% walking | as a mod |
| Spinning threads disturbing the execution thread through SMT | no effect: every stage has a core of its own | — |

## 4. Open leads

- **The execution thread's per-draw work** (section 1): the slow draws with 4× the instructions are the largest
  target (17-26 ms per frame). A draw on this fork costs ~4 µs where other KytyPS5 forks report 1-2 µs in other
  games. Start from a CPU profile of the slow operations (resource lookups, binding changes, read verification).
- **The frame-start GPU→CPU dependency** (8-9 ms): speculation with a later check, or doing that resource walk on the
  GPU.
- **Page protection** on the execution thread (~2.5 ms per frame) and waits for the resolve thread (4-7%).
- **A second GPU queue** for compute work the PS5 runs asynchronously (cloth: up to +10%).
- **The game's own load levels.** The game lowers its detail when its GPU load is high, but measures its own
  resource ring usage, not the CPU cost that limits the emulator; an emulator-side controller that follows the frame
  rate could use the same levels automatically.
