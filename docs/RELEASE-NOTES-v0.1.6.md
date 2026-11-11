## KytyPS5 Yōtei fork v0.1.6

Game version **2.002** only (1.512 still starts but is not tested). All numbers are from the test PC (Ryzen 7 9800X3D,
RTX 5070 Ti 16 GB, 64 GB RAM) in the game's *Performance* mode, on replays of the same recorded routes.

This release has two parts: **A** is the v0.1.6 package prepared on 7.11 (never published on its own), **B** is
everything added after it.

### In short

- **Town walk: about 14 fps in v0.1.5 → about 16 fps (part A) → about 19-20 fps now (part B).** Riding through the
  open world near the inn: about 26-27 fps. (Separate measurements on different days, not one side-by-side run.)
- **Much shorter starts and loads:** the shaders are ready in about 4 s instead of 33-35 s after the first start,
  "Continue" loads in about 8-9 s instead of about 15 s.
- **Fewer hitches:** no more 1-3 s freezes while walking, far fewer long frames in the first minutes, shorter freezes
  at camera cuts in cutscenes.
- **Bigger shader hints** (11,969 shaders, 16,117 pipelines) and a launcher that walks you through the first start.

### Part A – the v0.1.6 package (prepared 7.11)

Performance
- **Faster town walk: 13.9 → 15.8 fps (+13%)**, median frame 68-75 → 60-64 ms, worst 1% of frames 88-103 → 82-86 ms.
  Many small savings on the emulator's main GPU thread (texture tables refreshed only where they changed), plus a
  compiler profile trained on walking.

Fixes
- **The 1-3 second stalls while walking are gone.** When the game streamed in new data, a dozen of its threads could
  spin on one emulator lock at once, and the thread holding it got no CPU time. Waiting threads now back off.
- **Much better on PCs where reading back from the graphics card is slow.** On one tester's i9-12900K + RTX 5090,
  reading GPU results back over PCIe ran at ~19 MB/s (about 60× slower than on our test PC) and kept the game near
  10 fps. The emulator now times these reads at start and, when they are slow, copies the data through the GPU
  instead. The log shows `BAR_READS fast` or `BAR_READS slow`; `KYTY_BAR_READS=0` / `=1` forces the choice.
- **No more endless loading spinner on some dumps.** The game asks at boot whether the base game is owned; the emulator
  now says yes. Thanks to **CarRamRod-963** for finding and fixing this.
- **Missing faces:** "Enable readback" (launcher, Graphics & display) is now on by default and is switched on once for
  existing settings. Players reported that faces stayed invisible without it. It costs about 2% fps in town; you can
  turn it off again.

Launcher and look
- **New first steps: import the shader hints, then precompile, then play.** Many players skipped the hints and had
  hours of stutter. **Precompile** and **Play** stay disabled until the hints are imported (Shaders page →
  **Import hints…** → `hints\PPSA26344_02.002.000.kyhints`), and **Play** stays disabled until the precompile has
  finished. The Play page says which step is next, the Precompile buttons are framed red until the hints' shaders are
  ready and green when done, and the help buttons open the player guide at the right page.
- **New look:** shader preparation shows a brush ensō in the top-right corner (also over the loading screen), the
  F2 / Shift+F2 panel has brush corners and a brush-stroke frame graph, and the launcher has the brush style.
- Shader precompile uses **8 threads** by default (was 6).

### Part B – added after the v0.1.6 package

Performance
- **Town walk about 16 → about 20 fps.** The largest step: the graphics card now reads part of the game's per-draw
  resource data itself instead of the emulator preparing it for every draw (16.0 → 19.6 fps, +22%, 0 differences in its
  checking mode). Smaller measured steps on top: handling of the game's data ring by game frames (+2.4%), no needless
  waits on some GPU signals (+1.4%), cheaper resource lookups (+1.5%), a fresh optimisation profile (+3.1%).
- **Faster in-engine cutscenes.** Several of the heaviest cutscene shaders were made cheaper (inlined loads, faster
  ray-tracing node lookups, a quicker path for empty format reads): the two largest changes gave about +8.6% and +7.5%
  in the second part of the cutscene after the Snake fight. That cutscene now runs at about 11-12 fps in automatic runs
  (its heaviest part about 8 fps).
- **Much faster start.** Translated shaders are kept on disk (`_PipelineCache\<title>.reuse`, about 416 MB): after the
  first start, preparing the shaders takes about 4 s instead of 33-35 s.
- **Faster "Continue".** Loading a save from the main menu takes about 8-9 s instead of about 15 s: the game's graphics
  memory is made readable by the card when the game maps it, not later during play. Cost: about 4 GB of pinned system
  RAM instead of 1.9 GB (shown as the card's shared memory).
- From the main KytyPS5 project: faster kernel semaphores and signal handling (superkeka), a faster GPU page-fault scan
  that no longer loses pages past its limit (nmzik).

Fixes
- **Fewer hitches in the first minutes of play.** Late imports of game memory held the graphics driver for about 50 ms
  each and froze the frame; they are now done in larger groups. Frames over 100 ms in the town walk 18 / 11 → 3 / 4, in
  the first minute 14 → 5, and fps +5.0%.
- **Shorter freezes at camera cuts in cutscenes.** At a camera cut the game creates and frees about 2,100 small GPU
  buffers in one frame; they now share memory blocks instead of each getting its own allocation. The long frames at the
  first camera cut of the intro add up to about 1.9-2.8 s instead of about 3.3-3.6 s. About 0.5 s is still left and is
  being investigated.
- **No shader stutter right after starting.** At every game start the emulator waits until the prepared shaders are
  loaded (the window title shows **"preparing shaders N%"**; a few seconds after a finished precompile, about 30 s once
  after an update). In the town walk afterwards 1 shader was missing instead of 27-40.
- **Updates keep your shader cache.** The caches now depend only on the shader translator's version, not on the whole
  emulator file, so later versions that do not change the translator keep the prepared shaders. (This version does
  change it: the first start prepares the shaders again once.)
- The store of translated shaders may hold up to 4 GB (was 2 GB, which was full and dropped results).
- The GPU clock the game reads through its shaders ran about 25% too fast; it now runs at the right rate (Senaxx).

Shader hints and launcher
- **Shader hints: 11,969 shaders and 16,117 pipelines** (v0.1.5: 6,880 / 8,001): ours plus a large set a player shared
  from far into the game. The precompile takes several minutes (longer on CPUs with fewer cores).
- **New hints are imported again:** the launcher remembers which hints file the first steps were done with; a version
  with new hints asks for the import again, so players who did the steps before also get the new shaders.
- **Default window 1920×1080** for new settings (the game renders at 1080p; a 1280×720 window only made it blurrier).
  Saved settings keep their choice.
- **The performance panel's GPU row works on Windows** (it showed "n/a"): it shows how busy the graphics card is with
  this game.

Diagnostics
- **Texture relocation guard (on by default):** after textures are moved to free graphics memory, their old copies are
  kept for 10 s and any late use is logged as `TEXMOVE_STALE_USE`; every `TEXMOVE` line has the wall-clock time. No
  late use was found in our tests (1,888 moved textures). Cost: no measurable change in fps. `KYTY_TEXMOVE_GUARD=0`
  turns it off.
- Diagnostic and measurement tools for developers (off unless enabled).

### Known issues

- **Checkered, pixelated shadows after a short stutter or a cutscene (issue #11).** While a new shader compiles, the
  emulator skips the draws that need it for a few frames; an incomplete frame can get into an effect the game carries
  from frame to frame. **Fix: quit to the main menu and load your save.** It is rarer in this version (the shaders are
  prepared at the start, the hints cover much more); the proper fix (pipeline libraries) is planned.
- **Occasional fps drops during play.** Sometimes the frame rate drops for a while (for example to about 16 fps near the
  end of a town walk); the cause is being investigated.
- **Graphics memory grows over a long session:** watch "Vulkan budget" in Shift+F2; if it gets close to the limit and
  the game slows down, save, quit and start again.
- **Workarounds are still in place** (memory read through pointers before it has a buffer, image memory shared by
  several images, stencil, colour compression and others): see *Known workarounds* in `README-YOTEI.md`.
- **RAM:** about 4 GB of RAM is pinned for the card and the shader store holds about 2.2 GB; **32 GB of RAM** stays the
  recommendation.

When you report a problem: note the time on your screenshot, then Launcher → Advanced → **Export bug report**, and
attach the zip and the screenshot to an issue.

Thanks to **CarRamRod-963**, **Senaxx**, **nmzik** and **superkeka** for the changes taken from their work, and to the
player who shared the shader hints.

### Download

Download **KytyPS5-Yotei-v0.1.6-windows-x64.zip**, unpack it into a new folder and copy your `_SaveData` folder there.
Then follow `KytyPS5-Yotei-Player-Guide.pdf`: import the hints, let the precompile finish, play.
