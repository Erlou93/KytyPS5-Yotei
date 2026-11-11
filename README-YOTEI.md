# KytyPS5 Yōtei fork

An experimental fork of [KytyPS5](https://github.com/KytyPS5/KytyPS5) (itself based on
[Kyty](https://github.com/InoriRus/Kyty)) that targets **one game only: Ghost of Yōtei (PS5)**. Every change in this
tree is made, measured and tested against that game. Other titles are not tested and may run worse than on upstream
KytyPS5; use upstream for them. The upstream README ([KytyPS5](https://github.com/KytyPS5/KytyPS5#readme)) is kept
unchanged in the source tree as `README.md`.

Downloads and bug reports: [github.com/Erlou93/KytyPS5-Yotei](https://github.com/Erlou93/KytyPS5-Yotei)
([Releases](https://github.com/Erlou93/KytyPS5-Yotei/releases), [Issues](https://github.com/Erlou93/KytyPS5-Yotei/issues)).

This repository contains **no game files, no firmware, no system software and no material taken from the game's
code**. You need your own legally obtained dump of the game. Where the emulator relies on how the game behaves (for
example the layout of a table the game fills in memory), that knowledge is described in comments as behaviour; no game
code is copied.

## Status

- Work in progress. The game boots, loads saves and is playable, but runs well below the PS5's frame rate and some
  effects still differ from the PS5 (see [Known workarounds](#known-workarounds)).
- Correctness is checked against reference captures of the game's own GPU work: a fix counts as correct when the
  emulator computes the same results as the PS5 on the data the game really uses.
- Performance work is done behind switches that have a verify mode (old and new path compared, 0 differences) before
  they become the default.

## Requirements

- Windows 10/11, 64-bit; a CPU with AVX2 (the release build uses `/arch:AVX2`).
- A Vulkan 1.3 GPU with current drivers. Development and all testing are done on an **NVIDIA GeForce RTX** card
  (RTX 5070 Ti, 16 GB). AMD and Intel GPUs are untested; a few exact half-precision paths are enabled only on devices
  that were measured (NVIDIA), other vendors take slower but exact fallbacks.
- 32 GB of RAM recommended (the game needs about 20 GB of free commit) and several GB of free disk space for the shader
  and pipeline caches.

## Playing (release package)

Requirements: Windows 10/11 64-bit, a Vulkan GPU with a current driver, and **32 GB of RAM** (the game takes 22-28 GB
of system memory in play; with 16 GB Windows runs out of memory and pages to disk).

1. Unpack the zip into an empty folder (not under `Program Files`; the emulator writes its caches, saves and logs next
   to itself, the launcher its settings, `Kyty.ini`; nothing is written anywhere else).
2. Start `launcher.exe` (not `kyty_emulator.exe`: only the launcher prepares the shaders), choose your Ghost of Yōtei
   dump (game version **2.002**, the one supported and tested since v0.1.4; 1.512 still starts but is no longer
   tested and has no shader hints): its folder or its `.zar` archive.
3. **Import the shader hints, then precompile (required since v0.1.6):** the package ships shader hints for the game
   (`hints\`, a list of the shaders the game uses, without any game code). Shaders page → **Import hints…** → choose
   `hints\PPSA26344_02.002.000.kyhints`. The launcher finds those shaders in your game files and compiles them; this
   takes a few minutes and runs only once. Until the hints are imported, **Precompile** and **Play** stay disabled, and
   **Play** stays disabled until the precompile has finished (a paused one finishes with *Continue precompiling*).
   Without hints the first hours stutter while shaders compile during play. The v0.1.6 hints (11,969 shaders, 16,117
   pipelines: ours plus a large set a player shared from far into the game) cover the game **from the start to the
   Snake fight, the open world and much further**; the precompile takes several minutes (longer on CPUs with fewer
   cores). Other areas compile their shaders while you play, with short stutters the first time you reach them.
   Shaders compiled while playing are saved, so the next visit is smooth. When a later version ships new hints, the
   launcher asks for the import again. At every game start the window title shows **"preparing shaders N%"** while the
   game waits for the prepared shaders (a few seconds after a finished precompile).
4. **Graphics mode:** the game's own *Performance* mode is the one tested. In *Quality* mode the game changes its
   internal resolution all the time, which makes the screen flicker in this build; the **Render at 1080p** mod
   (launcher, Mods page; on by default on 1.512 and 2.002 since v0.1.3, also in cutscenes) keeps it steady. The
   **Shorter detail distance (LOD 0.4)** mod (1.512) gives the largest speed-up
   (about +16%, +29% while walking) at the cost of simpler distant objects.
   The launcher's default window is **1920×1080** (the game renders at 1080p; a smaller window only makes it blurrier).
5. **Performance panel:** F2 shows or hides it in game, Shift+F2 shows its details (frame time, CPU waiting on the GPU,
   draw calls, shader compiles, memory). The launcher's Overlay option shows it from the start.
6. **If the game crashes:** every run writes a log to `_Logs` next to the emulator (the newest 10 are kept). Launcher,
   Advanced page, **Export bug report** packs the newest logs, the settings and the system info into one zip, with your
   user and computer names replaced; attach it to an issue at
   [github.com/Erlou93/KytyPS5-Yotei/issues](https://github.com/Erlou93/KytyPS5-Yotei/issues).

The package has the same steps with pictures in `KytyPS5-Yotei-Player-Guide.pdf`.

What to expect: on the test PC (Ryzen 7 9800X3D, RTX 5070 Ti) the game runs at roughly 19-20 fps while walking through
towns and 26-27 fps riding through the open world with the default settings (test runs of 11.11), more with the mods above. The emulator's GPU work runs on four threads, each on its own
physical CPU core, chosen automatically for the processor (`KYTY_GPU_THREAD_CPUS=off` turns that off).

What changed in this version: see `docs/RELEASE-NOTES-v0.1.6.md` (all changes since v0.1.5).

Known slow spots:
- Some in-engine cutscenes are still slower than play: the cutscene after the Snake fight runs at about 11-12 fps
  (its heaviest part about 8 fps; 5.5 in v0.1.3). The **Render at 1080p** mod (on by default) keeps cutscenes at 1080p instead of up to 4K.
- Scenes with a lot of fire, such as parts of the Snake fight, can drop to about 10 fps.

**Known issue: graphics memory grows over a long session.** Each change of scene (especially cutscenes) leaves unused
gaps in the graphics card's memory. Since v0.1.5 the emulator moves textures out of nearly empty memory blocks so
whole blocks can be released (texture relocation); this reclaims part of it, not all. Since v0.1.6 a guard keeps the
old copies of moved textures for 10 s and logs any late use (`TEXMOVE_STALE_USE`; none in our tests over 1,888 moved
textures). **If you see textures, trees or other objects disappearing, flickering or turning black, please open an
issue** with a screenshot (with the time), the place and the bug report zip (Advanced page). On a 16 GB card the usage goes from about
12 GB after loading to 13.5-15 GB after 30-60 minutes; near the card's limit the driver moves data to system RAM and
the frame rate drops. Watch "Vulkan budget" in the performance panel (Shift+F2): if it gets close to the limit and the
game slows down, save, quit and start again.

**Known issue: checkered, pixelated shadows after a short stutter.** While a new shader compiles, the emulator skips
the draws that need it for a few frames instead of making the game wait (otherwise new areas would freeze for seconds).
Sometimes such an incomplete frame gets into an effect the game carries from frame to frame, and shadows and lit
surfaces turn into a blocky checkered pattern that stays. **Fix: quit to the main menu and load your save** (no
restart needed). Since v0.1.6 it is rarer: the game waits for the prepared shaders at the start and the hints cover
much more of the game, so fewer draws are skipped. A proper fix (pipeline libraries, so new shaders no longer need
draws skipped) is planned and replaces this workaround. Reported as issue #11.

**Touch pad (e.g. "Swipe up" prompts):** a DualSense's touch pad works as on the PS5. With a keyboard and mouse press
**F6**, move the mouse to swipe, and press F6 again to leave touch pad mode. With other controllers assign a button to
**TouchPadMode** in the launcher's input mapping: that button turns touch pad mode on and off, and the left stick moves
the finger.

## Building (Windows)

Toolchain used for this fork:

- Git (with submodules);
- CMake 3.22.1 or newer and Ninja (the copies shipped with Visual Studio work);
- Visual Studio 2022 or Build Tools 2022 with the **Desktop development with C++** workload (Windows SDK, `vcvars64.bat`);
- LLVM **20.1.8** for Windows (`clang-cl`, `lld-link`, `llvm-mt`); the MSVC compiler `cl.exe` is not supported;
- [glslang](https://github.com/KhronosGroup/glslang/releases) (`glslangValidator` or `glslang` on `PATH`, or
  `-DKYTY_GLSLANG_VALIDATOR=<path>`);
- FFmpeg is downloaded by CMake on the first configure (or point `-DFFMPEG_PREBUILT_DIR=<dir>` at an unpacked package);
- for the launcher only: Qt 6 for MSVC, 64-bit, with Concurrent, Network and Widgets (6.7.2 is used for this fork).

The fork has its own Qt launcher (`src/launcher`): game selection, display and graphics settings, shader
precompilation with a progress bar, the shader cache, log folder and a local bug-report export. Its design is
described in [docs/launcher/DESIGN-SPEC.md](docs/launcher/DESIGN-SPEC.md); how to build it (with or without the
emulator) is in [CONTRIBUTING.md](CONTRIBUTING.md). The emulator also runs on its own from the command line. To build
the emulator only, without Qt, from an **x64 Native Tools Command Prompt for VS 2022** (or after calling
`vcvars64.bat`), in the repository root, with `LLVM` set to your LLVM 20.1.8 folder:

```bat
git submodule update --init --recursive

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DKYTY_BUILD_LAUNCHER=OFF ^
  "-DCMAKE_C_COMPILER=%LLVM%/bin/clang-cl.exe" "-DCMAKE_CXX_COMPILER=%LLVM%/bin/clang-cl.exe" ^
  "-DCMAKE_LINKER=%LLVM%/bin/lld-link.exe" "-DCMAKE_MT=%LLVM%/bin/llvm-mt.exe" ^
  "-DCMAKE_C_FLAGS_RELEASE=/O2 /Ob2 /DNDEBUG /arch:AVX2 -flto=thin" ^
  "-DCMAKE_CXX_FLAGS_RELEASE=/O2 /Ob2 /DNDEBUG /arch:AVX2 -flto=thin"

cmake --build build --target kyty_emulator -j 4
```

The reference interpreter used for correctness checks (`tools/refinterp`, CPU only) builds with
`tools\refinterp\build.bat`; set `VCVARS64` (path to `vcvars64.bat`) and `CLANG_CL` (path to `clang-cl.exe`) unless
both are already on hand in your prompt.

## Running

Start `launcher.exe` (next to `kyty_emulator.exe`), choose the game folder and press **Play**, or start the emulator
directly:

```bat
kyty_emulator.exe --game "D:\Games\<your Ghost of Yotei dump>" --screen-width 1280 --screen-height 720
```

Run `kyty_emulator.exe --help` for the logging, validation and debugging options; `--perf-overlay` shows the
performance panel from the start (F2 / Shift+F2 in game). Tessellation is on by default
(`--no-tessellation` turns it off). The first start compiles shaders and pipelines in the background, so the first
minutes stutter; later starts reuse the caches. `--precompile-only` (the launcher's "Precompile shaders") compiles the
shaders recorded in earlier runs and exits without starting the game. `--export-hints <file>` with `--precompile-only`
writes the recorded shaders as a hints file (no game code), `--import-hints <file>` finds a hints file's shaders in the
game folder and adds them before compiling.

## Documentation

- [docs/PERFORMANCE.md](docs/PERFORMANCE.md) – where the frame time goes, how a change is measured, and every larger
  idea that was tried with its result (also the ones that did not work); read it before performance work;
- [docs/FIXES.md](docs/FIXES.md) – what was broken, why, how it was fixed and what the fix costs (newest first);
- [docs/FRAME.md](docs/FRAME.md) – a Ghost of Yōtei frame pass by pass (shadows, G-buffer, terrain, cloth, particles,
  probes and RTGI, fog, water, FSR3, post-processing) and how the emulator handles each pass;
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) – the GPU thread, preparation workers, command recording, shader
  translation and caches, memory tracking, readbacks, bindless textures, `envDefaults.h`;
- [docs/TESTING.md](docs/TESTING.md) – verify modes (`=2`), live A/B in one run, test places, pictures, workaround
  counters, warming the shader cache before measuring;
- [docs/launcher/DESIGN-SPEC.md](docs/launcher/DESIGN-SPEC.md) – the launcher's design;
- [CONTRIBUTING.md](CONTRIBUTING.md) – rules for changes, building, reporting problems.

## Settings are defaults in code

No launch script or settings file is needed: a plain `kyty_emulator.exe --game ...` uses the same settings as the
maintainer's test runs. All settings this fork depends on are defaults in the source:

- `src/common/envDefaults.h` holds the table of settings that used to come from the launch script, each marked as
  `config` (ordinary setting or measured performance switch), `unproven` (probably hardware behaviour, named
  `FIX_`) or `workaround` (named `WORKAROUND_`, see below), with the work item that will replace it;
- every other switch has its default where it is read (`std::getenv("KYTY_...")`), documented in the comment above it.

An environment variable still overrides any default (`KYTY_X=0` turns a switch off), and `KYTY_ENV_DEFAULTS=0` ignores
the whole table. At start the emulator prints one `ENV_DEFAULT KYTY_X=value source=env|code` line per table entry.
`KYTY_TRACE_*` and `KYTY_CAPTURE_*` switches are diagnostics, off by default.

Which settings keep the shader cache warm: only the settings listed in `Common::ShaderEnv::Names`
(`src/common/shaderEnv.h`) change the translated shaders or pipelines, and only their effective values (the
environment, else the code default) decide whether the pipelines of earlier runs count as compiled. Changing any other
`KYTY_` variable - reports (`KYTY_TRACE_*`), live A/B, pad scripts, dispatch skip lists such as the "No
particles" mod (`KYTY_DIAG_SKIP_CS_HASHES`), precompiler threads, VRAM budget - keeps the cache; changing a listed one
makes the next start compile in the background again. The start log has one `ENV_FINGERPRINT hash=...` line: two runs
with the same hash share the cache.

## Known workarounds

The goal of the fork is a build with **no workarounds**: every approximation is replaced by a proper implementation
(checked with a verify mode, 0 differences) and then removed. Until then the following are on by default. They are
the `workaround` entries of `src/common/envDefaults.h` plus a few defaults in code ("(code)"); names are given without
the `KYTY_` prefix. The environment can turn them off (`=0`), but the game then fails or shows errors: they stay on
until their replacement lands.

| Area (work item) | Switches | What may differ from the PS5 |
|---|---|---|
| Memory read through pointers before it has a buffer (M06) | BDA fallback (code) | such a read goes straight to the game's memory in RAM: the 2 MiB pieces the GPU used are made readable by the card (pinned RAM, shown as the card's shared memory). Since v0.1.6 the game's memory is made readable when the game maps it (about 4 GB pinned), so such reads no longer return zeros the first time |
| GPU-written memory (D6) | `DIAG_BDA_STORE_TRACK`, `DIAG_BDA_STORE_UNTRACKED`, `DIAG_BDA_PIN_FAULTED`, `DIAG_FAULT_WINDOW_KB`, `DIAG_SCALAR_READS_IGNORE_ADDRESS_WRITES` | pages the GPU writes through pointers are tracked late and kept generously; rare stale reads |
| Image memory shared by several images (D4/D5) | `WORKAROUND_ALIAS_SYNC` | transient targets copied between aliases; some post-processing (screen-space GI filter) may be off; in the game's Quality mode, where the internal resolution changes all the time, the screen can flicker and textures can vanish for a moment (use the "Render at 1080p" mod). The exact replacement exists (`KYTY_FIX_IMAGE_MEMORY_OWNER=1`: no flicker) but costs 10-15% fps, so it is not the default yet |
| Stencil and depth metadata (D8) | `WORKAROUND_STENCIL_PLANE_SYNC`, `WORKAROUND_STENCIL_STORAGE_SHADOW`, `WORKAROUND_PRESERVE_STENCIL`, `WORKAROUND_STENCIL_ALIAS_COPY`, `WORKAROUND_HTILE_FULL_RANGE`, `FIX_HTILE_RMW_CLEAR` (code), `WORKAROUND_DEPTH_ENCODING_LENIENT` | stencil kept as copies, HTILE emulated as "full range"; occasional missing masks or shadow tiles |
| Colour compression (A5) | `WORKAROUND_DCC_FAST_CLEAR`, `DIAG_DCC_KEEP_BUFFER_WRITES`, `DIAG_COLOR_META_UNALIGNED_SKIP` | simplified DCC clear handling |
| One fill compute shader (A5) | `WORKAROUND_FILL_CS_HASHES` (code: shader `2bd3cd129405f9a9`) | the shader is recognised by its hash and its stores are taken as a DCC fast clear (it still runs); with the list set by hand, a listed shader's output is filled instead of computed |
| GPU-selected (bindless) textures (D1/D2) | `DIAG_BINDLESS_EAGER`, `DIAG_NULL_TEXTURE_GRAY`, `DIAG_BINDLESS_IMMEDIATE_BUDGET` (code: 64 new textures per frame) | a new texture can show a placeholder for 1-3 frames; unresolved descriptors read as empty (e.g. snow and mud tracks). Fixed since 2026-10-29 (C9): texture tables over 1 MiB (particles, mist) are translated like every other table instead of showing placeholders past 1 MiB, and a table the table path cannot take stops with `BINDLESS_TABLE_REFUSED` instead of a silent placeholder |
| Shaders capped (D7) | `SHADER_LOOP_LIMIT_HASHES` (2 shaders), `DIAG_SKIP_UNSUPPORTED_CS`, global loop watchdog `KYTY_SHADER_LOOP_LIMIT` (code; a safety net, not a workaround: every shader loop ends after 65,536 iterations so a runaway loop cannot hang the GPU - it never fired in our tests, 0 of ~1,950 checks at the gate and in the village, 29.10). The particle-emit shader 142383a7 is no longer skipped (29.10; it runs rarely - if a crash follows the log line `W9_PARTICLE_EMIT`, report it) | rare shaders with a loop cap |
| Draws while a new pipeline compiles | (code) | the draw is skipped until its pipeline is ready (a few frames) instead of the game waiting; can leave checkered shadows until the save is loaded again (see the known issue above) |
| Terrain shaders (M42) | `WORKAROUND_WATERFALL_POINTER_RECORD0` | approximate material table in some terrain shaders |
| Texture edge cases | `SDK_FIX_MIP_CHAIN_CLAMP`, `SDK_FIX_NULL_COMPARE_DEPTH`, `SDK_FIX_INTEGER_POINT` (code) | rare sampling edge cases approximated (logged) |
| Game-specific table (D6) | `RECORD_KEYED_GAME_COUNTS` (code) | one shader's record count is read from the game's data layout instead of a generic bound |

The hardware behaviour that was once a switch (formatted stores through GPU-selected descriptors, the GDS base in M0,
textures less aligned than their swizzle block, mip-tail slices) is always on now. Two table entries are marked
`unproven` (`FIX_TEXTURE_CLAMP_LAST_LEVEL`, `FIX_RT_11_11_10_FLOAT`: believed to be the hardware's behaviour, an
exactness check is still open). The game's GPU timer reads host GPU busy time (`FIX_VIRTUAL_GPU_CLOCK`); because the emulated frame takes
longer than on a PS5, the game stays at a lower internal resolution than on the console (1080p at the test scene of
most measurements). Reporting the time a PS5 would take is open.

## Licenses

- This fork, like KytyPS5, is licensed under the **GNU General Public License version 2 only** ([LICENSE](LICENSE)).
  If you distribute a binary, you must also offer the complete corresponding source code.
- The original Kyty is MIT-licensed; its copyright and license notice are kept in
  [LICENSES/Kyty-MIT.txt](LICENSES/Kyty-MIT.txt).
- `tools/refinterp/generated/isa_tables.inc` is generated from AMD's machine-readable GPU ISA (AMD GPUOpen, MIT):
  [LICENSES/AMD-GPU-MR-ISA-MIT.txt](LICENSES/AMD-GPU-MR-ISA-MIT.txt).
- Code adapted from other KytyPS5 forks (chenxiao07/KytyPS5, TheCruZ/KytyPS5-GTA, both GPL-2.0) carries a "Based on ..." note with its origin
  in the file; ideas taken from other forks name their source in the comment.
- The launcher uses Qt 6.7.2 under the LGPL-3.0, linked dynamically and unmodified:
  [LICENSES/Qt-NOTICE.txt](LICENSES/Qt-NOTICE.txt), [LICENSES/LGPL-3.0-Qt.txt](LICENSES/LGPL-3.0-Qt.txt),
  [LICENSES/GPL-3.0.txt](LICENSES/GPL-3.0.txt).
- The launcher and the emulator's overlays use the Cormorant Garamond, Manrope and JetBrains Mono fonts under the SIL
  Open Font License 1.1 (`LICENSES/OFL-1.1-*.txt`).
- Third-party components in `3rdparty/` and the submodules remain under their own licenses, included with them.
- Ghost of Yōtei, PlayStation and PS5 are trademarks of their respective owners. This project is not affiliated with
  or endorsed by Sony Interactive Entertainment or Sucker Punch Productions.
