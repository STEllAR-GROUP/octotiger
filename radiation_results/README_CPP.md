# C++ radiation results tools

The radiation-results workflow is implemented in C++20. It invokes gnuplot
for still PNG/PDF plots. C++ reads Silo through its C API, writes native VisIt
XML sessions, and invokes VisIt's `-movie -sessionfile` command to render frames.
FFmpeg encodes browser-compatible H.264 MP4s with the recorded simulation timing.
The shipped workflow has no custom Python scripts or Python package dependencies;
VisIt uses its own bundled movie utility and Python runtime internally.
The small shell launchers build and execute the C++ program.

This update targets the September 17, 2026 M1 refactor **with the restored
conservation update**: camelCase regression methods, internal F/c, physical F
in the subgrid and output files. The archive contains results tools only. It
does not replace solver sources, configurations, reference data, or old results.
Existing Python files are left in place for rollback; use the `.sh` commands
below to select the new implementation.

## Install and build

Commands start from HOME. Save the archive in `~/Downloads` first.

```bash
cd ~
sudo apt install build-essential cmake gnuplot-nox ffmpeg libsilo-dev nlohmann-json3-dev libssl-dev libfftw3-dev
unzip ~/Downloads/radiation_results.zip -d ~/octotiger/src/octotiger
~/octotiger/src/octotiger/radiation_results/build_cpp.sh release
```

Build names accept Debug, Release, or RelWithDebInfo with any capitalization.
`CXX` selects the compiler on the first CMake configure; `JOBS` controls build
parallelism. For a custom Silo installation, export `SILO_ROOT` before building.
Still plots use gnuplot's Cairo terminals; gnuplot 6.0 was tested.
Movie rendering uses VisIt 3.4.2. The bundled reference generator and serial solver harness use
C++23 and the checkout's shared radiation headers. FFTW3 is required to build
the generator.

For movies, the tools find `visit` on PATH or use
`~/visit3_4_2.linux-x86_64/bin/visit`. Set `VISIT` or pass
`--visit /absolute/path/to/bin/visit` to choose another installation.
`bin/frontendlauncher` is also accepted. Still plots and session generation do
not require a running VisIt installation.

The project root is `~/octotiger`, the source checkout is
`~/octotiger/src/octotiger`, and application builds are under
`~/octotiger/build/octotiger/$TYPE`. The runner reads INX from the selected
CMake cache. It rebuilds
the application unless `--no-build` is supplied. It does not configure HPX,
CUDA, Kokkos, or the application build itself.

The restored solver update already has the slice-export hook. If it is absent,
inspect and install the hook with:

```bash
cd ~
~/octotiger/src/octotiger/radiation_results/results.sh install --check
~/octotiger/src/octotiger/radiation_results/results.sh install
```

The installer backs up `src/grid.cpp` before adding the hook. It keeps an existing
complete hook and its header. A changed source layout is reported instead of
being patched by guesswork. Rebuild Octo-TIGER after installing a new hook.

## Run

All four tests, level 2 first, then level 3, then level 4:

```bash
cd ~
~/octotiger/src/octotiger/radiation_results/run_live.sh all 2 3 4 release --threads 12
```

Single test and resolution, using the existing executable:

```bash
cd ~
~/octotiger/src/octotiger/radiation_results/run_live.sh wave 2 release --threads 12 --no-build
```

Aliases: `wave`, `front` / `streaming`, `gaussian` / `diffusion`, and `sphere`.
The full names streaming_wave, streaming_front, gaussian_pulse, and
equilibrium_sphere are also accepted. `run.sh` runs the same CGS workflow with
plots and without automatic movies. `--no-silo` disables snapshots as well.

Defaults are a domain of [−3×10¹⁰,+3×10¹⁰] cm on every axis, t=0–4 s,
c=2.99792458×10¹⁰ cm/s, 12 threads, and 61 requested snapshots. `run_live.sh`
defaults to levels 2,3,4 in Debug; `run.sh` defaults to levels 2,3 in Release.
The timestep cap accounts for Octo-TIGER's output-check interval.

The runner reads `~/octotiger/src/octotiger/radiation_results/configs/*.ini`, scales those
templates to CGS, and writes each run's effective configuration as
`run.ini`. Do not replace a template with a generated `run.ini`.
`--rad-reconstruction plm|ppm` passes the choice to a solver that supports that
option; this package adds no reconstruction method to the solver.

The Gaussian test uses the bundled `gen_radiation_reference` C++ generator and
its `OTRAD001` reference format. `build_cpp.sh` builds it next to
`.build/radiation-results`; the runner selects that copy automatically. The
application build is asked to build only `octotiger`, so it does not need a
`gen_radiation_reference` target. An explicit
`--generator /absolute/path/gen_radiation_reference` still overrides the default.
The generator was recovered from the September 13 source patch and adapted to
the current camelCase reference interfaces. It retains the original FFTW
telegraph solution, periodized Gaussian cell averages, Nyquist-plane filtering,
and physical flux units.

If an earlier run stopped with `No rule to make target 'gen_radiation_reference'`,
install this updated archive and `libfftw3-dev`, rebuild the results tools, then
start a fresh batch with `--no-build` to reuse the Octo-TIGER executable that
already compiled. That failure occurred before simulations started and before
`batch.json` was written, so `--resume` does not apply to that failed batch.

By default, each batch is written under
`~/octotiger/radiation_results/results/`. Each batch prints its `index.html`
path and normally opens it. `--no-open`
suppresses opening the browser. Reports and movies appear after each completed
simulation. The batch page refreshes every ten seconds, pauses while a video is
playing, and stops refreshing at completion. This generates static files for
your existing results hosting; it does not upload or deploy them.

## Problem pages

The batch's `index.html` is a responsive 2×2 index with one card per test,
a preview from the highest completed resolution with a plot, and a link to:

- `streaming_wave.html`: periodic exact M1 transport of a sinusoid.
- `streaming_front.html`: periodic exact transport of a discontinuous front.
- `gaussian_pulse.html`: a Fourier **telegraph-equation** reference for a small
  perturbation in a fixed scattering medium.
- `equilibrium_sphere.html`: the steady diffusion reference for a smooth
  Gaussian emitter.

Each problem page includes its setup, equations and averaging procedure,
reference limitations, provenance and literature links, all completed
resolutions, all four field norms and orders, field maps, profiles, every
available movie variant, conservation budgets, and downloadable data.
The original combined report remains at `plots/index.html`; `movies.html`
remains the combined movie gallery. Live updates refresh the index and problem
pages together. Missing runs and missing diagnostics are explicitly identified.
Pages are static and work offline; literature links require a connection.

Descriptions document the reviewed September 17 source. Saved run settings
take precedence over example values. Profile constants such as the streaming
mode are compiled into the solver and are not always recorded in `run.ini`;
the pages distinguish the reviewed definition from recorded run metadata.

To update only the HTML of an existing batch, without running the solver,
gnuplot, or FFmpeg:

```bash
cd ~
BATCH=~/octotiger/radiation_results/results/live-20260917-084723-426280
~/octotiger/src/octotiger/radiation_results/results.sh pages "$BATCH"
```

`plot` and `movies` also refresh the problem pages automatically. The `pages`
command uses existing images and movies; use `plot` to apply a changed plot
style, and `movies` to apply a changed movie style. An already running process
keeps its old reporter until restarted; apply the commands above after it
finishes, or resume after stopping it cleanly.

## Resume

Use the saved batch path, without repeating the case, levels, or build:

```bash
cd ~
~/octotiger/src/octotiger/radiation_results/run_live.sh \
  --resume ~/octotiger/radiation_results/results/live-YYYYMMDD-HHMMSS-XXXXXX \
  --threads 12
```

Completed simulations are validated and reused. The runner checks executable
and recorded source hashes, settings, final time, full-volume norms, slices,
and available conservation records before reusing data. It can read compatible
CGS batches created by `run_live.py`. Changed solver binaries or sources require
a new batch, including when the conservation update changed those sources.

Resume continues a batch, not a partially completed solver simulation. Incomplete
run directories remain untouched and require a new batch. Batches still marked
active are rejected; stop the original runner cleanly before resuming. Ctrl+C
retains completed results and marks the batch inactive.

### Fix for FFmpeg appearing frozen

An earlier version placed subprocesses in separate process groups while leaving
their input attached to the foreground terminal. FFmpeg could then be stopped by
terminal job control. This version redirects subprocess stdin to `/dev/null`,
passes `-nostdin` to FFmpeg, and displays frame-count/percentage progress during
encoding. The full machine-readable FFmpeg progress is saved in `encode.log`.

A tested VisIt 3.4.2 binary sometimes exits with code 250 during shutdown after
successfully saving its frames (also reproduced with a trivial VisIt script).
Only for that exit code, the runner can proceed after VisIt reports completion,
no rendering error is logged, every expected frame exists, and FFmpeg decodes
all frames successfully. It prints the exception and records the exit code in
movie metadata. Missing frames, rendering errors, failed image decoding, and
other nonzero exits stop the movie step.

If encoding is stopped in the previous version, press Ctrl+C and wait for the
shell prompt, install this update, rebuild the results tools, and use `--resume`
with that batch's path. Completed simulations are reused. An updated renderer
may regenerate movie frames; the numerical simulation does not need to repeat.

## Existing results and movies

```bash
cd ~
BATCH=~/octotiger/radiation_results/results/live-YYYYMMDD-HHMMSS-XXXXXX
~/octotiger/src/octotiger/radiation_results/results.sh check "$BATCH"
~/octotiger/src/octotiger/radiation_results/results.sh plot "$BATCH"
~/octotiger/src/octotiger/radiation_results/results.sh movies "$BATCH" --seconds 20 --fps 30
```

`plot`, `check`, and `movies` also accept a single completed run directory.
Plots include all four radiation fields, numerical/reference/error slices,
profiles, full-volume norm convergence, and conservation histories. CSV and JSON
summaries are retained. Convergence requires at least two completed, comparable
resolutions; mixed settings and duplicate resolutions are rejected. Old results
without conservation CSVs can still be plotted, with conservation marked missing.
New simulations require conservation output.

Gaussian energy maps, energy profiles, and `er` movies now display **E − Ebg
on a logarithmic scale**. Subtracting the uniform background makes the spreading
pulse visible. The automatic display spans six decades below the largest
positive excess (numerical/reference slice peak for a plot, all recorded movie
frames for a movie). Movie color limits stay fixed through time. If no positive
excess exists, a finite fallback range is used and every sample stays at the
floor. Values at or below the floor, including zero and negative excesses, are
drawn at the floor and labeled accordingly. Raw CSV/Silo data, norms,
conservation, signed errors, and signed flux displays are unchanged.
For Gaussian `er` movies, `--minimum` and `--maximum` refer to positive **energy
excess**, in physical energy-density units. The display transform and limits
are recorded in the plot/movie metadata. Other tests retain their prior scales.

To update existing plots and movies with this style (no solver rerun):

```bash
~/octotiger/src/octotiger/radiation_results/results.sh plot "$BATCH"
~/octotiger/src/octotiger/radiation_results/results.sh movies "$BATCH"
```

Conservation uses physical E, Fx, Fy, Fz and the residual

    current integral − initial integral + outward boundary budget − source budget.

Every conservation plot is now dimensionless. For a field integral Q, both the
measured total and predicted total `(Q0 - boundary + source)` are divided by
`abs(Q0)`. The residual panel shows the **signed** corrected residual divided
by the same fixed initial scale. No later source growth or boundary transport
can dilute the reported error. Negative initial flux components start at −1.

For zero or roundoff-level initial net flux, dividing by that component is not
meaningful. Those panels instead use `c * abs(initial radiation energy)`, with
the fallback written in the axis labels and report tables. The threshold is
`abs(initial component)/(c*abs(initial energy)) <= 64*double_epsilon`.
A zero initial energy is reported as an undefined normalization.

`conservation-fractions.csv` and its units file contain the dimensionless total,
prediction, residual, boundary, and source histories. Original physical-unit
CSV data and raw summary fields remain available. Explicit axis padding avoids
gnuplot warnings for nearly constant totals; zero residuals are labeled.
A valid ledger is not itself proof that its measured error is acceptably small.

Movie options include:

| Option | Meaning |
|---|---|
| `--field er|fx|fy|fz|fluxmag` | Physical field or flux magnitude |
| `--view slice --axis z --position 0` | Central slice, default |
| `--view 3d` | Field on the domain's exterior surface; no volume rendering |
| `--color hot|viridis|gray|diverging` | Colormap |
| `--minimum X --maximum X` | Fixed color limits; automatic limits span all snapshots |
| `--seconds 20 --fps 30 --hold 1` | Duration, frame rate, endpoint holds |
| `--width 1280 --height 960` | Frame dimensions |
| `--reuse-frames` | Re-encode cached frames after checking their render signature |
| `--allow-sparse` | Explicitly permit fewer snapshots than the saved capture request |
| `--visit PATH --ffmpeg PATH` | Movie executable overrides |
| `--session-dir DIR` | Override the flat session folder |

Movies read numerical `X.*.silo` and `final.silo`; they never substitute
`analytic.silo` for the final solution. Irregular simulation-time gaps are
preserved by holding recorded states. No interpolated states are invented.
The Silo reader supports the rectilinear, zone-centered uniform-grid output of
these regression tests. The C++ reader scans times and global color limits;
VisIt renders the Silo data using those fixed limits. A changed renderer or
movie style requires fresh frames; omit `--reuse-frames` for the first run of
this version. This does not rerun the solver.

## VisIt sessions

All generated sessions live together in one flat directory:

```
radiation_results/visit_sessions/
```

A typical filename is:

```
live-20260917-084723-426280--gaussian_pulse--l2-n32--er-slice-z-at0--6be79c426ace.session
```

The name identifies batch, problem, resolution, field, view, and slice position.
A short hash distinguishes data paths and display settings. The matching `.visit`
file lists the numerical snapshots and `.json` records times, limits, units,
and display settings; these companions are in the same folder, with no nested
batch or problem directories. Identical inputs/settings reuse the same name.
Session references are absolute paths on the machine where they are generated.
Keep the Silo data in place, or regenerate sessions after moving the results.

Every `movies` invocation creates its session before rendering. To create only
the sessions, without rendering frames or encoding movies:

```bash
cd ~
BATCH=~/octotiger/radiation_results/results/live-YYYYMMDD-HHMMSS-XXXXXX
~/octotiger/src/octotiger/radiation_results/results.sh sessions "$BATCH"
```

The same field/view/color options apply to `sessions` and `movies`. To inspect,
open VisIt, choose File → Restore session, and select a file in that folder.
Or pass its exact filename:

```bash
cd ~
SESSION=~/octotiger/radiation_results/visit_sessions/NAME.session
~/visit3_4_2.linux-x86_64/bin/visit -sessionfile "$SESSION"
```

The session contains the complete time series, selected variable, slice or 3D
camera, fixed color range, annotations, and Gaussian logarithmic energy-excess
expression. Use Save session as with a new name to preserve manual edits;
regeneration replaces the tool's generated file for that same name.

For a 3D inspection session without making a movie:

```bash
~/octotiger/src/octotiger/radiation_results/results.sh sessions "$BATCH" --view 3d
```

The generated 3D view is a Pseudocolor exterior surface. VisIt's GUI lets you
change operators, slices, views, or plot types interactively.

## Validation

```bash
cd ~
cmake -S ~/octotiger/src/octotiger/radiation_results \
  -B ~/octotiger/src/octotiger/radiation_results/.build -DBUILD_TESTING=ON
cmake --build ~/octotiger/src/octotiger/radiation_results/.build -j 4
ctest --test-dir ~/octotiger/src/octotiger/radiation_results/.build --output-on-failure
```

Verified with GCC 13.3, CMake 3.28, Silo 4.11, and gnuplot 6.0:

- 97 reader, fixed-initial conservation, scheduling, SHA256, multiblock Silo,
  and flat session naming/content checks.
- 29 integration checks: real gnuplot output, batch completion, resume without
  rerunning completed simulations, legacy workflow metadata, and changed
  source/executable/settings rejection; an all-case batch in an application
  build with only the `octotiger` target also invokes the real bundled FFTW
  generator. The solver is a test fixture here.
- Dedicated pages cover all problems and completed resolutions, and HTML-only
  regeneration reuses images without requiring gnuplot or FFmpeg. Gaussian
  log plots handle nonpositive excess values with a finite, labeled floor.
- The index and problem pages were checked in a headless browser at desktop
  and mobile widths, including navigation, local links, and image loading.
  A Gaussian energy-excess MP4 was encoded with a fixed logarithmic color scale
  and mixed positive/nonpositive samples; the all-nonpositive plot fallback
  also rendered successfully. These checks use synthetic fixtures.
- PNG/PDF still output, plus native VisIt session restore and movie rendering
  on synthetic multiblock data: all three slice axes, logarithmic Gaussian
  energy excess, and 3D flux magnitude. Paths with spaces are included.
- The bundled generator's complete 32³, t=4 s CGS reference file matched the
  recovered original generator byte for byte.
- A real controlling-terminal regression verifies noninteractive child stdin,
  successful FFmpeg encoding, visible 100% completion, and the saved MP4. A
  near-constant CGS conservation fixture also renders without gnuplot warnings.
  This automated terminal test uses a VisIt test double; real VisIt rendering
  is checked separately.
- The serial launcher compiled extracted production methods from the restored
  conservation update and ran all four branches on an 8³ grid in CGS. The
  Gaussian branch used a uniform-background reference fixture, testing execution
  and conservation I/O rather than Gaussian-pulse accuracy.
  The C++ readers accepted its norms, slices, and conservation histories; the
  reported residuals were at floating-point roundoff scale under the previous
  normalization. The updated fixed-initial normalization has dedicated fixtures.

Full HPX/MPI/CUDA application execution has not been verified in this environment.

To use the serial launcher with a real matching reference file:

```bash
cd ~
~/octotiger/src/octotiger/radiation_results/results.sh validate-serial \
  --reference /absolute/path/reference.bin --output ~/radiation-serial-results \
  --cells 32
```

Use `--cxx g++` to select its compiler and `--sanitize` for ASan/UBSan. This
extracts selected solver methods and uses fixtures for options and geometry;
it does not replace a distributed application test.

## Script mapping

| Previous Python script | C++ entry point |
|---|---|
| `run_live.py` | `run_live.sh`, or `results.sh live` |
| `run.py` | `run.sh`, or `results.sh run` |
| `plot.py` | `results.sh plot` |
| HTML-only regeneration | `results.sh pages` |
| `results.py` | `cpp/data.cpp`, `results.sh check` |
| `movies.py`, `visit_movie.py`, `movie_support.py` | `results.sh movies`, `cpp/movie.cpp` |
| VisIt session generation | `results.sh sessions`, `cpp/visit_session.cpp` |
| `install.py` | `results.sh install` |
| `validate_serial.py` | `results.sh validate-serial` |

`results.sh --help` prints the full command summary. The launcher rebuilds when
C++ sources or CMakeLists change and otherwise uses the cached executable.
