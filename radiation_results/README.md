# Radiation results for the new Octo-TIGER solver

## Opt-in gradient AMR

The uniform-grid workflow remains the default. Gradient runs allow exactly two
adjacent leaf levels: the positional level is the maximum, and the minimum
defaults to one less. For example, this prepares a short front run without
launching it:

```sh
radiation_results/run.sh front 3 --build cmake-build-hpx2rc-radiation \
  --gradient-refine-field radiation_energy --gradient-refine-mode absolute \
  --gradient-refine-threshold 1e-10 --gradient-refine-floor 1e-30 \
  --gradient-refine-buffer 2 --gradient-refine-coarsen-factor 0.8 \
  --time 0.02 --snapshots 3 --threads 2 --jobs 2 --no-build --no-open \
  --no-movies --no-silo --dry-run
```

The threshold above is illustrative, not an accuracy-approved setting. Absolute
mode measures a field gradient per code length; relative mode measures the
gradient of `log(max(field,floor))`, also per code length. The runner uses CGS,
so these are per cm, not cell-to-cell dimensionless jumps. Available fields are
`density` and `radiation_energy`; the latter is total radiation energy, without
background subtraction. `--min-level` can explicitly set the adjacent lower
level. All controls are recorded and restored by batch resume.

Adaptive reports read actual whole-mesh block/cell counts from
`gradient-amr.csv`. Raw final CSV files preserve native leaf widths and have
level-qualified filenames. Display plots project constant leaf averages onto
finest pixels, after checking that leaf footprints tile the plane without gaps
or overlaps. They show positive-side cells touching geometric z=0, whose centers
are at `z=leaf_dx/2`; mixed levels therefore do not share a cell-center plane.
Full-volume reference norms use each leaf's own cell average and volume. Uniform
grid convergence orders are suppressed for adaptive runs. AMR accuracy and cost
must be established against a uniform run at the same finest resolution.

## QB4 levels 2–5

The offline-first cluster workflow builds Octo-TIGER, runs all seven problems at
levels 2–5, and collects their plots and webpages. Start with
[`qb4/README.md`](qb4/README.md); `workflow.py prepare` creates the plan without
contacting Slurm. Submission requires a separate explicit `--submit` command.

## Coupled shocks and blast wave

Full-duration launcher (from the repository root):

```sh
radiation_results/run_ensman.sh 2 --dry-run
radiation_results/run_ensman.sh 2 --output radiation_results/results/ensman-full-l2
```

The second command runs the real simulations: blast → supercritical →
subcritical, sequentially, using their full configured durations, two workers,
two build jobs, and 21 Silo snapshots. It updates the batch website after each
completed case. Level 2 is a coarse first run. The shock cases require roughly
1.23 million and 6.30 million steps at this resolution with the physical light
speed; no long runs were launched while setting up the scripts.
See [ENSMAN.md](ENSMAN.md#full-duration-local-runs) for individual cases,
resolutions, result locations, and the limits of batch resume.


The local suite also includes `subcritical` (`subcritical_shock`), `supercritical`
(`supercritical_shock`), and `blast` (`radiating_blast`). `all` now selects all seven
problems. The three additions enable radiation hydrodynamics and retain their
physical CGS configuration directly; they are not rescaled to the radiation-only
6 × 10¹⁰ cm box. Their template final times are 38300, 7500, and 22000 s.
`--time` explicitly overrides the duration for each selected case.
`--radiation-reduced-c-factor F` opts the coupled cases into the
Skinner-Ostriker reduced-speed-of-light approximation, with 0 < F ≤ 1 and
default F=1. The four analytic cases retain F=1 even in an `all` batch. The
runner saves both speeds and the factor, preserves it on batch resume, and
labels modified-equation conservation budgets in the generated pages. Factor
convergence must be checked before drawing physical conclusions. The QB
configuration supports separate factors for the three coupled cases.

These are Cartesian adaptations of the problems in [Ensman (1994), *Test Problems
for Radiation and Radiation-Hydrodynamics Codes*](https://doi.org/10.1086/173889),
§§5.1–5.3. The original paper uses a spherical moving piston for the shocks and a
hollow, stratified shell with innermost-zone energy deposition for the blast.
The local shocks use a stationary reflecting planar wall and incoming gas. The
blast uses spherically symmetric initial profiles on the Cartesian mesh, a
filled core, Gaussian deposition width, and approximate LTE diffusion radiation
field with a realizability floor. The new web pages document these differences,
the original numerical targets, and each local run's actual parameters.

There is no supplied exact analytic profile for these coupled problems. Reports
show gas and radiation temperatures, density, velocity, radiation energy, signed radiation flux, and reduced flux as
numerical profiles, without invented reference curves, error norms, or convergence
orders. A run ending before its template final time is visibly labeled an
**early-time implementation check**. Such a run does not establish a developed
subcritical/supercritical structure or quantitative agreement with the paper.

For a deliberately short local preview, after building Octo-TIGER with radiation:

```sh
radiation_results/run.sh subcritical 1 --build cmake-build-hpx2rc-radiation \
  --time 0.01 --snapshots 3 --threads 2 --jobs 2 --no-build --no-open --no-movies --no-silo
```

The runner creates `radiation-hydro-slices/`, which opts the application into final
numerical CSV output independently of analytic comparisons. The CSV schema is
`t,dx,x,y,z,rho,vx,vy,vz,gas_temperature,radiation_temperature,er,fx,fy,fz`.
The full central plane is validated for cell coverage, time, spacing, positivity,
and radiation realizability. Temperature units are K; the default configurations
use CGS for the other fields. Plots use the plane at z=Δx/2 and the ray at
y=z=Δx/2. Both are half-cell offsets, not exact central-axis samples.

The C++ reader and integration tests exercise the new data validation, seven-page
navigation, preserved physical time/domain, and absence of fabricated accuracy
norms. Their synthetic fixtures test the reporting pipeline, not solver physics.
All generated pages remain local until separately published.


> **hpx2rc-radiation restoration:** These tools and the complete workflow reference
> below are restored from `dominic-wip2` onto `7485b798`. See
> [tests/README.md](../tests/README.md) for solver provenance and validation.
> The current tools use C++20. Default build jobs and HPX workers are **2**;
> concurrent work shares the local resource budget. FFmpeg decoder/filter/
> encoder worker counts are explicitly limited to one. No QB4 access is needed.
>
> This flat checkout and the older `PROJECT/src/octotiger` layout are both
> supported. Older directory examples below describe the latter layout.
> From this checkout, build and test without copying an archive:
>
> ```sh
> cmake -S radiation_results -B /tmp/octotiger-radiation-results -DBUILD_TESTING=ON
> cmake --build /tmp/octotiger-radiation-results --parallel 2
> ctest --test-dir /tmp/octotiger-radiation-results --output-on-failure --parallel 1
> radiation_results/run.sh wave 2 --build /path/to/application/build --threads 2 --jobs 2 --no-build --dry-run
> ```
>
> The normal shell launchers build `radiation_results/.build` on demand. The
> standalone Python solver-validation scripts remain available alongside the
> C++ results interface. Reference generation is also integrated into the
> application and standalone CTest suites through `reference.cmake`.

Run the four new radiation problems at several resolutions, then inspect paired
numerical/reference plots, signed-error maps, and convergence. The scripts use
the new four-field state: E and physical Fx,Fy,Fz in code units.

The current runner uses your checkout's built executable. Archived executables,
VisIt sessions and results are not installed by this restoration. Old and new
diffusion/sphere results use different problem definitions and are not one
convergence series.

## Install and run

The radiation CTest/reference update is included in this checkout. Put this `radiation_results` directory under
`~/octotiger/src/octotiger`. The launchers treat `~/octotiger` as the project
root, `~/octotiger/src/octotiger` as the source checkout, and
`~/octotiger/build/octotiger/$TYPE` as the application build directory.

From `~/octotiger`:

```bash
src/octotiger/radiation_results/results.sh install
src/octotiger/radiation_results/run.sh all 2 3 4 release
```

Use `debug` in place of `release` for that build. The runner rebuilds the required
targets in your already configured CMake build. It reads INX from that build's
`OCTOTIGER_WITH_GRIDDIM` cache entry; it does not assume INX=8. With INX=8, levels
2, 3, 4 mean 32³, 64³, 128³ cells. Both minimum and maximum levels are fixed,
with `unigrid=on`. Higher resolution requires more memory and timesteps.
All selected cases run at the lowest level first, then at each higher level.

For **movies as well as still plots**, add `--movies`:

```bash
bash radiation_results/run.sh all 2 3 4 release --movies
```

This targets 61 Silo states and creates a 20-second MP4 after each completed
case/level. It uses `hard_dt` as well as `odt` to control capture cadence.
VisIt and FFmpeg are required for rendering. See the movie options in [README_CPP.md](README_CPP.md) for
rendering controls and dependencies. The four test configurations use `omega=0`.

If the new generator target has not yet appeared in your build, first run your
normal configuration command (`./build.sh Release` or `./build.sh Debug`).
The Gaussian reference generator requires FFTW3 development files, as before.

The command prints the results directory at the start and the report's full path
when done. Open that `plots/index.html` in Firefox or another browser. It works
offline and lets you expand each resolution and its flux maps.

Individual cases and faster convergence-only runs:

```bash
bash radiation_results/run.sh wave 2 3 4 release
bash radiation_results/run.sh streaming 2 3 4 debug
bash radiation_results/run.sh diffusion 2 3 4 release --no-silo
bash radiation_results/run.sh sphere 2 3 4 release --threads 2
```

`--no-silo` retains the CSV slices and full-volume norms; it skips Silo output.
For new batches it also removes the snapshot-cadence timestep cap, since no
snapshots are being written. An explicit `--hard-dt` or positive template
`hard_dt` remains an upper bound, and the solver still applies its physical
CFL limit. Resuming an older batch retains that batch's original timestep policy.
`--no-build` reuses an already rebuilt executable. `--jobs` controls build
parallelism; `--threads` controls HPX worker threads. Debug does not imply ASan,
so the old unconditional ASan preload has been removed.

Use `--root /path/to/project` if the project directory is elsewhere. Use
`--build RelWithDebInfo` for `PROJECT/build/octotiger/RelWithDebInfo`, and
`--generator /path/to/gen_radiation_reference` if there is more than one
generator executable beneath it.

To choose a batch name or final time:

```bash
src/octotiger/radiation_results/run.sh all 2 3 release --output radiation_results/results/check1
src/octotiger/radiation_results/run.sh diffusion 2 3 release --time 0.4 --odt 0.1
```

The specified output directory must be new or empty. Otherwise the runner stops.
Default runs end at t=0.2, with Silo output interval 0.05. The Gaussian reference
is generated separately for **every resolution and requested final time**. No
reference interpolation in time occurs. All reference parameters come from the
saved configuration; these supplied tests normalize the true light speed to c=1.

Preview a run plan without building, executing, or writing results:

```bash
bash radiation_results/run.sh all 2 3 4 release --dry-run
```

## Files you can inspect

Each batch lives in `results/<UTC timestamp>/`. Each case/resolution has its own
working directory, for example `streaming_wave/l2/`.

| File | Contents |
|---|---|
| `run.ini`, `run.json`, `run.log` | Exact configuration, command, metadata, completion status, and full log |
| `L1.dat`, `L2.dat`, `Linf.dat` | Original full-domain error norms from Octo-TIGER |
| `radiation-conservation.csv` | Full-domain E, Fx, Fy, Fz integrals, cumulative outward boundary transport, and cumulative source changes |
| `radiation-slices/slice-*.csv` | Per-leaf final cell layer, including x,y,z,dx,t and four numerical/reference pairs at 17-digit precision |
| `final.silo` | Numerical final state, when Silo is enabled |
| `analytic.silo` | Reference final state, when Silo is enabled |
| `X.*.silo` | Time snapshots, when Silo is enabled |
| `reference.bin` | Gaussian initial/final Fourier reference, for the Gaussian case |
| `plots/index.html` at batch root | Offline report with all case/resolution plots |
| `plots/errors.csv`, `plots/errors.json` | Full-volume errors and adjacent-resolution observed orders for all four fields and all three norms |
| `plots/<case>/l<level>/slice.csv` | Combined portable slice table, including signed errors |
| `plots/<case>/l<level>/*.png`, `*.pdf` | Numerical/reference/error maps and line profiles |
| `plots/<case>/l<level>/conservation.png`, `conservation.pdf`, `conservation.csv` | Radiation totals and conservation balance histories |
| `movies.html` at batch root | Movie gallery, with `--movies` |
| `movies/er-slice-z/movie.mp4` inside each case/level | Time evolution, with saved PNG frames and rendering metadata |

Replot an existing batch without rerunning the solver:

```bash
radiation_results/results.sh plot radiation_results/results/check1
```

Replotting checks the actual log, norm files, and CSV cells. A nonzero executable
exit fails the run. Missing or duplicate completion markers, incorrect final time,
nonfinite values, mixed resolutions, duplicated/missing slice cells, and appended
norm records are rejected. Incomplete runs are marked failed and omitted from
plots; completed runs from that batch can still be replotted.

## What the errors mean

### Radiation conservation

After rebuilding Octo-TIGER, every radiation run writes
`<datadir>/radiation-conservation.csv`, independently of `disable_diagnostics`
and Silo output. Only active leaf cells contribute to the volume integrals;
covered parent grids and ghost cells are excluded. The run driver records an
initial sample, samples between batches of timesteps, and the final numerical
state before analytic comparison replaces the fields.

For each Q = integral(E, Fx, Fy, or Fz), the balance error is

```
balance error = Q(t) - Q(initial) + outward boundary transport - source change
```

Boundary transport uses the numerical face fluxes after AMR flux correction,
with the same timestep and face area used by the solver. Only physical exterior
faces count; periodic and internal interfaces do not. Sources record the actual
cell changes from emission, scattering, matter coupling, and rotation of the
flux components when applicable. Radiation momentum is integral(F)/c² for
constant c. The diagnostic measures the radiation equation balance; it does not
independently verify conservation of combined gas-plus-radiation energy or
momentum.

The streaming tests should retain their raw energy and flux integrals. In the
Gaussian and sphere tests, scattering can change flux; the sphere also emits
radiation and exchanges it through the boundary. Such physical changes appear
in the budget and are not counted as conservation errors.

Each resolution in `plots/index.html` includes totals, raw changes, source and
boundary contributions, signed balance errors, normalized final/maximum errors,
and history plots. `run_live.sh` also adds a Conservation link and expandable
plot to its live page. Normalization uses the energy budget scale and c times
that scale for flux, with component budget scales where larger, so an initially
zero net flux does not cause division by zero. The report states the exact
normalization. CSV values use the simulation units; `run_live.sh` uses CGS,
giving integral(E) in erg and integral(F) in erg cm/s.

Pending budgets are collected before regridding so they are not lost when
leaves are removed or moved. Totals are sampled after regridding so any change
introduced by remapping remains visible in the balance. A restarted executable
starts a new diagnostic baseline at its restart time and replaces the CSV in
that run directory; preserve the previous run directory to retain its history.
The regression plotting tools expect complete runs starting at t=0.

Older results lacking the CSV display **Conservation data unavailable**. These
budgets cannot be reconstructed from a final slice or existing error norms;
rebuild and rerun to obtain them. New runs through the updated runners require
the CSV, which also catches accidentally using an older binary with `--no-build`.

For a single CGS sphere run including plots and movies:

```bash
radiation_results/run_live.sh sphere 2 release --threads 2 \
    --visit "$HOME/visit3_4_2.linux-x86_64/bin/frontendlauncher"
```

### Reference errors

For e = numerical cell average minus reference cell average:

```
L1   = sum(|e| * cell volume) / domain volume
L2   = sqrt(sum(e^2 * cell volume) / domain volume)
Linf = max(|e|)
p    = log(error_coarse / error_fine) / log(dx_coarse / dx_fine)
```

The convergence norms come from the **whole 3D domain**, not the displayed slice.
Orders compare adjacent resolutions at the same final time. Zero errors have no
reported order; negative orders are retained. The Δx and Δx² curves are visual
guides, not fits or acceptance thresholds.

The plots show the cell layer at z=Δx/2, the layer immediately above z=0. Line
profiles use y=z=Δx/2. This is explicit because an even grid has no cell center
exactly at y=z=0. Both the numerical and reference data use the same cells. The
layer changes with resolution and tends toward the center plane. No interpolation
or averaging of unrelated rows is hidden in the plots.

For Gaussian and sphere energy plots, the uniform background is subtracted from
both displayed solutions so their structure is visible. Signed errors remain
absolute errors. No division by a nearly zero flux/reference is used. Numerical
and reference maps share a color scale at each resolution; the error map has its
own symmetric scale. Values in all figures/CSV files are code units, while Silo
fields follow Octo-TIGER's output-unit conversions.

| Case alias | Current problem | Reference |
|---|---|---|
| `wave` | `RADIATION_STREAMING_WAVE` | Exact periodic nonlinear M1 wave, direction (2,-3,1)/sqrt(14), matched cell averages |
| `streaming` or `front` | `RADIATION_STREAMING_FRONT` | Exact advected periodic square pulse with overlap-integrated cell averages |
| `diffusion` or `gaussian` | `RADIATION_GAUSSIAN_PULSE` | Periodic, linearized telegraph solution generated with FFTW3 |
| `sphere` | `RADIATION_EQUILIBRIUM_SPHERE` | Steady diffusion solution for the matching Gaussian bulb, with analytic boundary values |

The Gaussian and sphere references are linearized/diffusion limits, so nonlinear
model error can eventually limit refinement. The RK1 time update can limit smooth
problems to first order. A moving discontinuity can converge at different rates
in L1, L2, and Linf. These figures expose those rates; they do not label a run
passing merely because it finished. The original CTest tolerance checks remain
available separately:

```bash
ctest --test-dir release -R '^radiation\.' --output-on-failure
```

## VisIt

Open the new run's `final.silo` or the `X.*.silo` time series. Add a Pseudocolor
plot of `er` and a Slice operator for a central plane. Open `analytic.silo`
separately for the reference. Flux fields are `fx`, `fy`, and `fz`. Keep each
`.silo.data` directory beside its matching `.silo` file; the root file references
those blocks. The new runner retains the `final` and `analytic` names so the
reference is not accidentally appended as another numerical movie frame.

The legacy VisIt sessions reference old locations/fields. Start with the new
files when saving a session for these tests.

## Source change and validation

`results.sh install` inserts one optional capture hook in `grid::compute_analytic` and
installs `octotiger/test_problems/radiation/plot_output.hpp`. It first matches the
expected function context and saves `src/grid.cpp.before-radiation-plots`.
It is idempotent and refuses conflicting files. Use `radiation_results/results.sh install --check --root .` from the checkout
to print the source patch.

Capture is enabled only for a new radiation regression problem when
`<datadir>/radiation-slices/` already exists. The runner creates that directory.
The hook reads numerical fields before `compute_analytic` replaces them with the
reference, and writes one file per leaf intersecting the selected plane. Normal
CTest runs do not create this directory and therefore do not export slices.
For multiple localities, the output directory must be shared across them. The
provided runner starts one application process using the requested HPX threads.

The serial validation harnesses compile the production transport/source methods
on a uniform mesh. They check local numerics and diagnostics; they do not test
HPX actions, distributed boundaries, or checkpoint/regrid scheduling. Full
application builds/runs require your configured machine.

The included tests check stale and invalid data, norm/time/grid matching,
observed-order calculation, concurrent per-leaf output, and capture before
replacement. Production conservation tests cover interior-only sums, all six
boundary signs, periodic transport, source and rotation budgets, adjacent-block
cancellation, and deliberate unaccounted drift. A production-grid-to-CSV
roundtrip checks cumulative budgets and physical-flux units; the C++ reader
suite separately verifies its ingestion and presentation:

```bash
python3 -m unittest discover -s radiation_results/tests -v
```

`validate_serial.py` is an optional validation harness. It compiles the production
methods with a serial mesh fixture and requires a matching generated binary
reference. It is separate from the normal `run.sh` workflow.

The broader grid checks also exercise physical storage, flux restriction, matter
coupling, and AMR means in Debug, Release, and UBSan configurations:

```bash
python3 tests/validateRadiationGrid.py
```
