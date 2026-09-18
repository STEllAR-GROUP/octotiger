# Radiation results for the new Octo-TIGER solver

Run the four new radiation problems at several resolutions, then inspect paired
numerical/reference plots, signed-error maps, and convergence. The scripts use
the new four-field state: E and physical Fx,Fy,Fz in code units.

The original archive is preserved under `legacy/`. Its old executables, VisIt
sessions, and results are historical data. The current runner uses your checkout's
built executable. Old and new diffusion/sphere results use different problem
definitions and are not one convergence series.

## Install and run

This expects the radiation CTest/reference update from our last session to be
installed already. Put this `test_results` directory inside your Octo-TIGER
checkout. If an older directory is there, move it aside before extracting the ZIP.

From `~/workspace/octotiger`:

```bash
python3 test_results/install.py .
python3 -m venv test_results/.venv
source test_results/.venv/bin/activate
python3 -m pip install -r test_results/requirements.txt
bash test_results/run.sh all 2 3 4 release
```

Use `debug` in place of `release` for that build. The runner rebuilds the required
targets in your already configured CMake build. It reads INX from that build's
`OCTOTIGER_WITH_GRIDDIM` cache entry; it does not assume INX=8. With INX=8, levels
2, 3, 4 mean 32³, 64³, 128³ cells. Both minimum and maximum levels are fixed,
with `unigrid=on`. Higher resolution requires more memory and timesteps.
All selected cases run at the lowest level first, then at each higher level.

For **movies as well as still plots**, add `--movies`:

```bash
bash test_results/run.sh all 2 3 4 release --movies
```

This targets 61 Silo states and creates a 20-second MP4 after each completed
case/level. It uses `hard_dt` as well as `odt` to control capture cadence.
VisIt and FFmpeg are required for rendering. See [MOVIES.md](MOVIES.md) for
installation, line-by-line command explanations, longer movies, and rendering
on another machine. The four test configurations use `omega=0`.

If the new generator target has not yet appeared in your build, first run your
normal configuration command (`./build.sh Release` or `./build.sh Debug`).
The Gaussian reference generator requires FFTW3 development files, as before.

The command prints the results directory at the start and the report's full path
when done. Open that `plots/index.html` in Firefox or another browser. It works
offline and lets you expand each resolution and its flux maps.

Individual cases and faster convergence-only runs:

```bash
bash test_results/run.sh wave 2 3 4 release
bash test_results/run.sh streaming 2 3 4 debug
bash test_results/run.sh diffusion 2 3 4 release --no-silo
bash test_results/run.sh sphere 2 3 4 release --threads 8
```

`--no-silo` retains the CSV slices and full-volume norms; it skips Silo output.
`--no-build` reuses an already rebuilt executable. `--jobs` controls build
parallelism; `--threads` controls HPX worker threads. Debug does not imply ASan,
so the old unconditional ASan preload has been removed.

Use `--root /path/to/octotiger` if this directory is elsewhere. Use
`--build RelWithDebInfo` for a differently named build directory, and
`--generator /path/to/gen_radiation_reference` if there is more than one
generator executable beneath it.

To choose a batch name or final time:

```bash
bash test_results/run.sh all 2 3 release --output test_results/results/check1
bash test_results/run.sh diffusion 2 3 release --time 0.4 --odt 0.1
```

The specified output directory must be new or empty. Otherwise the runner stops.
Default runs end at t=0.2, with Silo output interval 0.05. The Gaussian reference
is generated separately for **every resolution and requested final time**. No
reference interpolation in time occurs. All reference parameters come from the
saved configuration; these supplied tests normalize the true light speed to c=1.

Preview a run plan without building, executing, or writing results:

```bash
bash test_results/run.sh all 2 3 4 release --dry-run
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
python3 test_results/plot.py test_results/results/check1
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
and history plots. `run_live.py` also adds a Conservation link and expandable
plot to its live page. Normalization uses the energy budget scale and c times
that scale for flux, with component budget scales where larger, so an initially
zero net flux does not cause division by zero. The report states the exact
normalization. CSV values use the simulation units; `run_live.py` uses CGS,
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
python3 test_results/run_live.py sphere 2 release --threads 12 \
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

`install.py` inserts one optional capture hook in `grid::compute_analytic` and
installs `octotiger/test_problems/radiation/plot_output.hpp`. It first matches the
expected function context and saves `src/grid.cpp.before-radiation-plots`.
It is idempotent and refuses conflicting files. Use `python3 test_results/install.py --check .` from the checkout
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
cancellation, and deliberate unaccounted drift. A production-grid-to-CSV-to-reader
roundtrip checks the cumulative budgets and physical-flux units:

```bash
python3 -m unittest discover -s test_results/tests -v
```

`validate_serial.py` is an optional validation harness. It compiles the production
methods with a serial mesh fixture and requires a matching generated binary
reference. It is separate from the normal `run.sh` workflow.

The broader grid checks also exercise physical storage, flux restriction, matter
coupling, and AMR means in Debug, Release, and UBSan configurations:

```bash
python3 tests/validateRadiationGrid.py
```
