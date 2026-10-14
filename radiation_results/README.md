# M1 RK1 and FLD radiation tests

The current local workflow compares **M1 RK1** (`radiation_model=m1`,
`radiation_integrator=euler`) with stationary **FLD backward Euler**
(`radiation_model=fld`). ADER2 is removed. Existing historical campaigns keep
their frozen scripts and results; do not resume them with these entry points.

## Supported problems

| Problem | M1 RK1 | FLD | Reference |
|---|---|---|---|
| Gaussian pulse | Yes | Yes | M1 linear moment evolution; FLD periodic constant-diffusion solution |
| Stationary Marshak heating | Yes | Yes | Su–Olson linear half-space diffusion solution for radiation and material energy |
| Streaming wave and front | Yes | Setup/reporting only; zero-opacity solve does not converge | Exact M1 transport profiles; no analytic FLD profile, with conservation/positivity/flux-bound diagnostics |
| Equilibrium sphere | Yes | Yes | Analytic constant-diffusion Gaussian-emitter equilibrium |
| Subcritical/supercritical shocks and radiating blast | Yes | Unsupported | Numerical profiles and conservation diagnostics |

FLD references include energy and flux. Its flux is derived from the radiation
energy gradient. The Gaussian heat-equation, steady Gaussian-emitter and Su–Olson solutions are
**diffusion-limit references**, not exact solutions of the nonlinear
Levermore–Pomraning flux-limited equations. The website labels reference
assumptions and slice-area reference discrepancies. Marshak additionally
compares material energy; its reference is for a half-space, whereas the
simulation has a finite domain. Gaussian references are cell averages; the
existing Marshak reference is sampled at cell centers. Cross-model differences
are reported separately from each model's reference discrepancies.

FLD currently requires physical light speed, CPU, stationary material, and the
supported ideal-gas/test emission laws. Its composite multigrid/GMRES solve
collects leaves on the root locality. This is a correctness milestone with a
serial root solve, not a distributed scaling or moving-gas implementation.

## Prepare locally without starting simulations

Build the solver and C++ reporting tools first. Paired comparison plots also
require Python 3 with NumPy and Matplotlib. Keep all simultaneous local
builds/tests within the shared resource budget.

```sh
cmake --build cmake-build-hpx2rc-radiation --target octotiger --parallel 2
cmake -S radiation_results -B radiation_results/.build -DBUILD_TESTING=ON
cmake --build radiation_results/.build --parallel 2
ctest --test-dir radiation_results/.build --output-on-failure --parallel 1
python3 radiation_results/run_both.py \
  --output output/m1-fld-local --prepare \
  --build cmake-build-hpx2rc-radiation --results-build radiation_results/.build
```

The C++ tests use synthetic solver output. `--prepare` freezes source, scripts,
binaries and shared libraries; checks all generated configurations with
`--dry-run`; and checks executable startup with `--help`. It does **not** start
any problem simulation. The HPX versioned library and `libhpx.so` alias refer
to the same saved file.
The kernel still loads the host ELF interpreter; its hash is pinned and checked
before execution, so a changed system loader requires preparing a new campaign.

The default [local-study.json](local-study.json) pilot has eight sequential runs:
Gaussian and Marshak, levels 0 and 1, both models, two CPU workers, three
snapshots, final time 0.25 and maximum timestep 0.03125. Gaussian uses seconds;
Marshak uses normalized time. These are short implementation comparisons, not
full-duration production qualification. M1 internally subcycles radiation, so
matching the maximum outer timestep does not imply matching transport steps.
A 10 GiB disk reserve and 8 GiB process-group memory ceiling are enforced.

Inspect `campaign.json`, `commands.txt`, and `index.html` in the output directory.
To print a plan without creating output, replace `--prepare` with `--dry-run`.
Custom `--cases`, `--levels`, `--time`, and `--hard-dt` belong at preparation;
resume cannot change frozen settings. `run_local.py` also accepts
`--models m1` or `--models fld`. Unsupported FLD selections are listed explicitly.

The full-suite preset selects all eight problems at levels 2, 3 and 4 (32, 64,
128 cells per side), with 101 requested snapshots and 12 HPX workers. Each
problem has its own duration: the transport cases use 0.2 s, Marshak uses
normalized time 1, and the shocks/blast use their full template times
(38300, 7500 and 22000 s). Maximum outer steps are one two-hundredth of each
duration. These are starting settings, not a demonstrated timestep convergence
result. M1 still subcycles for transport stability. New FLD runs check output
after each physical step, so their capture cap is one output interval instead
of the M1 driver's five-step batch. Explicit `--time` and `--hard-dt` override
the preset values for every selected problem.

```sh
python3 radiation_results/run_both.py --preset full \
  --output output/m1-fld-full --prepare
```

This currently prepares 33 runnable cases (eight M1 and three stationary FLD
problems at three levels). Fifteen FLD combinations remain explicitly
unavailable: six transparent wave/front cases fail nonlinear convergence at
exactly zero opacity, and nine shock/blast cases require moving-gas coupling.
Adding an opacity floor or turning off hydro would change those problems and
is not a substitute. Preparation performs configuration checks but starts no
benchmark. High-resolution root-gathered FLD remains subject to the configured
memory ceiling. Completed pilot campaigns keep their frozen inputs.

## Start only after authorization

**Preparation is separate from execution. Wait for the user's instruction to
begin.** Once authorized, use the saved runner:

```sh
python3 output/m1-fld-local/.campaign/run_campaign.py \
  --resume output/m1-fld-local --start
```

The runner executes one case/model at a time. Use `--run-limit 2` with
`--resume DIR --start` to pause after two newly completed results for website
staging; the next start continues the remaining plan without a STOP file.
`--threads` may use up to the available logical CPUs, subject to the current
user-directed resource budget. It verifies immutable settings,
source/runtime hashes and completed-run metadata, records each process identity,
and refreshes comparison plots after each successful run. It never deploys a
website. Create `STOP` inside the campaign directory or interrupt the runner to
stop. A later authorized start requires removing `STOP`. A manifest marked
active/running after a crash requires checking the recorded host PID and start
time before recovery; do not blindly launch a duplicate.

Incomplete attempts are retained under `attempts/` when an explicitly resumed
campaign retries. Historical ADER checkpoints are unsupported. The old
`start_scheduled_study.py` name now forwards to this explicit prepare/start
interface and does not start or resume an old scheduled study automatically.

## Local website and movies

Use `--preset movies` to prepare a longer version of the original eight-run
Gaussian/Marshak pilot: Gaussian evolves to 1.5 seconds, Marshak to normalized
time 60, with 121 requested snapshots at levels 0 and 1. The physical windows
show pulse spreading and substantial material heating; they are not a spatial
or timestep convergence study. Stationary M1 and FLD check output after every
physical step. M1 retains its existing accuracy cap and streaming restriction.
The preset uses ten solver workers, leaving two cores for rendering under the
shared twelve-core limit. Render these saved states with `--seconds 60` for a
one-minute movie; this changes playback without inventing intermediate states.

The prepared index is a local preview. After results complete, regenerate
comparisons without starting a solver:

```sh
python3 output/m1-fld-local/.campaign/run_campaign.py \
  --resume output/m1-fld-local --refresh-only
python3 output/m1-fld-local/.campaign/publish_campaign.py \
  output/m1-fld-local --output output/m1-fld-site
```

The publisher stages a new local website directory with a hash receipt. It
checks completed-run provenance and comparison compatibility, rejects synthetic
fixtures, and holds the campaign lock throughout staging. Analytic reference
binaries and norm downloads remain available. Solver binaries, checkpoints, and
private caches are excluded; checkpoint download links become labeled local-only
text. Local page links and asset paths are checked before the complete site is
placed in its destination; an existing nonempty destination is never overwritten.
Remote deployment is a separate explicitly authorized action. The current
publisher can upload the checked stage to `https://www.octotiger.org`:

```sh
python3 radiation_results/deploy_campaign.py output/m1-fld-local \
  --staged-site output/m1-fld-site --deploy
```

It verifies all uploaded file hashes, installs an immutable generation, archives
the previous homepage, and activates the new campaign link under a shared remote
lock. The homepage and campaign are then checked over HTTPS. `--prepare` performs
only local staging; `--inspect` reads remote deployment state. Historical campaign
URLs and results are retained.

Movies are optional, separate work after execution. `campaign_movies.py --campaign
DIR --plan` lists supported completed runs without rendering. Its metadata and
labels distinguish M1's evolved flux from FLD's derived flux. Rendering requires
VisIt/FFmpeg, must fit the shared CPU budget, and is not enabled by preparation.
After rendering, use the current publisher to create a new stage with a movie
gallery and embedded players in the run reports:

```sh
python3 radiation_results/publish_campaign.py output/m1-fld-local \
  --output output/m1-fld-site-movies
```

Only complete movies whose metadata and asset hashes match the saved run are
included. Rendering caches stay local. The gallery reports the available movie
count and stored snapshot times; playback holds saved states without solution
interpolation. The initial pilot saved three states per run. Adding the gallery
does not change solver inputs, saved run metadata, or prior website generations.

See [README_CPP.md](README_CPP.md) for individual runs and data contracts,
[../doc/radiation-fld.md](../doc/radiation-fld.md) for solver limitations, and
[ENSMAN.md](ENSMAN.md) for the M1-only coupled benchmarks. No QB access is needed
for this local workflow.
