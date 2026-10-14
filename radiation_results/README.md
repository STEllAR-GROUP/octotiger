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
| Streaming wave and front | Yes | Unsupported | Existing streaming references |
| Equilibrium sphere | Yes | Unsupported | Existing equilibrium reference |
| Subcritical/supercritical shocks and radiating blast | Yes | Unsupported | Numerical profiles and conservation diagnostics |

FLD references include energy and flux. Its flux is derived from the radiation
energy gradient. The Gaussian heat-equation solution and Su–Olson solution are
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

## Start only after authorization

**Preparation is separate from execution. Wait for the user's instruction to
begin.** Once authorized, use the saved runner:

```sh
python3 output/m1-fld-local/.campaign/run_campaign.py \
  --resume output/m1-fld-local --start
```

The runner executes one case/model at a time. It verifies immutable settings,
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
Remote deployment is a separate action. Historical websites are not relabeled.

Movies are optional, separate work after execution. `campaign_movies.py --campaign
DIR --plan` lists supported completed runs without rendering. Its metadata and
labels distinguish M1's evolved flux from FLD's derived flux. Rendering requires
VisIt/FFmpeg, must fit the shared CPU budget, and is not enabled by preparation.

See [README_CPP.md](README_CPP.md) for individual runs and data contracts,
[../doc/radiation-fld.md](../doc/radiation-fld.md) for solver limitations, and
[ENSMAN.md](ENSMAN.md) for the M1-only coupled benchmarks. No QB access is needed
for this local workflow.
