# C++ radiation result tools

Build `radiation_results` with CMake; it requires C++20, Silo, OpenSSL,
nlohmann-json and gnuplot for plots. `results.sh` manages a cached tool build.
`run.sh`/`run_live.sh` forward to the C++ runner. `--dry-run` prints/checks the
configuration without running a solver. For paired local work, use the frozen
campaign workflow in [README.md](README.md).

## Individual runs

```sh
radiation_results/run.sh gaussian 0 --build cmake-build-hpx2rc-radiation \
  --radiation-model fld --time .25 --hard-dt .03125 --threads 2 --jobs 2 \
  --snapshots 3 --no-build --no-open --no-movies --dry-run
radiation_results/run.sh marshak 0 --build cmake-build-hpx2rc-radiation \
  --radiation-model m1 --radiation-integrator rk1 --threads 2 --jobs 2 \
  --snapshots 3 --no-build --no-open --no-movies --dry-run
```

Removing `--dry-run` starts a simulation; do so only after authorization.
`--radiation-model` accepts `m1` and `fld`. `--radiation-integrator auto` selects
`euler` for M1 and `backward_euler` for FLD; `rk1` is an alias for `euler`.
ADER2 is rejected. The Gaussian and Marshak model comparisons use CPU, physical
light speed and the same stationary medium. The six remaining problems are
M1-only. `run_ensman.sh` retains the M1 RK1 shock/blast launcher.

A C++ batch contains one model. Resume restores saved model, algorithm version,
flux semantics and physical settings and verifies source/executable provenance.
Use a new batch to change them. `--resume` checks finished outputs and runs only
unfinished cases. Historical ADER batches/checkpoints are not continued.

## References and output

`gen_radiation_reference` generates periodic, filtered Gaussian cell averages.
The M1 branch evolves the linear moment equations; its FLD branch evolves the
constant-coefficient diffusion equation and derives flux from its analytic
gradient. The initial energy is shared. A nonlinear LP FLD result is compared
against the diffusion-limit reference with an explicit qualification.

Marshak compares radiation energy, flux and material energy against the existing
Su–Olson linear two-temperature half-space solution. Its sample locations are
cell centers. LP limiter effects and the finite simulation boundary can produce
reference discrepancies even with fine spatial/time resolution.

FLD export reads accepted states without replacing them with reference values.
Radiation slices have `t,dx,x,y,z,er,fx,fy,fz,er_ref,fx_ref,fy_ref,fz_ref`; material
slices add `egas,egas_ref` after the geometry columns. Reference-free problems
omit reference columns. Plots/exported CSV retain units and native leaf widths.
FLD norms describe the sampled slice area, not a full-volume solver reduction.

FLD diagnostics validate accepted timestep coverage, iteration counts and
nonlinear/linear residual tolerances. Radiation energy uses its boundary/source
ledger; the accepted-step diagnostics also retain material+radiation energy
balance. FLD flux is derived and does not carry M1's evolved-momentum budget.
Comparisons retain distinct analytic-reference and cross-model discrepancies.

`run.json` includes `radiation_model`, `radiation_integrator`,
`radiation_algorithm_version`, `radiation_flux_semantics`, executable hashes,
reference model/qualification, and physical/configuration settings. FLD
reference metadata uses `reference_kind=linear-diffusion-limit` and
`reference_exact_for_solver=false`. Gaussian uses
`reference_model=periodic-constant-diffusion`; Marshak uses
`reference_model=su-olson-linear-halfspace`.

## Checks and auxiliary commands

```sh
ctest --test-dir radiation_results/.build --output-on-failure --parallel 1
python3 -m unittest discover -s radiation_results -p 'test_*.py'
```

The current runner/comparison/publication unit tests use synthetic metadata or
mocked subprocesses. The C++ integration tests use a synthetic solver executable;
passing them is not physical benchmark validation. Application regression tests
in `tests/radiation` run real OctoTiger and require execution authorization.

`results.sh --help` lists `check`, `plot`, `pages`, `movies`, `sessions` and
`validate-serial`. Existing M1 adaptive-grid controls remain available through
`--gradient-refine-field`, `--gradient-refine-mode`, and the related threshold
options. FLD's initial paired pilot is uniform-grid; mixed-grid physics and
performance need separate validation. QB-specific scripts remain separate and
must not be used without a user-provided login budget.
