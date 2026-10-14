# Program Options

## Optional gradient refinement

`gradient_refine_field=none` is the default. Existing problem-specific density,
hydrodynamic, and gravity refinement criteria remain active without modification.
To replace that criterion with gradient tagging, set `unigrid=off` and select
`density` (total mass density) or `radiation_energy` (requires `radiation=on`).
Selecting a field together with `unigrid=on` is rejected instead of silently
ignoring the criterion. No shadow-grid evolution is used.

```ini
unigrid=off
min_level=4
max_level=5
gradient_refine_field=radiation_energy
gradient_refine_mode=relative
gradient_refine_threshold=10
gradient_refine_floor=1e-30
gradient_refine_buffer=0
gradient_refine_coarsen_factor=0.8
```

The example threshold is illustrative; physical field and coordinate units and
the problem determine its useful value. For every inspected cell and each of
its six face neighbors, the indicator is the maximum of:

- `absolute`: `abs(q_neighbor - q_cell) / dx`, in field units per code length;
- `relative`: `abs(log(max(q_neighbor,floor)) - log(max(q_cell,floor))) / dx`,
  in inverse code length.

These are one-sided differences, so an isolated peak is detected even when
opposite neighbors have equal values. The gradient is not multiplied by the
cell width, and it is not a Euclidean norm. `gradient_refine_floor` has the
selected field's units, only affects the logarithmic mode, and does not change
the evolved field. Threshold and floor must be finite and positive.

Any tagged cell refines its entire block. `gradient_refine_buffer` additionally
inspects zero (default), one, or two face-halo cell layers. A refined parent
remains refined while its indicator exceeds `gradient_refine_threshold` times
`gradient_refine_coarsen_factor` (default 0.8; allowed range `(0,1]`). Decisions
are evaluated at the parent's resolution. `min_level` is always enforced and
the new criterion never refines past `max_level`. Set `min_level=max_level-1`
to allow two adjacent active leaf levels; the retained internal tree nodes do
not count as leaves. Neither both-level occupancy nor a prescribed cell-count
ratio is guaranteed by a fixed physical threshold.

`gradient-amr.csv` records physical simulation code time, step, level, leaf
blocks, and leaf cells initially and after each evolution regrid, even with
other diagnostics disabled. Each block contains `INX^3` cells. For a desired
fine/coarse **leaf-cell count** ratio `R`, a fraction `R/(8+R)` of the coarse
blocks must be refined in three dimensions; block quantization can make a
requested ratio impossible. Uniform finest meshes are valid outcomes.

Enabled gradient options and level bounds are saved in checkpoint metadata.
A restart of such a checkpoint restores those settings. Older checkpoints
without these fields retain the supplied configuration (default: disabled).

`radiation_reduced_c_factor` is an opt-in ratio of the radiation transport
light speed to physical `c`. The default is `1` and retains the full-speed
model. Values must be finite and satisfy `0 < factor <= 1`. For example:

```sh
octotiger --config_file=run.ini --radiation_reduced_c_factor=0.1
```

The same name is accepted in configuration files. It does not change the
physical light-speed constant or the physical units of stored radiation flux.
Use a reduced factor only when its approximation is appropriate for the
problem and check sensitivity to that choice.

The factor is distributed with the other options and saved in radiation
checkpoint metadata version 2. A restart restores the saved factor even if
the command line or configuration requests a different value. Version 1
and older checkpoints use factor `1`. A version 2 checkpoint with a missing
or invalid factor is rejected.

## Hydrodynamic time integration

The time integrator is selected with `--time_integrator`:

| Value | Method |
| --- | --- |
| `RK3` | Existing three-stage Runge-Kutta integrator (default). |
| `SSP_MULTISTEP` | Third-order, two-stage strong stability preserving multistep integrator. |

The selection applies to hydrodynamics and its coupled gravity sources. It can
also be set as `time_integrator = SSP_MULTISTEP` in a configuration file. Values
are case-insensitive. For a fresh run, omitting the option preserves the existing integrator.

With `radiation=on`, `SSP_MULTISTEP` is disabled: a requested multistep
selection uses `RK3` for hydrodynamics. Radiation retains its configured
integration method; the multistep scheme is not adapted to radiation. Exact
SSP checkpoint payloads that declare radiation are rejected before advancing.
Restart restores the saved physics: a valid hydro-only SSP checkpoint restores
`radiation=off` and its saved integrator even if the command line requests
radiation. An ordinary radiation restart resolves a requested multistep
selection to `RK3` after restoring the saved radiation setting.

For example:

```sh
./octotiger --config_file=problem.ini --time_integrator=SSP_MULTISTEP
```

New SSP Silo checkpoints preserve the integrator and all state needed for
exact continuation. If the option is omitted on restart, the checkpoint's SSP
integrator is restored automatically. An explicitly conflicting integrator or
incompatible solver configuration is rejected. Older Silo files without the
lossless restart payload retain the legacy loading/bootstrap behavior.

`SSP_MULTISTEP` implements the manuscript's two-step general linear method,
retaining the previous state and its complete right-hand side, including gravity
sources and boundary outflow. The gravitational potential is recomputed for each
new stage density. It uses the exact nonnegative coefficient boundary (`c3 = 0`)
of the manuscript's coefficient family in place of its empirical `c4` fit.

With `alpha = previous_dt / current_dt`, the supported multistep range is
`alpha >= 1/1.9`: timestep growth is capped at 1.9, with no artificial decrease
limit. The timestep selector respects `hard_dt`, the stop time,
and the forward-Euler limits from both the current and cached derivatives:
`current_dt * v <= current_FE_limit` and
`current_dt * v5 <= previous_FE_limit`. The forward-Euler estimate uses the
existing `cfl * dx / wavespeed` calculation. As with SSPRK3, the SSP guarantee
requires that the spatial operator is stable under this forward-Euler bound.

A fresh run uses RK3 once to initialize history. Refinement conservatively
interpolates the previous state and complete derivative, coarsening averages
them, and integrated outflow histories are transferred consistently. Migration
and exact restart preserve this history, including at batch/regrid boundaries.
A step with missing history or no admissible positive timestep uses RK3. The log reports
`SSP_MULTISTEP stages=2 alpha=...` for multistep updates and `stages=3` for RK3
startup or fallback. Two stages reduce right-hand-side evaluations on those
steps; timestep restrictions and startup frequency determine the total cost.

See [SSP multistep integration](ssp-multistep.md) for the coefficients, gravity
energy accounting, history lifetime, stability assumptions, and tests.
The dedicated `--problem=AMR_ADVECTION` verification problem transports a
smooth compact density pulse at unit x velocity with constant pressure.
`--refinement_floor` controls its density-based moving refinement region.
See `test_problems/ssp_multistep/amr_accuracy.py` for the complete validation
configuration, exact cell-average reference, and checkpoint checks.
