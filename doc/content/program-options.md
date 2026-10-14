# Program Options

The time integrator is selected with `--time_integrator`:

| Value | Method |
| --- | --- |
| `RK3` | Existing three-stage Runge-Kutta integrator (default). |
| `SSP_MULTISTEP` | Third-order, two-stage strong stability preserving multistep integrator. |

The selection applies to hydrodynamics and its coupled gravity sources. It can
also be set as `time_integrator = SSP_MULTISTEP` in a configuration file. Values
are case-insensitive. For a fresh run, omitting the option preserves the existing integrator.

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
