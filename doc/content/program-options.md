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
`0.7 <= alpha <= 4`. The timestep selector respects `hard_dt`, the stop time,
and the forward-Euler limits from both the current and cached derivatives:
`current_dt * v <= current_FE_limit` and
`current_dt * v5 <= previous_FE_limit`. The forward-Euler estimate uses the
existing `cfl * dx / wavespeed` calculation. As with SSPRK3, the SSP guarantee
requires that the spatial operator is stable under this forward-Euler bound.

The first step at each scheduled batch boundary uses RK3 to initialize history
after regridding. Migration preserves histories. Restarting inside a batch
continues with the saved history and original batch phase; it does not add an
RK3 startup step. A step whose history or timestep ratio is unsuitable uses RK3. The log reports
`SSP_MULTISTEP stages=2 alpha=...` for multistep updates and `stages=3` for RK3
startup or fallback. Two stages reduce right-hand-side evaluations on those
steps; timestep restrictions and startup frequency determine the total cost.

See [SSP multistep integration](ssp-multistep.md) for the coefficients, gravity
energy accounting, history lifetime, stability assumptions, and tests.
