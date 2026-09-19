# M1 radiation: S&O transport, RSLA and subcycling

This patch is based on the supplied `code.tar(20260919-033724).gz`. It replaces the active M1 transport and coupling path. Hydrodynamic reconstruction, Riemann solvers, source evaluation, and RK stages are unchanged. The node driver gains radiation timestep metadata, the radiation contribution to its acoustic timestep bound, and communication safeguards.

## Numerical method

The radiation state remains four fields: energy density E and physical Fx, Fy, Fz. Numerical calculations use Q=F/c to keep all components in energy-density units. The normalization uses physical c, including when the transport speed is reduced.

* Direct Levermore M1 closure and the extreme characteristic speeds from Skinner & Ostriker (2013), equations (14)--(15) and (41). Hanawa H/beta variables and transformations are removed.
* Component PLM of E and F, using the harmonic van Leer slope limiter in Athena's primitive PLM implementation. Constant rescaling from F to Q gives the same reconstruction. A common slope factor enforces E >= 0 and |F| <= cE on both faces without changing the cell average.
* Athena-style unsplit VL: first-order HLL fluxes predict a half-time cell state; PLM of that state supplies the final HLL fluxes. The first predictor includes the explicit sources and direct implicit damping. The corrector evaluates explicit sources again at half time.
* The three-cell radiation halo supplies the predictor's two-cell halo locally. There is one radiation state exchange and one final flux/reflux phase per radiation substep. A final boundary/restriction phase synchronizes the end of the gas step. No half-time state exchange is introduced.
* Physical inflow/outflow conditions are applied to the predictor at half time. AMR reflux averages the final time-centered flux.
* A common face-flux limiter blends toward first-order global-speed HLL when needed for realizability. Floating-point corrections at vacuum/streaming boundaries are limited by the local arithmetic scale and included in the radiation conservation ledger. There is no fixed radiation energy floor.

The PLM stencil and limiter follow Athena's published implementation. This is an implementation of its VL/PLM structure, not a claim that it reproduces Hyperion bit for bit. In particular, this code adds M1 realizability safeguards and uses Octo-TIGER's AMR machinery.

## Subcycling and communication

The timestep reduction retains two independent minima: the gas timestep and the radiation CFL limit. They can come from different leaves. Every node, including refined parents, constructs the same subcycle count from those global values. Substep endpoints partition the gas interval directly; floating-point accumulation cannot cause different nodes to take a different number of steps.

The gas timestep includes a conservative radiation-pressure acoustic bound based on S&O equation (29). `hard_dt` remains a gas-step cap. `rad_max_subcycles` also caps the gas timestep; it never permits an oversized radiation step. With `hydro=0` or `rad_subcycling=0`, the global step obeys the radiation CFL limit directly.

The relevant counter fixes are:

1. State restriction, sibling exchange, and AMR prolongation continue to use the independent `rcycle`.
2. Radiation flux-correction actions now carry `rcycle`; their receive channels use explicit keys rather than assigning a substep according to arrival order. New or migrated blocks inherit the current radiation epoch.
3. The optimized hydro exchange promise allocation accounts for the additional material halo refresh between regrids. Radiation substeps do not consume hydro promises.
4. Radiation coupling waits for outstanding optimized gas-halo reads before modifying the gas. There is one frozen-material halo refresh per coupled gas step, independent of the radiation subcycle count.

## Sources and the hydro interface

The nonlinear thermal exchange uses S&O equations (44)--(49), once per gas step. The direct flux damping uses equation (43) during every radiation substep. The O(v/c) work and O(beta tau) momentum terms use the isotropic approximation in equation (18), and are evaluated in the predictor and corrector. Opacities and gas velocity are frozen across those substeps.

The physical-light-speed factors are retained in gas source terms and in F/c. Radiation transport, damping, and radiation-side work use c_hat. For local matter exchange, the conserved combinations are

    gas total energy + E / (c_hat/c)
    gas momentum     + F / (c * c_hat)

They reduce to the physical totals when c_hat=c. The separate radiation ledger still reports physical E and F, including boundary transport and source increments.

There are deliberate differences from S&O to preserve the existing hydro integrator:

* The thermal/radiation/source sequence runs after the existing hydro RK loop, rather than inserting a thermal solve and gas source stages inside Athena's hydro sequence.
* Gas feedback is accumulated from the radiation source increments actually applied, then committed after subcycling. This conserves the modified local totals instead of implementing their separate endpoint correction in equation (53).
* The existing unequal Planck/Rosseland opacity model remains available. The work term then uses the corresponding 2*chi_absorption-chi_total coefficient from equation (5a). A constant `rad_opacity` gives equal gray means and the source equations used in S&O.

Consequently, smooth source-free transport is second order, but the full hydro/radiation splitting remains formally first order. Theta=0.51 is only near trapezoidal; it is not exactly second order. A stiff theta update that loses a nonnegative thermal solution or reverses damping falls back to backward Euler.

As in a thermal-only operator split, a large thermal step can lower E below an existing beam's |F|/c before momentum damping occurs. This is rejected with a diagnostic, not repaired by removing physical flux. Reduce `hard_dt` for such thermal problems; use `rad_energy_mode=absorption` when the intended model is absorption with no emission in the evolved band. The explicit velocity terms assume the non-stiff regime discussed in S&O; subcycling by the transport CFL alone is not a guarantee in dynamic diffusion.

## Options

All controls are declared in `octotiger/options.hpp`, parsed from the CLI/config file, broadcast to all localities, printed at startup, and saved in Silo metadata. Old checkpoints retain configured defaults for absent fields. Explicit radiation CLI/config values override checkpoint defaults.

| Option | Default | Meaning |
|---|---:|---|
| `rad_subcycling` | `true` | Radiation substeps inside each gas step |
| `rad_c_ratio` | `1.0` | c_hat/c, with 0 < ratio <= 1 |
| `rad_cfl` | `0.4` | Sum-of-direction Courant number, with 0 < CFL <= 0.5 |
| `rad_max_subcycles` | `1024` | Maximum substeps; restricts the gas step if necessary |
| `rad_theta` | `1.0` | Source theta in [0.5,1], with stiff backward-Euler fallback |
| `rad_velocity_terms` | `true` | Include velocity-dependent work and momentum terms |
| `rad_opacity` | `-1.0` | Nonnegative: equal gray opacity per mass in code units; negative: existing opacity functions |
| `rad_energy_mode` | `thermal` | `thermal`, `absorption`, or `equilibrium` |
| `rad_log_subcycles` | `false` | Print actual gas step, global radiation limit, count and first substep |
| `rad_implicit` | existing `true` | Existing master switch for material coupling |

`thermal` includes gas/radiation absorption and emission. `absorption` removes energy from this radiation band without depositing that absorption energy in the gas, matching S&O's absorption-only case; velocity work and momentum feedback remain. `equilibrium` omits thermal exchange while retaining work and radiation forces.

Full c remains the default. To choose a reduced speed explicitly, add, for example, these lines to a problem configuration:

```ini
rad_subcycling = true
rad_c_ratio = 0.01
rad_theta = 1.0
rad_log_subcycles = true
```

The example ratio is a setting, not a suitability determination for a particular physical problem. S&O's timescale condition is c_hat much greater than v_max * max(1, tau_max).

For the prescribed-medium regression tests, RSLA rescales the analytic time and exact local source update while retaining physical F. A Gaussian reference must be generated for `reference_time = stop_time * rad_c_ratio`, with physical c. The existing movie runner defaults to full c and its reference files remain compatible with that default.

## Validation

The included serial validator compiles the production numerical methods and the production subcycle driver. Only mesh/HPX infrastructure and material fixtures are substituted. It exercises:

* M1 closure trace, isotropic/oblique-streaming/vacuum limits, equation (41) speeds, and realizable PLM faces.
* Thermal roots over multiple stiffnesses and reduced speeds, the modified energy invariant, theta convergence, and stiff fallback.
* Independent gas/radiation timestep reduction, identical parent/leaf substep schedules, exact final boundary time, and inherited radiation epochs.
* The actual gas feedback, thermal exchange, absorption-only beam, and independent gas halo count.
* Three-dimensional block transport with one old-state halo fill, smooth-wave convergence, RSLA propagation, and a discontinuous beam bordering vacuum.
* Half-time inflow in physical flux units, and the actual AMR fine-face restriction/coarse-face replacement.

The smooth streaming test at 8, 16, and 32 cells along the wave has L1 errors `9.529321e-03`, `4.447625e-03`, and `1.234238e-03`; the last refinement gives order `1.849414`.

Debug and release numerical checks passed. AddressSanitizer and UndefinedBehaviorSanitizer passed with leak detection disabled because this execution environment runs under ptrace. The existing reference/movie serial harness also compiles with the new interface.

**Not run here:** a full Octo-TIGER build, actual HPX channel delivery, MPI multi-locality runs, regridding/migration under the runtime, or a coupled astrophysical production run. This workspace lacks HPX, MPI and CMake. The serial driver checks and communication audit do not establish distributed runtime correctness. A distributed AMR/regridding run is still required before trusting this change in production.

## Apply and run from HOME

Unpack the update outside the source tree, then check and apply the patch against the supplied source version:

```bash
cd ~
unzip ~/Downloads/octotiger-so-radiation.zip -d ~/octotiger-so-update
git -C ~/octotiger/src/octotiger apply --check ~/octotiger-so-update/octotiger-so-radiation.patch
git -C ~/octotiger/src/octotiger apply ~/octotiger-so-update/octotiger-so-radiation.patch
python3 ~/octotiger/src/octotiger/verification_results/radiation/validate_so.py
```

If the check reports conflicts, the live source differs from the supplied archive; review the diff rather than copying over newer files. The archive also contains complete replacement files under `patched-files/`, validation logs, and SHA-256 identities of the base and patched files.

For local memory checks:

```bash
cd ~
python3 ~/octotiger/src/octotiger/verification_results/radiation/validate_so.py --sanitize
python3 ~/octotiger/src/octotiger/verification_results/radiation/validate_so.py --release
```

Rebuild the normal Octo-TIGER target after applying the patch. There is no change to its HPX/Kokkos/CUDA build configuration.

## References

Skinner & Ostriker (2013), [A Two-moment Radiation Hydrodynamics Module in Athena Using a Time-explicit Godunov Method](https://arxiv.org/abs/1306.0010), especially sections 3.1--3.4.

Athena's [primitive PLM reconstruction source](https://www.astro.princeton.edu/~jstone/Athena/doxygen/html.with_source/lr__states__prim2_8c_source.html), especially the harmonic differences and monotonicity constraints. The M1-specific cone and flux safeguards described above are additional implementation choices.
