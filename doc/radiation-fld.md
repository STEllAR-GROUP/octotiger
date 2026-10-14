# Mixed-frame flux-limited diffusion

`radiation_model=fld` selects a CPU reference implementation of grey,
two-temperature flux-limited diffusion (FLD). Radiation energy and material
heating/cooling are solved together with backward Euler. The existing M1
model remains the default, with its Euler/RK1 integrator. ADER2 has been
removed.

The CPU implementation uses physical light speed and ideal gas. Stationary
transport supports Cartesian leaf meshes with dyadic refinement and 2:1 face
balance. Moving gas currently requires a uniform mesh and one species. Both
paths are first order in time. Gravity, rotation, other equations of state,
accelerators, and distributed multigrid remain unsupported.

## Equations and thermal elimination

Let `E` be radiation energy density and `u` material internal-energy density.
The stationary equations advanced over a step are

```text
(E - E_old)/dt - div(D grad E) = c chi_P [B(u) - E] + q
(u - u_old)/dt                = -c chi_P [B(u) - E]

D = c lambda(R) / chi_R
R = |grad E| / (chi_R E)
lambda(R) = (2 + R) / (6 + 3 R + R^2)
```

For the ideal-gas thermal law, `B(u)=a_r T(u)^4`, with
`a_r=4 sigma_SB/c`. The `MARSHAK` verification problem instead retains its
separate linear emission law `B(u)=u` and its normalized constants. It is
not an ideal-gas LTE reference problem.

The optional prescribed emissivity `q` is independent of material energy.
It is zero except for the stationary Gaussian-emitter sphere benchmark.
`solve_with_source` adds `dt*q` to the radiation right-hand side in scratch;
the material equation retains its original energy. This is the coupled
backward-Euler source term, not a committed source split. A failed attempt
returns the original state and zero budgets; a smaller retry accounts only
for its own accepted interval. A diagnostic refresh (`dt=0`) injects nothing.

The material equation is eliminated locally through its positive thermal
root. Each modified-Newton linear solve freezes diffusion and opacity while
retaining the material-emission derivative. The full coupled residual is
reevaluated after updating material energy, opacity, and the flux limiter.
A converged frozen linear system alone does not accept a nonlinear step.
The Newton direction is limited locally to remain inside the nonnegative
domain before a global line search checks the full coupled residual. This
prevents signed Krylov noise in nearly dark cells from throttling every
cell's update. Accepted states are not repaired with an energy floor.

The limiter uses a full reconstructed gradient at each shared face. The
face energy used by the limiter is bounded by the neighboring cell energies;
the cells themselves are not clipped. Cell diagnostic fluxes and face
transport fluxes use their respective reconstructed gradients. A flat
transparent region has zero flux and uses a finite zero-diffusion
convention. Nonflat transparent regions use the streaming limit. Robin
boundary coefficients are solved consistently with their boundary trace;
an inadmissible or singular constitutive boundary state rejects an attempt.

The extinction coefficients `chi_R` and `chi_P` already include density and
have dimensions of inverse length. They must not be multiplied by density
again. The generic fitted-opacity path uses
`octotiger/radiation/physical_opacity.hpp`: code density is converted to CGS
before evaluating dimensional fitted constants, and the resulting inverse
centimeters are converted to inverse code length. Its temperature is in
kelvin. Dedicated benchmark opacity branches retain their problem-specific
code-unit normalization.

## Moving gas and frames

Radiation energy is lab-frame energy. Opacity and the FLD closure are evaluated
in the material frame: this is a mixed-frame formulation. For the LP pressure
tensor `P=f E`, the lab diagnostic flux is

```text
Fd = -D grad(E)
H = E v + P.v
F_lab = Fd + H
force = -lambda grad(E)
```

The explicit first-order finite-volume stage advances gas mass, momentum,
total energy, and radiation enthalpy transport with shared Rusanov face fluxes.
Its gas-force impulse uses midpoint velocity for work; the opposite work is
applied to radiation, preserving combined energy. The signal-speed bound
includes radiation pressure and enthalpy transport. Both this stage and the
implicit solve run in scratch; a failed implicit solve retries the entire
coupled substep, without duplicating gas updates or boundary budgets.

Density and velocity are frozen during each implicit solve. The material
thermal target is refreshed at every trial radiation energy and opacity:

```text
h = 1 - v.(I+f).v/c^2
Eeff = h E - 2 v.Fd/c^2
du/dt = c chi_P [Eeff - B(u)]
dE/dt = -du/dt - div(Fd)
```

The `v.H/c^2` correction retains the leading dynamic-diffusion exchange.
This follows the nonrelativistic leading-regime mixed-frame equations of
[Krumholz et al. (2007), equations 38–42](https://arxiv.org/abs/astro-ph/0611003).
It is not a full relativistic expansion; runtime states require `|v|/c<=0.1`.
The derivative used to precondition a linear correction lags the gradient and
closure, but the accepted nonlinear residual does not. Only `Fd` is bounded
by `c E`; clipping the reconstructed lab flux would remove physical advection.

The shock benchmarks retain their reflecting low-x wall, fixed upstream
high-x reservoir, and transverse extrapolated gas/zero-diffusion boundaries.
The blast uses gas outflow and vacuum diffusion boundaries. Their material
transport and radiation force are active. Algorithm version 2 identifies this
moving implementation; stationary FLD version 1 and M1 RK1 remain distinct.

The solver normalizes the coupled energy residual by material/radiation energy
scales and checks its volume-weighted L1 norm against total energy. The latter
prevents a localized energy peak from hiding a domain-wide imbalance.
In cold shocks gas energy can exceed radiation energy by many orders
of magnitude; a small coupled residual need not imply the same relative
radiation-only residual. Reports retain both, without treating them as equal.

## Composite operator and multigrid

`octotiger/elliptic/composite.hpp` contains the radiation-independent scalar
operator. Only leaf cells are physical unknowns. Geometry construction
checks dyadic alignment, domain coverage, face coverage, and 2:1 balance.
Each coarse/fine contact is split into fine subfaces, with one shared flux
entering the two adjacent volumes with opposite signs on every operator
evaluation. No final reflux correction substitutes for that composite
operator.

A least-squares linear reconstruction brings the coarse trace to the
tangential position of each fine subface. The normal gradient used by the
FLD limiter is the same gradient used by the scalar operator. In the thick
limit, distance-weighted extinction gives harmonic diffusion transmission
across an opacity jump.

The correction hierarchy uses volume-weighted residual restriction, signed
linear prolongation, and Galerkin coarse operators. Virtual levels continue
below an eight-cell-wide root, including `8^3 -> 4^3 -> 2^3`, followed by a
small pivoted-LU direct solve. Each V-cycle applies three forward and three
reverse Gauss-Seidel sweeps with relaxation weight 0.7. These geometric
V-cycles precondition right-preconditioned GMRES, restarted after 30 Krylov
iterations. This is a linear correction multigrid hierarchy inside the
nonlinear iteration, rather than nonlinear FAS multigrid. Coarse/fine reconstruction can make the
operator nonsymmetric, so the implementation makes no CG assumption.
Pure periodic/Neumann Poisson tests check compatibility and remove the
constant nullspace. The FLD time term normally removes that nullspace.

This reconstruction is not a proof of monotonicity or of convergence
independent of arbitrary coefficient contrast. Nonlinear positivity is
checked separately from linear convergence. Constant-state flux evaluation
uses differences to avoid subtracting large diffusion matrix entries, and
scaled residual norms avoid underflow for very small finite energies.

## Runtime integration and limitations

`src/radiation/fld.cpp` gathers the physical leaves through dedicated HPX
tree actions. The root builds the composite geometry and performs the CPU
nonlinear and multigrid solves serially, then returns accepted cell values
through dedicated reply channels. These packets do not reuse the M1 halo
or reflux channels. Existing radiation/hydro restriction is used after a
completed solve to synchronize stored physical states, not to transfer
multigrid corrections.

The root must hold the entire composite mesh, face stencils, hierarchy, and
Krylov vectors. This is a reference implementation, not distributed
multigrid. Distributing operator application, smoothing, reductions, and
per-node solver scratch is future work. No locality-scaling or production
memory claim follows from the standalone tests.

Topology is fixed during each attempted solve. The root retries rejected
attempts with a halved timestep, with a bounded retry count. Accepted
internal substeps remain scratch until the whole requested interval has
completed; only then are state and conservation budgets installed on the
leaves. Diagnostic flux is reconstructed after startup/restart and after
regridding. Scratch geometry is rebuilt from the current leaves, so it is
not persisted across migration or restart.

There is no M1 streaming CFL restriction on an FLD step. With material
transport disabled, `hard_dt` and the remaining stop-time batch control
the timestep. Implicit stability does not establish temporal accuracy;
choose and refine the timestep for the physical problem.

## Options and example configurations

The implementation requires `gravity=off`, `eos=IDEAL`, `omega=0`,
`radiation_backend=cpu`, `radiation_reduced_c_factor=1`, and `rad_implicit=on`.
With `hydro=off`, material momenta must be zero. With `hydro=on`, set
`unigrid=on`, `n_species=1`, and `0<cfl<=0.5`. Generic driven source terms
and inflow boundaries remain rejected; the shock benchmark reservoir is a
problem-specific boundary. Set a useful positive timestep or stop time.

| Option | Meaning/default |
| --- | --- |
| `radiation_model` | `m1` by default; select `fld` explicitly. |
| `radiation_integrator` | `auto` selects `euler` for M1 and `backward_euler` for FLD; `rk1` aliases M1 Euler. |
| `radiation_limiter` | `levermore_pomraning`. |
| `fld_boundary` | `auto` follows `reflect_bc`; explicit values are `reflecting` and `vacuum`. |
| `fld_nonlinear_tolerance` | Coupled relative residual tolerance, `1e-10`. |
| `fld_linear_tolerance` | Scalar relative residual tolerance, `1e-10`. |
| `fld_max_iterations` | Maximum nonlinear iterations per attempted step, `50`. |
| `fld_max_linear_iterations` | Maximum GMRES iterations per linear solve, `200`. |

The scalar library supports Dirichlet, prescribed outward flux/Neumann,
reflecting, Robin, and periodic boundaries. General per-face data are a
library API; the runtime configuration exposes reflecting, vacuum, periodic,
the prescribed Marshak problem, and the sphere's analytic Dirichlet bath.
Its Robin convention is

```text
alpha E_boundary + gamma D dE/dn = value
outward flux q = -D dE/dn
```

Thus reflecting is `{alpha=0,gamma=1,value=0}`, prescribed outward flux is
`{0,1,-q}`, and vacuum is `{c/2,1,0}`. `MARSHAK` overrides the generic outer
boundary: the low-x face is `{c/2,1,2}`, the high-x face is vacuum, and the
transverse faces reflect. The incoming half-space flux is one in its
normalized units.

The equilibrium sphere uses the same initial energy cell averages and
Gaussian emissivity as the existing fixed-medium benchmark. Its Rosseland
extinction is `rad_test_chi`, its absorption extinction is zero, and the gas
remains unchanged. Each composite boundary subface is assigned the
four-by-four Gauss average of the analytic sphere energy over the face;
ghost-cell volume averages are not used as boundary traces. The error-function
profile solves constant-coefficient linear diffusion. It is a qualified
diffusion-limit reference, not an exact steady solution of nonlinear LP FLD.
Its finite-box source integral, rather than the all-space luminosity, enters
the conservation ledger.

Streaming wave and front now use an explicitly configured small positive
scattering opacity with domain optical depth `chi_R L=1e-4` in both models.
Their energy profiles and periodic domains are unchanged. FLD derives flux
from the energy gradient; it cannot preserve the independently directed M1
flux. The translated zero-opacity M1 formulas are transparent-limit references
at this finite opacity, and are not FLD spatial analytic solutions. FLD has
exact periodic energy-integral invariants, plus positivity and flux-bound
checks. Standalone five-step level-2 probes passed for both cases. An 8-cubed
wave history needed timestep retries near its flat equilibrium; the front
completed without retries. These checks do not establish full level-2/3/4
runtime or convergence. Exact-vacuum nonlinear convergence is not claimed.

For a small uniform Marshak run, save the following as a configuration file.
These are example inputs; application validation must use the executable
actually built from the current source.

```ini
problem=MARSHAK
radiation=on
radiation_model=fld
radiation_integrator=backward_euler
radiation_backend=cpu
radiation_reduced_c_factor=1
rad_implicit=on
hydro=off
gravity=off
eos=IDEAL
omega=0
periodic=off
fld_boundary=reflecting
unigrid=on
min_level=1
max_level=1
xscale=1
n_species=1
atomic_mass=1
atomic_number=1
hard_dt=0.125
stop_time=0.25
stop_step=1
disable_analytic=on
hydro_host_kernel_type=LEGACY
hydro_device_kernel_type=OFF
monopole_device_kernel_type=OFF
multipole_device_kernel_type=OFF
```

Run with `octotiger --config_file=marshak-fld.ini --hpx:threads=2`, using the
path to the intended executable. For homogeneous ideal-gas thermal exchange,
use `problem=RADIATION_COUPLING`, `periodic=on`, and units
`code_to_g=1e30`, `code_to_cm=2.99792458e10`, `code_to_s=1`. These units make
the physical light speed one in code units; they do not reduce it.

## Diagnostics and restart contract

`FLD accepted` reports the requested interval, accepted internal substeps,
retries, nonlinear iterations, GMRES iterations, V-cycles, maximum accepted
linear/nonlinear residuals, boundary energy, and conservation error.
`boundary_energy` is positive **into** the domain. The conservation error is

```text
sum_leaves volume * [(E_new + u_new) - (E_old + u_old) - dt*q] - boundary_energy
```

Radiation source energy includes prescribed emission and material-to-radiation
exchange; the latter is positive when material loses energy.
The existing radiation CSV ledger uses the opposite boundary convention:
`er_boundary` is positive outward. Its energy contract is
`delta(er) + er_boundary - er_source = 0`, apart from the nonlinear tolerance
and roundoff. For moving runs the combined energy balance includes gas total
(internal plus kinetic) energy and its advective boundary budget. The solver
also requires a volume-weighted L1 residual, preventing a hot localized cell
from masking a domain-wide imbalance.

The stored `fx`, `fy`, and `fz` fields are derived physical flux diagnostics
`-D grad(E)` for stationary runs and `-D grad(E)+E v+P.v` for moving runs.
They are not independently evolved radiation momentum. Gas recoil comes from
the explicitly discretized force, never from a change of the diagnostic flux.
The M1 flux-moment conservation columns are not FLD momentum invariants.

`momentum-diagnostics.csv` contains full-volume gas/radiation momenta and
energies, cumulative gas boundary budgets, and available radiation boundary
budgets. Unknown FLD boundary traction is left unavailable, not interpreted as
zero. Paired momentum plots cover the two shocks and blast. Fixed-material
benchmarks omit these plots because their external material reservoir prevents
a closed gas-plus-radiation momentum test.

New Silo metadata records the model, limiter, integrator, algorithm version,
flux semantics, and FLD solver options. Saved physics takes precedence over
conflicting restart command-line settings. Legacy checkpoints without model
metadata retain M1 semantics. Unknown or inconsistent method identifiers
are rejected.

## Validation evidence and commands

The standalone elliptic test was compiled with
`g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -I.` and passed without compiler
warnings. Recorded checks include mixed boundary conditions, constant and
linear preservation, leaf-volume conservation, periodic/Neumann nullspaces,
signed corrections, an opacity contrast of `1e6`, constant-state diffusion
coefficients of `1e100`, and right-hand sides of order `1e-200`.

The uniform sine manufactured solution had L2 error `0.00436543` at `8^3`
and `0.00108553` at `16^3`. A half-domain composite refinement test had
error `0.0112185` at base resolution four and `0.0028271` at eight, for all
three interface orientations. Quadratic composite error ratios were
`3.91`, `3.87`, and `3.81` under the same refinement. Those systems required
5–10 GMRES iterations; this is measured coverage, not an arbitrary-contrast
or resolution-independent guarantee.

An independent standalone cold Marshak probe used 288 composite cells,
initial `E=u=0`, `chi_R=chi_P=10`, `c=3e10`, `dt=1e-10`, the low-x Robin
bath, and reflecting remaining faces. It converged in 6–7 nonlinear
iterations for each of the three refinement orientations, with nonlinear
residuals below `4.5e-11` and physical conservation defects below `9e-22`.

Local validation on October 13, 2026 built the HPX application and passed
12,056 standalone FLD checks, including two successive application-size
Marshak steps initialized with `E=0`, `u=1e-20`. The standalone radiation
suite passed all 17 tests. Application evidence is saved under
`output/fld-validation/20261013/`:

- Homogeneous thermal exchange at levels zero and one conserved total
  energy to `4.72e-15` relative error and restarted bitwise identically.
- Uniform level-one Marshak transport conserved boundary-adjusted energy
  to `6.46e-12`. The adaptive run took eight further steps on a mixed
  level-one/two mesh, with total relative balance error `8.63e-11`.
- One- and two-worker outputs were bitwise identical for both thermal
  cases and both Marshak meshes. These are single-locality tests.
- With two nonlinear iterations allowed, each Marshak macrostep retried
  twice and committed four smaller substeps. With one iteration and an
  intentionally stringent tolerance of `1e-16`, the solve exhausted its
  16 halvings, drained the tree, and terminated without an accepted step
  or completed checkpoint.
- The default unlimited `stop_time` works with `hard_dt` and `stop_step`
  without overflowing timestep arithmetic.
- Retained M1 Euler passed subcycling, coupled-wave conservation and
  restart checks at both levels. The explicit `rk1` alias also passed a
  level-one restart test at reduced-light-speed factor `0.1`, including
  rejection of inconsistent checkpoint metadata.

The final ten-run FLD report is
`output/fld-validation/20261013/accepted-thread2/validation.json`; it also
records six rejected configurations, one forced solver failure, and hashes
of the executable and loaded libraries. The same cases passed with one
worker. The updated results launcher's three synthetic tests passed too;
those test the launcher/reporting pipeline, not additional solver physics.

The FLD application checks use `dt=0.125`, exceeding the corresponding
M1 streaming timestep by factors six and twelve on levels zero and one.
They verify positivity, bounded diagnostic flux, unchanged material
momentum, consistent thermal tracers, and accepted boundary budgets.
These short runs do not establish moving-gas accuracy or production
scalability. `linear_residual` records the Krylov solve before any local
Newton-direction safeguard; acceptance uses `nonlinear_residual`.

The subsequent stationary sphere extension passed 12,744 standalone checks,
including exact homogeneous coupled source injection, finite-box emitter
integration, face-only boundary quadrature, source/boundary energy balance,
diagnostic refresh without injection, and rollback/retry without repeated
emission. Application checks then completed the sphere, Gaussian and Marshak
at uniform level 0 through `t=0.25`, each in two `dt=0.125` steps with three
saved states and zero retries. The maximum nonlinear residual was
`9.37405e-11`; the largest normalized radiation energy ledger residual was
`5.86963e-13`. Evidence is in
`output/full-suite-validation-20261013/application-summary.json`. These runs
verify the source integration and larger output-compatible steps, not temporal
convergence. Transparent streaming still fails the separate vacuum probes.

On October 14, bounded moving-gas checks advanced both shocks and the blast
through four one-second FLD steps on a uniform level-one mesh. Combined
gas-plus-radiation energy balances, including boundary transport, were
`1.74e-14`, `1.69e-16`, and `5.82e-12` relative to initial total energy.
These short checks exercise coupling and budgets, not full-duration shock
structure or moving-diffusion convergence. At levels zero and one, a two-step
checkpoint followed by two resumed steps reproduced all ten saved gas and
radiation fields bitwise, including restoration of conflicting CLI settings.
The stationary application regression also passed nine runs, six rejected
configurations, and the forced solver-exhaustion check.

The new streaming configurations use domain optical depth `1e-4` in both
models. Bounded M1 wave/front checks measured the expected implicit scattering
attenuation while leaving gas fields bitwise unchanged; zero-opacity controls
retain their previous behavior. Positive-opacity FLD probes covered both
profiles, positivity, flux bounds, and energy balance. Exact-vacuum probe
failures do not qualify this positive-opacity case. Evidence is under
`output/full-readiness-20261014/`. The full 48-run campaign remains unstarted.

Example checks, subject to the shared local worker budget:

```sh
cmake -S tests/radiation -B /tmp/octotiger-radiation-checks
cmake --build /tmp/octotiger-radiation-checks --target radiation_elliptic_test test_fld --parallel 2
ctest --test-dir /tmp/octotiger-radiation-checks -R '^(radiation_elliptic|radiation_fld)$' --output-on-failure -j1
ctest --test-dir BUILD -R '^radiation\.fld\.application$' --output-on-failure -j1
```

Use the real application build directory in place of `BUILD`. Distributed
failure handling, migration across localities, full regrid qualification,
second-order stiff temporal convergence, moving diffusion, production scale,
and accelerator performance remain separate acceptance work.

## Design references

[Zhang et al., CASTRO II: Gray Radiation Hydrodynamics](https://arxiv.org/abs/1105.2466)
provides a grey FLD radiation-hydrodynamics reference with implicit diffusion
and thermal coupling. [Teunissen and Keppens, A geometric multigrid library
for quadtree/octree AMR grids](https://arxiv.org/abs/1901.11370) provides an
octree multigrid design reference. The code here is a new implementation;
it does not incorporate either solver, and it does not inherit their
validation or parallel scaling results.
