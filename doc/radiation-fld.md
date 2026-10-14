# Stationary flux-limited diffusion

`radiation_model=fld` selects a CPU reference implementation of grey,
two-temperature flux-limited diffusion (FLD). Radiation energy and material
heating/cooling are solved together with backward Euler. The existing M1
model remains the default, with its Euler/RK1 integrator. ADER2 has been
removed.

This milestone supports stationary ideal gas, physical light speed, and
Cartesian leaf meshes with dyadic refinement and 2:1 face balance. It is
first order in time. Moving radiation hydrodynamics, radiation force and
work, dynamic diffusion, higher-order time integration, gravity, rotation,
other equations of state, and accelerator execution remain outside this
implementation's supported scope.

## Equations and thermal elimination

Let `E` be radiation energy density and `u` material internal-energy density.
The stationary equations advanced over a step are

```text
(E - E_old)/dt - div(D grad E) = c chi_P [B(u) - E]
(u - u_old)/dt                = -c chi_P [B(u) - E]

D = c lambda(R) / chi_R
R = |grad E| / (chi_R E)
lambda(R) = (2 + R) / (6 + 3 R + R^2)
```

For the ideal-gas thermal law, `B(u)=a_r T(u)^4`, with
`a_r=4 sigma_SB/c`. The `MARSHAK` verification problem instead retains its
separate linear emission law `B(u)=u` and its normalized constants. It is
not an ideal-gas LTE reference problem.

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
small direct solve. Damped Gauss-Seidel smoothing and geometric V-cycles
precondition restarted GMRES. Coarse/fine reconstruction can make the
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

The implementation requires `hydro=off`, `gravity=off`, `eos=IDEAL`,
`omega=0`, `radiation_backend=cpu`, `radiation_reduced_c_factor=1`, and
`rad_implicit=on`. Cell momenta must be zero. Driven source terms and inflow
boundaries are rejected. Explicitly set a useful positive timestep or stop
time.

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
and the prescribed Marshak problem. Its Robin convention is

```text
alpha E_boundary + gamma D dE/dn = value
outward flux q = -D dE/dn
```

Thus reflecting is `{alpha=0,gamma=1,value=0}`, prescribed outward flux is
`{0,1,-q}`, and vacuum is `{c/2,1,0}`. `MARSHAK` overrides the generic outer
boundary: the low-x face is `{c/2,1,2}`, the high-x face is vacuum, and the
transverse faces reflect. The incoming half-space flux is one in its
normalized units.

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
sum_leaves volume * [(E_new + u_new) - (E_old + u_old)] - boundary_energy
```

Material-to-radiation source energy is positive when material loses energy.
The existing radiation CSV ledger uses the opposite boundary convention:
`er_boundary` is positive outward. Its energy contract is
`delta(er) + er_boundary - er_source = 0`, apart from the nonlinear tolerance
and roundoff.

The stored `fx`, `fy`, and `fz` fields are derived physical flux diagnostics
`-D grad(E)`. They are not evolved radiation momentum, their change imparts
no material recoil, and the M1 flux-moment conservation columns are not FLD
momentum invariants. The current stationary model has no advected radiation
enthalpy contribution. Moving-frame flux semantics require a future moving
gas formulation.

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
