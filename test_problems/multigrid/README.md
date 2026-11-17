# Structured Poisson multigrid

The numerical kernel in `octotiger/multigrid/block.hpp` and
`src/multigrid/block.cpp` solves the cell-centered Poisson equation

```text
laplacian(phi) = source,        source = 4 pi G rho for gravity.
```

Each block stores a regular `N x N x N` interior and a one-cell ghost layer.
The seven-point stencil uses Cartesian geometry and the block's cell width;
there are no sparse matrix rows. The physical solution comprises all uncovered
cells, including coarse cells beside refined regions. Covered parent cells
store coarse solver data. At refinement interfaces, ghost interpolation and
shared face-flux averaging supply the composite operator's wider couplings.

The stencil and transfer infrastructure originated in
[`dmarce1/scorpio`, revision `b39beb91df1c15631a102f8502867ddabc6eca3e`](https://github.com/dmarce1/scorpio/tree/b39beb91df1c15631a102f8502867ddabc6eca3e).
The historical iteration that restricted smoothed corrections and replaced fine
octet means has been superseded by the residual-correction cycle described
below. Historical convergence counts do not describe the current solver.

## Residual-correction cycle

`octotiger/multigrid/poisson_cycle.hpp` defines the cycle schedule used by both
the distributed application adapter and the standalone numerical fixture.
Every stage callback completes the selected level before the next stage starts.
The hierarchy follows the octree parents; refinement always creates all eight
children. The root has no additional sub-root hierarchy or exact bottom solve.

Let `D` denote the dimensionless seven-point Laplacian numerator, `h` the cell
width, and `R` volume averaging over eight fine cells. The normalized correction
equation on a uniform level is

```text
H_h c_h = b_h,       H_h = -D / 6.
```

At the start of a cycle, correction fields and their RHS fields are cleared.
Physical leaves receive

```text
b_h = h^2 * (composite_laplacian(phi) - source) / 6.
```

This uses the current physical residual after fine-face fluxes have replaced
coarse interface fluxes. On descent:

1. Covered parents receive the restricted remaining correction residual from
   their children:

   ```text
   b_H = 4 R(b_h - H_h c_h) = 4 R(b_h + D c_h / 6).
   ```

   The factor four comes from `H = 2 h`; the residuals here include the
   `h^2 / 6` normalization. The coarse correction starts at zero.
2. Uncovered coarse leaves receive the fine correction's interface-flux
   contribution in their RHS. For a positive-axis jump payload `J`, the lower
   face contributes `-J / 6` and the upper face `+J / 6`. The payload is half
   the sum of four fine correction differences, expressing the fine gradient
   in coarse-cell-width units. The physical-potential flux discrepancy is
   already present in the initial composite residual and is not added again.
3. Each level takes two red/black pairs. Same-level halos are exchanged before
   each complete pair and again after the last pair, so residual restriction
   and fine correction fluxes use fresh neighboring values. Correction ghosts
   bordering an as-yet-unsolved coarser region are zero on descent.

On ascent, the parent correction is prolonged **additively** into the existing
fine correction. Cell-centered trilinear interpolation uses weights `3/4` and
`1/4` along each axis. It retains the fine correction already obtained by
presmoothing; it does not replace its octet mean. Parent face, edge, and corner
halos are refreshed immediately before prolongation. Each level then takes
four red/black pairs, again with an exchange before each pair. The root therefore
receives six pairs in total. Finally the corrections are added to the physical
potential, which is synchronized before forces and the composite residual are
recomputed.

Interior parent-to-child prolongation and AMR boundary interpolation have
different roles: missing same-level correction ghosts on ascent use the same
**quadratic coarse interpolation** as physical-potential ghosts. This keeps the
correction boundary treatment consistent with the composite physical operator.
Correction exchanges include all 26 neighboring directions, supplying the edges
and corners required by interpolation. The local red/black pair itself retains
fixed halo snapshots; there is no exchange between its two colors.

Physical Dirichlet potentials are evaluated directly at each level's ghost-cell
centers. Physical correction ghosts are homogeneous. This convention does not
reflect interior values to impose Dirichlet data at the domain face. Physical
face accelerations are restricted by averaging four fine faces. The coarse
physical residual uses those same restricted fluxes, so internal interface
contributions cancel in the integrated conservation check.

## Implementation boundaries

- Arrays use contiguous `z` storage. The Laplacian sums six neighbor-minus-center
  differences, annihilating an exactly constant field without cancellation.
- The block kernels have no HPX, MPI, octree ownership, or scheduler dependency.
  The application adapter supplies ownership, communication, and stage barriers;
  the fixture supplies serial data movement through the same cycle schedule.
- Block geometry, payload sizes, and interpolation support are internal
  preconditions checked with assertions when `NDEBUG` is not defined. Runtime
  option validation and numerical-failure checks remain active in Release.
- The serial fixture uses direct level/location lookup so a deep uniform
  hierarchy does not incur a linear node search for every ghost sample.
- These tests cover the Poisson gravity operator. They do not validate separate
  variable-coefficient reaction/diffusion or FLD solvers.

## Standalone verification

The tests require a C++17 compiler and CMake, without HPX or application
dependencies:

```sh
cmake -S test_problems/multigrid -B /tmp/scorpio_multigrid_build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/scorpio_multigrid_build --parallel 1
ctest --test-dir /tmp/scorpio_multigrid_build --output-on-failure --parallel 1
```

The three standalone CTests are:

- `multigrid_numerics`: stencil and transfer identities, additive trilinear
  linear-function reproduction, normalized correction initialization, discrete
  manufactured solves on uniform and mixed AMR trees, composite flux
  conservation, and continuum convergence. AMR cases include a corner patch,
  nested physical leaf levels 1/2/3, and an interior patch with all six
  coarse/fine faces.
- `multigrid_numerics_deep_uniform`: a uniform `128^3` problem with block `N=8`
  and four refinement levels. It must recover the discrete manufactured
  potential and reach its residual tolerance within 120 cycles. This catches
  the depth-dependent failure of the historical iteration.
- `multigrid_isolated_boundary`: independent multipole-boundary identities,
  including rotational invariance and the Legendre addition theorem.

On 2026-11-17, all three passed in a GCC 15.2 Release standalone build, pinned
to one CPU. Total CTest time was 24.28 seconds, including 24.06 seconds for the
deep test. These are individual verification timings, not scaling benchmarks.

| Discrete manufactured problem | Cycles | Relative L1 residual | Maximum potential error |
| --- | ---: | ---: | ---: |
| Uniform, block `N=6`, level 1 | 14 | `4.87546e-11` | `1.29442e-10` |
| Corner AMR, block `N=6` | 14 | `3.60605e-11` | `9.33509e-11` |
| Nested AMR, block `N=6`, leaf levels 1/2/3 | 14 | `3.59082e-11` | `9.33349e-11` |
| Interior AMR patch, block `N=6` | 13 | `6.30276e-11` | `1.61033e-10` |
| Uniform `128^3`, block `N=8`, level 4 | 30 | `9.69057e-10` | `6.82201e-10` |

The fixture reports the volume-weighted **L1 residual divided by its initial
value**. This differs from the application's L2 residual divided by the source
norm. The basic discrete cases require a relative residual below `1e-10`; the
deep test requires `1e-9`. Discrete sources are constructed from the prescribed
potential using the composite operator, so these cases check algebraic
consistency and convergence, not continuum truncation error. Integrated
residual and physical-boundary flux agree within `2e-11` in the basic cases
and `5e-11` in the deep case.

A separate continuum-source test uses
`laplacian(phi) = -3 pi^2 sin(pi x) sin(pi y) sin(pi z)`, with the matching
potential plus an affine function and exact ghost values. Increasing block `N`
from 4 to 8 on the corner AMR pattern reduces volume-weighted L2 error from
`0.00506812` to `0.00115094`, a ratio of `4.40348`.

The current standalone evidence is retained locally in
`output/scf_mg_implementation_20261117/numerics_build/Testing/Temporary/LastTest.log`.
Older standalone and application runs under `output/radiation_mg_validation`
validated the superseded historical iteration. Their cycle counts, timings,
sanitizer results, and distributed-runtime checks are not validation of this
residual-correction revision.

## Static OctoTiger gravity comparison

`src/multigrid/gravity.cpp` adapts the block kernels to OctoTiger's HPX node
components. Each node owns its arrays; the root issues level barriers and
receives scalar reductions. Source and potential arrays are not gathered onto
the root. Covered parent physical data are restricted before use, and the
shared cycle schedules correction transfers and halo exchanges.

With an application build configured using `OCTOTIGER_WITH_TESTS=ON`, run:

```sh
cmake --build /path/to/build --target octotiger multigrid_numerics multigrid_isolated_boundary --parallel 1
ctest --test-dir /path/to/build --output-on-failure --parallel 1 -R '^multigrid_'
```

An individual static comparison, with optional per-leaf CSV output, is:

```sh
python3 test_problems/multigrid/check_gravity.py \
  --exe /path/to/build/octotiger --work-dir /tmp/gravity_uniform \
  --level 2 --write-fields
python3 test_problems/multigrid/check_gravity.py \
  --exe /path/to/build/octotiger --work-dir /tmp/gravity_amr \
  --level 3 --amr --boundary multipole
```

The checker records the command, complete application log, and metrics. The
sphere has unit mass, radius 0.3, and center `(0.17,-0.11,0.07)`. It requires
hydro and radiation off, zero density floors, and a fresh `solid_sphere` run.
OctoTiger's sphere refinement predicate is centered at the origin: maximum
level 2 refines every level-one block, so the AMR test uses level 3 and verifies
that physical leaves have mixed resolution.

`gravity_mg_boundary=analytic` evaluates exact sphere potentials at ghost
centers. `gravity_mg_boundary=multipole` computes COM-centered moments through
degree four from the leaf sources, then distributes the small boundary context
to every node. It requires the boundary to lie outside the source-enclosing
sphere. The boundary is a truncated exterior approximation; its independent
tests include the correction of Scorpio's negative-even-m normalization error.

The application reports the true volume-weighted relative **L2 residual after
coarse/fine face reconciliation**, along with potential and force errors
against the analytic sphere, MG–FMM differences, represented mass, cycles, and
elapsed times. A cycle-limit failure returns an error. Potential L1/L2 and
vector-force L2 errors are normalized by total physical volume; potential Linf
is absolute.

Both solvers use the same frozen leaf density and topology. Sphere cut-cell
densities use the existing 25-cubed midpoint quadrature. FMM represents them as
cell-center point monopoles and omits self interaction; multigrid solves a
seven-point Poisson discretization. Their difference is not an independent MG
error estimate, and residual tolerance does not measure continuum accuracy.
The analytic boundary uses nominal sphere mass; `represented_mass` exposes the
small discrepancy from density quadrature.

The static comparison sets `correct_am_grav=off`; it rejects an enabled
angular-momentum correction for this vacuum test. Empty FMM source groups use
a geometric expansion center to avoid a zero-mass COM division, which does not
establish general vacuum support for that correction. Static comparison MG
results remain separate from FMM fields and gas energy. Production SCF models
and their stability require separate validation.

For `B` total blocks, including covered parents, each V-cycle performs
`6 B N^3` red/black cell updates. With `L` leaves in a full octree,
`B=(8 L-1)/7`. These counts exclude transfers, halo exchanges, residuals, and
boundary evaluation. Comparison timings include setup and convergence
monitoring and are not controlled scaling benchmarks.

## Application validation of the residual-correction revision

On 2026-11-17, an isolated GCC 13 Release application build containing this MG
revision, with no pending SCF edits, passed all seven `multigrid_` CTests:
the three standalone checks above and the uniform, AMR, multipole-boundary,
and expected-nonconvergence application cases. The last verifies that an
insufficient one-cycle budget returns failure. The broader `multigrid` label
also selects the separate elliptic composite test; that executable was initially
unbuilt, then built and passed in a separate invocation. The command above
selects just the seven Poisson solver checks.

| Mesh and boundary | Leaves | Cycles | Relative L2 residual | Potential L2 error | Force L2 error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform level 1, analytic | 8 | 22 | `3.79119e-10` | `3.17455e-3` | `1.97515e-1` |
| AMR through level 3, analytic | 288 | 19 | `6.36688e-10` | `2.08676e-4` | `2.41877e-2` |
| Uniform level 1, multipole | 8 | 22 | `3.79155e-10` | `3.17843e-3` | `1.97516e-1` |

The application results use block `N=8`, one HPX worker, and tolerance `1e-9`.
The combined working build, which also contains pending SCF changes, separately
passed the same four application cases with identical non-timing metrics.
These static tests do not establish a converged binary SCF model or its stability.

A further AMR level-3 multipole comparison used one locality with one worker,
then two local HPX processes with one worker each over loopback TCP. Both
converged in 19 cycles to relative residual `6.3668536740464582e-10`, with 288
leaves and 147,456 physical cells. Every reported non-timing metric matched
exactly. Commands, logs, build fingerprints, and the parity report are in
`publication_two_localities/` under the evidence directory below. This checks
local distributed dispatch, not multi-machine scaling.

Evidence is under `output/mg_residual_fix_20261117`: `publication_ctest.log`,
`elliptic_ctest.log`, `publication_build/test_problems/multigrid/`, and
`combined_ctest.log`. `publication_source_hashes.json` records the exact staged
solver sources used for the isolated build. These are local validation artifacts,
not required repository inputs.
