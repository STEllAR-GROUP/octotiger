# Structured multigrid numerical port

The numerical kernel in `octotiger/multigrid/block.hpp` and
`src/multigrid/block.cpp` ports the gravity solver from
[`dmarce1/scorpio`, revision `b39beb91df1c15631a102f8502867ddabc6eca3e`](https://github.com/dmarce1/scorpio/tree/b39beb91df1c15631a102f8502867ddabc6eca3e).
It solves the cell-centered Poisson equation

```text
laplacian(phi) = source,        source = 4 pi G rho for gravity.
```

Each block stores a regular `N x N x N` interior and a one-cell ghost layer.
The seven-point stencil uses implicit Cartesian geometry and the block's cell
width; there are no sparse matrix rows. The physical solution comprises all
uncovered cells, including coarse cells beside refined regions. Covered parent
cells store the coarse representation used by the iteration.
At refinement interfaces, ghost interpolation and shared face-flux averaging
create wider couplings in the composite operator. Those couplings are applied
through structured transfers, without assembling sparse rows.

## Preserved numerical behavior

The reference locations below are relative to the pinned Scorpio revision.

| Behavior | Scorpio source |
| --- | --- |
| Seven-point Laplacian and eight-cell averaging | `src/array3d.h`, lines 191–199 |
| Initial correction `(D phi - h^2 source)/6` in uncovered cells | `src/multigrid/multigrid.cpp`, lines 917–929 |
| Two red/black pairs on descent, four on ascent | `src/multigrid/multigrid.cpp`, lines 773–817 |
| Same-level halo snapshot before each complete red/black pair | `src/multigrid/multigrid.cpp`, lines 820–850 |
| Save the correction as its normalized RHS without clearing the correction | `src/multigrid/multigrid.cpp`, lines 1068–1075 |
| Restrict the correction after descent smoothing | `src/multigrid/multigrid.cpp`, lines 1079–1101 |
| Replace each fine octet's correction mean on ascent, preserving its variation | `src/multigrid/multigrid.cpp`, lines 1131–1180 |
| Tensor quadratic interpolation for physical potential ghosts | `src/array3d.h`, lines 21–39; `src/multigrid/multigrid.cpp`, lines 505–524 |
| Piecewise constant injection for correction ghosts | `src/multigrid/multigrid.cpp`, lines 98–117 |
| Coarse/fine flux-defect adjustment to the coarse correction RHS | `src/multigrid/multigrid.cpp`, lines 931–1065 |
| Restrict physical face acceleration by averaging four fine faces | `src/multigrid/multigrid.cpp`, lines 131–184 |

The hierarchy follows the existing octree parents. The root block receives the
same two descent and four ascent smoothing pairs; the historical solver has no
sub-root hierarchy or separate exact bottom solve. Physical Dirichlet values
are evaluated at ghost-cell centers, with homogeneous correction ghosts at the
physical boundary. This is the historical convention, not a reflected
cell-center construction for prescribed values at the domain face.

For a fine/coarse interface, the potential-jump payload is half the sum of four
fine potential differences. This expresses the fine gradient using the coarse
cell width. The force payload is instead one quarter of the four physical face
accelerations. The coarse residual uses that same restricted physical flux, so
interior coarse/fine contributions cancel in the integrated conservation check.

This port preserves Scorpio's iteration, including its restriction of smoothed
corrections. It does not replace the iteration with a Galerkin matrix hierarchy
or a textbook residual-restriction cycle. Variable reaction/diffusion operators
and FLD coupling are subsequent work.

## Deliberate implementation differences

- OctoTiger refines all eight children or none, so each block stores one
  refinement flag. Correction initialization and residual sums make that
  decision once per block. Coarse/fine flux adjustments act only on leaf block
  faces; Scorpio's partially covered parent octants are not represented.
  Descendants can still have different depths, recorded by the octree; transfers
  operate one level at a time. `sampleNearestCell` returns a sampled value for
  piecewise-constant injection, leaving the destination assignment to its caller.
- Arrays use contiguous `z` storage and visit that direction innermost. Scorpio
  visited the other index innermost. Within a color, seven-point updates are
  independent; the two-color order and fixed halo snapshot are preserved.
- The Laplacian is evaluated as the sum of six neighbor-minus-center differences
  so a constant field is annihilated exactly. Floating-point summation order may
  differ from the original expression.
- The numerical block has no MPI, HPX, octree ownership, or scheduler dependency.
  The application adapter supplies ownership, communication, and stage barriers.
- Block geometry, payload sizes, and interpolation support are internal
  preconditions checked with assertions when `NDEBUG` is not defined. Runtime
  validation of user options and numerical failures remains active in Release
  builds. Residual diagnostics return infinity for nonfinite values.
- `numerical.cpp` supplies a small serial tree only as a standalone verification
  fixture. It is not the production communication or ownership architecture.

## Standalone verification

The tests require a C++17 compiler and CMake, without HPX or application
dependencies:

```sh
cmake -S test_problems/multigrid -B /tmp/scorpio_multigrid_build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/scorpio_multigrid_build --parallel 1
ctest --test-dir /tmp/scorpio_multigrid_build --output-on-failure --parallel 1
```

On 2026-11-15, GCC 15.2 Release builds passed both `multigrid_numerics` and
`multigrid_isolated_boundary`. Builds and tests were pinned to one CPU under the
shared resource allocation. The numerical fixture checks stencil identities,
quadratic interpolation, octet mean replacement, payload ordering, and exact
annihilation of a constant. It then solves discrete manufactured problems on a
uniform tree and a 2:1 AMR tree with one of the eight level-one blocks refined.

| Problem | Cycles | Relative residual | Maximum potential error |
| --- | ---: | ---: | ---: |
| Uniform, block `N=6` | 15 | `2.35935e-11` | `6.16616e-11` |
| 2:1 AMR, block `N=6` | 15 | `9.91194e-11` | `2.0629e-10` |

These manufactured discrete sources check consistency and convergence of the
implemented operator; they do not independently measure its truncation error.
A separate continuum-source test uses
`laplacian(phi) = -3 pi^2 sin(pi x) sin(pi y) sin(pi z)`, with the matching
potential plus an affine function and exact ghost values. On the same AMR
pattern, increasing block `N` from 4 to 8 reduces the volume-weighted L2 error
from `0.00506812` to `0.00115094`, a ratio of `4.40348`. The integrated residual
and physical-boundary flux agree within the test tolerance of `2e-12`.

The numerical fixture also passed AddressSanitizer and UndefinedBehaviorSanitizer
with `ASAN_OPTIONS=detect_leaks=0`. LeakSanitizer could not run under the execution
environment's ptrace restriction; that limitation is not a completed leak check.

Application-level comparison with the existing FMM, distributed execution, and
gravity boundary selection are validated separately by the application adapter.

## Static OctoTiger gravity comparison

`src/multigrid/gravity.cpp` adapts the block kernels to OctoTiger's HPX node
components. Each node owns its arrays; the root issues level barriers and
receives scalar reductions. Source and potential arrays are never gathered
onto the root. Same-level correction exchanges send the six face interiors;
physical-potential exchanges also supply edges and corners for quadratic
coarse interpolation. Covered parent data are restricted before a level is used.

With an application build configured using `OCTOTIGER_WITH_TESTS=ON`, run:

```sh
cmake --build /path/to/build --target octotiger multigrid_numerics multigrid_isolated_boundary --parallel 1
ctest --test-dir /path/to/build --output-on-failure --parallel 1 -L multigrid
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

The checker records the command, complete application log, and numerical
metrics. The sphere has unit mass, radius 0.3, and center `(0.17,-0.11,0.07)`.
The comparison requires hydro and radiation off, zero density floors, and a
fresh `solid_sphere` run. OctoTiger's sphere refinement predicate is centered at
the origin: maximum level 2 refines every level-one block, so the AMR test uses
level 3 and explicitly verifies that the leaf mesh has mixed resolution.

`gravity_mg_boundary=analytic` uses exact sphere potentials at ghost centers.
`gravity_mg_boundary=multipole` computes COM-centered moments through degree
four from the same leaf sources, then distributes the small boundary context
to every node. It requires the boundary to lie outside the source-enclosing
sphere. The independent multipole test corrects an original Scorpio
negative-even-m normalization error and checks rotational invariance and the
ordinary Legendre addition theorem. This boundary remains a truncated exterior
approximation.

The application reports the true volume-weighted relative L2 residual after
coarse/fine face reconciliation, along with potential and force errors against
the analytic sphere, MG–FMM differences, represented mass, cycles, and elapsed
times. A cycle-limit failure returns an error. Potential L1/L2 and vector-force
L2 errors are normalized by total physical volume; potential Linf is absolute.

Both solvers use the same frozen leaf density and topology. Sphere cut-cell
densities use the existing 25-cubed midpoint quadrature. FMM represents them as
cell-center point monopoles and omits self interaction; multigrid solves a
seven-point Poisson discretization. Their difference is not an independent MG
error estimate, and residual tolerance does not measure continuum accuracy.
The analytic boundary uses nominal sphere mass; `represented_mass` exposes
the small mass discrepancy from density quadrature.

The comparison sets `correct_am_grav=off`. Empty FMM source groups now use a
geometric expansion center to avoid a zero-mass COM division; this narrow fix
does not establish general vacuum support for the angular-momentum correction.
MG results remain separate from FMM fields and gas energy.
The driver rejects an enabled angular-momentum correction for this vacuum test.

For `B` total blocks, including covered parents, each V-cycle performs exactly
`6 B N^3` red/black cell updates. With `L` leaves in a full octree,
`B=(8 L-1)/7`. These counts exclude transfers, halo exchanges, residuals and
boundary evaluation. Timings include MG setup and convergence monitoring,
and one additional FMM solve on the already-initialized tree; they are initial
diagnostics rather than a controlled scaling benchmark.

## Measured application validation, 2026-11-15

The GCC 13 Release application build used block `N=8`, CPU execution, and
relative residual tolerance `1e-9`. All six permanent CTests passed:
`multigrid_numerics`, `multigrid_isolated_boundary`,
`multigrid_gravity_uniform`, `multigrid_gravity_amr`,
`multigrid_gravity_multipole`, and `multigrid_gravity_nonconvergence`.
The last test verifies that a one-cycle limit reports nonconvergence and returns
failure; it does not require that solve to converge. This was the six-test
multigrid selection, not the entire application test suite.

The following results use one HPX worker. Uniform levels 2 and 3 and AMR with
multipole boundaries were additional checks, beyond those six registered tests.
Potential and force errors use the analytic continuum sphere reference described
above; the force column is the multigrid vector-force L2 error.

| Mesh and boundary | Leaves | Cycles | Relative residual | MG potential L2 | FMM potential L2 | MG force L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform level 1, analytic | 8 | 22 | `4.00062e-10` | `3.17455e-3` | `4.87591e-2` | `1.97515e-1` |
| Uniform level 2, analytic | 64 | 18 | `9.97900e-10` | `8.04827e-4` | `1.30048e-2` | `7.06625e-2` |
| Uniform level 3, analytic | 512 | 46 | `8.94467e-10` | `2.06917e-4` | `3.37242e-3` | `2.41537e-2` |
| AMR through level 3, analytic | 288 | 28 | `9.89393e-10` | `2.08676e-4` | `3.37220e-3` | `2.41877e-2` |
| Uniform level 1, multipole | 8 | 22 | `4.00088e-10` | `3.17843e-3` | `4.87591e-2` | `1.97516e-1` |
| AMR through level 3, multipole | 288 | 28 | `9.89377e-10` | `2.09323e-4` | `3.37220e-3` | `2.41879e-2` |

Halving the uniform cell width reduced MG potential L2 error by `3.94439`
and then `3.88961`, consistent with second-order potential convergence on this
problem. The represented masses at levels 1, 2, and 3 were respectively
`1.00020039792`, `1.00005008492`, and `0.99998432298`; the AMR mass matched
level 3. These finite-resolution errors include the density quadrature and
solver differences already noted, rather than measuring solver iteration error
alone.

Two further AMR/multipole checks used one locality with two HPX workers, then
two local HPX processes with one worker each over loopback TCP. Both reproduced
all reported non-timing metrics of the one-worker run exactly, including 28
cycles and 147,456 physical cells. This checks those worker/locality
configurations; it is not a multi-machine or scaling study.

The 288-leaf AMR tree has 329 total blocks and performs 1,010,688 smoother cell
updates per cycle. The uniform level-3 tree has 585 total blocks and performs
1,797,120 per cycle. Its 46 cycles, compared with 18 at uniform level 2, show
that these runs do not establish mesh-independent iteration counts. Saved
elapsed times are individual diagnostic runs, with the setup and monitoring
costs described above; no speedup claim follows from them. MG elapsed time was
higher than FMM in these diagnostics: the one-worker uniform `64^3` run took
`14.97 s` for MG versus `0.821 s` for FMM, with the differing setup and timing
scopes noted above. This validation
covers static Poisson gravity, not a variable-coefficient or FLD implementation.

The saved evidence is under
`/home/dmarce1/workspace/octotiger/output/radiation_mg_validation`: registered
test results in `build/Testing/Temporary/LastTest.log` and
`build/test_problems/multigrid/`, and additional results in `uniform_l2/`,
`uniform_l3/`, `amr_multipole/`, `amr_two_workers/`, and
`amr_two_localities/`. These directories retain the metrics, commands, and
application or locality logs; they are local validation artifacts, not required
repository inputs.

After converting internal invariant checks to assertions, specializing block
coverage to all eight children or none, and renaming the sampling helper to
`sampleNearestCell`, both standalone Debug tests passed with assertions enabled
and `-Wall -Wextra -Werror`. The Release application rebuilt successfully and
all six multigrid CTests passed again. Uniform, AMR, and multipole-boundary runs
matched every reported non-timing metric from the preceding implementation
exactly, including cycle counts, residuals, and potential/force errors. Evidence
is in `assert_refinement_build.log`, `assert_refinement_ctest.log`, and
`assert_refinement_comparison.json` under the validation directory above. This
checks numerical preservation; it is not a controlled performance comparison.
