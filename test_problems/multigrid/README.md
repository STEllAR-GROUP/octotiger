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
