# FLD integration, step 1: composite elliptic multigrid

This is the first radiation integration commit on `dominic_wip`. It imports the
radiation-independent linear solver from `hpxr2c_radiation` at
`393d5de64af5b0bd313bcdfd3296b18ac5ef720b`, together with its numerical regressions.
The numerical implementation is retained; include ordering, const placement,
formatting, and test registration follow this branch's conventions.

## Operator and hierarchy

For leaf-cell values u, the operator is

    L(u) = a u - ∇·(β ∇u) = f.                                  (1)

`cell::center` and `cell::width` describe the physical leaf cells;
`scalar_operator` takes the cell reaction coefficient a and face coefficient β.
Both must be finite and nonnegative. `face_fluxes(u)` evaluates −β∇u: interior
fluxes point along the positive coordinate axis, exterior fluxes point outward.
Robin data use α u + γ β ∂u/∂n = g, so an imposed outward flux q uses
`boundary_condition{0, 1, -q}`. `apply` is homogeneous; `solve` adds
`boundary_rhs()` to f. The mesh must outlive its `scalar_operator`.

`composite_mesh` validates Cartesian, dyadic, face-balanced 2:1 leaves covering
the domain. A shared coarse/fine contact is split into fine subfaces. Each
subface has one flux that enters the adjoining cell-volume balances with
opposite signs. Tangential least-squares reconstruction preserves linear
fields across refinement interfaces.

The physical unknowns remain on leaves. The linear solver builds virtual
coarser meshes, volume-weighted restriction R, signed linear prolongation P,
and Galerkin operators A_coarse = R A_fine P. Its V-cycle uses three forward
and three reverse relaxed Gauss-Seidel sweeps (weight 0.7) and a pivoted LU
bottom solve. On an 8³ uniform mesh the hierarchy continues through 4³ to 2³.

V-cycles precondition restarted GMRES (default restart 30). The reconstructed
operator can be nonsymmetric, so conjugate-gradient assumptions do not apply.
Periodic or all-Neumann Poisson solves check RHS compatibility and remove the
constant nullspace. Corrections may have either sign; there is no positivity
clipping. `solve_report` exposes convergence, residuals, iterations, V-cycles,
hierarchy levels, and a termination reason. Initial guesses already satisfying
the tolerance return before building a hierarchy.

This is a serial CPU reference solver. It has no HPX tree gather/scatter,
distributed multigrid, GPU path, radiation closure, or time integrator.
Disconnected zero-diffusion regions can introduce nullspaces beyond the
single constant mode handled here. The method does not promise monotonicity
or coefficient-independent convergence for arbitrary meshes and contrasts.

## Checks

Build independently of the application and its dependencies:

```sh
cmake -S test_problems/elliptic -B build/elliptic -DCMAKE_BUILD_TYPE=Release
cmake --build build/elliptic --parallel 1
ctest --test-dir build/elliptic --output-on-failure -j1
```

The same test is registered in a normal application build when
`OCTOTIGER_WITH_TESTS=ON`. It checks affine reproduction, shared-face flux
conservation, mixed Dirichlet/Neumann/Robin boundaries, uniform and composite
manufactured-solution convergence in all three refinement orientations,
periodic/reflecting Poisson compatibility, signed corrections, a 10⁶
coefficient jump, a constant state with diffusion 10¹⁰⁰, a finite RHS of order
10⁻²⁰⁰, and rejection of incomplete topology.

## Following commits

1. Add stationary nonlinear FLD closure and local thermal elimination on this
   operator, with tests of the coupled implicit solve.
2. Connect stationary FLD to the application: options, leaf exchange,
   boundaries, diagnostics, and restart validation.
3. Add moving-medium FLD and its coupled benchmarks.
4. Integrate M1 separately after the FLD stages are reviewable.

Those stages are not included in this commit. Existing hydro, WD, stacktrace,
and GPU initialization changes remain the integration baseline.
