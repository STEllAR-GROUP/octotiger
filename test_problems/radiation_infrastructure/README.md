# Shared radiation infrastructure

Second staged import from `hpxr2c_radiation` snapshot
`393d5de64af5b0bd313bcdfd3296b18ac5ef720b`, following the standalone elliptic
multigrid foundation. The destination already had a legacy radiation runtime.
This step separates and repairs its generic machinery; it does not yet connect
the composite solver to an FLD nonlinear solve or import the newer M1 model.

`src/radiation/exchange.cpp` contains sibling face/edge/corner exchange,
bottom-up volume restriction, fine-face flux restriction, and coarse-to-fine
halo interpolation. AMR envelopes are received concurrently but unpacked
serially because their shadow stencils overlap. Shadow allocation/indexing is
separate from radiation indexing. Transfer payload lengths are checked before
unpacking. New child creation uses conservative piecewise-constant prolongation,
including a restart at time zero. Halo interpolation uses limited linear slopes
and one common slope reduction for all eight children and every field, preserving
the parent average. The generic helper accepts a model admissibility predicate;
the current caller requires finite fields and nonnegative radiation energy.

`src/radiation/iteration.cpp` provides the ordered tree maximum collective needed
by shared subcycling and subsequent nonlinear iterations. Every node participates;
only leaves contribute a timestep constraint and advance physical state. The
legacy explicit light-crossing limit and RK2/source kernels are retained for now.
All nodes use the same substep count. Each interval is formed from fractions of
the hydro step, with the last endpoint exactly equal to the requested duration.
Physical radiation boundaries use RK stage times. The final exchange refreshes
parents and ghosts before regridding; hydro is refreshed at the same endpoint
after material coupling. The node clock advances after this work.

The seven-field legacy storage layout and checkpoint format are unchanged.
Model selection, nonlinear FLD/material coupling, moving-medium FLD, new M1
transport, and accelerators remain later imports.

## Tests

The standalone `radiation.infrastructure` test checks affine interpolation,
conservation through positivity limiting, signed moments, vacuum, invalid states,
substep adjacency, exact final endpoints, and count/interval rejection:

```sh
cmake -S test_problems/radiation_infrastructure -B build_radiation_infrastructure
cmake --build build_radiation_infrastructure --parallel 1
ctest --test-dir build_radiation_infrastructure --output-on-failure
```

In an application build with `OCTOTIGER_WITH_TESTS=ON`, the
`radiation.grid_transfers` test links the production `octolib` and runs under HPX.
It checks all 26 sibling directions, refinement/restriction round trips in all
eight octants, area-averaged flux restriction, and overlapping coarse halo
interpolation against an analytic affine field. It also verifies interior cells
are untouched by halo interpolation. The `radiation.tree_smoke` case evolves a nine-node tree through three small
hydro timesteps with radiation enabled and checks the final clock. The small
step isolates the exchange/integrator path from the legacy near-streaming
transport limitation described below. Run the suite with:

```sh
cmake --build build --target octotiger radiation_infrastructure_test radiation_grid_transfers_test --parallel 1
ctest --test-dir build -R '^radiation\.' --output-on-failure --parallel 1
```

These tests cover the shared numerical/data-transfer layer. They do not establish
FLD convergence, M1 realizability, multi-locality correctness, GPU behavior, or
full coupled radiation-hydrodynamic accuracy.

A larger-step exploratory run of the existing `radiation_test` problem
(`hard_dt=0.03`, `stop_time=1`, `stop_step=1`) reaches the legacy flux-limit
assertion in `rad_grid::advance`. That function is unchanged apart from const
placement in this import. Consequently the small-step tree smoke and transfer
checks must not be interpreted as validation of larger-step legacy transport.
Resolve/retest that path when importing the transport model.
