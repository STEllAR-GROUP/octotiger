# Radiation validation

`radiation_model=m1` is the default and uses the original first-order Euler
(RK1) method. `radiation_model=fld` selects stationary two-temperature flux
limited diffusion with implicit backward Euler and scalar multigrid. FLD
requires CPU execution, physical light speed, ideal gas, and hydro/gravity/rotation
disabled. Its stored flux is a derived diagnostic, so the M1 momentum ledger
is not an FLD invariant.

## Stationary FLD checks

`radiation_elliptic` and `radiation_fld` exercise the standalone composite
operator/multigrid and nonlinear FLD physics. `radiation.fld.application`
checks homogeneous thermal exchange on levels 0 and 1, physical total-energy
balance, derived flux bounds, timesteps above the M1 streaming restriction,
and checkpoint continuation with deliberately conflicting model options. A
Marshak case exercises spatial transport and injected boundary energy using
the verification problem's dimensionless units and linear emission law.
Successful timestep retry checks accepted-substep conservation, and forced
solver exhaustion checks bounded collective failure without a completed
checkpoint. The optional adaptive case evolves further after refinement.

```sh
ctest --test-dir build -L fld --output-on-failure --parallel 1
python3 tests/radiation/test_fld_application.py \
  --executable build/octotiger --threads 2 --mixed
```

The optional adaptive run requires both level-1 and level-2 leaves. These
application smoke checks do not establish convergence or moving-gas support.
Each run keeps its configuration, log, and validation report in the printed
output directory.

The October 13, 2026 validation passed all 17 standalone tests and the
[ten-run FLD application suite](../output/fld-validation/20261013/accepted-thread2/validation.json),
including six unsupported configurations and one exhausted-solver check.
See [the method scope and measured results](../doc/radiation-fld.md) for
conservation, restart, thread-count comparison, and remaining limitations.

## Euler radiation subcycling

The focused check forces hydro timesteps above the radiation CFL and verifies
that the original method performs radiation substeps:

```sh
python3 tests/radiation/test_euler_subcycling.py \
  --executable cmake-build-hpx2rc-radiation/octotiger \
  --threads 2 --work-dir output/euler-subcycling-rerun
```

The [completed eight-run report](../output/ader2-validation/euler-subcycling/validation.json)
checks smooth coupled configurations at levels 0 and 1 with four and eight
radiation substeps. Stop/restart errors were `9.95e-15` and `5.10e-15`; the
largest total-energy drift was `3.03e-16`. Homogeneous repeated Euler substeps
agreed with the matching global `dt=0.02` update to `2.99e-16`. Level 1 exercises
parent nodes and halos, but this is not mixed-level AMR or a temporal-order test.

## Application build

Use the dependency paths and other options required by your existing HPX 2RC
build, adding these options:

```sh
cmake -S . -B build -DOCTOTIGER_WITH_RADIATION=ON -DOCTOTIGER_WITH_TESTS=ON
cmake --build build --parallel 2
ctest --test-dir build -L radiation --output-on-failure --parallel 1
```

The four application regressions use 32 cells per side (`GRIDDIM=8`, level 2),
physical radiation fluxes, two HPX workers and isolated output directories.
Their explicit `code_to_s=1/2.99792458e10`, `code_to_cm=1` unit conversion makes
c=1 for the dimensionless analytic problems. Their final time is 0.2. The
Gaussian reference is generated automatically by a CTest fixture; FFTW3 is
required. Numerical thresholds were recovered from the comparator bytecode
tracked on `dominic-wip2`; they have not been loosened for this restoration.
`radiation.application.restart` additionally compares uninterrupted and
checkpoint/resumed runs and checks restored radiation metadata.
`radiation.application.restart.reduced_c` repeats that check with
`radiation_reduced_c_factor=0.1` and a deliberately conflicting restart
configuration. `radiation.application.reduced_c_options` rejects zero,
negative, greater-than-one and nonfinite values through both CLI and config,
checks help/metadata, and compares omitted-factor output with explicit factor
1. Radiation metadata version 3 stores the model, integrator, algorithm version,
limiter, flux interpretation, and reduced light speed. FLD checkpoints also
store boundary and solver settings. Version 1 and legacy M1 checkpoints
restore factor 1 rather than taking a conflicting restart value.

## Coupled shock and blast benchmarks

The `subcritical_shock` and `supercritical_shock` configurations use a planar
reflecting wall. `radiating_blast` initializes a spherical atmosphere and
explosion on the three-dimensional Cartesian mesh. Their parameters come from
Ensman (1994); geometric and initial-condition differences are documented in
[`radiation_results/ENSMAN.md`](../radiation_results/ENSMAN.md).

```sh
ctest --test-dir build -L radiation-hydro --output-on-failure --parallel 1
```

`radiation.unit.hydro_benchmarks` checks the CGS equation of state, initial LTE
radiation and its physical advective flux, exact shock cell averages, unit
conversions, the regularized blast atmosphere, and the deposited energy at
8, 16 and 32 cells per side. `radiation.production.hydro_boundaries` compiles
the actual new gas and radiation boundary branches and opacity functions with
UBSan. It checks all six faces and three ghost layers, wall reflection,
upstream inflow, unclipped transverse values, angular momentum and opacity
units. It uses fixture arrays and is separate from the application tests.

The three `radiation.application.*.smoke` checks each run four coupled CPU
steps with two HPX workers and 16 cells per side. They inspect every cell in
the initial/final Silo files for finite, positive matter and radiation states,
physical flux realizability and evolution of both components. The shock runs
must compress the gas; each run must exchange thermal energy and close its
radiation energy budget including source and boundary terms. These short runs
also check that the website export contains a complete central slice, matches
the checkpoint time, and obeys the CGS radiation temperature relation. They
exercise the implementation; they do **not** establish subcritical or
supercritical shock structure at the paper's final times, or validate the
blast against the paper's spherical calculation. Temporary run directories
are printed and retained with logs and `validation.json` for inspection.

## Optional gradient refinement checks

Gradient refinement is disabled by default. The `gradient` CTest label covers
absolute and logarithmic indicators, the actual grid tagging branch (including
different hydro/radiation halo widths), invalid options, and checkpoint settings.
The numerical/grid checks run with `-ffast-math`; the extracted grid check also
uses UBSan. `amr.gradient.options` compares default and explicit `none` results
in the current executable, including hydro and gravity fields, and compares a
resumed density-gradient run with an uninterrupted run.
`amr.gradient.restart_zero` refines a radiation checkpoint saved at time zero
and verifies every child's radiation field against its saved coarse parent,
including the volume-integrated energy and fluxes.

```sh
ctest --test-dir build -L gradient --output-on-failure --parallel 1
```

`amr.gradient.radiation` runs six steps of the illuminated sphere with mixed
level-3/4 leaves and four HPX workers. It checks initial and post-regrid leaf-cell
counts, exact domain coverage, and the radiation energy ledger. This catches
startup ghost-coordinate initialization and mixed-level exchange failures that
uniform-grid tests cannot exercise. Logs and actual run configurations are
retained in the printed temporary directory. This is a short integration check,
not a full-duration accuracy or convergence result. Threshold proposals and
bounded checks for all seven problems at levels 2–5 are described in
[`radiation_results/amr/README.md`](../radiation_results/amr/README.md).

## Standalone numerical tests

These need a C++20 compiler, CMake, Python 3 and FFTW3, but no HPX or Silo:

```sh
cmake -S tests -B /tmp/octotiger-radiation-tests -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/octotiger-radiation-tests --parallel 2
ctest --test-dir /tmp/octotiger-radiation-tests --output-on-failure --parallel 1
```

The suites cover M1 closure and realizability across scalar types/dimensions,
thermal coupling, FPE guards, actual production grid/source/AMR methods,
physical flux storage and conservation budgets, driver collection lifecycle,
Fourier/reference data, strict result comparison, and all four analytic
problems on a serial mesh. The production grid harness also runs debug,
release and UBSan variants. These extracted-method checks complement the
actual application tests; they do not provide distributed MPI/GPU coverage.

Legacy `tests/radiation/test_m1.cpp` and `test_grid.py` entry points now forward
to their maintained suites. The reference tests use the current camelCase
API. The conservation solver tests use standard-library CSV verification;
they no longer depend on the removed Python results application.

Application tests use at most four HPX workers and the illustrated builds use
two parallel jobs. Concurrent work shares the user's global local resource
budget. No QB4 access is needed.

The restored legacy Marshak configuration and CMake file remain present, with
its CTest disabled as on `dominic-wip2`. Its historical analytic normalization
assumes c=1 and has not been validated by this port. CPU validation does not
establish GPU or multi-locality runtime coverage.
