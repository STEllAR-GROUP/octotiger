# Radiation restoration and validation

Branch `hpx2rc-radiation` starts at `7485b798` (the HPX 2.0 release-candidate
port) and restores the radiation solver, diagnostics, regression problems and
results tools from `dominic-wip2` at
`2e7e57b7e91fb3cbeec2bf80b364d774e700f650`. Application integration uses C++20
and retains the base branch's `real` and `integer` types and HPX interfaces.

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

All commands above reserve at most two local compute threads. Concurrent work
shares the user's global local resource budget. No QB4 access is needed.

The restored legacy Marshak configuration and CMake file remain present, with
its CTest disabled as on `dominic-wip2`. Its historical analytic normalization
assumes c=1 and has not been validated by this port. CPU validation does not
establish GPU or multi-locality runtime coverage.
