# Hydro scenarios

The default adapter reads a configured build's authoritative CTest JSON inventory
and runs every enabled matching scenario variant with its original field regexes,
Silo comparisons, fixture setup and cleanup. It does not copy tolerances or
reimplement numerical checks. The descriptor regex subset is used only by the
explicit `--exe`-without-`--build` diagnostic smoke interface, which cannot pass
regression certification.

```bash
python3 verification_results/runner.py run hydro Release --build /path/to/fresh/build --ctest /path/to/ctest --output /path/to/new/results
```

CTest >=3.21, a built application, enabled legacy tests, Silo browser/reference
data, and all enabled variant dependencies are required. CTest fixtures always
run serially to avoid shared-output collisions. `--threads=N` sets the HPX
worker count for each Octo-TIGER process through `HPX_COMMANDLINE_OPTIONS` while
leaving CTest parallelism at one.

Each variant executes serially to avoid shared `final.silo`/log collisions. Do not
run other CTests concurrently in the same build. Raw
Silo/data/log files are copied, SHA-256 checked and synced before registered cleanup
runs. JUnit, verbose output, original registrations, config hashes and per-variant
results are retained. Existing scenario data in the build tree causes refusal;
use a fresh configured build. Missing/disabled registrations are conditional,
not a pass. A configured-suite pass does not establish pre/post migration
numerical equivalence or cover unconfigured CUDA/HIP/VC/Kokkos/SYCL options.

The historical `hydro.amr.sod_big_amr` identifier remains for compatibility but
means **higher-resolution Sod** (`sod_big.ini`), not the separate `amr_test`
problem in `test_problems/amr/amr.ini`. The latter has no authoritative CTest
registration and remains unvalidated. Sod and Sod-big selections do not overlap.
