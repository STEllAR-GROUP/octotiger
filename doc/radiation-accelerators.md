# Radiation CUDA and Kokkos backends

`radiation_backend=cpu` is the default and retains the existing solver. Select
`cuda` or `kokkos` in an Octo-Tiger configuration file, or pass
`--radiation_backend=cuda` on the executable command line. A backend missing
from the build is an error; selection never silently falls back to CPU.

Both accelerator paths implement the current M1 primitive conversion, limited
reconstruction, HLL fluxes, conservative update, rotating-frame flux rotation,
and local radiation–matter coupling. They preserve physical flux storage,
reduced-light-speed factors, and the host's long-double source accounting.
Device results are checked before host outputs are committed. AMR boundary
exchange and refluxing remain on the host: corrected face fluxes are uploaded
before advancing. EOS/composition-dependent opacity evaluation and the
prescribed-medium regression source remain on the host as well.

The old `cpu_kernel.hpp`/`cuda_kernel.hpp` compatibility interface is not the
active M1 solver. Runtime dispatch is in `rad_grid.cpp`, with shared scalar
device equations in `accelerator_math.hpp` and separate CUDA/Kokkos launchers.

## Build and run on this workstation

The validated local executable is
`cmake-build-radiation-accelerators/octotiger`. Its configuration uses the
existing HPX dependencies, GCC 13, CUDA 12.8, and Kokkos 5.2.2 with CUDA enabled.
The original CPU build directory is separate. To reproduce the GPU build:

```sh
cmake -S . -B cmake-build-radiation-accelerators \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/hpx2rcDependencies/install/gcc;$PWD/hpx2rcDependencies/install/common" \
  -DSilo_DIR="$PWD/hpx2rcDependencies/install/common" \
  -DFETCHCONTENT_SOURCE_DIR_KOKKOSSIMD="$PWD/hpx2rcDependencies/build/gcc/octotiger/_deps/kokkossimd-src" \
  -DFETCHCONTENT_SOURCE_DIR_SVESIMD="$PWD/hpx2rcDependencies/build/gcc/octotiger/_deps/svesimd-src" \
  -DOCTOTIGER_WITH_RADIATION=ON \
  -DOCTOTIGER_WITH_RADIATION_CUDA=ON -DOCTOTIGER_WITH_RADIATION_KOKKOS=ON \
  -DOCTOTIGER_WITH_CUDA=OFF -DOCTOTIGER_WITH_KOKKOS=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build cmake-build-radiation-accelerators --target octotiger --parallel 2
```

The radiation-specific build flags enable these kernels without enabling the
hydro/gravity accelerator implementations. Architectures and dependency paths
must match another machine; the QB A100 microbenchmark uses architecture 80.

The existing website runner accepts the execution choice and saves it with the
batch, including the generated solver configurations and resume metadata:

```sh
bash radiation_results/run_live.sh all 2 3 4 \
  --build cmake-build-radiation-accelerators \
  --study-config radiation_results/local-study.json \
  --radiation-backend cuda --threads 12 --jobs 2 \
  --output radiation_results/results/local-cuda-l234 --no-build --no-movies
```

Use `kokkos` instead of `cuda` to select Kokkos. This command starts the full
physical study; the tests described below run only short validation samples.
The runner rejects a conflicting backend when resuming an existing batch.

## Implicit solve and measured algorithm choice

With frozen opacity and the ideal-gas thermal relation, the implicit thermal
equation is `e + A e^4 = R`. Scaling `e=s*y` with
`s=min(R,(R/A)^(1/4))` gives `a*y+b*y^4=1`, where the coefficients are in
`[0,1]` and at least one is exactly one. For a positive right-hand side, the
positive dimensionless root is in `[0.724491959...,1]`; the physical energy
has no corresponding bound. Zero-opacity and zero-energy cases are handled
separately in the solver.

`positive_quartic.hpp` contains a bounded, rationalized analytic formula and a
fixed six-step Newton alternative. Both passed 137,532 CPU coefficient cases
against independent long-double bisection. The CUDA microbenchmark compares
those against the existing safeguarded Newton method with identical input
sets, coherent and shuffled warp placement, five rotated measurement rounds,
and device results checked against the same independent reference.

On QB's A100 80GB PCIe, median kernel times for 262,144 roots were:

| Method | Coherent coefficients | Shuffled coefficients |
| --- | ---: | ---: |
| Existing safeguarded Newton | 8.3 us | 10.5 us |
| Analytic formula plus fixed polish | 10.2 us | 16.1 us |
| Fixed six-step Newton | 11.7 us | 17.6 us |

The existing method also won on the local RTX 3090 for this coefficient sample.
The production device kernels therefore retain the same scaled safeguarded
Newton solve as the CPU: an 80-iteration safety limit and a 16-epsilon residual
criterion. These timings cover only the normalized root calculation, not
scaling, the complete source update, data transfers, or an application speedup.
The analytic/fixed-step methods achieve tighter residuals in this experiment;
the methods are compared against the existing solver's accuracy requirement.

Raw evidence is under
`radiation_results/results/accelerator-validation-20261011/`, including
`quartic-comparison.json`, the local CSV, and QB job 1083966's sources, hashes,
compiler metadata, five timing rounds, and Slurm accounting. QB used CUDA
12.2.1 and GCC 8.5 for this standalone C++17 benchmark. That job completed in
four seconds: 64 billing units/hour implies approximately 0.0711 SUs. Login
use for the task was 4 of the authorized 12, and the SSH session was closed.

## Validation and limits

`test_accelerator.cpp` compares actual CUDA/Kokkos outputs with independent
host M1 fluxes and coupling, including vacuum, streaming, rotation, physical
units, reduced light speed, stiff sources, invalid states, and corrected face
fluxes. `test_accelerator_application.py` runs four-step coupled shock/blast
checks and compares all saved fields against the CPU, checking conservation
and realizability through the existing application tests. These are not
completed Ensman benchmark runs or RSLA accuracy studies.

Local validation on 2026-10-11 passed 18 CPU radiation/gradient tests, four
quartic/backend tests (405,521 parity checks per backend), and all three
website-runner tests. All six CUDA/Kokkos versus CPU shock/blast comparisons
passed, with worst field-scaled difference 5.3e-15. Both backends passed the
six-step mixed L3/L4 illuminated-sphere test: the fine/coarse leaf-cell ratio
was 0.5333 and the worst radiation energy ledger residual was 1.4e-16.
A short L2 streaming-wave run through the original website runner produced
verified CUDA output, profile plots, and HTML pages in the validation directory.

CUDA reuses up to twelve device-specific workspaces with independent streams;
additional simultaneous calls use temporary workspaces. Kokkos reuses one
workspace protected by a mutex. Calls currently synchronize with the host at
each stage; batching many leaf blocks into a launch remains future work.
The Kokkos backend uses its configured default execution space. CUDA has been
tested here. An HPX default execution space is explicitly rejected to avoid
holding a blocking mutex across an HPX task suspension; use the CPU radiation
backend for that configuration. Serial/OpenMP configurations are not validated
by the CUDA results. Multiple GPUs and multiple localities have not been
validated for these radiation launchers.
