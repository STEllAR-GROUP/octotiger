# QB4 build preparation

These files were prepared and checked locally. They have not been run on QB4.
This build script does not log in, submit a job, query Slurm, or manage access.

The root `build.sh` is a new allocation-only entry point. The older
`build_octotiger.sh` is unchanged and is not used by the QB workflow: it deletes
the requested directory, hard-codes a local Debug HPX prefix, and uses an
unbounded `make -j`. The separate OctotigerII project's `build.sh` targets a
different source tree and HPX 1.11; it is not a build recipe for this branch.

## Dependency environment

Use the same `QB_ENV_FILE` for the build and all run jobs. It must select a
consistent compiler and installed HPX **2** stack with networking and the MPI
parcelport enabled. Set `HPX_DIR` to its installed CMake package, and load the
same MPI implementation used to build it. `tools/qb4/hpx-check` verifies the
package's version and exported transport settings before configuring Octo-Tiger.
Setting consumer CMake flags cannot add MPI to a TCP-only HPX installation.

Also provide installed CPPuddle, Vc, Silo/HDF5 and Boost. The results tools need
FFTW3, OpenSSL and nlohmann_json; plotting/collection needs gnuplot. FFmpeg is
needed only for optional movies; the QB workflow disables movies by default.
Dependency paths can be supplied through `CMAKE_PREFIX_PATH` and the
package variables listed by `./build.sh --help`. `Silo_DIR` and `SILO_ROOT` are
normalized to the same prefix for the application and results tools. The
script neither builds dependencies nor installs system packages.

Copy [environment.sh.example](environment.sh.example) to your chosen
`QB_ENV_FILE` and edit its paths and compiler setup. An environment file for a
staged layout can contain the following.
These are **example destination paths**, not detected or verified installations;
replace them with the actual cluster prefixes and load the compiler module that
built HPX before exporting `CC`/`CXX`. A prefix using `lib64/cmake` instead of
`lib/cmake` needs its package-directory variables adjusted accordingly.

```bash
# Load the verified compiler module here. gcc/13.2.0 appears in older QB records,
# but no module load is performed automatically by build.sh.
QB_STACK_PREFIX=/work/$USER/octotiger-stack/hpx2-mpi
QB_COMMON_PREFIX=/work/$USER/octotiger-stack/common
QB_MPI_PREFIX=/work/$USER/octotiger-stack/openmpi
QB_SOURCE_PREFIX=/work/$USER/octotiger-stack/sources
export CC=gcc CXX=g++
export HPX_DIR="$QB_STACK_PREFIX/lib/cmake/HPX"
export CPPuddle_DIR="$QB_STACK_PREFIX/lib/cmake/CPPuddle"
export Vc_DIR="$QB_COMMON_PREFIX/lib/cmake/Vc"
export Silo_DIR="$QB_COMMON_PREFIX" SILO_ROOT="$QB_COMMON_PREFIX"
export HDF5_ROOT="$QB_COMMON_PREFIX"
export CMAKE_PREFIX_PATH="$QB_STACK_PREFIX:$QB_COMMON_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export MPI_C_COMPILER="$QB_MPI_PREFIX/bin/mpicc"
export MPI_CXX_COMPILER="$QB_MPI_PREFIX/bin/mpicxx"
export MPIEXEC_EXECUTABLE="$QB_MPI_PREFIX/bin/mpiexec"
export PATH="$QB_MPI_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$QB_MPI_PREFIX/lib:$QB_MPI_PREFIX/lib64:$QB_STACK_PREFIX/lib:$QB_STACK_PREFIX/lib64:$QB_COMMON_PREFIX/lib:$QB_COMMON_PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export FETCHCONTENT_SOURCE_DIR_KOKKOSSIMD="$QB_SOURCE_PREFIX/simd-math"
export FETCHCONTENT_SOURCE_DIR_SVESIMD="$QB_SOURCE_PREFIX/sve"
```

The common prefix in this example contains Vc, Silo, HDF5, Boost, hwloc, FFTW3,
OpenSSL and nlohmann_json, or these packages are provided by compatible loaded
modules. CPPuddle is built against the selected HPX stack. A site with separate
prefixes can list each in `CMAKE_PREFIX_PATH` and use the package-specific hints
from `--help`; there is no dependency bootstrap hidden in the build step.

Stage these two source-only dependencies before the job, preserving the
revisions pinned by the root CMakeLists.txt:

| Environment variable | Source revision |
| --- | --- |
| `FETCHCONTENT_SOURCE_DIR_KOKKOSSIMD` | `kokkos/simd-math`, `c845b7d9d8da461457cc5aa88fdda5923773a4bf` |
| `FETCHCONTENT_SOURCE_DIR_SVESIMD` | `srinivasyadav18/sve`, `a0582754c96f265b10e3c19d4fbd3e8d6debc321` |

Kokkos SIMD is header-only: its directory must contain `simd.hpp`. The SVE
directory must contain `CMakeLists.txt`.
The build disables FetchContent downloads and updates. Copy source checkouts,
not workstation binaries, CMake caches, or installed dependency prefixes.
The application test-data submodule must also be present when staging this
branch because the build enables the test targets.

## Build inside the allocated compute job

After the site-approved environment is available on a compute node:

```bash
QB_ENV_FILE=/path/to/verified-qb-environment.sh ./build.sh -j 8
```

The script requires `SLURM_JOB_ID`, `SLURM_CPUS_PER_TASK`, and a matching
`SLURMD_NODENAME`/hostname. An interactive `salloc` shell left on a login host
is insufficient. The default is at most eight jobs, bounded by both the
per-task allocation and process affinity; an explicit excessive `-j` is an
error. Build phases run sequentially. OpenMP/BLAS helper threading is set to
one and inherited unbounded Make settings are replaced.

The default outputs are:

```text
build-qb4-radiation/octotiger
build-qb4-radiation/radiation-results-tools/radiation-results
build-qb4-radiation/radiation-results-tools/gen_radiation_reference
build-qb4-radiation/qb4-build-manifest.txt
```

Use `--build-dir DIR` to choose another destination. Existing directories are
reused; select a fresh directory when changing compiler or dependency ABI.
The build is CPU Release, radiation enabled, with test targets built but not
executed. It does not establish MPI runtime correctness or benchmark accuracy.
The manifest records the job, node, parallel count, HPX/MPI selection and
executable hashes.

## Historical local evidence

The retained OctotigerII artifacts at
`docs/validation/isolated-solvers-2026-10-01/job.sh` and
`build/build-solvers.sh` in that separate checkout record `workq`, 64 CPUs per
node, `gcc/13.2.0`, and a private Open MPI 5.0.11 installation under
`/work/dmarce1/OctotigerII/packages/openmpi-5.0.11-gcc13.2/install`. They used
HPX 1.11, not this branch's HPX 2 stack. The retained documentation also records
Silo/HDF5, Boost and hwloc prefixes for those older jobs. Those records are
historical evidence, not confirmation that current modules or paths still
exist. No verified QB4 HPX 2 MPI installation was found in the local records.
The workstation HPX 2 package explicitly has its MPI parcelport disabled.

The script deliberately does not guess a replacement install prefix, module,
account, partition, MPI launcher plugin, or walltime. Supply the verified site
environment to the job scripts prepared under `radiation_results`.

Local checks require no cluster tools or dependencies:

```bash
bash -n build.sh
python3 tests/test_qb4_build.py
```

The Python checks replace all CMake and build commands with local mocks. They
cover allocation/host guards, CPU limits, a failed dependency probe, missing
staged sources, path quoting, ordered build phases, and expected output paths.
