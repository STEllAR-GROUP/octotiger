# CPU HPX 2 MPI dependency bootstrap

`bootstrap_cpu_stack.sh` builds HPX and CPPuddle from already staged sources in
an allocated compute job. It performs no download, login, submission, or system
installation. The selected environment must already provide a consistent C++20
compiler/MPI implementation and Boost/hwloc. Supply Vc, Silo/HDF5, FFTW3, OpenSSL,
nlohmann_json and gnuplot separately through existing common prefixes/modules.
Kokkos, HPXKokkos and CUDA are not needed for this CPU radiation build.

Stage the **current source contents**, including untracked compatibility files,
from these local paths into the indicated children of a shared `sources` folder:

| Destination | Local path below `hpx2rcDependencies` | Base revision |
|---|---|---|
| `hpx` | `src/hpx` | `2b2018189ab8ffd190c7ad04fd4da5534286726b` (2.0.0-rc1) |
| `cppuddle` | `src/cppuddle` | `5d189e5b910d03901b358ffa5c2539e2e71edb06` |
| `stdexec` | `build/gcc/hpx/_deps/stdexec-src` | `04de75de5807e73a229acf047c6976575e313359` |
| `asio` | `build/gcc/hpx/_deps/asio-src` | `12e0ce9e0500bf0f247dbd1ae894272656456079` (1.30.2) |

HPX, CPPuddle, and stdexec contain local compatibility edits. In particular,
the current HPX source has `cmake/HPX_PatchStdexecSyclAtomic.cmake`, which is newer
than the October 7 timestamped patch snapshot. Do not substitute that old
snapshot for the current tree. `FETCHCONTENT_SOURCE_DIR_STDEXEC` bypasses HPX's
patch step, so stage the already patched stdexec tree, not a clean checkout at
the same commit. The helper hashes the actual four source trees into
`BUILD/source-manifest.json`, excluding `.git` internals.

Inside the allocated compute job, after choosing real shared paths:

```bash
tools/qb4/bootstrap_cpu_stack.sh \
  --env /absolute/shared/path/compiler-mpi-common.sh \
  --sources /absolute/shared/path/sources \
  --build-dir /absolute/shared/path/stack-build \
  --prefix /absolute/shared/path/hpx2-mpi -j 8
```

Set `CC`, `CXX`, `MPI_C_COMPILER`, and `MPI_CXX_COMPILER` in that environment.
Optional hints include `CMAKE_PREFIX_PATH`, `Boost_ROOT`, `Hwloc_ROOT`,
`Hwloc_INCLUDE_DIR`, `Hwloc_LIBRARY`, and `MPIEXEC_EXECUTABLE`. The job count must
fit both its per-task allocation and CPU affinity. All configuration/build/install
steps stop on failure. Configure is fully disconnected and uses the supplied
Asio/stdexec sources; both CPU tests and examples are disabled during bootstrap.
`HPX_WITH_THREAD_STACK_MMAP=OFF` preserves the previously established QB stack
allocation workaround; MPI execution still needs an actual compute-node check.
`HPX_WITH_PKGCONFIG=OFF` avoids an HPX 2 RC package-generation failure when
FindMPI supplies `HOST_LINK` expressions. Octo-Tiger consumes the installed
CMake package; that package remains enabled.

Afterward set `HPX_DIR=PREFIX/lib/cmake/HPX` and
`CPPuddle_DIR=PREFIX/lib/cmake/CPPuddle` in the shared runtime environment used by
`build.sh` and the radiation workflow. That application build independently
checks that the selected HPX is version 2 with MPI/networking enabled.

The application also needs the following staged sources, already available
under `hpx2rcDependencies/build/gcc/octotiger/_deps/`:

- `kokkossimd-src`: `c845b7d9d8da461457cc5aa88fdda5923773a4bf`.
- `svesimd-src`: `a0582754c96f265b10e3c19d4fbd3e8d6debc321`.

If Vc or Silo must be rebuilt, their local clean sources are `src/vc`
(`288454deb621be1fa71d2e85cc0d81f3ee7056ea`, Vc 1.4.5) and `src/silo`
(`f05c5a8e6c22b784b7f5eb958d50218cae873d07`, Silo 4.12.1). Use the matching
compiler and an isolated common install prefix. Vc needs `BUILD_TESTING=OFF`
and `BUILD_EXAMPLES=OFF`; Silo uses `SILO_ENABLE_FORTRAN=OFF`,
`SILO_ENABLE_ZFP=OFF`, `SILO_ENABLE_FPZIP=OFF`, `SILO_ENABLE_HZIP=OFF`,
`SILO_ENABLE_SILOCK=OFF`, and `SILO_ENABLE_BROWSER=ON`, with its HDF5 prefix.
The older workstation `buildStack.py` hard-codes CUDA/compiler paths and MPI
OFF, so it is not a QB bootstrap command.

Offline checks: `bash -n tools/qb4/bootstrap_cpu_stack.sh` and
`python3 tools/qb4/test_bootstrap.py`. The tests use fake CMake and compile nothing.
