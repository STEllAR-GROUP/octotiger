The hpx2rc Jenkins dependency stack
=================================

Jenkins runs the updated PowerTiger build driver with a private Spack recipe
repository. PowerTiger is fetched at the commit in powertiger-lock.json and
the reviewed local update is applied after checking its SHA256. No moving
PowerTiger branch or separate upstream pull request is required.

stack-lock.json records the locally validated HPX 2.0.0-rc1, pristine Kokkos
5.2.2, HPXKokkos, CPPuddle and Silo 4.12.1 commits. Adjacent patches reproduce the local
source trees. HPX applies the nested stdexec patches during configuration.
Kokkos has no source patch. C++20 is used throughout this stack.

Silo 4.12.1 includes the upstream HDF5 1.14 driver-table and curve-name fixes
needed by modern Clang. It builds from pristine source with CMake and without
the unused Fortran interface. The ROCm HPX configuration disables discovery
of optional hipBLAS, which Octotiger does not use, while keeping HIP compute
enabled. Neither correction changes the numerical tests.

The driver stages recipes, patches, compiler configuration and package-index
cache per matrix row. Dependency source and build directories are also private
to each row's fresh checkout, so interrupted shared stages cannot be reused.
Installed packages and downloaded source archives remain shared through Spack.
GCC rows request GCC 13.2.0: a working existing compiler
is detected on the current operating system, or GCC is built from source with
the available GCC 11 compiler. Compiler bootstrapping is serialized and stays
within the row's Slurm allocation. Installation reuse is handled by Spack's
dependency and patch hashes; old site HPX/Kokkos recipes are not modified.

The original eighteen CPU, CUDA, HIP and SYCL configurations remain. Clang CUDA
uses native Clang and static Kokkos, as in local validation. The SYCL row builds
Intel LLVM v6.0.1 (oneAPI 2025.0 series, exposed as dpcpp@2025.0.1) from pinned
source with its CUDA adapter. This release supports the row's Volta target;
the old site's SYCL compiler is below Kokkos 5.2's supported compiler version.
Its GCC host configuration enables the pinned release's compatibility option
for GCC_INSTALL_PREFIX, preserving the selected GCC's headers and runtime.
The official build instructions are at:
https://github.com/intel/llvm/blob/v6.0.1/sycl/doc/GetStartedGuide.md

NVIDIA CUDA and SYCL rows explicitly request one GPU with --gres=gpu:1.
Rostam requires this even for exclusive node allocations; an exclusive CPU
allocation alone does not grant NVIDIA device access. Scheduler visibility
settings are preserved. See https://wiki.rostam.cct.lsu.edu/en/slurm/gpu.

CTest remains serial within a row, streams results, and limits individual
tests to one hour. All existing tests and numerical tolerances are retained.
Jenkins archives build provenance, private recipes, compiler configuration,
CTest results and redirected per-problem logs, including on failure.

Validation scope: the dependency patches reproduce the previously passing
GCC13/Clang21 CUDA12.8 local stack. Jenkins uses GCC13.2/Clang20/CUDA12.9 and
separate HIP/SYCL configurations; those require actual Jenkins results.
Octotiger's HPX 1.9.1 source compatibility is retained, but this pipeline
intentionally tests HPX 2.0.0-rc1.
