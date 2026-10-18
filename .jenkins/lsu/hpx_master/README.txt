Jenkins HPX master dependency stack
==================================

All eighteen matrix entries use the same exact HPX master revision recorded in
stack-lock.json. The private HPX recipe checks and prints its actual Git HEAD
before configuration (or verifies identifying file hashes for Spack source
archives without Git metadata). The source pin is the top of master resolved for this
change; a moving branch is not fetched independently for each matrix row.

This bundle reuses the successful hpx2rc PowerTiger/Spack scaffolding with a new
private namespace. It selects C++20, Kokkos 5.2.2, HPX-Kokkos 0.4.1, CPPuddle
0.4.0 and Silo 4.12.1. GCC rows use GCC 13.2.0, Clang rows use LLVM 20, the HIP
row uses ROCm 6.4.1, and the SYCL row builds the pinned DPC++/oneAPI 2025.0.1
compiler with its CUDA adapter. These match the modern dependency requirements;
the previous HPX 1.9.1/Kokkos 4/GCC 11 stack is not reused.

HPX master supplies the initial stdexec/NVCC workarounds. GCC/CUDA also applies
the reviewed NVCC compatibility subset from HPX PR #7711 revision 8767d155,
rebased onto this exact master revision. It fixes host sender instantiation,
exports STDEXEC_DEMANGLE_SENDER_NAMES consistently to C++ and CUDA consumers,
and preserves the NVCC operation-state destructor workaround. The recipe
checks all three patched source hashes; stack-lock.json records the full
upstream revision, patch checksum, and resulting hashes. Rebuilding HPX and
its consumers together is required because the macro changes sender types.
Only SYCL adds the separate atomic-header opt-out: CUDA CCCL atomic declarations
conflict with the pinned DPC++ headers. HPX exports that choice to every
consumer. HPX-Kokkos replaces the removed HPX range trait with std::ranges and
uses Kokkos's public sycl_queue() accessor. CPPuddle uses the current HPX
futures header and hpx::post executor API. Kokkos itself remains pristine.

PowerTiger, all private dependency sources, and compatibility patches are
locked with commits/checksums. Each row uses private stages, recipes, compiler
configuration and package caches; installed dependencies may be reused only
through their complete Spack dependency/patch hashes. The Jenkins log and
archived overlay manifest record the exact pinned stack and compiler choices.

The driver limits build jobs and HPX workers to the Slurm allocation. CTest
runs one case at a time. GPU rows request a GPU explicitly and preserve Slurm's
device visibility; the SYCL row prints the visible devices before testing.
The pipeline does not send notification email.

Historical local validation of hydro_changes passed 255/255 full CPU tests
and 70/70 targeted tests against this HPX master revision, using GCC13 and
Kokkos HPX/Serial. These results predate the added NVCC patch and do not
validate that patch or CUDA/HIP/SYCL execution. Fresh validation of the
patched dependency stack is required before claiming GPU matrix success.
