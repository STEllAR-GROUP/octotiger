# CMake options

## GPGPU Integration
* `OCTOTIGER_WITH_CUDA`: Enable [CUDA](https://docs.nvidia.com/cuda/) FMM kernels. Default value is `OFF`.
* `OCTOTIGER_WITH_KOKKOS`: Enable the build with [Kokkos](https://github.com/kokkos/kokkos). Default value is `OFF`.

## Octo-Tiger Experiment Parameters
* `OCTOTIGER_WITH_GRIDDIM`: Grid size. Default value is `8`.
* `OCTOTIGER_THETA_MINIMUM`: Minimal allowed theta value - important for optimizations. Default value is `0.34`.
* `OCTOTIGER_WITH_GRAV_PAR`: Enable parallelism in gravitational solver. Default value is `OFF`.
* `OCTOTIGER_WITH_RADIATION`: Enable radiation transport solver. Default value is `OFF`.

## Enable test targets
* `OCTOTIGER_WITH_TESTS`: Enable tests cases. Default value is `ON`.

## Vc Library Integration
The [Vc library](https://github.com/VcDevel/Vc) is used for x86 vectorization in Octo-Tiger.
* `OCTOTIGER_WITH_Vc`: . Default value is `ON`.

The following options also exist can can be set if it necessary to override the configuration detected by Vc:
* `OCTOTIGER_WITH_AVX`: . Default value is `OFF`.
* `OCTOTIGER_WITH_AVX2`: . Default value is `OFF`.
* `OCTOTIGER_WITH_AVX512`: . Default value is `OFF`.

## Spack Integration
[Spack](https://github.com/spack/spack) is a relatively popular package manager software available on HPC machines.
* `OCTOTIGER_SPACK_BUILD`: Build project with the Spack package. Default value is `OFF`.

## Windows-specific options
This group of options exists because getting some features or tests in Octo-Tiger to build may not be difficult and unnecessary with MSVC on Windows.
* `OCTOTIGER_WITH_BOOST_MULTIPRECISION`: Use `Boost.Multiprecision` Instead of GCC Quad-Precision Math Library. Default value is `OFF`.
* `OCTOTIGER_WITH_QUADMATH`: Enable sections using GCC Quad-Precision Math Library. Default value is `ON`.
* `OCTOTIGER_WITH_BLAST_TEST`: Enable the Blast test. Default value is `ON`.

## Documentation
* `OCTOTIGER_WITH_DOCU`: Enable the target to build the documentation. Default value is `OFF`.
