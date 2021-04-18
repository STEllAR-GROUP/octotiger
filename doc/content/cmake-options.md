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
* `OCTOTIGER_WITH_Vc`: Enable the use of Vc in Octo-Tiger. Default value is `ON`.

The following options also exist can can be set if it necessary to override the configuration detected by Vc:
* `OCTOTIGER_WITH_AVX`: Force Vc to use [AVX1 instructions](https://software.intel.com/content/www/us/en/develop/articles/introduction-to-intel-advanced-vector-extensions.html). Default value is `OFF`.
* `OCTOTIGER_WITH_AVX2`: Force Vc to use [AVX-2 instructions](https://software.intel.com/content/www/us/en/develop/blogs/haswell-new-instruction-descriptions-now-available.html). Default value is `OFF`.
* `OCTOTIGER_WITH_AVX512`: Force Vc to use [AVX-512 instructions](https://software.intel.com/content/www/us/en/develop/articles/intel-avx-512-instructions.html). Default value is `OFF`.

## Spack Integration
[Spack](https://github.com/spack/spack) is a relatively popular package manager software available on HPC machines.
* `OCTOTIGER_SPACK_BUILD`: Build project with the Spack package. Default value is `OFF`.

## Blast Test Problem 
This group of options exists because getting the [Blast test problem](https://github.com/STEllAR-GROUP/octotiger/tree/master/src/test_problems/blast) in Octo-Tiger to build may be difficult and unnecessary when the [GCC Quad-Precision Math Library](https://gcc.gnu.org/onlinedocs/libquadmath/) is unavailable.

* `OCTOTIGER_WITH_BLAST_TEST`: Enable or disable the Blast test problem. Default value is `ON`.
* `OCTOTIGER_WITH_BOOST_MULTIPRECISION`: Use [`Boost.Multiprecision`](http://www.boost.org/doc/libs/release/libs/multiprecision/index.html) instead of GCC Quad-Precision Math Library in the Blast test problem. Default value is `OFF`.
* `OCTOTIGER_WITH_QUADMATH`: Force the use of GCC Quad-Precision Math Library. Default value is `ON`.

## Documentation
* `OCTOTIGER_WITH_DOCU`: Enable the CMake target for building Octo-Tiger documentation. Default value is `OFF`.
