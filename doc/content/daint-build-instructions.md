# Build and use Octo-Tiger on Piz Daint


### General
Piz Daint requires additional bugfixes to get around some float128 issues with Boost!
Hence we require some special branches containing these machine-specific bugfixes, in order to build Octo-Tiger on Daint!

Branches to use:
- For the [buildscripts](https://github.com/STEllAR-GROUP/OctoTigerBuildChain/tree/main) use the branch [daint-build](https://github.com/STEllAR-GROUP/OctoTigerBuildChain/tree/daint-build)
- For [Octo-Tiger](https://github.com/STEllAR-GROUP/octotiger) use the branch [daint_production_build](https://github.com/STEllAR-GROUP/octotiger/tree/daint_production_build)

Currently we use Vc + CUDA for the kernel configuration, either with the Cray or the Gnu compiler. 

### CUDA + Cray Compiler Version

#### Modules to load
- module load cray-mpich
- module load cudatoolkit
- module load daint-gpu

#### Build Command
For the root-directory of the buildscripts, run
> ./build-all.sh Release with-CC with-cuda with-mpi without-papi without-apex without-kokkos with-simd without-hpx-backend-multipole without-hpx-backend-monopole with-hpx-cuda-polling

In case the buildscripts fail because they cannot download silo, run 
> cp -r /project/s1078/silo/ src/
to use the backup we have in the project directory. Rerun the buildscripts afterwards.

#### Recommended Kernel Parameters
Add these runtime parameters to your Octo-Tiger call in addition to any scenario parameters:
> \-\-cuda_number_gpus=1 \-\-cuda_streams_per_gpu=128 \-\-cuda_buffer_capacity=1024 \-\-monopole_host_kernel_type=VC \-\-multipole_host_kernel_type=VC \-\-monopole_device_kernel_type=CUDA \-\-multipole_device_kernel_type=CUDA \-\-hydro_device_kernel_type=CUDA \-\-hydro_host_kernel_type=LEGACY

### CUDA + GNU Compiler Version

#### Modules to load
- module load cray-mpich
- module swap PrgEnv-cray/6.0.9 PrgEnv-gnu
- module load cudatoolkit
- module load daint-gpu

#### Build Command
For the root-directory of the buildscripts, run
> ./build-all.sh Release with-CC with-cuda with-mpi without-papi without-apex without-kokkos with-simd without-hpx-backend-multipole without-hpx-backend-monopole with-hpx-cuda-polling

In case the buildscripts fail because they cannot download silo, run 
> cp -r /project/s1078/silo/ src/
to use the backup we have in the project directory. Rerun the buildscripts afterwards.

#### Recommended Kernel Parameters
Add these runtime parameters to your Octo-Tiger call in addition to any scenario parameters:
> \-\-cuda_number_gpus=1 \-\-cuda_streams_per_gpu=128 \-\-cuda_buffer_capacity=1024 \-\-monopole_host_kernel_type=VC \-\-multipole_host_kernel_type=VC \-\-monopole_device_kernel_type=CUDA \-\-multipole_device_kernel_type=CUDA \-\-hydro_device_kernel_type=CUDA \-\-hydro_host_kernel_type=LEGACY
