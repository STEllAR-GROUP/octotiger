#!/bin/bash -l
#SBATCH --job-name="Octo-Tiger Aggregation Performance Test CUDA"
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=cuda-A100
#SBATCH --nodelist=toranj0
#SBATCH -e aggregation_test_cuda.out
#SBATCH -o aggregation_test_cuda.out

module load llvm/12.0.1 cuda hwloc
./build-all.sh Release with-CC-clang with-cuda without-mpi without-papi without-apex with-kokkos with-simd without-hpx-backend-multipole without-hpx-backend-monopole with-hpx-cuda-polling without-otf2 boost jemalloc hdf5 silo vc hpx kokkos cppuddle octotiger 
./src/octotiger/octotiger-performance-tests/rostam/aggregation-test/blast_aggregation_performance_cuda.sh 
pip3 install --upgrade matplotlib pandas 
python3 ./src/octotiger/octotiger-performance-tests/rostam/aggregation-test/plot_blast_aggregation_performance.py --filename=performance_results.log --gpu-name='NVIDIA A100' | tee plot.log
