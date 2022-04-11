#!/bin/bash -l
#SBATCH --job-name="Octo-Tiger Aggregation Performance Test HIP"
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=mi100
#SBATCH -e aggregation_test_hip.out
#SBATCH -o aggregation_test_hip.out

module load gcc/9.4.0 rocm/4.3.1 hwloc
pip3 install --upgrade matplotlib pandas
srun ./build-all.sh Release with-CC-clang without-cuda without-mpi without-papi without-apex with-kokkos with-simd without-hpx-backend-multipole without-hpx-backend-monopole with-hpx-cuda-polling without-otf2 boost jemalloc hdf5 silo vc hpx kokkos cppuddle octotiger 
srun ./src/octotiger/octotiger-performance-tests/rostam/aggregation-test/blast_aggregation_performance_hip.sh 
python3 ./src/octotiger/octotiger-performance-tests/rostam/aggregation-test/plot_blast_aggregation_performance.py --filename=performance_results.log --gpu-name='AMD MI100' | tee plot.log
