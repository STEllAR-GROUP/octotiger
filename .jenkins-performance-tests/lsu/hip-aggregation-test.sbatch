#!/bin/bash -l
#SBATCH --job-name="Octo-Tiger Aggregation Performance Test HIP"
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=mi100
#SBATCH --nodelist=kamand0
#SBATCH -e aggregation_test_hip.out
#SBATCH -o aggregation_test_hip.out

echo "Aggregation Test running on $(hostname):"
echo "Loading modules..."
module load llvm/14 rocm/5 hwloc cmake
module list
echo "Building Octo-Tiger and toolchain..."
echo ""
srun ./build-all.sh Release with-CC-clang without-cuda without-mpi without-papi without-apex without-kokkos with-simd without-hpx-backend-multipole without-hpx-backend-monopole with-hpx-cuda-polling without-otf2 boost jemalloc hdf5 silo vc hpx kokkos cppuddle octotiger 
echo "Running aggregation performance test..."
srun ./src/octotiger/octotiger-performance-tests/rostam/aggregation-test/blast_aggregation_performance_hip.sh 
echo "Upgrading required python modules..."
pip3 install --upgrade matplotlib pandas
echo "Creating plots..."
python3 ./src/octotiger/octotiger-performance-tests/rostam/aggregation-test/plot_blast_aggregation_performance.py --filename=performance_results.log --gpu-name='AMD MI100' | tee plot.log
