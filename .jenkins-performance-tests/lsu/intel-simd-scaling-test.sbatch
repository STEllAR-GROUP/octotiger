#!/bin/bash -l
#SBATCH --job-name="Octo-Tiger SIMD Scaling Test Intel"
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=cuda-A100
#SBATCH --nodelist=toranj1
#SBATCH -e intel_simd_scaling_test.out
#SBATCH -o intel_simd_scaling_test.out

module load gcc/11.2.1 hwloc cmake
module list
./build-all.sh Release with-CC without-cuda without-mpi without-papi with-apex with-kokkos with-simd with-hpx-backend-multipole with-hpx-backend-monopole with-hpx-cuda-polling without-otf2 boost hdf5 silo jemalloc vc hpx kokkos cppuddle octotiger
./build/octotiger/build/tools/gen_rotating_star_init
time ./src/octotiger/octotiger-performance-tests/rostam/node-level-scaling-tests/test_cpu_performance.sh src/octotiger/octotiger-performance-tests/rostam/node-level-scaling-tests/simd_rotating_star_icelake_scenario
cat LOG.txt
cp LOG.txt combined_scenario_runtimes.csv
pip3 install --upgrade matplotlib pandas numpy
python3 src/octotiger/octotiger-performance-tests/rostam/node-level-scaling-tests/plot/plot_node_level_scaling.py --filename=LOG.txt --title_prefix="Hydro + Gravity Scenario " --output_prefix="intel_avx_combined_" --simd_key="AVX"
python3 src/octotiger/octotiger-performance-tests/rostam/node-level-scaling-tests/plot/plot_node_level_scaling.py --filename=LOG.txt --title_prefix="Hydro + Gravity Scenario " --output_prefix="intel_avx512_combined_" --simd_key="AVX512"

time ./src/octotiger/octotiger-performance-tests/rostam/node-level-scaling-tests/test_cpu_performance.sh src/octotiger/octotiger-performance-tests/rostam/node-level-scaling-tests/simd_blast_icelake_scenario
cat LOG.txt
cp LOG.txt hydro_scenario_runtimes.csv
python3 src/octotiger/octotiger-performance-tests/rostam/node-level-scaling-tests/plot/plot_node_level_scaling.py --filename=LOG.txt --title_prefix="Hydro Scenario " --output_prefix="intel_avx_hydro_" --simd_key="AVX"
python3 src/octotiger/octotiger-performance-tests/rostam/node-level-scaling-tests/plot/plot_node_level_scaling.py --filename=LOG.txt --title_prefix="Hydro Scenario " --output_prefix="intel_avx512_hydro_" --simd_key="AVX512"

time ./src/octotiger/octotiger-performance-tests/rostam/node-level-scaling-tests/test_cpu_performance.sh src/octotiger/octotiger-performance-tests/rostam/node-level-scaling-tests/simd_sphere_icelake_scenario
cat LOG.txt
cp LOG.txt gravity_scenario_runtimes.csv
python3 src/octotiger/octotiger-performance-tests/rostam/node-level-scaling-tests/plot/plot_node_level_scaling.py --filename=LOG.txt --title_prefix="Gravity Scenario " --output_prefix="intel_avx_gravity_" --simd_key="AVX"
python3 src/octotiger/octotiger-performance-tests/rostam/node-level-scaling-tests/plot/plot_node_level_scaling.py --filename=LOG.txt --title_prefix="Gravity Scenario " --output_prefix="intel_avx512_gravity_" --simd_key="AVX512"
