#!/usr/bin/env bash
# Build the CPU radiation application and results tools within a compute job.
# This script does not submit jobs, connect to a cluster, or install dependencies.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./build.sh [--env FILE] [--build-dir DIR] [-j JOBS]

Run on an allocated Slurm compute node after loading the matching compiler,
HPX 2 MPI, CPPuddle, Vc, Silo/HDF5, Boost, FFTW3, OpenSSL and nlohmann_json.
QB_ENV_FILE (or --env) is sourced for that environment. HPX_DIR must name
the installed HPX CMake package. Workstation TCP-only HPX is rejected.

Defaults: Release CPU build with radiation and tests enabled, build directory
build-qb4-radiation, at most 8 build jobs. JOBS/-j may select another count up
to both SLURM_CPUS_PER_TASK and the current process's CPU affinity. Tests are
built but not run. Existing build directories are reused, never removed.

The two pinned FetchContent sources must already exist. Set
FETCHCONTENT_SOURCE_DIR_KOKKOSSIMD and FETCHCONTENT_SOURCE_DIR_SVESIMD.
No dependency download or package installation is performed by this script.

Output:
  BUILD_DIR/octotiger
  BUILD_DIR/radiation-results-tools/radiation-results
  BUILD_DIR/radiation-results-tools/gen_radiation_reference

Dependency hints accepted from the environment include CMAKE_PREFIX_PATH,
CPPuddle_DIR, Vc_DIR, Silo_DIR (or SILO_ROOT), HDF5_ROOT, Boost_ROOT,
MPI_CXX_COMPILER, MPI_C_COMPILER, MPIEXEC_EXECUTABLE, FFTW3_INCLUDE_DIR,
FFTW3_LIBRARY, OPENSSL_ROOT_DIR, nlohmann_json_DIR and Python3_EXECUTABLE.
CC/CXX select the
same compiler used for HPX. Keep that environment for subsequent runs.
EOF
}

fail() { printf 'build.sh: %s\n' "$*" >&2; exit 2; }
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build_dir="$project_dir/build-qb4-radiation"
env_file=${QB_ENV_FILE:-}
requested_jobs=${JOBS:-}
while (($#)); do
    case "$1" in
        --env|--build-dir|-j|--jobs)
            (($# >= 2)) || fail "$1 requires a value"
            case "$1" in
                --env) env_file=$2 ;;
                --build-dir) build_dir=$2 ;;
                *) requested_jobs=$2 ;;
            esac
            shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) fail "unknown argument: $1" ;;
    esac
done

# salloc may leave a shell on the login host. A job id alone is insufficient.
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || fail 'run inside an allocated Slurm compute job'
[[ ${SLURM_CPUS_PER_TASK:-} =~ ^[1-9][0-9]*$ ]] || fail 'SLURM_CPUS_PER_TASK must be set by the allocation'
[[ -n ${SLURMD_NODENAME:-} ]] || fail 'SLURMD_NODENAME is required; do not build on the login host'
node=$(hostname -s)
[[ "$node" == "${SLURMD_NODENAME%%.*}" ]] || fail 'the current host is not the allocated compute node'
if [[ -n "$env_file" ]]; then
    [[ -r "$env_file" ]] || fail "cannot read QB_ENV_FILE: $env_file"
    # An explicitly supplied environment file may load site modules and paths.
    source "$env_file"
fi

for program in cmake nproc sha256sum; do
    command -v "$program" >/dev/null || fail "missing program: $program"
done
affinity_cpus=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)
[[ "$affinity_cpus" =~ ^[1-9][0-9]*$ ]] || fail 'could not determine the CPU affinity limit'
cpu_limit=$SLURM_CPUS_PER_TASK
if ((affinity_cpus < cpu_limit)); then cpu_limit=$affinity_cpus; fi
jobs=${requested_jobs:-8}
if [[ -z "$requested_jobs" ]] && ((jobs > cpu_limit)); then jobs=$cpu_limit; fi
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || fail 'JOBS must be a positive integer'
((jobs <= cpu_limit)) || fail "requested $jobs jobs, but this task allows at most $cpu_limit"
export CMAKE_BUILD_PARALLEL_LEVEL=$jobs
export MAKEFLAGS="-j$jobs"
unset MFLAGS
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

[[ -n ${HPX_DIR:-} && -f "$HPX_DIR/HPXConfig.cmake" ]] || fail 'set HPX_DIR to the installed HPX 2 MPI CMake package'
for variable in FETCHCONTENT_SOURCE_DIR_KOKKOSSIMD FETCHCONTENT_SOURCE_DIR_SVESIMD; do
    source_dir=${!variable:-}
    # simd-math is header-only and intentionally has no CMakeLists.txt.
    required_file=CMakeLists.txt
    if [[ "$variable" == FETCHCONTENT_SOURCE_DIR_KOKKOSSIMD ]]; then required_file=simd.hpp; fi
    [[ -n "$source_dir" && -f "$source_dir/$required_file" ]] || fail "set $variable to the staged source checkout containing $required_file (downloads are disabled)"
done
if [[ -z ${Silo_DIR:-} && -n ${SILO_ROOT:-} ]]; then export Silo_DIR=$SILO_ROOT; fi
if [[ -z ${SILO_ROOT:-} && -n ${Silo_DIR:-} ]]; then export SILO_ROOT=$Silo_DIR; fi
case "$build_dir" in
    /*) ;;
    *) build_dir="$PWD/$build_dir" ;;
esac
[[ "$build_dir" != "$project_dir" && "$build_dir" != / ]] || fail 'use a separate build directory'
mkdir -p -- "$build_dir"
build_dir=$(cd -- "$build_dir" && pwd)
[[ "$build_dir" != "$project_dir" ]] || fail 'use a separate build directory'

# Use identical dependency/compiler hints for the probe and both consumers.
dependency_args=("-DHPX_DIR=$HPX_DIR" -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON)
for variable in CPPuddle_DIR Vc_DIR Silo_DIR HDF5_ROOT Boost_ROOT BOOST_ROOT MPI_CXX_COMPILER MPI_C_COMPILER MPIEXEC_EXECUTABLE FFTW3_INCLUDE_DIR FFTW3_LIBRARY OPENSSL_ROOT_DIR nlohmann_json_DIR Python3_EXECUTABLE; do
    if [[ -n ${!variable:-} ]]; then dependency_args+=("-D$variable=${!variable}"); fi
done
if [[ -n ${CMAKE_PREFIX_PATH:-} ]]; then
    dependency_args+=("-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH//:/;}")
fi
printf 'Building on %s: job %s, %s parallel jobs (task/affinity limit %s)\n' "$node" "$SLURM_JOB_ID" "$jobs" "$cpu_limit"
cmake -S "$project_dir/tools/qb4/hpx-check" -B "$build_dir/hpx-check" \
    -DCMAKE_BUILD_TYPE=Release "${dependency_args[@]}"
cmake -S "$project_dir" -B "$build_dir" "${dependency_args[@]}" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DOCTOTIGER_WITH_RADIATION=ON -DOCTOTIGER_WITH_TESTS=ON \
    -DOCTOTIGER_WITH_GRIDDIM=8 -DOCTOTIGER_WITH_FAST_FP_CONTRACT=OFF \
    -DOCTOTIGER_WITH_CXX17=OFF -DOCTOTIGER_WITH_CXX20=ON \
    -DOCTOTIGER_WITH_CUDA=OFF -DOCTOTIGER_WITH_HIP=OFF -DOCTOTIGER_WITH_KOKKOS=OFF \
    "-DFETCHCONTENT_SOURCE_DIR_KOKKOSSIMD=$FETCHCONTENT_SOURCE_DIR_KOKKOSSIMD" \
    "-DFETCHCONTENT_SOURCE_DIR_SVESIMD=$FETCHCONTENT_SOURCE_DIR_SVESIMD"
cmake --build "$build_dir" --parallel "$jobs"
tools_dir="$build_dir/radiation-results-tools"
cmake -S "$project_dir/radiation_results" -B "$tools_dir" "${dependency_args[@]}" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build "$tools_dir" --parallel "$jobs"
for executable in "$build_dir/octotiger" "$tools_dir/radiation-results" "$tools_dir/gen_radiation_reference"; do
    [[ -x "$executable" ]] || fail "build did not produce $executable"
done
{
    printf 'job=%s\nnode=%s\njobs=%s\nenvironment_file=%s\n' "$SLURM_JOB_ID" "$node" "$jobs" "$env_file"
    cat "$build_dir/hpx-check/hpx-mpi-selection.txt"
    sha256sum "$build_dir/octotiger" "$tools_dir/radiation-results" "$tools_dir/gen_radiation_reference"
} > "$build_dir/qb4-build-manifest.txt"
printf 'Build complete. Executable: %s/octotiger\nResults tools: %s\n' "$build_dir" "$tools_dir"
