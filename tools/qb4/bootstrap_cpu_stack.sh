#!/usr/bin/env bash
# Build staged HPX 2 + CPPuddle sources inside a Slurm compute allocation.
# No connection, submission, download, or system installation is performed.
set -euo pipefail
fail() { printf 'bootstrap_cpu_stack.sh: %s\n' "$*" >&2; exit 2; }
usage() {
    cat <<'EOF'
Usage: bootstrap_cpu_stack.sh --env FILE --sources DIR --build-dir DIR --prefix DIR [-j N]

DIR under --sources must contain hpx/, cppuddle/, stdexec/, and asio/.
Stage current patched local sources, including all untracked compatibility files.
stdexec must already carry the HPX patches: source overrides skip FetchContent
patch commands. The environment supplies matching CC/CXX, MPI_C_COMPILER,
MPI_CXX_COMPILER, CMAKE_PREFIX_PATH and installed Boost/hwloc prerequisites.
This builds a CPU-only HPX MPI stack, then installs CPPuddle against it.
Application/common dependencies (Vc, Silo/HDF5, FFTW3, OpenSSL, JSON, gnuplot)
must already be provided separately. Parallel builds default to eight jobs and
must fit both SLURM_CPUS_PER_TASK and process affinity. No tests are run here.
EOF
}
environment= sources= build= prefix= requested_jobs=8
while (($#)); do
    case "$1" in
        --env|--sources|--build-dir|--prefix|-j|--jobs)
            (($# >= 2)) || fail "$1 needs a value"
            case "$1" in
                --env) environment=$2 ;;
                --sources) sources=$2 ;;
                --build-dir) build=$2 ;;
                --prefix) prefix=$2 ;;
                *) requested_jobs=$2 ;;
            esac
            shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) fail "unknown argument: $1" ;;
    esac
done
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || fail 'run in an allocated Slurm compute job'
[[ ${SLURM_CPUS_PER_TASK:-} =~ ^[1-9][0-9]*$ ]] || fail 'missing allocated CPU count'
[[ -n ${SLURMD_NODENAME:-} && "$(hostname -s)" == "${SLURMD_NODENAME%%.*}" ]] || fail 'not on the allocated compute node'
[[ -r "$environment" ]] || fail '--env must name a readable environment file'
source "$environment"
for program in cmake nproc python3; do
    command -v "$program" >/dev/null || fail "missing $program"
done
for variable in CC CXX MPI_C_COMPILER MPI_CXX_COMPILER; do
    [[ -n ${!variable:-} ]] && command -v "${!variable}" >/dev/null || fail "set $variable to a matching executable"
done
for directory in "$sources" "$build" "$prefix"; do
    [[ "$directory" == /* && "$directory" != / ]] || fail 'source, build and install paths must be absolute non-root directories'
done
[[ "$sources" != "$build" && "$sources" != "$prefix" && "$build" != "$prefix" ]] || fail 'use distinct source/build/install directories'
[[ -f "$sources/hpx/CMakeLists.txt" && -f "$sources/cppuddle/CMakeLists.txt" ]] || fail 'missing staged HPX/CPPuddle sources'
[[ -f "$sources/stdexec/include/stdexec/execution.hpp" && -f "$sources/asio/asio/include/asio.hpp" ]] || fail 'missing staged stdexec or Asio1.30.2 headers'
affinity=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)
[[ "$affinity" =~ ^[1-9][0-9]*$ && "$requested_jobs" =~ ^[1-9][0-9]*$ ]] || fail 'invalid CPU count'
((requested_jobs <= SLURM_CPUS_PER_TASK && requested_jobs <= affinity)) || fail 'jobs exceed allocation or affinity'
export CMAKE_BUILD_PARALLEL_LEVEL=$requested_jobs MAKEFLAGS="-j$requested_jobs"
unset MFLAGS
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p -- "$build" "$prefix"
# Record actual staged contents, including local/untracked compatibility patches.
python3 - "$sources" "$build/source-manifest.json" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
record = {}
for name in ('hpx', 'cppuddle', 'stdexec', 'asio'):
    digest = hashlib.sha256()
    count = 0
    for path in sorted((root / name).rglob('*')):
        if '.git' in path.relative_to(root).parts or not path.is_file():
            continue
        relative = str(path.relative_to(root / name))
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update((relative + '\0' + value + '\n').encode())
        count += 1
    record[name] = {'files': count, 'tree_sha256': digest.hexdigest()}
pathlib.Path(sys.argv[2]).write_text(json.dumps(record, indent=2) + '\n')
PY
dependency_prefixes=${CMAKE_PREFIX_PATH:-}
common=("-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_INSTALL_PREFIX=$prefix" "-DCMAKE_INSTALL_LIBDIR=lib"
    "-DCMAKE_C_COMPILER=$CC" "-DCMAKE_CXX_COMPILER=$CXX"
    "-DCMAKE_PREFIX_PATH=$prefix;${dependency_prefixes//:/;}"
    "-DMPI_C_COMPILER=$MPI_C_COMPILER" "-DMPI_CXX_COMPILER=$MPI_CXX_COMPILER"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON)
for variable in Boost_ROOT BOOST_ROOT Hwloc_ROOT Hwloc_INCLUDE_DIR Hwloc_LIBRARY MPIEXEC_EXECUTABLE; do
    if [[ -n ${!variable:-} ]]; then common+=("-D$variable=${!variable}"); fi
done
cmake -S "$sources/hpx" -B "$build/hpx" "${common[@]}" \
    -DHPX_WITH_CXX_STANDARD=20 -DHPX_WITH_CUDA=OFF -DHPX_WITH_HIP=OFF -DHPX_WITH_SYCL=OFF \
    -DHPX_WITH_DISTRIBUTED_RUNTIME=ON -DHPX_WITH_NETWORKING=ON \
    -DHPX_WITH_PARCELPORT_MPI=ON -DHPX_WITH_PARCELPORT_MPI_MULTITHREADED=ON \
    -DHPX_WITH_PARCELPORT_TCP=OFF -DHPX_WITH_MALLOC=system \
    -DHPX_WITH_THREAD_STACK_MMAP=OFF \
    -DHPX_WITH_TESTS=OFF -DHPX_WITH_EXAMPLES=OFF -DHPX_WITH_APEX=OFF \
    -DHPX_WITH_PKGCONFIG=OFF \
    -DHPX_WITH_FETCH_STDEXEC=ON -DHPX_WITH_FETCH_ASIO=ON \
    "-DFETCHCONTENT_SOURCE_DIR_STDEXEC=$sources/stdexec" \
    "-DFETCHCONTENT_SOURCE_DIR_ASIO=$sources/asio" -DHPX_WITH_ASIO_TAG=asio-1-30-2
cmake --build "$build/hpx" --parallel "$requested_jobs"
cmake --install "$build/hpx"
cmake -S "$sources/cppuddle" -B "$build/cppuddle" "${common[@]}" \
    "-DHPX_DIR=$prefix/lib/cmake/HPX" -DCPPUDDLE_WITH_HPX=ON \
    -DCPPUDDLE_WITH_KOKKOS=OFF -DCPPUDDLE_WITH_CUDA=OFF -DCPPUDDLE_WITH_TESTS=OFF
cmake --build "$build/cppuddle" --parallel "$requested_jobs"
cmake --install "$build/cppuddle"
printf 'CPU MPI stack installed in %s\nHPX_DIR=%s/lib/cmake/HPX\nCPPuddle_DIR=%s/lib/cmake/CPPuddle\n' "$prefix" "$prefix" "$prefix"
