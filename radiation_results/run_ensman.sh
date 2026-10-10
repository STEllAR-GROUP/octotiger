#!/usr/bin/env bash
# Full-duration coupled tests through the existing sequential results runner.
set -euo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$DIR/.." && pwd)"

usage() {
    cat <<'EOF'
Usage: radiation_results/run_ensman.sh [ensman|blast|supercritical|subcritical] [levels...] [options]
       radiation_results/run_ensman.sh --resume BATCH [--threads N] [--dry-run]

Defaults: level 2 (32^3 with this build), two HPX workers, two build jobs,
          cmake-build-hpx2rc-radiation, 21 Silo snapshots, plots, no movies.
Cases run sequentially: blast (22000 s), supercritical (7500 s),
                        subcritical (38300 s). Physical light speed is the default.

  --dry-run           Print the plan; do not build/run the solver or write a batch.
  --build PATH        Use another configured application build.
  --output DIR        Use a new/empty batch directory (otherwise timestamped).
  --threads N         HPX workers, default 2; shared local budget still applies.
  --jobs N            Build jobs for both helpers and application, default 2.
  --no-build          Reuse the current application executable.
  --no-silo           Omit snapshots; final numerical CSV and plots remain.
  --snapshots N       Requested Silo snapshots, default 21.
  --time SECONDS      Override all selected durations; shorter runs are labeled.
  --radiation-reduced-c-factor F
                      Opt-in RSLA for coupled cases only, 0 < F <= 1 (default 1).
  --resume BATCH      Reuse completed runs and start untouched runs. An incomplete
                      simulation is refused; this is not a checkpoint restart.
  --help              Show this help without building helper tools.

Other radiation_results/run.sh options are forwarded. The C++ helper tools may
build on first use, including dry-run; the application is never run by dry-run.
Physical-c shock runs require millions of timesteps even at level 2. See ENSMAN.md.
EOF
}

selector=ensman
explicit_selector=false
case "${1:-}" in
    ensman|blast|supercritical|subcritical|radiating_blast|supercritical_shock|subcritical_shock)
        selector="$1"
        explicit_selector=true
        shift
        ;;
esac
args=("$@")
has_levels=false
has_build=false
resume=false
build_jobs=2
workers=2
for ((i=0; i<${#args[@]}; ++i)); do
    arg="${args[i]}"
    key="${arg%%=*}"
    case "$key" in
        --help|-h) usage; exit 0 ;;
        --root)
            echo 'run_ensman.sh uses its own checkout; use run.sh for --root.' >&2
            exit 2
            ;;
        --build|--resume|--threads|--jobs|--output|--exe|--generator|--launcher|--time|--radiation-reduced-c-factor|--min-level|--gradient-refine-field|--gradient-refine-mode|--gradient-refine-threshold|--gradient-refine-floor|--gradient-refine-buffer|--gradient-refine-coarsen-factor|--snapshots|--odt|--hard-dt|--rad-reconstruction|--gnuplot|--visit|--ffmpeg|--cxx|--seconds|--fps|--hold|--field|--view|--axis|--position|--width|--height|--color|--minimum|--maximum|--session-dir)
            if [[ "$arg" == *=* ]]; then
                value="${arg#*=}"
            else
                if ((i+1 >= ${#args[@]})); then
                    echo "Missing value for $key" >&2
                    exit 2
                fi
                i=$((i+1))
                value="${args[i]}"
            fi
            case "$key" in
                --build) has_build=true ;;
                --resume) resume=true ;;
                --threads) workers="$value" ;;
                --jobs) build_jobs="$value" ;;
            esac
            ;;
        --*) ;; # The C++ parser validates forwarded flags.
        *)
            if [[ ! "$arg" =~ ^[0-9]+$ ]]; then
                echo "Expected a level or option, got: $arg (use --build PATH)" >&2
                exit 2
            fi
            has_levels=true
            ;;
    esac
done

# This host's standing maximum is 12 cores; smaller hosts use half their cores.
available="$(getconf _NPROCESSORS_ONLN)"
limit=$((available/2))
if ((limit < 1)); then
    echo 'At least two available CPU cores are required to respect the half-core limit.' >&2
    exit 2
fi
if ((limit > 12)); then limit=12; fi
for count in "$workers" "$build_jobs"; do
    if [[ ! "$count" =~ ^[1-9][0-9]*$ ]] || ((${#count} > 2)) || ((count > limit)); then
        echo "Threads and jobs must be between 1 and $limit; concurrent work shares that budget." >&2
        exit 2
    fi
done
export JOBS="$build_jobs"

defaults=(--threads 2 --jobs 2 --no-open --no-movies)
if "$resume"; then
    if "$explicit_selector" || "$has_levels"; then
        echo '--resume takes its cases and levels from the saved batch; omit positional arguments.' >&2
        exit 2
    fi
else
    defaults=("$selector" "${defaults[@]}" --snapshots 21)
    if ! "$has_levels"; then defaults+=(2); fi
    if ! "$has_build"; then defaults+=(--build "$ROOT/cmake-build-hpx2rc-radiation"); fi
fi

echo 'Sequential local Ensman runs; physical-c full shock durations require millions of steps.'
exec "$DIR/run.sh" "${defaults[@]}" "${args[@]}"
