#!/usr/bin/env bash
set -euo pipefail
# Compatibility entry point. The implementation and generated results now
# belong to verification_results/radiation; retain this path for one cycle.
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "run" || "${1:-}" == "live" ]]; then
	mode="$1"
	shift
	exec "$DIR/../verification_results/run.sh" "$mode" "$@"
fi
exec "$DIR/../verification_results/radiation/results.sh" "$@"
