#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$DIR/.." && pwd)"
if [[ "$ROOT" == */src/octotiger ]]; then
    ROOT="$(cd -- "$ROOT/../.." && pwd)"
fi
exec "$DIR/results.sh" live --root "$ROOT" "$@"
