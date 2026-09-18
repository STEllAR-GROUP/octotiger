#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
needs_build=false
if [[ ! -x "$DIR/.build/radiation-results" || ! -x "$DIR/.build/gen_radiation_reference" ]]; then
    needs_build=true
else
    for source in "$DIR/CMakeLists.txt" "$DIR"/cpp/*; do
        if [[ "$source" -nt "$DIR/.build/radiation-results" ]]; then
            needs_build=true
            break
        fi
    done
fi
if "$needs_build"; then "$DIR/build_cpp.sh"; fi
exec "$DIR/.build/radiation-results" "$@"
