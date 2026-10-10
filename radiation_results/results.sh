#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
needs_build=false
if [[ ! -x "$DIR/.build/radiation-results" || ! -x "$DIR/.build/gen_radiation_reference" ]]; then
    needs_build=true
else
    for source in "$DIR/CMakeLists.txt" "$DIR/reference.cmake" "$DIR"/cpp/* \
        "$DIR"/../test_problems/radiation/*.hpp "$DIR"/../octotiger/math/*.hpp; do
        if [[ "$source" -nt "$DIR/.build/radiation-results" || "$source" -nt "$DIR/.build/gen_radiation_reference" ]]; then
            needs_build=true
            break
        fi
    done
fi
if "$needs_build"; then "$DIR/build_cpp.sh"; fi
exec "$DIR/.build/radiation-results" "$@"
