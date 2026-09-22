#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
needsBuild=false
if [[ ! -x "$DIR/.build/radiation-results" || ! -x "$DIR/.build/gen_radiation_reference" ]]; then
    needsBuild=true
else
    for source in "$DIR/CMakeLists.txt" "$DIR"/cpp/*; do
        if [[ "$source" -nt "$DIR/.build/radiation-results" ]]; then
            needsBuild=true
            break
        fi
    done
fi
if "$needsBuild"; then "$DIR/build_cpp.sh"; fi
exec "$DIR/.build/radiation-results" "$@"
