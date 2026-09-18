#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TYPE="${1:-Release}"
case "${TYPE,,}" in
 debug) TYPE=Debug;; release) TYPE=Release;; relwithdebinfo) TYPE=RelWithDebInfo;;
 *) echo "Usage: $0 [Debug|Release|RelWithDebInfo]" >&2; exit 2;;
esac
cmake -S "$DIR" -B "$DIR/.build" -DCMAKE_BUILD_TYPE="$TYPE" -DBUILD_TESTING=OFF
cmake --build "$DIR/.build" -j "${JOBS:-4}"
