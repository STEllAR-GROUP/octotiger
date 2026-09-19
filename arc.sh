#!/usr/bin/env bash

set -Eeuo pipefail

source_dir="$HOME/octotiger/src/octotiger"
git_root="$(git -C "$source_dir" rev-parse --show-toplevel)"
archive="$HOME/Desktop/code.tar.gz"
status_file="$HOME/Desktop/code-status-before-archive.txt"

git -C "$git_root" status --short > "$status_file"

rm -f "$archive"

(
    cd "$git_root"

    {
        printf '%s\0' .git
        git ls-files -z --cached --others --exclude-standard
    } |
        tar \
            --null \
            --files-from=- \
            --create \
            --gzip \
            --verbose \
            --file="$archive"
)

if ! tar -tzf "$archive" | grep -Eq '(^|/)\.git(/|$)'; then
    echo "ERROR: archive does not contain .git" >&2
    exit 1
fi

echo
echo "Git metadata included."
echo "Pre-archive status: $status_file"
du -h "$archive"
