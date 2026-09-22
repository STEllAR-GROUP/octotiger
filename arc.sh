#!/usr/bin/env bash

set -Eeuo pipefail

sourceDir="$HOME/octotiger/src/octotiger"
gitRoot="$(git -C "$sourceDir" rev-parse --show-toplevel)"

codeArchive="$HOME/Desktop/code.tar.gz"
gitArchive="$HOME/Desktop/git.tar.gz"
statusFile="$HOME/Desktop/code-status-before-archive.txt"

git -C "$gitRoot" status --short > "$statusFile"

rm -f "$codeArchive" "$gitArchive"

cd "$gitRoot"

# Archive tracked and non-ignored untracked files that currently exist.
git ls-files -z --cached --others --exclude-standard |
    while IFS= read -r -d '' path; do
        if [[ -e "$path" || -L "$path" ]]; then
            printf '%s\0' "$path"
        fi
    done |
    tar \
        --null \
        --files-from=- \
        --create \
        --gzip \
        --verbose \
        --file="$codeArchive"

# Archive Git metadata separately.
tar \
    --create \
    --gzip \
    --verbose \
    --file="$gitArchive" \
    .git

if tar -tzf "$codeArchive" | grep -Eq '(^|/)\.git(/|$)'; then
    echo "ERROR: code archive unexpectedly contains .git" >&2
    exit 1
fi

if ! tar -tzf "$gitArchive" | grep -Eq '(^|/)\.git(/|$)'; then
    echo "ERROR: Git archive does not contain .git" >&2
    exit 1
fi

echo
echo "Archives created successfully."
echo "Code archive:       $codeArchive"
echo "Git archive:        $gitArchive"
echo "Pre-archive status: $statusFile"
echo
du -h "$codeArchive" "$gitArchive"
