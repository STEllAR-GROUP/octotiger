#!/usr/bin/env bash

set -euo pipefail

repo="$HOME/workspace/octotiger"
archive="$HOME/Desktop/code.tar.gz"

rm -f "$archive"

tar -czvf "$archive" \
	-C "$repo" \
	--exclude-vcs \
	--exclude='*.silo' \
	--exclude='*.silo.*' \
	--exclude='*.h5' \
	--exclude='*.hdf5' \
	--exclude='*.bin' \
	--exclude='*.session' \
	--exclude='*.dat' \
	--exclude='*.log' \
	--exclude='*.mp4' \
	--exclude='*.avi' \
	--exclude='*.webm' \
	--exclude='core' \
	--exclude='core.*' \
	--exclude='__pycache__' \
	--exclude='.venv' \
	--exclude='CMakeFiles' \
	--exclude='CMakeCache.txt' \
	--exclude='test_results/**/octotiger' \
	--exclude='test_results/results' \
	--exclude='test_results/serial-preview' \
	src \
	octotiger \
	frontend \
	cmake \
	test_problems \
	test_results \
	CMakeLists.txt

du -h "$archive"
