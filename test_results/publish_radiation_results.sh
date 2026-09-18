#!/usr/bin/env bash

set -Eeuo pipefail

# Publish the generated Octo-TIGER radiation-results website from Sagan to the
# DigitalOcean Nginx document root.  Raw simulation data (including SILO files)
# and intermediate movie frames are deliberately not uploaded.

OCTOTIGER_ROOT="${OCTOTIGER_ROOT:-$HOME/workspace/octotiger}"
RESULTS_ROOT="${OCTOTIGER_RESULTS_ROOT:-$OCTOTIGER_ROOT/test_results/results}"
PUBLISH_HOST="${OCTOTIGER_PUBLISH_HOST:-dmarce1@129.212.177.35}"
PUBLISH_DIR="${OCTOTIGER_PUBLISH_DIR:-/var/www/octotiger}"
PUBLIC_URL="${OCTOTIGER_PUBLISH_URL:-https://www.octotiger.org/}"
INTERVAL="${OCTOTIGER_PUBLISH_INTERVAL:-10}"

WATCH=0
DRY_RUN=0
REQUESTED_BATCH=""
WAIT_PID=""

usage() {
	cat <<'EOF'
Usage:
  publish_radiation_results.sh [OPTIONS] [BATCH_DIRECTORY]

Publish the newest generated radiation-results website once.  If an explicit
BATCH_DIRECTORY is supplied, publish that batch instead.

Options:
  --watch             Repeat indefinitely, selecting the newest batch each time
  --hourly            Upload now, then wait one hour between uploads (Ctrl+C stops)
  --interval SECONDS  Watch interval (default: 10)
  --dry-run           Show what rsync would change without changing the server
  -h, --help          Show this help

Examples:
  publish_radiation_results.sh
  publish_radiation_results.sh --watch
  publish_radiation_results.sh --hourly
  publish_radiation_results.sh --watch --interval 3600
  publish_radiation_results.sh --watch --interval 20
  publish_radiation_results.sh "$HOME/workspace/octotiger/test_results/results/live-20260915-144628-575781"

Environment overrides:
  OCTOTIGER_ROOT
  OCTOTIGER_RESULTS_ROOT
  OCTOTIGER_PUBLISH_HOST
  OCTOTIGER_PUBLISH_DIR
  OCTOTIGER_PUBLISH_URL
  OCTOTIGER_PUBLISH_INTERVAL
EOF
}

die() {
	printf 'Error: %s\n' "$*" >&2
	exit 1
}

while (($#)); do
	case "$1" in
		--watch)
			WATCH=1
			;;
		--hourly)
			WATCH=1
			INTERVAL=3600
			;;
		--interval)
			shift
			(($#)) || die '--interval requires a value'
			INTERVAL="$1"
			;;
		--dry-run)
			DRY_RUN=1
			;;
		-h | --help)
			usage
			exit 0
			;;
		--)
			shift
			if (($#)); then
				[[ -z "$REQUESTED_BATCH" ]] || die 'only one batch directory may be supplied'
				REQUESTED_BATCH="$1"
				shift
			fi
			(($# == 0)) || die 'only one batch directory may be supplied'
			break
			;;
		-*)
			die "unknown option: $1"
			;;
		*)
			[[ -z "$REQUESTED_BATCH" ]] || die 'only one batch directory may be supplied'
			REQUESTED_BATCH="$1"
			;;
	esac
	shift
done

[[ "$INTERVAL" =~ ^[1-9][0-9]*$ ]] || die 'the interval must be a positive integer'

for command_name in date find realpath rsync sed sleep sort ssh; do
	command -v "$command_name" >/dev/null 2>&1 || die "required command not found: $command_name"
done

if [[ -n "$REQUESTED_BATCH" ]]; then
	REQUESTED_BATCH="$(realpath -e "$REQUESTED_BATCH")"
	[[ -f "$REQUESTED_BATCH/index.html" ]] ||
		die "batch does not contain a top-level index.html: $REQUESTED_BATCH"
else
	[[ -d "$RESULTS_ROOT" ]] || die "results directory does not exist: $RESULTS_ROOT"
fi

# This directory is mirrored with rsync deletion of stale, included web files.
# Refuse an empty/root destination, including after removing trailing slashes.
while [[ "$PUBLISH_DIR" == */ ]]; do PUBLISH_DIR="${PUBLISH_DIR%/}"; done
[[ "$PUBLISH_DIR" == /* ]] || die 'the remote publish directory must be an absolute non-root path'

latest_batch() {
	local newest

	newest="$({
		find "$RESULTS_ROOT" \
			-mindepth 2 -maxdepth 2 \
			-type f -name index.html \
			-printf '%T@\t%h\n' 2>/dev/null || true
	} | sort -nr | sed -n '1p')"

	[[ -n "$newest" ]] || return 1
	printf '%s\n' "${newest#*$'\t'}"
}

SSH_OPTIONS=(
	-o BatchMode=yes
	-o ConnectTimeout=10
	-o ServerAliveInterval=30
	-o ServerAliveCountMax=3
)
# Quote for the remote shell, including an apostrophe in an overridden path.
REMOTE_DIR_QUOTED="'${PUBLISH_DIR//\'/\'\\\'\'}'"

RSYNC_OPTIONS=(
	--archive
	--human-readable
	--itemize-changes
	--partial
	--partial-dir=.rsync-partial
	--delay-updates
	--delete-delay
	--prune-empty-dirs
	--chmod=D755,F644
	'--exclude=*/renders/***'
	'--exclude=encode-*/'
	'--exclude=*.silo.data/'
	'--exclude=*.silo'
	'--exclude=*.tmp'
	'--include=*/'
	'--include=*.html'
	'--include=*.css'
	'--include=*.js'
	'--include=*.json'
	'--include=*.csv'
	'--include=*.tsv'
	'--include=*.ini'
	'--include=*.log'
	'--include=*.txt'
	'--include=*.out'
	'--include=*.session'
	'--include=*.png'
	'--include=*.jpg'
	'--include=*.jpeg'
	'--include=*.gif'
	'--include=*.svg'
	'--include=*.webp'
	'--include=*.ico'
	'--include=*.pdf'
	'--include=*.mp4'
	'--include=*.m4v'
	'--include=*.webm'
	'--include=*.woff'
	'--include=*.woff2'
	'--include=*.ttf'
	'--include=*.webmanifest'
	'--include=L1.dat'
	'--include=L2.dat'
	'--include=Linf.dat'
	'--exclude=*'
)

if ((DRY_RUN)); then
	RSYNC_OPTIONS+=(--dry-run)
fi

publish_once() {
	local batch="$1"

	[[ -f "$batch/index.html" ]] || {
		printf 'Warning: skipping batch without index.html: %s\n' "$batch" >&2
		return 1
	}

	printf '\nPublishing: %s\n' "$batch"
	printf 'Destination: %s:%s\n' "$PUBLISH_HOST" "$PUBLISH_DIR"
	if ! ssh "${SSH_OPTIONS[@]}" "$PUBLISH_HOST" \
		"test -d $REMOTE_DIR_QUOTED && test -w $REMOTE_DIR_QUOTED"; then
		printf 'Cannot access writable remote directory %s:%s\n' "$PUBLISH_HOST" "$PUBLISH_DIR" >&2
		return 1
	fi

	# Explicitly propagate errors: errexit is disabled inside a function tested
	# by `if ! publish_once`, so a bare failed rsync would otherwise look successful.
	rsync "${RSYNC_OPTIONS[@]}" \
		-e "ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=3" \
		"$batch/" \
		"$PUBLISH_HOST:$PUBLISH_DIR/" || return "$?"

	if ((DRY_RUN)); then
		printf 'Dry run complete; the server was not changed.\n'
	else
		printf 'Published successfully: %s\n' "$PUBLIC_URL"
	fi
}

stop_publishing() {
	trap - INT TERM
	if [[ -n "$WAIT_PID" ]]; then
		kill "$WAIT_PID" 2>/dev/null || true
		wait "$WAIT_PID" 2>/dev/null || true
	fi
	printf '\nPublishing stopped.\n'
	exit 0
}
trap stop_publishing INT TERM

wait_for_refresh() {
	printf '[%s] Waiting %s seconds before the next upload. Ctrl+C stops.\n' \
		"$(date '+%Y-%m-%d %H:%M:%S %Z')" "$INTERVAL"
	# Waiting on a background sleep lets Bash run the signal trap immediately,
	# even for an hour-long wait (including when sent SIGTERM by a terminal).
	sleep "$INTERVAL" &
	WAIT_PID=$!
	wait "$WAIT_PID" || true
	WAIT_PID=""
}

while :; do
	if [[ -n "$REQUESTED_BATCH" ]]; then
		batch="$REQUESTED_BATCH"
	elif ! batch="$(latest_batch)"; then
		if ((WATCH)); then
			printf 'No generated site found under %s; retrying in %s seconds.\n' \
				"$RESULTS_ROOT" "$INTERVAL" >&2
			wait_for_refresh
			continue
		fi
		die "no generated site containing index.html found under $RESULTS_ROOT"
	fi

	if ! publish_once "$batch"; then
		((WATCH)) || exit 1
		printf 'Publish failed; retrying in %s seconds.\n' "$INTERVAL" >&2
	fi

	((WATCH)) || break
	wait_for_refresh
done
