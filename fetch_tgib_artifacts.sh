#!/usr/bin/env bash
# Download TGIB run artifacts from the GPU server over SFTP.
#
#   ./fetch_tgib_artifacts.sh                  # processed, saved_checkpoints, saved_models
#   ./fetch_tgib_artifacts.sh saved_models     # only the directories named
#   ./fetch_tgib_artifacts.sh -n               # dry run: remote sizes + local free space
#
# Settings (environment variables):
#   REMOTE      user@host                    (default: nahoms@10.5.10.171)
#   REMOTE_DIR  TGIB dir, relative to the remote home  (default: projects/CTGIB/TGIB)
#   LOCAL_DIR   where the directories land   (default: ./TGIB next to this script)
#
# How it works:
#   - One SSH master connection is opened, so the password (or key passphrase) is
#     asked once. sftp then runs in batch mode over that connection and stops at
#     the first error.
#   - Files are downloaded into a staging directory first. An interrupted run
#     resumes there (`get -a`), and resuming only ever appends to this script's own
#     partial files. It never appends onto an older local copy with different
#     contents.
#   - The staged copy is checked against the server (file count and total bytes).
#     Only then is it merged into LOCAL_DIR: same-named files are overwritten, and
#     local-only files are kept.
#   - Host keys are verified as usual. The first connection asks you to confirm
#     the server's fingerprint.
set -euo pipefail

REMOTE="${REMOTE:-nahoms@10.5.10.171}"
REMOTE_DIR="${REMOTE_DIR:-projects/CTGIB/TGIB}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="${LOCAL_DIR:-$SCRIPT_DIR/TGIB}"
DEFAULT_DIRS=(processed saved_checkpoints saved_models)

usage() {
    sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

DRY_RUN=0
DIRS=()
while [ $# -gt 0 ]; do
    case "$1" in
        -n|--dry-run) DRY_RUN=1 ;;
        -h|--help) usage 0 ;;
        -*) echo "unknown option: $1" >&2; usage 2 ;;
        *) DIRS+=("$1") ;;
    esac
    shift
done
[ ${#DIRS[@]} -gt 0 ] || DIRS=("${DEFAULT_DIRS[@]}")

for d in "${DIRS[@]}"; do
    case "$d" in
        */*|.|..|"") echo "directory names only, no paths: '$d'" >&2; exit 2 ;;
    esac
done

STAGING="$LOCAL_DIR/.sftp-incoming"
SOCKET_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tgib-ssh.XXXXXX")"
CTL="$SOCKET_DIR/ctl"
BATCH="$SOCKET_DIR/batch"

cleanup() {
    ssh -S "$CTL" -O exit "$REMOTE" >/dev/null 2>&1 || true
    rm -rf "$SOCKET_DIR"
}
trap cleanup EXIT

echo "Connecting to $REMOTE ..."
ssh -M -S "$CTL" -fN -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "$REMOTE"

remote() { ssh -S "$CTL" "$REMOTE" "$@"; }

# Remote inventory: one line per directory, "<name> <files> <bytes>". GNU find on the server.
quoted_dirs=""
for d in "${DIRS[@]}"; do quoted_dirs="$quoted_dirs $(printf '%q' "$d")"; done
inventory="$(remote "cd $(printf '%q' "$REMOTE_DIR") && for d in $quoted_dirs; do
    if [ ! -d \"\$d\" ]; then echo \"MISSING \$d\"; continue; fi
    find \"\$d\" -type f -printf '%s\n' | awk -v d=\"\$d\" '{n++; b+=\$1} END {printf \"%s %d %.0f\n\", d, n, b}'
done")"

if echo "$inventory" | grep -q '^MISSING '; then
    echo "$inventory" | sed -n 's/^MISSING /not found on server: '"$REMOTE_DIR"'\//p' >&2
    exit 1
fi

total_bytes=0
echo
printf '  %-20s %8s %12s\n' directory files size
while read -r name files bytes; do
    printf '  %-20s %8s %10s MB\n' "$name" "$files" "$((bytes / 1024 / 1024))"
    total_bytes=$((total_bytes + bytes))
done <<< "$inventory"

mkdir -p "$LOCAL_DIR"
free_kb="$(df -Pk "$LOCAL_DIR" | awk 'NR==2 {print $4}')"
need_kb=$(( (total_bytes / 1024) * 2 ))   # staging + merged copy can coexist briefly
echo
echo "  total $((total_bytes / 1024 / 1024)) MB; local free $((free_kb / 1024)) MB at $LOCAL_DIR"
if [ "$free_kb" -lt "$need_kb" ]; then
    echo "not enough local disk space: need about $((need_kb / 1024)) MB (download + merge)" >&2
    exit 1
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo; echo "dry run: nothing downloaded"
    exit 0
fi

mkdir -p "$STAGING"
: > "$BATCH"
for d in "${DIRS[@]}"; do
    echo "get -a -p -R \"$REMOTE_DIR/$d\" \"$STAGING/\"" >> "$BATCH"
done

echo
echo "Downloading into $STAGING ..."
sftp -o ControlPath="$CTL" -b "$BATCH" "$REMOTE"

# Verify every staged directory against the server before touching LOCAL_DIR.
echo
status=0
while read -r name files bytes; do
    got_files="$(find "$STAGING/$name" -type f | wc -l | tr -d ' ')"
    got_bytes="$(find "$STAGING/$name" -type f -exec stat -f %z {} + 2>/dev/null | awk '{b+=$1} END {printf "%.0f", b}')"
    if [ "$got_files" = "$files" ] && [ "$got_bytes" = "$bytes" ]; then
        echo "  ok        $name ($got_files files)"
    else
        echo "  MISMATCH  $name: server $files files / $bytes B, local $got_files files / $got_bytes B" >&2
        status=1
    fi
done <<< "$inventory"
if [ "$status" -ne 0 ]; then
    echo "verification failed; staged files kept in $STAGING. Re-run to resume." >&2
    exit 1
fi

for d in "${DIRS[@]}"; do
    mkdir -p "$LOCAL_DIR/$d"
    cp -Rp "$STAGING/$d/." "$LOCAL_DIR/$d/"
    rm -rf "${STAGING:?}/$d"
done
rmdir "$STAGING" 2>/dev/null || true

echo
echo "Done: ${DIRS[*]} -> $LOCAL_DIR"
