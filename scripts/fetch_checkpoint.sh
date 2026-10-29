#!/usr/bin/env bash
# Downloads the newest training checkpoint (artifact "checkpoint" of the train workflow).
# Usage: scripts/fetch_checkpoint.sh [dest-dir]      (default runs/latest)
# Then:  build/parcour_viewer --policy runs/latest/policy.bin
# Needs the GitHub CLI (gh), logged in. Exits with status 2 if there is no checkpoint yet.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$ROOT/runs/latest}"
REPO="${GITHUB_REPOSITORY:-$(cd "$ROOT" && gh repo view --json nameWithOwner --jq .nameWithOwner)}"
ID=$(gh api "repos/$REPO/actions/artifacts?name=checkpoint&per_page=20" \
       --jq '[.artifacts[] | select(.expired | not)][0].id // empty')
if [[ -z "$ID" ]]; then echo "no checkpoint artifact found in $REPO"; exit 2; fi
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
gh api "repos/$REPO/actions/artifacts/$ID/zip" > "$TMP/ckpt.zip"
rm -rf "$DEST"
mkdir -p "$DEST"
python3 -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$TMP/ckpt.zip" "$DEST"
echo "checkpoint $ID -> $DEST"
[[ -f "$DEST/summary.md" ]] && head -n 8 "$DEST/summary.md" || true
