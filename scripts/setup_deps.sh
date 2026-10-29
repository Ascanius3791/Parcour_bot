#!/usr/bin/env bash
# Downloads the prebuilt MuJoCo release into third_party/ (checksum verified).
set -euo pipefail
VER=${MUJOCO_VERSION:-3.17.0}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party"
if [[ -d "$DEST/mujoco-$VER" ]]; then echo "MuJoCo $VER already present"; exit 0; fi
mkdir -p "$DEST"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
URL="https://github.com/google-deepmind/mujoco/releases/download/$VER/mujoco-$VER-linux-x86_64.tar.gz"
curl -sSfL -o "$TMP/mj.tgz" "$URL"
curl -sSfL -o "$TMP/mj.sha" "$URL.sha256"
(cd "$TMP" && echo "$(cut -d' ' -f1 mj.sha)  mj.tgz" | sha256sum -c -)
tar xzf "$TMP/mj.tgz" -C "$DEST"
echo "MuJoCo $VER installed to $DEST/mujoco-$VER"
