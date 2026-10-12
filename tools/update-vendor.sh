#!/usr/bin/env bash
# Met à jour la bibliothèque JavaScript embarquée dans docs/vendor/ :
#   mediabunny — lecture en flux des fichiers audio/vidéo (WebCodecs), licence MPL-2.0.
# (Transformers.js est chargé depuis jsDelivr en version figée, voir docs/asr-worker.js.)
#
# Usage : tools/update-vendor.sh [version-mediabunny]
set -euo pipefail
MB_VERSION="${1:-1.61.3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/docs/vendor"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$OUT"
(cd "$TMP" && npm pack --silent "mediabunny@$MB_VERSION" >/dev/null && tar -xzf ./*.tgz)
cp "$TMP/package/dist/bundles/mediabunny.min.mjs" "$OUT/mediabunny.min.mjs"
cp "$TMP/package/LICENSE" "$OUT/mediabunny.LICENSE"
echo "mediabunny $MB_VERSION (dist/bundles/mediabunny.min.mjs)" > "$OUT/VERSIONS.txt"
echo "OK : mediabunny $MB_VERSION"
