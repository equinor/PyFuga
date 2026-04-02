#!/usr/bin/env bash
# Build sphinx-multiversion output, add a root index.html redirect (like GitLab Pages), then serve.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "${ROOT}/docs"
OUT="${1:-_build_multiversion_serve}"
PORT="${2:-8765}"
rm -rf "${OUT}"
sphinx-multiversion . "${OUT}"
LATEST_TAG="$(git describe --tags "$(git rev-list --tags --max-count=1)" 2>/dev/null || true)"
if [[ -n "${LATEST_TAG}" && -d "${OUT}/${LATEST_TAG}" ]]; then
  DEFAULT_REF="${LATEST_TAG}"
else
  DEFAULT_REF="main"
fi
sed "s|TAG|${DEFAULT_REF}|g" root_index_template.html > "${OUT}/index.html"
echo "Landing redirect: / → ${DEFAULT_REF}/index.html"
echo "Open http://127.0.0.1:${PORT}/ (serving ${OUT}/)"
exec python -m http.server "${PORT}" --directory "${OUT}"
