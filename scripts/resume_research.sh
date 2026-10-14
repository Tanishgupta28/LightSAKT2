#!/usr/bin/env bash
set -euo pipefail
LIGHTSAKT_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LIGHTSAKT_PROJECT_ROOT"
exec python -m scripts.recover --start
