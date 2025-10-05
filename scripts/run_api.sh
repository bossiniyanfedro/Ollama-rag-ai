#!/usr/bin/env bash
set -euo pipefail

uvicorn app.api.server:app --host 0.0.0.0 --port ${PORT:-8000} --reload

