#!/usr/bin/env bash
set -euo pipefail

exec python -u worker.py --queue-type jd
