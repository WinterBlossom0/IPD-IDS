#!/usr/bin/env bash
# Everything downstream of training. Stages 3-5 need no GPU beyond latent extraction.
set -u
cd "$(dirname "$0")"

echo "=== [1/5] extract frozen latents ($(date -u +%H:%M:%S)) ==="
torch-python extract_latents.py || { echo "FAILED extract"; exit 1; }

echo "=== [2/5] deployment cost benchmark ($(date -u +%H:%M:%S)) ==="
torch-python bench_efficiency.py || echo "WARN: efficiency bench failed"

echo "=== [3/5] frontier evaluation: MDL probes, LEACE, detection ($(date -u +%H:%M:%S)) ==="
ds-python evaluate_frontier.py || { echo "FAILED evaluate"; exit 1; }

echo "=== [4/5] LaTeX tables ($(date -u +%H:%M:%S)) ==="
ds-python ../paper/make_tables.py || echo "WARN: tables failed"

echo "=== [5/5] figures ($(date -u +%H:%M:%S)) ==="
ds-python make_figures.py || echo "WARN: figures failed"

echo "=== pipeline done $(date -u) ==="
