#!/usr/bin/env bash
# Capacity x erasure grid. Capacity is the primary axis (edge deployment); erasure
# strength is secondary. The hypothesis under test: shortcut reliance RISES as capacity
# falls, because a small model is pushed toward the cheapest predictive route - which
# would explain the documented cross-domain failure of lightweight NIDS models.
#
# Sizes are named by deployed ENCODER parameter count (decoder never ships):
#   L  h48 l32 b2  ~40.6k params  158 KB fp32
#   M  h32 l8  b2  ~18.1k params   71 KB
#   S  h16 l4  b1  ~ 3.0k params   12 KB
#   XS h8  l4  b1  ~ 1.1k params    4 KB   (below IIoT-TinyDNN's 2,255)
set -u
cd "$(dirname "$0")"
mkdir -p runs

run() { # size_name hidden latent blocks lam [extra...]
  echo "--- $1 lam=$5 ---"
  torch-python train_invariant.py --size-name "$1" --hidden "$2" --latent "$3" \
    --blocks "$4" --lam "$5" --epochs 8 "${@:6}" || echo "FAILED $1 lam=$5"
}

echo "=== grid start $(date -u) ==="
for LAM in 0 0.25 1.0; do
  run L  48 32 2 "$LAM"
  run M  32  8 2 "$LAM"
  run S  16  4 1 "$LAM"
  run XS  8  4 1 "$LAM"
done

echo "--- ablation: port column deleted (S size) ---"
run S 16 4 1 0.0 --drop-port

echo "=== grid done $(date -u) ==="
