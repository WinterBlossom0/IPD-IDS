#!/usr/bin/env bash
# Extra seed repeats to lift the capacity comparison from n=3 to n=5.
# Each run peaks around 1.8 GB (model + fit/val feature arrays resident on device), so four
# concurrently stays near 7 GB - inside the 12 GB budget for this process group.
set -u
cd "$(dirname "$0")"
echo "=== parallel seeds start $(date -u) ==="
for SEED in 45 46; do
  torch-python train_invariant.py --size-name L  --hidden 48 --latent 32 --blocks 2 --lam 0 --epochs 8 --seed $SEED &
  torch-python train_invariant.py --size-name M  --hidden 32 --latent 8  --blocks 2 --lam 0 --epochs 8 --seed $SEED &
  torch-python train_invariant.py --size-name S  --hidden 16 --latent 4  --blocks 1 --lam 0 --epochs 8 --seed $SEED &
  torch-python train_invariant.py --size-name XS --hidden 8  --latent 4  --blocks 1 --lam 0 --epochs 8 --seed $SEED &
  wait
  echo "--- seed $SEED wave complete $(date -u) ---"
done
echo "=== parallel seeds done $(date -u) ==="
