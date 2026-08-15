#!/usr/bin/env bash
# Seed repeats at lambda=0 so the capacity comparison carries error bars. With n=1 the
# between-size range (0.144) sits inside the within-size spread (up to 0.199), i.e. the
# ordering is not separable from run-to-run noise.
set -u
cd "$(dirname "$0")"
echo "=== seeds start $(date -u) ==="
for SEED in 43 44; do
  torch-python train_invariant.py --size-name L  --hidden 48 --latent 32 --blocks 2 --lam 0 --epochs 8 --seed $SEED
  torch-python train_invariant.py --size-name M  --hidden 32 --latent 8  --blocks 2 --lam 0 --epochs 8 --seed $SEED
  torch-python train_invariant.py --size-name S  --hidden 16 --latent 4  --blocks 1 --lam 0 --epochs 8 --seed $SEED
  torch-python train_invariant.py --size-name XS --hidden 8  --latent 4  --blocks 1 --lam 0 --epochs 8 --seed $SEED
done
echo "=== seeds done $(date -u) ==="
