# GRPO VMC Benchmarks

This repository contains a set of Variational Monte Carlo benchmarks and
cluster-launch utilities developed around comparing GRPO-style optimization with
standard VMC optimizers.

## Main scripts

- `grpo_vmc_benchmark.py`
  1D harmonic oscillator benchmark with analytic local energy and SGD/SR/GRPO.
- `neural_qho_vmc_benchmark.py`
  Neural-network QHO benchmark with SGD, exact SR, SR-KFAC, and GRPO.
- `neural_helium_vmc_benchmark.py`
  Two-electron helium benchmark with independent reevaluation of final states.
- `q2vmc_cluster_benchmark.py`
  Generator for a cluster-ready LapNet/Psiformer benchmark pack.
- `q2vmc_wandb_runner.py`
  Runtime wrapper that tails `train_stats.csv` and streams metrics to W&B.

## Ready-to-run cluster pack

`q2vmc_h200_48h_wandb/` is a generated 48-hour benchmark pack for the public
LapNet codebase. It includes:

- pinned upstream revisions for LapNet and LapJAX,
- an H200-oriented Slurm array script,
- per-run launcher scripts,
- W&B tracking via `train_stats.csv`,
- summary tooling for completed runs.

See `Q2VMC_CLUSTER_README.md` and `q2vmc_h200_48h_wandb/README.md` for cluster
usage details.
