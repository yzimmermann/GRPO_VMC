# Q2VMC Cluster Benchmark

Use [q2vmc_cluster_benchmark.py](/Users/yoel/GPRO_VMC/q2vmc_cluster_benchmark.py) to
materialize a Slurm-ready benchmark pack for the public LapNet codebase. The
generated pack now includes a `48:00:00` default wall-clock limit, conservative
Slurm array throttling, and W&B streaming via
[q2vmc_wandb_runner.py](/Users/yoel/GPRO_VMC/q2vmc_wandb_runner.py).

Example:

```bash
python q2vmc_cluster_benchmark.py generate \
  --outdir q2vmc_h200_paper \
  --preset paper \
  --time 48:00:00 \
  --partition gpu \
  --account YOUR_ACCOUNT
```

That will create:

- a pinned manifest for the public `LapNet` and `LapJAX` repos,
- the full `lapnet`/`psiformer` x system run matrix,
- an H200-oriented Slurm array launcher,
- a bootstrap script for the runtime environment,
- a W&B wrapper that tails `train_stats.csv`,
- a summary tool that reads `train_stats.csv` after the jobs finish.

The heavy GPU training is not exercised in this local workspace, so the intended
flow is: generate here, push the generated directory to the cluster, bootstrap
the env there, then `sbatch` the array script.
