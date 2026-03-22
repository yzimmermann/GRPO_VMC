#!/bin/bash
set -euo pipefail

echo "This helper moved into the generated benchmark directory."
echo "Run:"
echo "  python q2vmc_cluster_benchmark.py generate --outdir q2vmc_runs"
echo "and then:"
echo "  bash q2vmc_runs/bootstrap_q2vmc_env.sh"
