#!/bin/bash
#SBATCH --job-name=q2vmc
#SBATCH --array=1-24%1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export Q2VMC_RESULTS_ROOT="${SCRIPT_DIR}"
export Q2VMC_LAPNET_ROOT="${Q2VMC_LAPNET_ROOT:-${SLURM_SUBMIT_DIR}/external/LapNet}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"
export Q2VMC_WANDB_PROJECT="${Q2VMC_WANDB_PROJECT:-q2vmc-cluster}"
export Q2VMC_WANDB_ENTITY="${Q2VMC_WANDB_ENTITY:-}"
export Q2VMC_GRPO_INNER_STEPS="${Q2VMC_GRPO_INNER_STEPS:-4}"
export Q2VMC_GRPO_CLIP_EPSILON="${Q2VMC_GRPO_CLIP_EPSILON:-0.2}"
export Q2VMC_GRPO_MAX_ADVANTAGE="${Q2VMC_GRPO_MAX_ADVANTAGE:-5.0}"
export Q2VMC_GRPO_INNER_OPTIMIZER="${Q2VMC_GRPO_INNER_OPTIMIZER:-adam}"
export Q2VMC_GRPO_MAX_GRAD_NORM="${Q2VMC_GRPO_MAX_GRAD_NORM:-1.0}"
export Q2VMC_GRPO_LR_RATE="${Q2VMC_GRPO_LR_RATE:-0.001}"

mkdir -p "${SCRIPT_DIR}/slurm_logs"

if [[ -n "${Q2VMC_MODULES:-}" ]]; then
  eval "${Q2VMC_MODULES}"
fi

if [[ -n "${Q2VMC_VENV:-}" ]]; then
  # shellcheck disable=SC1090
  source "${Q2VMC_VENV}/bin/activate"
fi

COMMAND_FILE="${SCRIPT_DIR}/commands.txt"
CMD="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${COMMAND_FILE}")"
if [[ -z "${CMD}" ]]; then
  echo "No command found for array index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

echo "[$(date)] Q2VMC benchmark root: ${SCRIPT_DIR}"
echo "[$(date)] LapNet checkout: ${Q2VMC_LAPNET_ROOT}"
echo "[$(date)] Running command ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_MAX}: ${CMD}"
bash -lc "${CMD}"
