#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-$SCRIPT_DIR/external}"
PYTHON_BIN="${Q2VMC_PYTHON:-python3.10}"
VENV_DIR="${Q2VMC_VENV:-$SCRIPT_DIR/.venv-q2vmc}"
LAPNET_ROOT="${Q2VMC_LAPNET_ROOT:-$ROOT/LapNet}"
LAPJAX_ROOT="${Q2VMC_LAPJAX_ROOT:-$ROOT/lapjax}"

mkdir -p "$ROOT"

if [[ ! -d "$LAPNET_ROOT/.git" ]]; then
  git clone "https://github.com/bytedance/LapNet.git" "$LAPNET_ROOT"
fi
git -C "$LAPNET_ROOT" fetch --all --tags
git -C "$LAPNET_ROOT" checkout "65101311fd1da759cd112b5f8f15e71d0520ed83"

if [[ ! -d "$LAPJAX_ROOT/.git" ]]; then
  git clone "https://github.com/YWolfeee/lapjax.git" "$LAPJAX_ROOT"
fi
git -C "$LAPJAX_ROOT" fetch --all --tags
git -C "$LAPJAX_ROOT" checkout "f50f734d6f289c468264835f36fc4f9cc6667db0"

if git -C "$LAPNET_ROOT" apply --check "$SCRIPT_DIR/lapnet_grpo.patch" >/dev/null 2>&1; then
  git -C "$LAPNET_ROOT" apply "$SCRIPT_DIR/lapnet_grpo.patch"
elif git -C "$LAPNET_ROOT" apply --reverse --check "$SCRIPT_DIR/lapnet_grpo.patch" >/dev/null 2>&1; then
  echo "LapNet GRPO patch already applied."
else
  echo "Could not apply lapnet_grpo.patch cleanly." >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [[ -n "${JAX_INSTALL_CMD:-}" ]]; then
  echo "Running custom JAX install command"
  eval "$JAX_INSTALL_CMD"
else
  cat <<'EOF'
No JAX install command was supplied.

For H200s, install a CUDA-12-compatible JAX build first, for example by setting:

  export JAX_INSTALL_CMD='pip install -U "jax[cuda12]"'

or by loading your site-provided JAX module before running this script.
EOF
fi

python -m pip install -U \
  absl-py attrs chex dm-haiku flax h5py kfac_jax ml-collections \
  numpy optax pandas pyblock pyscf scipy tables typing_extensions wandb

python -m pip install --no-build-isolation "$LAPJAX_ROOT"
python -m pip install --no-build-isolation --no-deps -e "$LAPNET_ROOT"

cat <<EOF
Bootstrap complete.

Activate the environment with:
  source "$VENV_DIR/bin/activate"

Recommended exports before sbatch:
  export Q2VMC_VENV="$VENV_DIR"
  export Q2VMC_LAPNET_ROOT="$LAPNET_ROOT"

Pinned upstream revisions:
  LapNet: 65101311fd1da759cd112b5f8f15e71d0520ed83
  LapJAX: f50f734d6f289c468264835f36fc4f9cc6667db0
EOF
