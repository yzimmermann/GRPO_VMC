#!/usr/bin/env python3
"""Generate and summarize a cluster-ready Q2VMC-style benchmark sweep.

This harness targets the public LapNet repository, which already exposes both
the `lapnet` and `psiformer` wavefunction architectures and includes the small
and medium molecular systems used in the Q2VMC comparison:

  Li2, NH3, CO, CH3NH2 (methylamine), C2H6O (ethanol), C4H6 (bicbut)

The script does not reimplement LapNet/Psiformer. Instead, it materializes a
reproducible run matrix, Slurm array launcher, and a result summarizer so the
benchmark can be pushed to a GPU cluster and run there.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class UpstreamRepo:
    name: str
    url: str
    commit: str


@dataclass(frozen=True)
class SystemSpec:
    paper_name: str
    config_name: str
    config_path: str
    electrons: int


@dataclass(frozen=True)
class BenchmarkPreset:
    name: str
    batch_size: int
    iterations: int
    pretrain_iterations: int
    optimizer: str
    forward_laplacian: bool
    deterministic: bool
    use_x64: bool
    stats_frequency: int = 1
    save_frequency_minutes: float = 10.0
    mcmc_steps: int = 30
    move_width: float = 0.02
    burn_in: int = 100
    pretrain_basis: str = "ccpvdz"


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    optimizer: str
    architecture: str
    system_key: str
    system: SystemSpec
    preset: BenchmarkPreset

    @property
    def results_subdir(self) -> str:
        return f"results/{self.run_id}"


UPSTREAM = {
    "lapnet": UpstreamRepo(
        name="LapNet",
        url="https://github.com/bytedance/LapNet.git",
        commit="65101311fd1da759cd112b5f8f15e71d0520ed83",
    ),
    "lapjax": UpstreamRepo(
        name="LapJAX",
        url="https://github.com/YWolfeee/lapjax.git",
        commit="f50f734d6f289c468264835f36fc4f9cc6667db0",
    ),
}

SYSTEMS = {
    "Li2": SystemSpec("Li2", "Li2", "lapnet/configs/ferminet_system_configs.py", 6),
    "NH3": SystemSpec("NH3", "NH3", "lapnet/configs/ferminet_system_configs.py", 10),
    "CO": SystemSpec("CO", "CO", "lapnet/configs/ferminet_system_configs.py", 14),
    "CH3NH2": SystemSpec(
        "CH3NH2", "methylamine", "lapnet/configs/ferminet_system_configs.py", 16
    ),
    "C2H6O": SystemSpec(
        "C2H6O", "ethanol", "lapnet/configs/ferminet_system_configs.py", 26
    ),
    "C4H6": SystemSpec(
        "C4H6", "bicbut", "lapnet/configs/ferminet_system_configs.py", 30
    ),
}

PRESETS = {
    "smoke": BenchmarkPreset(
        name="smoke",
        batch_size=256,
        iterations=200,
        pretrain_iterations=50,
        optimizer="kfac",
        forward_laplacian=True,
        deterministic=True,
        use_x64=False,
        save_frequency_minutes=2.0,
        burn_in=20,
        mcmc_steps=10,
    ),
    "pilot": BenchmarkPreset(
        name="pilot",
        batch_size=1024,
        iterations=20_000,
        pretrain_iterations=1_000,
        optimizer="kfac",
        forward_laplacian=True,
        deterministic=True,
        use_x64=False,
        save_frequency_minutes=5.0,
        burn_in=50,
        mcmc_steps=20,
    ),
    "paper": BenchmarkPreset(
        name="paper",
        batch_size=4096,
        iterations=200_000,
        pretrain_iterations=5_000,
        optimizer="kfac",
        forward_laplacian=True,
        deterministic=True,
        use_x64=False,
    ),
}

DEFAULT_ARCHITECTURES = ("lapnet", "psiformer")
DEFAULT_SYSTEMS = tuple(SYSTEMS.keys())
DEFAULT_OPTIMIZERS = ("kfac", "grpo")
DEFAULT_WANDB_PROJECT = "q2vmc-cluster"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate",
        help="Materialize the benchmark manifest, commands, and Slurm launcher.",
    )
    generate.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Directory to populate with the benchmark pack.",
    )
    generate.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="paper",
        help="Training budget preset.",
    )
    generate.add_argument(
        "--architectures",
        nargs="+",
        default=list(DEFAULT_ARCHITECTURES),
        choices=sorted(DEFAULT_ARCHITECTURES),
        help="Architectures to benchmark.",
    )
    generate.add_argument(
        "--optimizers",
        nargs="+",
        default=list(DEFAULT_OPTIMIZERS),
        choices=sorted(DEFAULT_OPTIMIZERS),
        help="Optimizers to benchmark.",
    )
    generate.add_argument(
        "--systems",
        nargs="+",
        default=list(DEFAULT_SYSTEMS),
        choices=sorted(DEFAULT_SYSTEMS),
        help="Paper systems to benchmark.",
    )
    generate.add_argument("--job-name", default="q2vmc", help="Slurm job name.")
    generate.add_argument(
        "--gpus-per-job", type=int, default=4, help="GPUs requested per Slurm task."
    )
    generate.add_argument(
        "--cpus-per-task", type=int, default=32, help="CPUs requested per Slurm task."
    )
    generate.add_argument(
        "--mem",
        default="256G",
        help="Memory requested per Slurm task, or 0 for all node memory.",
    )
    generate.add_argument(
        "--time", default="48:00:00", help="Wall-clock limit for each Slurm task."
    )
    generate.add_argument(
        "--array-parallelism",
        type=int,
        default=1,
        help="Max number of array tasks allowed to run concurrently.",
    )
    generate.add_argument(
        "--partition", default="", help="Optional Slurm partition."
    )
    generate.add_argument("--account", default="", help="Optional Slurm account.")
    generate.add_argument("--qos", default="", help="Optional Slurm QoS.")
    generate.add_argument(
        "--constraint", default="", help="Optional Slurm constraint."
    )
    generate.add_argument(
        "--wandb-project",
        default=DEFAULT_WANDB_PROJECT,
        help="Weights & Biases project name.",
    )
    generate.add_argument(
        "--wandb-entity",
        default="",
        help="Optional Weights & Biases entity/team name.",
    )
    generate.add_argument(
        "--wandb-group",
        default="",
        help="Optional Weights & Biases group name. Defaults to q2vmc-<preset>.",
    )
    generate.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="Weights & Biases logging mode.",
    )
    generate.add_argument(
        "--wandb-poll-seconds",
        type=float,
        default=30.0,
        help="How often to ingest new LapNet CSV rows into W&B.",
    )
    generate.add_argument(
        "--grpo-inner-steps",
        type=int,
        default=4,
        help="Number of inner GRPO updates per outer iteration.",
    )
    generate.add_argument(
        "--grpo-clip-epsilon",
        type=float,
        default=0.2,
        help="Clipping epsilon for the GRPO importance ratio.",
    )
    generate.add_argument(
        "--grpo-max-advantage",
        type=float,
        default=5.0,
        help="Absolute cap applied to normalized GRPO advantages.",
    )
    generate.add_argument(
        "--grpo-inner-optimizer",
        choices=("adam", "lamb", "sgd"),
        default="adam",
        help="Inner optimizer used for GRPO updates.",
    )
    generate.add_argument(
        "--grpo-max-grad-norm",
        type=float,
        default=1.0,
        help="Global gradient clipping threshold for GRPO inner updates.",
    )
    generate.add_argument(
        "--grpo-lr-rate",
        type=float,
        default=0.001,
        help="Learning rate used for GRPO runs.",
    )

    summarize = subparsers.add_parser(
        "summarize",
        help="Collect the current status of the generated benchmark runs.",
    )
    summarize.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Benchmark directory previously created by `generate`.",
    )

    return parser.parse_args()


def bool_flag(value: bool) -> str:
    return "true" if value else "false"


def make_run_id(optimizer: str, architecture: str, system_key: str) -> str:
    return f"{optimizer}__{architecture}__{system_key}"


def build_runs(
    optimizers: Iterable[str],
    architectures: Iterable[str],
    systems: Iterable[str],
    preset: BenchmarkPreset,
) -> list[RunSpec]:
    runs = []
    for optimizer in optimizers:
        for architecture in architectures:
            for system_key in systems:
                system = SYSTEMS[system_key]
                runs.append(
                    RunSpec(
                        run_id=make_run_id(optimizer, architecture, system_key),
                        optimizer=optimizer,
                        architecture=architecture,
                        system_key=system_key,
                        system=system,
                        preset=preset,
                    )
                )
    return runs


def build_lapnet_command(run: RunSpec) -> str:
    flags = [
        "python",
        "main.py",
        f"--config={run.system.config_path}",
        f"--config.system.molecule_name={run.system.config_name}",
        f"--config.network.name={run.architecture}",
        f"--config.batch_size={run.preset.batch_size}",
        f"--config.optim.iterations={run.preset.iterations}",
        f"--config.optim.optimizer={run.optimizer}",
        f"--config.optim.forward_laplacian={bool_flag(run.preset.forward_laplacian)}",
        f"--config.log.stats_frequency={run.preset.stats_frequency}",
        f"--config.log.save_frequency={run.preset.save_frequency_minutes}",
        f"--config.log.save_path=${{Q2VMC_RESULTS_ROOT}}/{run.results_subdir}",
        f"--config.mcmc.burn_in={run.preset.burn_in}",
        f"--config.mcmc.steps={run.preset.mcmc_steps}",
        f"--config.mcmc.move_width={run.preset.move_width}",
        f"--config.pretrain.iterations={run.preset.pretrain_iterations}",
        f"--config.pretrain.basis={run.preset.pretrain_basis}",
        f"--config.debug.deterministic={bool_flag(run.preset.deterministic)}",
        f"--config.use_x64={bool_flag(run.preset.use_x64)}",
    ]
    if run.optimizer == "grpo":
        flags.extend(
            [
                f"--config.optim.lr.rate=${{Q2VMC_GRPO_LR_RATE:-0.001}}",
                f"--config.optim.grpo.inner_steps=${{Q2VMC_GRPO_INNER_STEPS:-4}}",
                f"--config.optim.grpo.clip_epsilon=${{Q2VMC_GRPO_CLIP_EPSILON:-0.2}}",
                f"--config.optim.grpo.max_advantage=${{Q2VMC_GRPO_MAX_ADVANTAGE:-5.0}}",
                f"--config.optim.grpo.inner_optimizer=${{Q2VMC_GRPO_INNER_OPTIMIZER:-adam}}",
                f"--config.optim.grpo.max_grad_norm=${{Q2VMC_GRPO_MAX_GRAD_NORM:-1.0}}",
            ]
        )
    return (
        'cd "${Q2VMC_LAPNET_ROOT}" && '
        + " ".join(flags)
        + f' > "${{Q2VMC_RESULTS_ROOT}}/{run.results_subdir}/stdout.log" 2>&1'
    )


def build_run_command(run: RunSpec, args: argparse.Namespace) -> str:
    run_dir = f"${{Q2VMC_RESULTS_ROOT}}/{run.results_subdir}"
    launch_script = f"{run_dir}/run_command.sh"
    if args.wandb_mode == "disabled":
        return f'bash "{launch_script}"'

    group = args.wandb_group or f"q2vmc-{run.preset.name}"
    project = f"${{Q2VMC_WANDB_PROJECT:-{args.wandb_project}}}"
    entity = f"${{Q2VMC_WANDB_ENTITY:-{args.wandb_entity}}}"
    command = [
        "python",
        "${Q2VMC_RESULTS_ROOT}/q2vmc_wandb_runner.py",
        f"--run-dir={run_dir}",
        f"--run-id={run.run_id}",
        f"--metadata={run_dir}/metadata.json",
        f"--stats-file={run_dir}/train_stats.csv",
        f"--launch-script={launch_script}",
        f"--project={project}",
        f"--mode={args.wandb_mode}",
        f"--group={group}",
        f"--poll-seconds={args.wandb_poll_seconds}",
        "--tags",
        run.preset.name,
        run.optimizer,
        run.architecture,
        run.system_key,
    ]
    command.append(f"--entity={entity}")
    return " ".join(command)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_manifest(
    outdir: Path, runs: list[RunSpec], preset: BenchmarkPreset, args: argparse.Namespace
) -> None:
    manifest = {
        "benchmark": "Q2VMC Psiformer/LapNet public baseline harness",
        "preset": asdict(preset),
        "launcher": {
            "gpus_per_job": args.gpus_per_job,
            "array_parallelism": args.array_parallelism,
            "time": args.time,
            "optimizers": list(args.optimizers),
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
            "wandb_group": args.wandb_group or f"q2vmc-{preset.name}",
            "wandb_mode": args.wandb_mode,
            "wandb_poll_seconds": args.wandb_poll_seconds,
            "grpo": {
                "inner_steps": args.grpo_inner_steps,
                "clip_epsilon": args.grpo_clip_epsilon,
                "max_advantage": args.grpo_max_advantage,
                "inner_optimizer": args.grpo_inner_optimizer,
                "max_grad_norm": args.grpo_max_grad_norm,
                "lr_rate": args.grpo_lr_rate,
            },
        },
        "upstream": {name: asdict(repo) for name, repo in UPSTREAM.items()},
        "runs": [
            {
                "run_id": run.run_id,
                "optimizer": run.optimizer,
                "architecture": run.architecture,
                "system_key": run.system_key,
                "system": asdict(run.system),
                "results_subdir": run.results_subdir,
                "launch_command": build_lapnet_command(run),
                "command": build_run_command(run, args),
            }
            for run in runs
        ],
    }
    write_text(outdir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True))


def write_commands(
    outdir: Path, runs: list[RunSpec], args: argparse.Namespace
) -> None:
    commands_path = outdir / "commands.txt"
    with commands_path.open("w", encoding="utf-8") as handle:
        for run in runs:
            handle.write(build_run_command(run, args))
            handle.write("\n")


def write_metadata_csv(outdir: Path, runs: list[RunSpec]) -> None:
    metadata_path = outdir / "run_metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "optimizer",
                "architecture",
                "system_key",
                "paper_name",
                "config_name",
                "electrons",
                "results_subdir",
            ],
        )
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "run_id": run.run_id,
                    "optimizer": run.optimizer,
                    "architecture": run.architecture,
                    "system_key": run.system_key,
                    "paper_name": run.system.paper_name,
                    "config_name": run.system.config_name,
                    "electrons": run.system.electrons,
                    "results_subdir": run.results_subdir,
                }
            )


def write_run_scaffolding(
    outdir: Path, runs: list[RunSpec], args: argparse.Namespace
) -> None:
    for run in runs:
        run_dir = outdir / run.results_subdir
        run_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "run_id": run.run_id,
            "optimizer": run.optimizer,
            "architecture": run.architecture,
            "system_key": run.system_key,
            "system": asdict(run.system),
            "preset": asdict(run.preset),
            "launch_command": build_lapnet_command(run),
            "command": build_run_command(run, args),
            "grpo": {
                "inner_steps": args.grpo_inner_steps,
                "clip_epsilon": args.grpo_clip_epsilon,
                "max_advantage": args.grpo_max_advantage,
                "inner_optimizer": args.grpo_inner_optimizer,
                "max_grad_norm": args.grpo_max_grad_norm,
                "lr_rate": args.grpo_lr_rate,
            },
            "wandb": {
                "project": args.wandb_project,
                "entity": args.wandb_entity,
                "group": args.wandb_group or f"q2vmc-{run.preset.name}",
                "mode": args.wandb_mode,
                "tags": [run.preset.name, run.optimizer, run.architecture, run.system_key],
            },
        }
        write_text(run_dir / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True))
        launch_script = textwrap.dedent(
            f"""\
            #!/bin/bash
            set -euo pipefail
            {build_lapnet_command(run)}
            """
        )
        launch_path = run_dir / "run_command.sh"
        write_text(launch_path, launch_script)
        launch_path.chmod(0o755)


def write_slurm_script(
    outdir: Path, job_name: str, num_runs: int, args: argparse.Namespace
) -> None:
    if args.array_parallelism and args.array_parallelism > 0:
        array_spec = f"1-{num_runs}%{args.array_parallelism}"
    else:
        array_spec = f"1-{num_runs}"
    directives = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --array={array_spec}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --gres=gpu:{args.gpus_per_job}",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH --time={args.time}",
        f"#SBATCH --output=slurm_logs/%x_%A_%a.out",
        f"#SBATCH --error=slurm_logs/%x_%A_%a.err",
    ]
    if args.partition:
        directives.append(f"#SBATCH --partition={args.partition}")
    if args.account:
        directives.append(f"#SBATCH --account={args.account}")
    if args.qos:
        directives.append(f"#SBATCH --qos={args.qos}")
    if args.constraint:
        directives.append(f"#SBATCH --constraint={args.constraint}")

    script = "\n".join(directives) + "\n\n" + textwrap.dedent(
        """\
        set -euo pipefail

        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        export Q2VMC_RESULTS_ROOT="${SCRIPT_DIR}"
        export Q2VMC_LAPNET_ROOT="${Q2VMC_LAPNET_ROOT:-${SLURM_SUBMIT_DIR}/external/LapNet}"
        export PYTHONUNBUFFERED=1
        export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
        export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"
        export Q2VMC_WANDB_PROJECT="${Q2VMC_WANDB_PROJECT:-%s}"
        export Q2VMC_WANDB_ENTITY="${Q2VMC_WANDB_ENTITY:-%s}"
        export Q2VMC_GRPO_INNER_STEPS="${Q2VMC_GRPO_INNER_STEPS:-%s}"
        export Q2VMC_GRPO_CLIP_EPSILON="${Q2VMC_GRPO_CLIP_EPSILON:-%s}"
        export Q2VMC_GRPO_MAX_ADVANTAGE="${Q2VMC_GRPO_MAX_ADVANTAGE:-%s}"
        export Q2VMC_GRPO_INNER_OPTIMIZER="${Q2VMC_GRPO_INNER_OPTIMIZER:-%s}"
        export Q2VMC_GRPO_MAX_GRAD_NORM="${Q2VMC_GRPO_MAX_GRAD_NORM:-%s}"
        export Q2VMC_GRPO_LR_RATE="${Q2VMC_GRPO_LR_RATE:-%s}"

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
        """
        % (
            args.wandb_project,
            args.wandb_entity,
            args.grpo_inner_steps,
            args.grpo_clip_epsilon,
            args.grpo_max_advantage,
            args.grpo_inner_optimizer,
            args.grpo_max_grad_norm,
            args.grpo_lr_rate,
        )
    )
    write_text(outdir / "submit_q2vmc_array.slurm", script)


def write_bootstrap_script(outdir: Path) -> None:
    script = f"""#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
ROOT="${{1:-$SCRIPT_DIR/external}}"
PYTHON_BIN="${{Q2VMC_PYTHON:-python3.10}}"
VENV_DIR="${{Q2VMC_VENV:-$SCRIPT_DIR/.venv-q2vmc}}"
LAPNET_ROOT="${{Q2VMC_LAPNET_ROOT:-$ROOT/LapNet}}"
LAPJAX_ROOT="${{Q2VMC_LAPJAX_ROOT:-$ROOT/lapjax}}"

mkdir -p "$ROOT"

if [[ ! -d "$LAPNET_ROOT/.git" ]]; then
  git clone "{UPSTREAM["lapnet"].url}" "$LAPNET_ROOT"
fi
git -C "$LAPNET_ROOT" fetch --all --tags
git -C "$LAPNET_ROOT" checkout "{UPSTREAM["lapnet"].commit}"

if [[ ! -d "$LAPJAX_ROOT/.git" ]]; then
  git clone "{UPSTREAM["lapjax"].url}" "$LAPJAX_ROOT"
fi
git -C "$LAPJAX_ROOT" fetch --all --tags
git -C "$LAPJAX_ROOT" checkout "{UPSTREAM["lapjax"].commit}"

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

if [[ -n "${{JAX_INSTALL_CMD:-}}" ]]; then
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

python -m pip install -U \\
  absl-py==1.4.0 attrs==21.2.0 chex==0.1.5 dm-haiku==0.0.9 \\
  dm-tree==0.1.8 distrax==0.1.2 flax==0.6.1 h5py==3.8.0 \\
  immutabledict==2.2.3 kfac_jax==0.0.3 ml-collections==0.1.1 \\
  numpy==1.21.5 optax==0.1.4 pandas==1.3.5 pyblock==0.6 \\
  pyscf==2.1.1 scipy==1.7.3 tables==3.7.0 \\
  tensorflow-probability==0.19.0 typing_extensions==4.5.0

python -m pip install -U wandb

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
  LapNet: {UPSTREAM["lapnet"].commit}
  LapJAX: {UPSTREAM["lapjax"].commit}
EOF
"""
    path = outdir / "bootstrap_q2vmc_env.sh"
    write_text(path, script)
    path.chmod(0o755)


def write_wandb_runner(outdir: Path) -> None:
    template = Path(__file__).with_name("q2vmc_wandb_runner.py")
    write_text(outdir / "q2vmc_wandb_runner.py", template.read_text(encoding="utf-8"))


def write_cluster_driver(outdir: Path) -> None:
    template = Path(__file__)
    write_text(
        outdir / "q2vmc_cluster_benchmark.py", template.read_text(encoding="utf-8")
    )


def write_lapnet_patch(outdir: Path) -> None:
    template = Path(__file__).parent / "patches" / "lapnet_grpo.patch"
    write_text(outdir / "lapnet_grpo.patch", template.read_text(encoding="utf-8"))


def write_readme(outdir: Path, runs: list[RunSpec], args: argparse.Namespace) -> None:
    architectures = ", ".join(sorted({run.architecture for run in runs}))
    optimizers = ", ".join(sorted({run.optimizer for run in runs}))
    systems = ", ".join(sorted({run.system_key for run in runs}))
    content = textwrap.dedent(
        f"""\
        # Q2VMC Cluster Benchmark

        This directory was generated by `q2vmc_cluster_benchmark.py` and is meant
        to be pushed to a GPU cluster and launched as a Slurm array job.

        ## What It Benchmarks

        - Architectures: {architectures}
        - Systems: {systems}
        - Preset: `{args.preset}`
        - Optimizers: {optimizers}
        - W&B mode: `{args.wandb_mode}`

        ## Files

        - `manifest.json`: full run manifest with pinned upstream revisions
        - `commands.txt`: one shell command per run
        - `run_metadata.csv`: flat metadata table
        - `submit_q2vmc_array.slurm`: Slurm launcher
        - `bootstrap_q2vmc_env.sh`: reproducible checkout/bootstrap helper
        - `lapnet_grpo.patch`: patch applied to the pinned LapNet checkout to add GRPO
        - `q2vmc_cluster_benchmark.py`: self-contained summarize/generate helper
        - `q2vmc_wandb_runner.py`: CSV-to-W&B streaming wrapper
        - `results/<run_id>/`: output directory for each benchmark run

        ## Suggested Workflow

        1. On the cluster login node, bootstrap the environment:

           `bash bootstrap_q2vmc_env.sh`

        2. Export the environment variables expected by the Slurm script:

           `export Q2VMC_VENV="$PWD/.venv-q2vmc"`
           `export Q2VMC_LAPNET_ROOT="$PWD/external/LapNet"`
           `export Q2VMC_WANDB_PROJECT="{args.wandb_project}"`

        3. Submit the array job:

           `sbatch submit_q2vmc_array.slurm`

        4. Summarize finished runs:

           `python q2vmc_cluster_benchmark.py summarize --outdir "$PWD"`

        ## Notes

        - The harness uses the public LapNet repository for both `lapnet` and
          `psiformer`, because that repo already contains both network types and
          the exact small-to-medium molecule configs we want.
        - `bootstrap_q2vmc_env.sh` applies `lapnet_grpo.patch` after checking out
          the pinned upstream commit, so the generated pack can launch either
          baseline `kfac` runs or GRPO runs from the same code snapshot.
        - Each benchmark run is launched through `q2vmc_wandb_runner.py`, which
          tails `train_stats.csv` and forwards new rows to W&B.
        - `config.debug.deterministic=true` is enabled in the generated preset to
          make optimizer comparisons easier to reproduce.
        - H200-compatible JAX installation varies by cluster image, so the
          bootstrap script intentionally leaves the JAX install line configurable
          via `JAX_INSTALL_CMD`.
        - The heavy training itself was not executed in this local workspace,
          because it requires a CUDA/JAX GPU environment.
        """
    )
    write_text(outdir / "README.md", content)


def load_manifest(outdir: Path) -> dict:
    manifest_path = outdir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def summarize_runs(outdir: Path) -> list[dict[str, object]]:
    manifest = load_manifest(outdir)
    summaries = []
    for run in manifest["runs"]:
        run_dir = outdir / run["results_subdir"]
        stats_path = run_dir / "train_stats.csv"
        stdout_path = run_dir / "stdout.log"
        summary = {
            "run_id": run["run_id"],
            "optimizer": run.get("optimizer", ""),
            "architecture": run["architecture"],
            "system_key": run["system_key"],
            "status": "missing",
            "steps_logged": 0,
            "final_energy": "",
            "final_ewmean": "",
            "best_ewmean": "",
            "final_pmove": "",
            "stdout_log": str(stdout_path),
            "wandb_url": "",
        }
        wandb_path = run_dir / "wandb_run.json"
        if stats_path.exists():
            with stats_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if rows:
                final = rows[-1]
                ewmeans = [float(row["ewmean"]) for row in rows if row.get("ewmean")]
                summary.update(
                    {
                        "status": "completed",
                        "steps_logged": len(rows),
                        "final_energy": final.get("energy", ""),
                        "final_ewmean": final.get("ewmean", ""),
                        "best_ewmean": min(ewmeans) if ewmeans else "",
                        "final_pmove": final.get("pmove", ""),
                    }
                )
            else:
                summary["status"] = "empty"
        elif stdout_path.exists():
            summary["status"] = "started"
        if wandb_path.exists():
            wandb_data = json.loads(wandb_path.read_text(encoding="utf-8"))
            summary["wandb_url"] = wandb_data.get("url", "")
        summaries.append(summary)
    return summaries


def write_summary(outdir: Path, rows: list[dict[str, object]]) -> None:
    path = outdir / "summary.csv"
    fieldnames = [
        "run_id",
        "optimizer",
        "architecture",
        "system_key",
        "status",
        "steps_logged",
        "final_energy",
        "final_ewmean",
        "best_ewmean",
        "final_pmove",
        "stdout_log",
        "wandb_url",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary_table(rows: list[dict[str, object]]) -> None:
    headers = [
        "run_id",
        "optimizer",
        "architecture",
        "system_key",
        "status",
        "steps_logged",
        "final_energy",
        "final_ewmean",
        "best_ewmean",
        "final_pmove",
        "wandb_url",
    ]
    widths = {key: len(key) for key in headers}
    for row in rows:
        for key in headers:
            widths[key] = max(widths[key], len(str(row.get(key, ""))))

    def fmt(row: dict[str, object]) -> str:
        return " | ".join(str(row.get(key, "")).ljust(widths[key]) for key in headers)

    print(fmt({key: key for key in headers}))
    print("-+-".join("-" * widths[key] for key in headers))
    for row in rows:
        print(fmt(row))


def handle_generate(args: argparse.Namespace) -> int:
    preset = PRESETS[args.preset]
    outdir = args.outdir.resolve()
    runs = build_runs(args.optimizers, args.architectures, args.systems, preset)

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results").mkdir(exist_ok=True)
    (outdir / "slurm_logs").mkdir(exist_ok=True)

    write_manifest(outdir, runs, preset, args)
    write_commands(outdir, runs, args)
    write_metadata_csv(outdir, runs)
    write_run_scaffolding(outdir, runs, args)
    write_slurm_script(outdir, args.job_name, len(runs), args)
    write_bootstrap_script(outdir)
    write_lapnet_patch(outdir)
    write_cluster_driver(outdir)
    write_wandb_runner(outdir)
    write_readme(outdir, runs, args)

    print(f"Wrote benchmark pack to {outdir}")
    print(f"Runs: {len(runs)}")
    print(f"Slurm launcher: {outdir / 'submit_q2vmc_array.slurm'}")
    print(f"Bootstrap helper: {outdir / 'bootstrap_q2vmc_env.sh'}")
    return 0


def handle_summarize(args: argparse.Namespace) -> int:
    outdir = args.outdir.resolve()
    rows = summarize_runs(outdir)
    write_summary(outdir, rows)
    print_summary_table(rows)
    print(f"\nSummary CSV: {outdir / 'summary.csv'}")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "generate":
        return handle_generate(args)
    if args.command == "summarize":
        return handle_summarize(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
