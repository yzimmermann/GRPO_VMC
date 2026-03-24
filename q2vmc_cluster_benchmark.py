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
import re
import shlex
import sys
import textwrap
from dataclasses import asdict, dataclass, replace
from itertools import product
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
class GRPOConfig:
    inner_steps: int
    clip_epsilon: float
    max_advantage: float
    inner_optimizer: str
    max_grad_norm: float
    lr_rate: float


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    optimizer: str
    architecture: str
    system_key: str
    system: SystemSpec
    preset: BenchmarkPreset
    seed: int | None = None
    grpo: GRPOConfig | None = None

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
    "tune": BenchmarkPreset(
        name="tune",
        batch_size=1024,
        iterations=5_000,
        pretrain_iterations=500,
        optimizer="kfac",
        forward_laplacian=True,
        deterministic=True,
        use_x64=False,
        save_frequency_minutes=3.0,
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

    sweep = subparsers.add_parser(
        "generate-grpo-sweep",
        help="Materialize a focused GRPO hyperparameter sweep with KFAC baselines.",
    )
    sweep.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Directory to populate with the benchmark pack.",
    )
    sweep.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="tune",
        help="Training budget preset for the sweep.",
    )
    sweep.add_argument(
        "--architectures",
        nargs="+",
        default=["lapnet"],
        choices=sorted(DEFAULT_ARCHITECTURES),
        help="Architectures to include in the sweep.",
    )
    sweep.add_argument(
        "--systems",
        nargs="+",
        default=["Li2"],
        choices=sorted(DEFAULT_SYSTEMS),
        help="Systems to include in the sweep.",
    )
    sweep.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="Explicit debug seeds for the sweep runs.",
    )
    sweep.add_argument(
        "--grpo-inner-steps-grid",
        nargs="+",
        type=int,
        default=[1, 2],
        help="GRPO inner-step values to sweep.",
    )
    sweep.add_argument(
        "--grpo-lr-grid",
        nargs="+",
        type=float,
        default=[1e-4, 3e-4, 1e-3],
        help="GRPO learning rates to sweep.",
    )
    sweep.add_argument(
        "--grpo-clip-grid",
        nargs="+",
        type=float,
        default=[0.1, 0.2],
        help="GRPO clipping epsilons to sweep.",
    )
    sweep.add_argument(
        "--grpo-max-advantage-grid",
        nargs="+",
        type=float,
        default=[2.0],
        help="GRPO max-advantage caps to sweep.",
    )
    sweep.add_argument(
        "--grpo-max-grad-norm-grid",
        nargs="+",
        type=float,
        default=[0.5],
        help="GRPO gradient-clipping thresholds to sweep.",
    )
    sweep.add_argument(
        "--grpo-inner-optimizer",
        choices=("adam", "lamb", "sgd"),
        default="adam",
        help="GRPO inner optimizer to use across the sweep.",
    )
    sweep.add_argument(
        "--no-kfac-baseline",
        action="store_true",
        help="Skip the KFAC baseline runs and generate only GRPO sweeps.",
    )
    sweep.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch-size override for the chosen preset.",
    )
    sweep.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Optional iteration override for the chosen preset.",
    )
    sweep.add_argument(
        "--pretrain-iterations",
        type=int,
        default=None,
        help="Optional pretrain-iteration override for the chosen preset.",
    )
    sweep.add_argument("--job-name", default="q2vmc", help="Slurm job name.")
    sweep.add_argument(
        "--gpus-per-job", type=int, default=1, help="GPUs requested per Slurm task."
    )
    sweep.add_argument(
        "--cpus-per-task", type=int, default=16, help="CPUs requested per Slurm task."
    )
    sweep.add_argument(
        "--mem",
        default="128G",
        help="Memory requested per Slurm task, or 0 for all node memory.",
    )
    sweep.add_argument(
        "--time", default="12:00:00", help="Wall-clock limit for each Slurm task."
    )
    sweep.add_argument(
        "--array-parallelism",
        type=int,
        default=4,
        help="Max number of array tasks allowed to run concurrently.",
    )
    sweep.add_argument("--partition", default="", help="Optional Slurm partition.")
    sweep.add_argument("--account", default="", help="Optional Slurm account.")
    sweep.add_argument("--qos", default="", help="Optional Slurm QoS.")
    sweep.add_argument(
        "--constraint", default="", help="Optional Slurm constraint."
    )
    sweep.add_argument(
        "--wandb-project",
        default=DEFAULT_WANDB_PROJECT,
        help="Weights & Biases project name.",
    )
    sweep.add_argument(
        "--wandb-entity",
        default="",
        help="Optional Weights & Biases entity/team name.",
    )
    sweep.add_argument(
        "--wandb-group",
        default="grpo-tune",
        help="Weights & Biases group name for the sweep.",
    )
    sweep.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="Weights & Biases logging mode.",
    )
    sweep.add_argument(
        "--wandb-poll-seconds",
        type=float,
        default=30.0,
        help="How often to ingest new LapNet CSV rows into W&B.",
    )

    plot = subparsers.add_parser(
        "plot",
        help="Generate overview and GRPO-diagnostic plots from finished runs.",
    )
    plot.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Benchmark directory previously created by `generate`.",
    )
    plot.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write plots into. Defaults to <outdir>/plots.",
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


def slugify_float(value: float) -> str:
    text = f"{value:g}"
    text = text.replace("-", "m").replace("+", "")
    text = text.replace(".", "p")
    return text


def sweep_run_id(
    optimizer: str,
    architecture: str,
    system_key: str,
    *,
    seed: int | None = None,
    grpo: GRPOConfig | None = None,
) -> str:
    parts = [optimizer, architecture, system_key]
    if grpo is not None:
        parts.extend(
            [
                f"is{grpo.inner_steps}",
                f"lr{slugify_float(grpo.lr_rate)}",
                f"clip{slugify_float(grpo.clip_epsilon)}",
                f"adv{slugify_float(grpo.max_advantage)}",
                f"gn{slugify_float(grpo.max_grad_norm)}",
                grpo.inner_optimizer,
            ]
        )
    if seed is not None:
        parts.append(f"seed{seed}")
    return "__".join(parts)


def run_wandb_tags(run: RunSpec) -> list[str]:
    tags = [run.preset.name, run.optimizer, run.architecture, run.system_key]
    if run.seed is not None:
        tags.append(f"seed{run.seed}")
    if run.grpo is not None:
        tags.extend(
            [
                f"K{run.grpo.inner_steps}",
                f"lr{run.grpo.lr_rate:g}",
                f"clip{run.grpo.clip_epsilon:g}",
                f"adv{run.grpo.max_advantage:g}",
                f"gn{run.grpo.max_grad_norm:g}",
                run.grpo.inner_optimizer,
            ]
        )
    return tags


def override_preset_from_args(
    preset: BenchmarkPreset,
    *,
    batch_size: int | None = None,
    iterations: int | None = None,
    pretrain_iterations: int | None = None,
) -> BenchmarkPreset:
    updates = {}
    if batch_size is not None:
        updates["batch_size"] = batch_size
    if iterations is not None:
        updates["iterations"] = iterations
    if pretrain_iterations is not None:
        updates["pretrain_iterations"] = pretrain_iterations
    return replace(preset, **updates) if updates else preset


def build_grpo_sweep_runs(
    args: argparse.Namespace,
    preset: BenchmarkPreset,
) -> list[RunSpec]:
    runs: list[RunSpec] = []
    preset = override_preset_from_args(
        preset,
        batch_size=args.batch_size,
        iterations=args.iterations,
        pretrain_iterations=args.pretrain_iterations,
    )

    for architecture in args.architectures:
        for system_key in args.systems:
            system = SYSTEMS[system_key]
            for seed in args.seeds:
                if not args.no_kfac_baseline:
                    runs.append(
                        RunSpec(
                            run_id=sweep_run_id(
                                "kfac", architecture, system_key, seed=seed
                            ),
                            optimizer="kfac",
                            architecture=architecture,
                            system_key=system_key,
                            system=system,
                            preset=preset,
                            seed=seed,
                        )
                    )

                for inner_steps, lr_rate, clip_epsilon, max_advantage, max_grad_norm in product(
                    args.grpo_inner_steps_grid,
                    args.grpo_lr_grid,
                    args.grpo_clip_grid,
                    args.grpo_max_advantage_grid,
                    args.grpo_max_grad_norm_grid,
                ):
                    grpo = GRPOConfig(
                        inner_steps=inner_steps,
                        clip_epsilon=clip_epsilon,
                        max_advantage=max_advantage,
                        inner_optimizer=args.grpo_inner_optimizer,
                        max_grad_norm=max_grad_norm,
                        lr_rate=lr_rate,
                    )
                    runs.append(
                        RunSpec(
                            run_id=sweep_run_id(
                                "grpo",
                                architecture,
                                system_key,
                                seed=seed,
                                grpo=grpo,
                            ),
                            optimizer="grpo",
                            architecture=architecture,
                            system_key=system_key,
                            system=system,
                            preset=preset,
                            seed=seed,
                            grpo=grpo,
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
    if run.seed is not None:
        flags.append(f"--config.debug.seed={run.seed}")
    if run.optimizer == "grpo":
        if run.grpo is not None:
            flags.extend(
                [
                    f"--config.optim.lr.rate={run.grpo.lr_rate}",
                    f"--config.optim.grpo.inner_steps={run.grpo.inner_steps}",
                    f"--config.optim.grpo.clip_epsilon={run.grpo.clip_epsilon}",
                    f"--config.optim.grpo.max_advantage={run.grpo.max_advantage}",
                    f"--config.optim.grpo.inner_optimizer={run.grpo.inner_optimizer}",
                    f"--config.optim.grpo.max_grad_norm={run.grpo.max_grad_norm}",
                ]
            )
        else:
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


def launcher_grpo_defaults(args: argparse.Namespace) -> dict[str, object]:
    if hasattr(args, "grpo_inner_steps"):
        return {
            "inner_steps": args.grpo_inner_steps,
            "clip_epsilon": args.grpo_clip_epsilon,
            "max_advantage": args.grpo_max_advantage,
            "inner_optimizer": args.grpo_inner_optimizer,
            "max_grad_norm": args.grpo_max_grad_norm,
            "lr_rate": args.grpo_lr_rate,
        }
    return {
        "inner_steps": args.grpo_inner_steps_grid[0],
        "clip_epsilon": args.grpo_clip_grid[0],
        "max_advantage": args.grpo_max_advantage_grid[0],
        "inner_optimizer": args.grpo_inner_optimizer,
        "max_grad_norm": args.grpo_max_grad_norm_grid[0],
        "lr_rate": args.grpo_lr_grid[0],
    }


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
        *run_wandb_tags(run),
    ]
    command.append(f"--entity={entity}")
    return " ".join(command)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_manifest(
    outdir: Path, runs: list[RunSpec], preset: BenchmarkPreset, args: argparse.Namespace
) -> None:
    grpo_defaults = launcher_grpo_defaults(args)
    manifest = {
        "benchmark": "Q2VMC Psiformer/LapNet public baseline harness",
        "preset": asdict(preset),
        "launcher": {
            "gpus_per_job": args.gpus_per_job,
            "array_parallelism": args.array_parallelism,
            "time": args.time,
            "optimizers": sorted({run.optimizer for run in runs}),
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
            "wandb_group": args.wandb_group or f"q2vmc-{preset.name}",
            "wandb_mode": args.wandb_mode,
            "wandb_poll_seconds": args.wandb_poll_seconds,
            "grpo": grpo_defaults,
        },
        "upstream": {name: asdict(repo) for name, repo in UPSTREAM.items()},
        "runs": [
            {
                "run_id": run.run_id,
                "optimizer": run.optimizer,
                "architecture": run.architecture,
                "system_key": run.system_key,
                "seed": run.seed,
                "grpo": asdict(run.grpo) if run.grpo is not None else None,
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
                "seed",
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
                    "seed": run.seed if run.seed is not None else "",
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
            "seed": run.seed,
            "system": asdict(run.system),
            "preset": asdict(run.preset),
            "launch_command": build_lapnet_command(run),
            "command": build_run_command(run, args),
            "grpo": (
                asdict(run.grpo)
                if run.grpo is not None
                else {
                    "inner_steps": getattr(args, "grpo_inner_steps", None),
                    "clip_epsilon": getattr(args, "grpo_clip_epsilon", None),
                    "max_advantage": getattr(args, "grpo_max_advantage", None),
                    "inner_optimizer": getattr(args, "grpo_inner_optimizer", None),
                    "max_grad_norm": getattr(args, "grpo_max_grad_norm", None),
                    "lr_rate": getattr(args, "grpo_lr_rate", None),
                }
            ),
            "wandb": {
                "project": args.wandb_project,
                "entity": args.wandb_entity,
                "group": args.wandb_group or f"q2vmc-{run.preset.name}",
                "mode": args.wandb_mode,
                "tags": run_wandb_tags(run),
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
    grpo_defaults = launcher_grpo_defaults(args)
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

        SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
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
            grpo_defaults["inner_steps"],
            grpo_defaults["clip_epsilon"],
            grpo_defaults["max_advantage"],
            grpo_defaults["inner_optimizer"],
            grpo_defaults["max_grad_norm"],
            grpo_defaults["lr_rate"],
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

This LapNet/LapJAX stack expects the older pinned JAX build used by the
benchmark harness. For the Harvard H200 setup we used, set:

  export JAX_INSTALL_CMD='pip install --upgrade "jax==0.3.24" "jaxlib==0.3.24+cuda11.cudnn82" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'

and load a compatible CUDA 11 module (for example `cuda/11.8.0-fasrc01`)
before running this script.
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

# Clean stale build artifacts from previous failed installs.
rm -rf "$LAPJAX_ROOT/build" "$LAPJAX_ROOT/dist"
find "$LAPJAX_ROOT" -maxdepth 1 -name '*.egg-info' -exec rm -rf {{}} +

# LapJAX's legacy setup.py creates a temporary generated package tree and then
# deletes it in post_setup(); make that cleanup non-fatal and use setup.py
# directly to avoid PEP517 metadata-generation issues on clusters.
python - <<'PY'
from pathlib import Path

setup_path = Path(r"$LAPJAX_ROOT") / "setup.py"
text = setup_path.read_text(encoding="utf-8")
old = "  shutil.rmtree('lapjax')"
new = "  shutil.rmtree('lapjax', ignore_errors=True)"
if old in text:
    setup_path.write_text(text.replace(old, new), encoding="utf-8")
PY

(cd "$LAPJAX_ROOT" && python setup.py install)
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


def maybe_float(value: str) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_train_stats(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    columns = rows[0].keys()
    series: dict[str, list[float]] = {column: [] for column in columns}
    for row in rows:
        for column in columns:
            value = maybe_float(row.get(column, ""))
            if value is None:
                value = float("nan")
            series[column].append(value)
    return series


def load_plot_data(outdir: Path) -> list[dict[str, object]]:
    manifest = load_manifest(outdir)
    rows: list[dict[str, object]] = []
    for run in manifest["runs"]:
        run_dir = outdir / run["results_subdir"]
        stats_path = run_dir / "train_stats.csv"
        if not stats_path.exists():
            continue
        series = load_train_stats(stats_path)
        if not series:
            continue
        rows.append(
            {
                "run_id": run["run_id"],
                "optimizer": run["optimizer"],
                "architecture": run["architecture"],
                "system_key": run["system_key"],
                "stats_path": stats_path,
                "stdout_path": run_dir / "stdout.log",
                "series": series,
            }
        )
    return rows


def load_grpo_stdout_diagnostics(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}

    pattern = re.compile(
        r"Step\s+(?P<t>\d+):.*?grpo_obj=(?P<objective>[-+0-9.eE]+),\s+"
        r"clip_mean=(?P<clip_mean>[-+0-9.eE]+),\s+"
        r"clip_max=(?P<clip_max>[-+0-9.eE]+)"
    )
    series = {
        "t": [],
        "grpo_objective": [],
        "grpo_clip_mean": [],
        "grpo_clip_max": [],
    }

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = pattern.search(line)
            if not match:
                continue
            series["t"].append(float(match.group("t")))
            series["grpo_objective"].append(float(match.group("objective")))
            series["grpo_clip_mean"].append(float(match.group("clip_mean")))
            series["grpo_clip_max"].append(float(match.group("clip_max")))

    if not series["t"]:
        return {}
    return series


def _filtered_xy(
    xs: list[float],
    ys: list[float],
    *,
    max_abs_y: float | None = None,
    positive_only: bool = False,
) -> tuple[list[float], list[float], int]:
    import math

    keep_x: list[float] = []
    keep_y: list[float] = []
    dropped = 0
    for x, y in zip(xs, ys):
        if not math.isfinite(x) or not math.isfinite(y):
            dropped += 1
            continue
        if max_abs_y is not None and abs(y) > max_abs_y:
            dropped += 1
            continue
        if positive_only and y <= 0.0:
            dropped += 1
            continue
        keep_x.append(x)
        keep_y.append(y)
    return keep_x, keep_y, dropped


def make_plots(outdir: Path, output_dir: Path | None = None) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for `plot`. Install it in the current environment "
            "or run the plotting snippet from a plotting-capable Python."
        ) from exc

    plot_rows = load_plot_data(outdir)
    if not plot_rows:
        raise SystemExit(f"No non-empty train_stats.csv files found under {outdir}")

    output_dir = (output_dir or (outdir / "plots")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    systems = list(dict.fromkeys(row["system_key"] for row in plot_rows))
    optimizer_colors = {"kfac": "#1f77b4", "grpo": "#ff7f0e"}
    architecture_styles = {"lapnet": "-", "psiformer": "--"}
    architecture_markers = {"lapnet": None, "psiformer": None}

    plt.style.use("default")

    overview_fig, overview_axes = plt.subplots(
        3,
        len(systems),
        figsize=(5.2 * len(systems), 10.0),
        squeeze=False,
        sharex="col",
    )
    legend_handles = []
    legend_labels = []

    for col, system_key in enumerate(systems):
        system_rows = [row for row in plot_rows if row["system_key"] == system_key]
        hidden_energy_points = 0

        for row in system_rows:
            series = row["series"]
            xs = series.get("t") or series.get("step") or []
            optimizer = str(row["optimizer"])
            architecture = str(row["architecture"])
            label = f"{optimizer}/{architecture}"
            color = optimizer_colors.get(optimizer, "#444444")
            linestyle = architecture_styles.get(architecture, "-")
            marker = architecture_markers.get(architecture, None)

            energy_x, energy_y, dropped_energy = _filtered_xy(
                xs, series.get("ewmean", []), max_abs_y=1.0e6
            )
            hidden_energy_points += dropped_energy
            if energy_x:
                line = overview_axes[0][col].plot(
                    energy_x,
                    energy_y,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=1.8,
                    alpha=0.95,
                    label=label,
                )[0]
                if label not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(label)

            pmove_x, pmove_y, _ = _filtered_xy(xs, series.get("pmove", []))
            if pmove_x:
                overview_axes[1][col].plot(
                    pmove_x,
                    pmove_y,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.5,
                    alpha=0.95,
                )

            var_x, var_y, _ = _filtered_xy(
                xs, series.get("var", []), positive_only=True, max_abs_y=1.0e12
            )
            if var_x:
                overview_axes[2][col].plot(
                    var_x,
                    var_y,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.5,
                    alpha=0.95,
                )

        overview_axes[0][col].set_title(system_key)
        overview_axes[0][col].set_ylabel("EWMA Energy")
        overview_axes[1][col].set_ylabel("pmove")
        overview_axes[1][col].set_ylim(0.0, 1.0)
        overview_axes[2][col].set_ylabel("Batch Var")
        overview_axes[2][col].set_yscale("log")
        overview_axes[2][col].set_xlabel("Iteration")
        if hidden_energy_points:
            overview_axes[0][col].text(
                0.02,
                0.02,
                f"{hidden_energy_points} extreme energy points hidden",
                transform=overview_axes[0][col].transAxes,
                fontsize=8,
                color="#666666",
                ha="left",
                va="bottom",
            )

    for ax_row in overview_axes:
        for ax in ax_row:
            ax.grid(True, alpha=0.25)

    overview_fig.suptitle("Q2VMC Benchmark Overview", fontsize=14)
    if legend_handles:
        overview_fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=min(4, len(legend_handles)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.98),
        )
    overview_fig.tight_layout(rect=(0, 0, 1, 0.94))
    overview_path = output_dir / "q2vmc_overview.png"
    overview_fig.savefig(overview_path, dpi=180)
    plt.close(overview_fig)

    grpo_rows = [row for row in plot_rows if row["optimizer"] == "grpo"]
    output_paths = [overview_path]
    if grpo_rows:
        diag_fig, diag_axes = plt.subplots(
            4,
            len(systems),
            figsize=(5.2 * len(systems), 12.5),
            squeeze=False,
            sharex="col",
        )
        diag_handles = []
        diag_labels = []
        arch_colors = {"lapnet": "#2ca02c", "psiformer": "#d62728"}

        for col, system_key in enumerate(systems):
            system_rows = [
                row
                for row in grpo_rows
                if row["system_key"] == system_key
            ]
            found_any = False
            for row in system_rows:
                series = dict(row["series"])
                if "grpo_objective" not in series:
                    stdout_series = load_grpo_stdout_diagnostics(Path(row["stdout_path"]))
                    if stdout_series:
                        series.update(stdout_series)
                if "grpo_objective" not in series:
                    continue

                found_any = True
                xs = series.get("t") or series.get("step") or []
                architecture = str(row["architecture"])
                color = arch_colors.get(architecture, "#444444")
                line = diag_axes[0][col].plot(
                    xs,
                    series.get("grpo_objective", []),
                    color=color,
                    linewidth=1.7,
                    label=architecture,
                )[0]
                if architecture not in diag_labels:
                    diag_handles.append(line)
                    diag_labels.append(architecture)

                if "grpo_advantage_std" in series:
                    diag_axes[1][col].plot(
                        xs,
                        series.get("grpo_advantage_std", []),
                        color=color,
                        linewidth=1.5,
                    )

                if "grpo_clip_mean" in series:
                    diag_axes[2][col].plot(
                        xs,
                        series.get("grpo_clip_mean", []),
                        color=color,
                        linewidth=1.5,
                    )
                if "grpo_clip_max" in series:
                    diag_axes[2][col].plot(
                        xs,
                        series.get("grpo_clip_max", []),
                        color=color,
                        linewidth=1.2,
                        alpha=0.45,
                    )
                kl_x, kl_mean, _ = _filtered_xy(
                    xs, series.get("grpo_kl_mean", []), positive_only=True
                )
                kl_x2, kl_max, _ = _filtered_xy(
                    xs, series.get("grpo_kl_max", []), positive_only=True
                )
                if kl_x:
                    diag_axes[3][col].plot(
                        kl_x,
                        kl_mean,
                        color=color,
                        linewidth=1.5,
                    )
                if kl_x2:
                    diag_axes[3][col].plot(
                        kl_x2,
                        kl_max,
                        color=color,
                        linewidth=1.2,
                        alpha=0.45,
                    )

            if not found_any:
                diag_axes[0][col].text(
                    0.5,
                    0.5,
                    "No GRPO diagnostics\nfound in CSV/stdout",
                    transform=diag_axes[0][col].transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#666666",
                )

            diag_axes[0][col].set_title(system_key)
            diag_axes[0][col].set_ylabel("GRPO Objective")
            diag_axes[1][col].set_ylabel("Advantage Std")
            diag_axes[2][col].set_ylabel("Clip Mean / Max")
            diag_axes[2][col].set_ylim(0.0, 1.05)
            diag_axes[3][col].set_ylabel("KL Mean / Max")
            diag_axes[3][col].set_xlabel("Iteration")
            diag_axes[3][col].set_yscale("log")

        for ax_row in diag_axes:
            for ax in ax_row:
                ax.grid(True, alpha=0.25)

        diag_fig.suptitle("GRPO Diagnostics", fontsize=14)
        if diag_handles:
            diag_fig.legend(
                diag_handles,
                diag_labels,
                loc="upper center",
                ncol=min(2, len(diag_handles)),
                frameon=False,
                bbox_to_anchor=(0.5, 0.98),
            )
        diag_fig.tight_layout(rect=(0, 0, 1, 0.95))
        diag_path = output_dir / "q2vmc_grpo_diagnostics.png"
        diag_fig.savefig(diag_path, dpi=180)
        plt.close(diag_fig)
        output_paths.append(diag_path)

    return output_paths


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


def handle_generate_grpo_sweep(args: argparse.Namespace) -> int:
    base_preset = PRESETS[args.preset]
    effective_preset = override_preset_from_args(
        base_preset,
        batch_size=args.batch_size,
        iterations=args.iterations,
        pretrain_iterations=args.pretrain_iterations,
    )
    outdir = args.outdir.resolve()
    runs = build_grpo_sweep_runs(args, base_preset)

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results").mkdir(exist_ok=True)
    (outdir / "slurm_logs").mkdir(exist_ok=True)

    write_manifest(outdir, runs, effective_preset, args)
    write_commands(outdir, runs, args)
    write_metadata_csv(outdir, runs)
    write_run_scaffolding(outdir, runs, args)
    write_slurm_script(outdir, args.job_name, len(runs), args)
    write_bootstrap_script(outdir)
    write_lapnet_patch(outdir)
    write_cluster_driver(outdir)
    write_wandb_runner(outdir)
    write_readme(outdir, runs, args)

    print(f"Wrote GRPO sweep pack to {outdir}")
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


def handle_plot(args: argparse.Namespace) -> int:
    outdir = args.outdir.resolve()
    output_paths = make_plots(outdir, args.output_dir)
    print("Wrote plots:")
    for path in output_paths:
        print(path)
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "generate":
        return handle_generate(args)
    if args.command == "generate-grpo-sweep":
        return handle_generate_grpo_sweep(args)
    if args.command == "summarize":
        return handle_summarize(args)
    if args.command == "plot":
        return handle_plot(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
