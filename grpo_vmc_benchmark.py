import matplotlib

matplotlib.use("Agg")

from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class BenchmarkConfig:
    n_iterations: int = 300
    G: int = 1000
    alpha_init: float = 1.5
    eta_sgd: float = 0.05
    eta_sr: float = 0.5
    eta_grpo_inner: float = 0.1
    K_inner: int = 5
    eps_clip: float = 0.2
    seed: int = 42
    clip_guard_threshold: float = 0.4
    clip_guard_step: int = 2
    sweep_K_values: tuple[int, ...] = (1, 3, 5, 10)
    sweep_eps_values: tuple[float, ...] = (0.1, 0.2, 0.3)


def sample_configs(alpha, n_samples, step_size=0.5, n_burn=200):
    """Return Metropolis-Hastings samples from |psi_alpha(x)|^2."""
    samples = np.empty(n_samples, dtype=np.float64)
    x_curr = 0.0
    logp_curr = log_prob(x_curr, alpha)

    total_steps = n_burn + n_samples
    sample_idx = 0
    for step in range(total_steps):
        x_prop = x_curr + step_size * np.random.randn()
        logp_prop = log_prob(x_prop, alpha)
        if np.log(np.random.rand()) < (logp_prop - logp_curr):
            x_curr = x_prop
            logp_curr = logp_prop

        if step >= n_burn:
            samples[sample_idx] = x_curr
            sample_idx += 1

    return samples


def local_energy(x, alpha):
    """Analytical local energy for the 1D harmonic oscillator."""
    return 0.5 * np.exp(alpha) + (0.5 - 0.5 * np.exp(2.0 * alpha)) * x**2


def log_prob(x, alpha):
    """Unnormalized log |psi_alpha(x)|^2."""
    return -np.exp(alpha) * x**2


def score_function(x, alpha):
    """Derivative of log |psi_alpha(x)|^2 up to an additive constant."""
    return -np.exp(alpha) * x**2


def vmc_gradient(local_energies, scores):
    centered_energies = local_energies - np.mean(local_energies)
    return 2.0 * np.mean(centered_energies * scores)


def fisher_scalar(scores, reg=1e-3):
    return np.mean(scores**2) - np.mean(scores) ** 2 + reg


def rolling_mean(values, window=10):
    values = np.asarray(values, dtype=np.float64)
    cumsum = np.cumsum(values)
    smoothed = np.empty_like(values)
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        total = cumsum[idx] - (cumsum[start - 1] if start > 0 else 0.0)
        smoothed[idx] = total / (idx - start + 1)
    return smoothed


def run_sgd(cfg):
    np.random.seed(cfg.seed)
    alpha = float(cfg.alpha_init)

    energies = []
    alphas = []
    el_evals = []
    grad_steps = []
    total_el_evals = 0
    total_grad_steps = 0

    for _ in range(cfg.n_iterations):
        x = sample_configs(alpha, cfg.G)
        e_loc = local_energy(x, alpha)
        scores = score_function(x, alpha)

        grad = vmc_gradient(e_loc, scores)

        total_el_evals += cfg.G
        energies.append(np.mean(e_loc))
        alphas.append(alpha)
        el_evals.append(total_el_evals)
        grad_steps.append(total_grad_steps)

        alpha -= cfg.eta_sgd * grad
        total_grad_steps += 1

    return {
        "energies": np.asarray(energies),
        "alphas": np.asarray(alphas),
        "el_evals": np.asarray(el_evals),
        "grad_steps": np.asarray(grad_steps),
    }


def run_sr(cfg):
    np.random.seed(cfg.seed)
    alpha = float(cfg.alpha_init)

    energies = []
    alphas = []
    el_evals = []
    grad_steps = []
    total_el_evals = 0
    total_grad_steps = 0

    for _ in range(cfg.n_iterations):
        x = sample_configs(alpha, cfg.G)
        e_loc = local_energy(x, alpha)
        scores = score_function(x, alpha)

        grad = vmc_gradient(e_loc, scores)
        sr_metric = fisher_scalar(scores, reg=1e-3)

        total_el_evals += cfg.G
        energies.append(np.mean(e_loc))
        alphas.append(alpha)
        el_evals.append(total_el_evals)
        grad_steps.append(total_grad_steps)

        alpha -= cfg.eta_sr * grad / sr_metric
        total_grad_steps += 1

    return {
        "energies": np.asarray(energies),
        "alphas": np.asarray(alphas),
        "el_evals": np.asarray(el_evals),
        "grad_steps": np.asarray(grad_steps),
    }


def grpo_objective_and_grad(alpha, alpha_old, x, advantages, eps_clip):
    log_ratio = log_prob(x, alpha) - log_prob(x, alpha_old)
    ratio = np.exp(np.clip(log_ratio, -60.0, 60.0))
    clipped_ratio = np.clip(ratio, 1.0 - eps_clip, 1.0 + eps_clip)

    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages
    use_unclipped = unclipped <= clipped
    frac_clipped = np.mean(~use_unclipped)

    d_log_ratio = score_function(x, alpha)
    d_ratio = ratio * d_log_ratio
    grad = np.mean(np.where(use_unclipped, advantages * d_ratio, 0.0))
    obj = np.mean(np.minimum(unclipped, clipped))
    return obj, grad, frac_clipped


def run_grpo(cfg):
    np.random.seed(cfg.seed)
    alpha = float(cfg.alpha_init)
    adv_eps = 1e-8
    eta_inner = float(cfg.eta_grpo_inner)

    energies = []
    alphas = []
    el_evals = []
    grad_steps = []
    frac_clipped_history = np.zeros((cfg.n_iterations, cfg.K_inner), dtype=np.float64)
    eta_inner_history = []
    total_el_evals = 0
    total_grad_steps = 0
    halving_events = []

    for outer_idx in range(cfg.n_iterations):
        alpha_old = alpha
        x = sample_configs(alpha_old, cfg.G)
        e_loc_old = local_energy(x, alpha_old)

        centered = e_loc_old - np.mean(e_loc_old)
        advantages = -centered / (np.std(e_loc_old) + adv_eps)

        total_el_evals += cfg.G
        energies.append(np.mean(e_loc_old))
        alphas.append(alpha_old)
        el_evals.append(total_el_evals)
        grad_steps.append(total_grad_steps)
        eta_inner_history.append(eta_inner)

        for inner_idx in range(cfg.K_inner):
            _, grad, frac_clipped = grpo_objective_and_grad(
                alpha=alpha,
                alpha_old=alpha_old,
                x=x,
                advantages=advantages,
                eps_clip=cfg.eps_clip,
            )
            frac_clipped_history[outer_idx, inner_idx] = frac_clipped

            if (
                inner_idx + 1 == cfg.clip_guard_step
                and frac_clipped > cfg.clip_guard_threshold
            ):
                eta_inner *= 0.5
                halving_events.append(
                    {
                        "outer_iteration": outer_idx,
                        "inner_step": inner_idx + 1,
                        "frac_clipped": float(frac_clipped),
                        "new_eta_inner": float(eta_inner),
                    }
                )

            alpha += eta_inner * grad
            total_grad_steps += 1

    mean_frac_clipped = np.mean(frac_clipped_history, axis=0)
    max_frac_clipped = np.max(frac_clipped_history, axis=0)

    return {
        "energies": np.asarray(energies),
        "alphas": np.asarray(alphas),
        "el_evals": np.asarray(el_evals),
        "grad_steps": np.asarray(grad_steps),
        "frac_clipped_history": frac_clipped_history,
        "mean_frac_clipped": mean_frac_clipped,
        "max_frac_clipped": max_frac_clipped,
        "eta_inner_history": np.asarray(eta_inner_history),
        "final_eta_inner": float(eta_inner),
        "num_eta_halvings": len(halving_events),
        "halving_events": halving_events,
    }


def first_hit(result, threshold=0.51):
    energies = np.asarray(result["energies"])
    hit_indices = np.flatnonzero(np.asarray(energies) < threshold)
    if hit_indices.size == 0:
        return "n/a", "n/a", "n/a"

    idx = int(hit_indices[0])
    return (
        str(idx + 1),
        str(int(result["grad_steps"][idx])),
        str(int(result["el_evals"][idx])),
    )


def print_summary(results):
    print(
        "Optimizer | Final E  | Iters to E<0.51 | Grad steps to E<0.51 | E_L evals to E<0.51\n"
        "----------|----------|-----------------|----------------------|--------------------"
    )

    for name, result in results.items():
        final_e = result["energies"][-1]
        hit_iter, hit_grad_steps, hit_evals = first_hit(result)
        print(
            f"{name:<9} | {final_e:>8.5f} | {hit_iter:>15} | {hit_grad_steps:>20} | {hit_evals:>20}"
        )


def make_plots(results, cfg, output_path="grpo_vmc_benchmark.png"):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    colors = {"SGD": "tab:blue", "SR": "tab:orange", "GRPO-VMC": "tab:green"}

    for name, result in results.items():
        smoothed = rolling_mean(result["energies"], window=10)
        iterations = np.arange(len(smoothed))

        axes[0].plot(iterations, smoothed, label=name, color=colors[name], lw=2)
        axes[1].plot(
            result["grad_steps"], smoothed, label=name, color=colors[name], lw=2
        )
        axes[2].plot(
            result["el_evals"], smoothed, label=name, color=colors[name], lw=2
        )

    for ax in axes:
        ax.axhline(
            0.5, color="black", linestyle="--", linewidth=1.25, label="E0 = 0.5"
        )
        ax.set_ylabel(r"$\langle E_L \rangle$")
        ax.grid(alpha=0.25)

    axes[0].set_xlim(0, cfg.n_iterations)
    axes[0].set_xlabel("Outer iteration")
    axes[0].set_title("Convergence: Energy vs. Iterations")

    axes[1].set_xlabel("Cumulative gradient steps")
    axes[1].set_title("Optimization Efficiency: Energy vs. Gradient Steps")

    axes[2].set_xlabel(r"Cumulative $E_L$ evaluations")
    axes[2].set_title(r"Sample Efficiency: Energy vs. $E_L$ Evaluations")

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), frameon=False)
    axes[1].legend(unique.values(), unique.keys(), frameon=False)
    axes[2].legend(unique.values(), unique.keys(), frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")


def format_clip_fracs(fracs):
    return "[" + ", ".join(f"{frac:.3f}" for frac in fracs) + "]"


def print_grpo_diagnostics(label, result):
    print(f"\n{label} clip diagnostics:")
    for inner_step, frac_clipped in enumerate(result["mean_frac_clipped"], start=1):
        max_frac = result["max_frac_clipped"][inner_step - 1]
        print(
            f"  frac_clipped[{inner_step}] mean={frac_clipped:.3f}, max={max_frac:.3f}"
        )

    print(
        "  eta_inner:"
        f" start={result['eta_inner_history'][0]:.5f},"
        f" final={result['final_eta_inner']:.5f},"
        f" halvings={result['num_eta_halvings']}"
    )

    if result["halving_events"]:
        first_event = result["halving_events"][0]
        print(
            "  first halving:"
            f" outer_iteration={first_event['outer_iteration']},"
            f" inner_step={first_event['inner_step']},"
            f" frac_clipped={first_event['frac_clipped']:.3f},"
            f" new_eta_inner={first_event['new_eta_inner']:.5f}"
        )


def run_grpo_sweep(cfg, cached_results=None):
    cached_results = {} if cached_results is None else cached_results
    sweep_rows = []

    for k_inner in cfg.sweep_K_values:
        for eps_clip in cfg.sweep_eps_values:
            cache_key = (k_inner, eps_clip)
            sweep_cfg = replace(cfg, K_inner=k_inner, eps_clip=eps_clip)
            result = cached_results.get(cache_key)
            if result is None:
                result = run_grpo(sweep_cfg)

            hit_iter, hit_grad_steps, hit_evals = first_hit(result)
            step2_clip = (
                result["mean_frac_clipped"][1]
                if result["mean_frac_clipped"].size >= 2
                else np.nan
            )
            step2_clip_max = (
                result["max_frac_clipped"][1]
                if result["max_frac_clipped"].size >= 2
                else np.nan
            )

            sweep_rows.append(
                {
                    "K_inner": k_inner,
                    "eps_clip": eps_clip,
                    "result": result,
                    "final_energy": float(result["energies"][-1]),
                    "iters_to_threshold": hit_iter,
                    "grad_steps_to_threshold": hit_grad_steps,
                    "el_evals_to_threshold": hit_evals,
                    "step2_clip": step2_clip,
                    "step2_clip_max": step2_clip_max,
                    "final_eta_inner": float(result["final_eta_inner"]),
                    "num_eta_halvings": int(result["num_eta_halvings"]),
                }
            )

    return sweep_rows


def _sort_key_for_threshold(value):
    return int(value) if value != "n/a" else 10**12


def print_grpo_sweep(sweep_rows):
    ordered_rows = sorted(
        sweep_rows,
        key=lambda row: (
            _sort_key_for_threshold(row["el_evals_to_threshold"]),
            row["final_energy"],
            row["K_inner"],
            row["eps_clip"],
        ),
    )

    print(
        "\nGRPO sweep over K_inner and eps_clip:\n"
        "K | eps_clip | Final E  | Iters to E<0.51 | Grad steps to E<0.51 | E_L evals to E<0.51 | Step-2 clip mean/max | Final eta | Halvings\n"
        "--|----------|----------|-----------------|----------------------|--------------------|----------------------|-----------|---------"
    )

    for row in ordered_rows:
        if np.isnan(row["step2_clip"]):
            step2_clip = "n/a"
        else:
            step2_clip = f"{row['step2_clip']:.3f}/{row['step2_clip_max']:.3f}"
        print(
            f"{row['K_inner']:>1} | {row['eps_clip']:^8.1f} | {row['final_energy']:>8.5f} | "
            f"{row['iters_to_threshold']:>15} | {row['grad_steps_to_threshold']:>20} | "
            f"{row['el_evals_to_threshold']:>20} | {step2_clip:>20} | "
            f"{row['final_eta_inner']:>9.5f} | {row['num_eta_halvings']:>8}"
        )

    print("\nPer-config clip diagnostics:")
    for row in ordered_rows:
        print(
            f"  K={row['K_inner']}, eps_clip={row['eps_clip']:.1f}: "
            f"frac_clipped={format_clip_fracs(row['result']['mean_frac_clipped'])}, "
            f"final_eta={row['final_eta_inner']:.5f}, "
            f"halvings={row['num_eta_halvings']}"
        )


def main():
    cfg = BenchmarkConfig()

    results = {
        "SGD": run_sgd(cfg),
        "SR": run_sr(cfg),
        "GRPO-VMC": run_grpo(cfg),
    }
    sweep_rows = run_grpo_sweep(
        cfg,
        cached_results={(cfg.K_inner, cfg.eps_clip): results["GRPO-VMC"]},
    )

    make_plots(results, cfg)
    print_summary(results)
    print_grpo_diagnostics("GRPO-VMC", results["GRPO-VMC"])
    print_grpo_sweep(sweep_rows)
    print("\nSaved plot to grpo_vmc_benchmark.png")


if __name__ == "__main__":
    main()
