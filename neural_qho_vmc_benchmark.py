import math
import time
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class NeuralBenchmarkConfig:
    n_iterations: int = 120
    G: int = 512
    n_burn: int = 200
    mh_step_size: float = 0.9
    hidden_dim: int = 32
    log_a_init: float = 1.5
    weight_scale: float = 0.05
    eta_sgd: float = 0.02
    eta_sr: float = 0.2
    eta_kfac: float = 0.15
    eta_grpo_inner: float = 0.02
    K_inner: int = 5
    eps_clip: float = 0.3
    sr_reg: float = 1e-3
    kfac_reg: float = 5e-3
    threshold_energy: float = 0.51
    clip_guard_threshold: float = 0.4
    clip_guard_step: int = 2
    seed: int = 7
    width_sweep_hidden_dims: tuple[int, ...] = (32, 64, 128, 256)
    width_sweep_iterations: int = 80


def rolling_mean(values, window=10):
    values = np.asarray(values, dtype=np.float64)
    cumsum = np.cumsum(values)
    smoothed = np.empty_like(values)
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        total = cumsum[idx] - (cumsum[start - 1] if start > 0 else 0.0)
        smoothed[idx] = total / (idx - start + 1)
    return smoothed


def init_theta(cfg):
    rng = np.random.default_rng(cfg.seed)
    hidden_dim = cfg.hidden_dim

    theta = np.zeros(1 + 3 * hidden_dim, dtype=np.float64)
    theta[0] = cfg.log_a_init

    idx = 1
    theta[idx : idx + hidden_dim] = cfg.weight_scale * rng.standard_normal(hidden_dim)
    idx += hidden_dim
    theta[idx : idx + hidden_dim] = cfg.weight_scale * rng.standard_normal(hidden_dim)
    idx += hidden_dim
    theta[idx : idx + hidden_dim] = cfg.weight_scale * rng.standard_normal(hidden_dim)
    return theta


def unpack_theta(theta, hidden_dim):
    idx = 0
    log_a = theta[idx]
    idx += 1
    w1 = theta[idx : idx + hidden_dim]
    idx += hidden_dim
    b1 = theta[idx : idx + hidden_dim]
    idx += hidden_dim
    w2 = theta[idx : idx + hidden_dim]
    return log_a, w1, b1, w2


def log_prob_scalar(x, theta, hidden_dim):
    log_a, w1, b1, w2 = unpack_theta(theta, hidden_dim)
    exp_a = math.exp(log_a)
    z = w1 * x + b1
    h = np.tanh(z)
    log_psi = -0.5 * exp_a * x * x + np.dot(w2, h)
    return 2.0 * log_psi


def sample_configs(theta, cfg):
    samples = np.empty(cfg.G, dtype=np.float64)
    x_curr = 0.0
    logp_curr = log_prob_scalar(x_curr, theta, cfg.hidden_dim)

    total_steps = cfg.n_burn + cfg.G
    sample_idx = 0
    for step in range(total_steps):
        x_prop = x_curr + cfg.mh_step_size * np.random.randn()
        logp_prop = log_prob_scalar(x_prop, theta, cfg.hidden_dim)
        if np.log(np.random.rand()) < (logp_prop - logp_curr):
            x_curr = x_prop
            logp_curr = logp_prop

        if step >= cfg.n_burn:
            samples[sample_idx] = x_curr
            sample_idx += 1

    return samples


def batch_stats(theta, x, hidden_dim):
    log_a, w1, b1, w2 = unpack_theta(theta, hidden_dim)
    exp_a = np.exp(log_a)

    z = x[:, None] * w1[None, :] + b1[None, :]
    h = np.tanh(z)
    sech2 = 1.0 - h**2

    log_psi = -0.5 * exp_a * x**2 + h @ w2
    log_prob = 2.0 * log_psi

    dlogpsi_dx = -exp_a * x + np.sum(sech2 * (w1 * w2)[None, :], axis=1)
    d2logpsi_dx2 = -exp_a + np.sum(
        -2.0 * h * sech2 * (w1**2 * w2)[None, :], axis=1
    )
    local_energies = -0.5 * (d2logpsi_dx2 + dlogpsi_dx**2) + 0.5 * x**2

    o_log_a = -0.5 * exp_a * x**2
    o_w1 = x[:, None] * sech2 * w2[None, :]
    o_b1 = sech2 * w2[None, :]
    o_w2 = h
    scores = np.concatenate(
        [o_log_a[:, None], o_w1, o_b1, o_w2],
        axis=1,
    )

    g1 = w2[None, :] * sech2
    a0_aug = np.stack([x, np.ones_like(x)], axis=1)

    return {
        "log_prob": log_prob,
        "local_energies": local_energies,
        "scores": scores,
        "a0_aug": a0_aug,
        "g1": g1,
        "h": h,
    }


def energy_gradient(local_energies, scores):
    centered_e = local_energies - np.mean(local_energies)
    return 2.0 * np.mean(centered_e[:, None] * scores, axis=0)


def exact_sr_preconditioner(scores, reg):
    centered_scores = scores - np.mean(scores, axis=0, keepdims=True)
    fisher = centered_scores.T @ centered_scores / scores.shape[0]
    fisher.flat[:: fisher.shape[0] + 1] += reg
    return fisher


def kfac_precondition(grad, stats, cfg):
    hidden_dim = cfg.hidden_dim
    reg = cfg.kfac_reg

    grad_log_a = grad[0]
    grad_w1 = grad[1 : 1 + hidden_dim]
    grad_b1 = grad[1 + hidden_dim : 1 + 2 * hidden_dim]
    grad_w2 = grad[1 + 2 * hidden_dim :]

    grad_layer1 = np.stack([grad_w1, grad_b1], axis=1)

    a0 = stats["a0_aug"]
    g1 = stats["g1"]
    h = stats["h"]

    a0_factor = (a0.T @ a0) / a0.shape[0]
    g1_factor = (g1.T @ g1) / g1.shape[0]
    h_factor = (h.T @ h) / h.shape[0]

    a0_factor.flat[:: a0_factor.shape[0] + 1] += reg
    g1_factor.flat[:: g1_factor.shape[0] + 1] += reg
    h_factor.flat[:: h_factor.shape[0] + 1] += reg

    nat_layer1 = np.linalg.solve(g1_factor, grad_layer1)
    nat_layer1 = np.linalg.solve(a0_factor.T, nat_layer1.T).T
    nat_w2 = np.linalg.solve(h_factor, grad_w2)

    precond_grad = np.concatenate(
        [
            np.array([grad_log_a / (np.var(stats["scores"][:, 0]) + reg)]),
            nat_layer1[:, 0],
            nat_layer1[:, 1],
            nat_w2,
        ]
    )
    return precond_grad


def grpo_objective_and_grad(theta, theta_old, x, advantages, cfg, log_prob_old):
    stats = batch_stats(theta, x, cfg.hidden_dim)
    log_ratio = stats["log_prob"] - log_prob_old
    ratio = np.exp(np.clip(log_ratio, -60.0, 60.0))
    clipped_ratio = np.clip(ratio, 1.0 - cfg.eps_clip, 1.0 + cfg.eps_clip)

    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages
    use_unclipped = unclipped <= clipped

    grad_weights = np.where(use_unclipped, advantages * ratio, 0.0)
    grad = np.mean(2.0 * grad_weights[:, None] * stats["scores"], axis=0)
    frac_clipped = np.mean(~use_unclipped)
    objective = np.mean(np.minimum(unclipped, clipped))
    return objective, grad, frac_clipped


def make_result_dict(energies, grad_steps, el_evals, wall_times, thetas, extras=None):
    result = {
        "energies": np.asarray(energies),
        "grad_steps": np.asarray(grad_steps),
        "el_evals": np.asarray(el_evals),
        "wall_times": np.asarray(wall_times),
        "thetas": np.asarray(thetas),
    }
    if extras:
        result.update(extras)
    return result


def run_sgd(cfg):
    np.random.seed(cfg.seed)
    theta = init_theta(cfg)

    energies = []
    grad_steps = []
    el_evals = []
    wall_times = []
    thetas = []
    cumulative_time = 0.0
    total_grad_steps = 0
    total_el_evals = 0

    for _ in range(cfg.n_iterations):
        t0 = time.perf_counter()
        x = sample_configs(theta, cfg)
        stats = batch_stats(theta, x, cfg.hidden_dim)
        grad = energy_gradient(stats["local_energies"], stats["scores"])
        theta -= cfg.eta_sgd * grad

        cumulative_time += time.perf_counter() - t0
        total_grad_steps += 1
        total_el_evals += cfg.G

        energies.append(np.mean(stats["local_energies"]))
        grad_steps.append(total_grad_steps)
        el_evals.append(total_el_evals)
        wall_times.append(cumulative_time)
        thetas.append(theta.copy())

    return make_result_dict(energies, grad_steps, el_evals, wall_times, thetas)


def run_sr_exact(cfg):
    np.random.seed(cfg.seed)
    theta = init_theta(cfg)

    energies = []
    grad_steps = []
    el_evals = []
    wall_times = []
    thetas = []
    cumulative_time = 0.0
    total_grad_steps = 0
    total_el_evals = 0

    for _ in range(cfg.n_iterations):
        t0 = time.perf_counter()
        x = sample_configs(theta, cfg)
        stats = batch_stats(theta, x, cfg.hidden_dim)
        grad = energy_gradient(stats["local_energies"], stats["scores"])
        fisher = exact_sr_preconditioner(stats["scores"], cfg.sr_reg)
        delta = np.linalg.solve(fisher, grad)
        theta -= cfg.eta_sr * delta

        cumulative_time += time.perf_counter() - t0
        total_grad_steps += 1
        total_el_evals += cfg.G

        energies.append(np.mean(stats["local_energies"]))
        grad_steps.append(total_grad_steps)
        el_evals.append(total_el_evals)
        wall_times.append(cumulative_time)
        thetas.append(theta.copy())

    return make_result_dict(energies, grad_steps, el_evals, wall_times, thetas)


def run_sr_kfac(cfg):
    np.random.seed(cfg.seed)
    theta = init_theta(cfg)

    energies = []
    grad_steps = []
    el_evals = []
    wall_times = []
    thetas = []
    cumulative_time = 0.0
    total_grad_steps = 0
    total_el_evals = 0

    for _ in range(cfg.n_iterations):
        t0 = time.perf_counter()
        x = sample_configs(theta, cfg)
        stats = batch_stats(theta, x, cfg.hidden_dim)
        grad = energy_gradient(stats["local_energies"], stats["scores"])
        nat_grad = kfac_precondition(grad, stats, cfg)
        theta -= cfg.eta_kfac * nat_grad

        cumulative_time += time.perf_counter() - t0
        total_grad_steps += 1
        total_el_evals += cfg.G

        energies.append(np.mean(stats["local_energies"]))
        grad_steps.append(total_grad_steps)
        el_evals.append(total_el_evals)
        wall_times.append(cumulative_time)
        thetas.append(theta.copy())

    return make_result_dict(energies, grad_steps, el_evals, wall_times, thetas)


def run_grpo(cfg):
    np.random.seed(cfg.seed)
    theta = init_theta(cfg)
    eta_inner = cfg.eta_grpo_inner

    energies = []
    grad_steps = []
    el_evals = []
    wall_times = []
    thetas = []
    cumulative_time = 0.0
    total_grad_steps = 0
    total_el_evals = 0
    frac_clipped_history = np.zeros((cfg.n_iterations, cfg.K_inner), dtype=np.float64)
    halving_events = []

    for outer_idx in range(cfg.n_iterations):
        t0 = time.perf_counter()
        theta_old = theta.copy()
        x = sample_configs(theta_old, cfg)
        old_stats = batch_stats(theta_old, x, cfg.hidden_dim)
        local_energies = old_stats["local_energies"]
        advantages = -(
            (local_energies - np.mean(local_energies))
            / (np.std(local_energies) + 1e-8)
        )

        for inner_idx in range(cfg.K_inner):
            _, grad, frac_clipped = grpo_objective_and_grad(
                theta=theta,
                theta_old=theta_old,
                x=x,
                advantages=advantages,
                cfg=cfg,
                log_prob_old=old_stats["log_prob"],
            )
            frac_clipped_history[outer_idx, inner_idx] = frac_clipped

            if (
                inner_idx + 1 == cfg.clip_guard_step
                and frac_clipped > cfg.clip_guard_threshold
            ):
                eta_inner *= 0.5
                halving_events.append(
                    {
                        "outer_iteration": outer_idx + 1,
                        "inner_step": inner_idx + 1,
                        "frac_clipped": float(frac_clipped),
                        "new_eta_inner": float(eta_inner),
                    }
                )

            theta += eta_inner * grad
            total_grad_steps += 1

        cumulative_time += time.perf_counter() - t0
        total_el_evals += cfg.G

        energies.append(np.mean(local_energies))
        grad_steps.append(total_grad_steps)
        el_evals.append(total_el_evals)
        wall_times.append(cumulative_time)
        thetas.append(theta.copy())

    extras = {
        "mean_frac_clipped": np.mean(frac_clipped_history, axis=0),
        "max_frac_clipped": np.max(frac_clipped_history, axis=0),
        "final_eta_inner": float(eta_inner),
        "num_eta_halvings": len(halving_events),
        "halving_events": halving_events,
    }
    return make_result_dict(
        energies, grad_steps, el_evals, wall_times, thetas, extras=extras
    )


def first_hit(result, threshold):
    hits = np.flatnonzero(np.asarray(result["energies"]) < threshold)
    if hits.size == 0:
        return "n/a", "n/a", "n/a"

    idx = int(hits[0])
    return (
        str(int(result["grad_steps"][idx])),
        str(int(result["el_evals"][idx])),
        f"{result['wall_times'][idx]:.3f}",
    )


def print_summary(results, cfg):
    print(
        "Optimizer | Final E  | Grad steps to E<thr | E_L evals to E<thr | Wall sec to E<thr\n"
        "----------|----------|---------------------|--------------------|------------------"
    )
    for name, result in results.items():
        grad_hit, eval_hit, wall_hit = first_hit(result, cfg.threshold_energy)
        print(
            f"{name:<10}| {result['energies'][-1]:>8.5f} |"
            f" {grad_hit:>19} | {eval_hit:>18} | {wall_hit:>16}"
        )


def print_grpo_diagnostics(result, cfg):
    print("\nGRPO clip diagnostics:")
    for inner_step, mean_clip in enumerate(result["mean_frac_clipped"], start=1):
        max_clip = result["max_frac_clipped"][inner_step - 1]
        print(
            f"  frac_clipped[{inner_step}] mean={mean_clip:.3f}, max={max_clip:.3f}"
        )
    print(
        "  eta_inner:"
        f" start={cfg.eta_grpo_inner:.5f},"
        f" final={result['final_eta_inner']:.5f},"
        f" halvings={result['num_eta_halvings']}"
    )


def make_plots(results, cfg, output_path="neural_qho_vmc_benchmark.png"):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    colors = {
        "SGD": "tab:blue",
        "SR-exact": "tab:orange",
        "SR-KFAC": "tab:red",
        "GRPO-VMC": "tab:green",
    }

    for name, result in results.items():
        smoothed = rolling_mean(result["energies"], window=10)
        axes[0].plot(result["wall_times"], smoothed, label=name, color=colors[name], lw=2)
        axes[1].plot(result["grad_steps"], smoothed, label=name, color=colors[name], lw=2)
        axes[2].plot(result["el_evals"], smoothed, label=name, color=colors[name], lw=2)

    for ax in axes:
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1.25, label="E0 = 0.5")
        ax.set_ylabel(r"$\langle E_L \rangle$")
        ax.grid(alpha=0.25)

    axes[0].set_xlabel("Cumulative wall-clock time (s)")
    axes[0].set_title("Neural QHO: Energy vs. Wall Time")

    axes[1].set_xlabel("Cumulative gradient steps")
    axes[1].set_title("Neural QHO: Energy vs. Gradient Steps")

    axes[2].set_xlabel(r"Cumulative $E_L$ evaluations")
    axes[2].set_title(r"Neural QHO: Energy vs. $E_L$ Evaluations")

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    for ax in axes:
        ax.legend(unique.values(), unique.keys(), frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")


def run_width_sweep(cfg):
    rows = []
    for hidden_dim in cfg.width_sweep_hidden_dims:
        sweep_cfg = NeuralBenchmarkConfig(
            n_iterations=cfg.width_sweep_iterations,
            G=cfg.G,
            n_burn=cfg.n_burn,
            mh_step_size=cfg.mh_step_size,
            hidden_dim=hidden_dim,
            log_a_init=cfg.log_a_init,
            weight_scale=cfg.weight_scale,
            eta_sgd=cfg.eta_sgd,
            eta_sr=cfg.eta_sr,
            eta_kfac=cfg.eta_kfac,
            eta_grpo_inner=cfg.eta_grpo_inner,
            K_inner=cfg.K_inner,
            eps_clip=cfg.eps_clip,
            sr_reg=cfg.sr_reg,
            kfac_reg=cfg.kfac_reg,
            threshold_energy=cfg.threshold_energy,
            clip_guard_threshold=cfg.clip_guard_threshold,
            clip_guard_step=cfg.clip_guard_step,
            seed=cfg.seed,
            width_sweep_hidden_dims=cfg.width_sweep_hidden_dims,
            width_sweep_iterations=cfg.width_sweep_iterations,
        )

        sr_kfac = run_sr_kfac(sweep_cfg)
        grpo = run_grpo(sweep_cfg)
        sr_grad, sr_eval, sr_wall = first_hit(sr_kfac, sweep_cfg.threshold_energy)
        grpo_grad, grpo_eval, grpo_wall = first_hit(grpo, sweep_cfg.threshold_energy)

        rows.append(
            {
                "hidden_dim": hidden_dim,
                "num_params": 1 + 3 * hidden_dim,
                "sr_kfac_final": float(sr_kfac["energies"][-1]),
                "sr_kfac_grad_hit": sr_grad,
                "sr_kfac_eval_hit": sr_eval,
                "sr_kfac_wall_hit": sr_wall,
                "grpo_final": float(grpo["energies"][-1]),
                "grpo_grad_hit": grpo_grad,
                "grpo_eval_hit": grpo_eval,
                "grpo_wall_hit": grpo_wall,
            }
        )
    return rows


def print_width_sweep(rows):
    print(
        "\nWidth sweep: SR-KFAC vs GRPO-VMC wall-clock crossover\n"
        "H | P | SR-KFAC final E | SR-KFAC wall sec to E<thr | GRPO final E | GRPO wall sec to E<thr\n"
        "--|---|-----------------|--------------------------|--------------|------------------------"
    )
    for row in rows:
        print(
            f"{row['hidden_dim']:>2} | {row['num_params']:>3} |"
            f" {row['sr_kfac_final']:>15.5f} | {row['sr_kfac_wall_hit']:>24} |"
            f" {row['grpo_final']:>12.5f} | {row['grpo_wall_hit']:>22}"
        )


def main():
    cfg = NeuralBenchmarkConfig()

    results = {
        "SGD": run_sgd(cfg),
        "SR-exact": run_sr_exact(cfg),
        "SR-KFAC": run_sr_kfac(cfg),
        "GRPO-VMC": run_grpo(cfg),
    }

    make_plots(results, cfg)
    width_rows = run_width_sweep(cfg)
    print(
        f"Neural ansatz benchmark for 1D QHO with hidden_dim={cfg.hidden_dim},"
        f" P={1 + 3 * cfg.hidden_dim}, G={cfg.G}, threshold={cfg.threshold_energy:.2f}"
    )
    print_summary(results, cfg)
    print_grpo_diagnostics(results["GRPO-VMC"], cfg)
    print_width_sweep(width_rows)
    print("\nSaved plot to neural_qho_vmc_benchmark.png")


if __name__ == "__main__":
    main()
