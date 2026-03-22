import math
import time
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


HELIUM_GROUND_STATE_ENERGY = -2.9037243770341196


@dataclass
class HeliumBenchmarkConfig:
    n_iterations: int = 80
    G: int = 256
    n_burn: int = 200
    mh_step_size: float = 0.55
    hidden_dim: int = 32
    log_z_init: float = math.log(1.4)
    log_beta_init: float = math.log(0.6)
    weight_scale: float = 0.03
    eta_sgd: float = 0.0005
    eta_sr: float = 0.02
    eta_kfac: float = 0.015
    eta_grpo_inner: float = 0.0015
    K_inner: int = 5
    eps_clip: float = 0.2
    sr_reg: float = 1e-3
    kfac_reg: float = 5e-3
    threshold_energy: float = -2.90
    clip_guard_threshold: float = 0.4
    clip_guard_step: int = 2
    seed: int = 17
    width_sweep_hidden_dims: tuple[int, ...] = (16, 32, 64)
    width_sweep_iterations: int = 50
    coordinate_eps: float = 1e-8
    n_eval_samples: int = 12000
    eval_burn: int = 4000
    n_eval_repeats: int = 5


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

    theta = np.zeros(3 + 5 * hidden_dim, dtype=np.float64)
    theta[0] = cfg.log_z_init
    theta[1] = cfg.log_beta_init

    idx = 2
    theta[idx : idx + 3 * hidden_dim] = (
        cfg.weight_scale * rng.standard_normal(3 * hidden_dim)
    )
    idx += 3 * hidden_dim
    theta[idx : idx + hidden_dim] = cfg.weight_scale * rng.standard_normal(hidden_dim)
    idx += hidden_dim
    theta[idx : idx + hidden_dim] = cfg.weight_scale * rng.standard_normal(hidden_dim)
    idx += hidden_dim
    theta[idx] = 0.0
    return theta


def unpack_theta(theta, hidden_dim):
    idx = 0
    log_z = theta[idx]
    idx += 1
    log_beta = theta[idx]
    idx += 1
    w1 = theta[idx : idx + 3 * hidden_dim].reshape(hidden_dim, 3)
    idx += 3 * hidden_dim
    b1 = theta[idx : idx + hidden_dim]
    idx += hidden_dim
    w2 = theta[idx : idx + hidden_dim]
    idx += hidden_dim
    b2 = theta[idx]
    return log_z, log_beta, w1, b1, w2, b2


def compute_features(configs, eps):
    r1_vec = configs[:, 0, :]
    r2_vec = configs[:, 1, :]
    r12_vec = r1_vec - r2_vec

    r1 = np.linalg.norm(r1_vec, axis=1)
    r2 = np.linalg.norm(r2_vec, axis=1)
    r12 = np.linalg.norm(r12_vec, axis=1)

    r1_safe = np.maximum(r1, eps)
    r2_safe = np.maximum(r2, eps)
    r12_safe = np.maximum(r12, eps)

    e1 = r1_vec / r1_safe[:, None]
    e2 = r2_vec / r2_safe[:, None]
    e12 = r12_vec / r12_safe[:, None]

    features = np.stack([r1 + r2, r1 * r2, r12], axis=1)

    return {
        "r1": r1,
        "r2": r2,
        "r12": r12,
        "r1_safe": r1_safe,
        "r2_safe": r2_safe,
        "r12_safe": r12_safe,
        "e1": e1,
        "e2": e2,
        "e12": e12,
        "features": features,
    }


def log_prob_scalar(config, theta, cfg):
    configs = np.asarray(config, dtype=np.float64)[None, :, :]
    feat = compute_features(configs, cfg.coordinate_eps)
    log_z, log_beta, w1, b1, w2, b2 = unpack_theta(theta, cfg.hidden_dim)
    zeta = math.exp(log_z)
    beta = math.exp(log_beta)

    phi = feat["features"][0]
    z = w1 @ phi + b1
    h = np.tanh(z)
    r12 = feat["r12"][0]

    jastrow = 0.5 * r12 / (1.0 + beta * r12)
    log_psi = -zeta * phi[0] + jastrow + np.dot(w2, h) + b2
    return 2.0 * log_psi


def sample_configs(
    theta,
    cfg,
    n_samples=None,
    n_burn=None,
    step_size=None,
    return_acceptance=False,
):
    n_samples = cfg.G if n_samples is None else n_samples
    n_burn = cfg.n_burn if n_burn is None else n_burn
    step_size = cfg.mh_step_size if step_size is None else step_size

    samples = np.empty((n_samples, 2, 3), dtype=np.float64)
    x_curr = 0.5 * np.random.randn(2, 3)
    logp_curr = log_prob_scalar(x_curr, theta, cfg)

    total_steps = n_burn + n_samples
    sample_idx = 0
    accepted = 0
    for step in range(total_steps):
        x_prop = x_curr + step_size * np.random.randn(2, 3)
        logp_prop = log_prob_scalar(x_prop, theta, cfg)
        if np.log(np.random.rand()) < (logp_prop - logp_curr):
            x_curr = x_prop
            logp_curr = logp_prop
            accepted += 1

        if step >= n_burn:
            samples[sample_idx] = x_curr
            sample_idx += 1

    if return_acceptance:
        return samples, accepted / total_steps
    return samples


def batch_stats(theta, configs, cfg):
    log_z, log_beta, w1, b1, w2, b2 = unpack_theta(theta, cfg.hidden_dim)
    zeta = np.exp(log_z)
    beta = np.exp(log_beta)

    feat = compute_features(configs, cfg.coordinate_eps)
    phi = feat["features"]
    r1 = feat["r1"]
    r2 = feat["r2"]
    r12 = feat["r12"]
    r1_safe = feat["r1_safe"]
    r2_safe = feat["r2_safe"]
    r12_safe = feat["r12_safe"]
    e1 = feat["e1"]
    e2 = feat["e2"]
    e12 = feat["e12"]

    z = phi @ w1.T + b1[None, :]
    h = np.tanh(z)
    sech2 = 1.0 - h**2
    mlp_out = h @ w2 + b2

    jastrow = 0.5 * r12 / (1.0 + beta * r12)
    log_psi = -zeta * phi[:, 0] + jastrow + mlp_out
    log_prob = 2.0 * log_psi

    jastrow_d1 = 0.5 / (1.0 + beta * r12) ** 2
    jastrow_d2 = -beta / (1.0 + beta * r12) ** 3

    mlp_feature_grad = (sech2 * w2[None, :]) @ w1
    mlp_feature_hess = np.einsum(
        "gh,hij->gij",
        w2[None, :] * (-2.0 * h * sech2),
        np.einsum("hi,hj->hij", w1, w1),
    )

    feature_grad = mlp_feature_grad.copy()
    feature_grad[:, 0] -= zeta
    feature_grad[:, 2] += jastrow_d1

    feature_hess = mlp_feature_hess.copy()
    feature_hess[:, 2, 2] += jastrow_d2

    grad_phi_1 = np.stack(
        [e1, r2[:, None] * e1, e12],
        axis=1,
    )
    grad_phi_2 = np.stack(
        [e2, r1[:, None] * e2, -e12],
        axis=1,
    )
    lap_phi_1 = np.stack([2.0 / r1_safe, 2.0 * r2 / r1_safe, 2.0 / r12_safe], axis=1)
    lap_phi_2 = np.stack([2.0 / r2_safe, 2.0 * r1 / r2_safe, 2.0 / r12_safe], axis=1)

    grad_logpsi_1 = np.einsum("gi,gij->gj", feature_grad, grad_phi_1)
    grad_logpsi_2 = np.einsum("gi,gij->gj", feature_grad, grad_phi_2)

    gram_1 = np.einsum("gik,gjk->gij", grad_phi_1, grad_phi_1)
    gram_2 = np.einsum("gik,gjk->gij", grad_phi_2, grad_phi_2)

    lap_logpsi_1 = np.einsum("gi,gi->g", feature_grad, lap_phi_1) + np.einsum(
        "gij,gij->g", feature_hess, gram_1
    )
    lap_logpsi_2 = np.einsum("gi,gi->g", feature_grad, lap_phi_2) + np.einsum(
        "gij,gij->g", feature_hess, gram_2
    )

    potential = -2.0 / r1_safe - 2.0 / r2_safe + 1.0 / r12_safe
    local_energies = (
        -0.5
        * (
            lap_logpsi_1
            + np.sum(grad_logpsi_1**2, axis=1)
            + lap_logpsi_2
            + np.sum(grad_logpsi_2**2, axis=1)
        )
        + potential
    )

    score_log_z = -zeta * phi[:, 0]
    score_log_beta = -0.5 * beta * r12**2 / (1.0 + beta * r12) ** 2
    score_w1 = np.einsum("gh,gi->ghi", sech2 * w2[None, :], phi).reshape(
        configs.shape[0], -1
    )
    score_b1 = sech2 * w2[None, :]
    score_w2 = h
    score_b2 = np.ones((configs.shape[0], 1), dtype=np.float64)

    scores = np.concatenate(
        [
            score_log_z[:, None],
            score_log_beta[:, None],
            score_w1,
            score_b1,
            score_w2,
            score_b2,
        ],
        axis=1,
    )

    g1 = sech2 * w2[None, :]
    a0_aug = np.concatenate(
        [phi, np.ones((configs.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    h_aug = np.concatenate(
        [h, np.ones((configs.shape[0], 1), dtype=np.float64)],
        axis=1,
    )

    return {
        "log_prob": log_prob,
        "local_energies": local_energies,
        "scores": scores,
        "a0_aug": a0_aug,
        "g1": g1,
        "h_aug": h_aug,
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

    grad_env = grad[:2]

    idx = 2
    grad_w1 = grad[idx : idx + 3 * hidden_dim].reshape(hidden_dim, 3)
    idx += 3 * hidden_dim
    grad_b1 = grad[idx : idx + hidden_dim]
    idx += hidden_dim
    grad_w2 = grad[idx : idx + hidden_dim]
    idx += hidden_dim
    grad_b2 = grad[idx]

    grad_layer1 = np.concatenate([grad_w1, grad_b1[:, None]], axis=1)
    grad_out = np.concatenate([grad_w2, np.array([grad_b2])])

    a0_factor = (stats["a0_aug"].T @ stats["a0_aug"]) / stats["a0_aug"].shape[0]
    g1_factor = (stats["g1"].T @ stats["g1"]) / stats["g1"].shape[0]
    h_factor = (stats["h_aug"].T @ stats["h_aug"]) / stats["h_aug"].shape[0]

    a0_factor.flat[:: a0_factor.shape[0] + 1] += reg
    g1_factor.flat[:: g1_factor.shape[0] + 1] += reg
    h_factor.flat[:: h_factor.shape[0] + 1] += reg

    env_scores = stats["scores"][:, :2]
    env_scores = env_scores - np.mean(env_scores, axis=0, keepdims=True)
    env_fisher = env_scores.T @ env_scores / env_scores.shape[0]
    env_fisher.flat[:: env_fisher.shape[0] + 1] += reg

    nat_env = np.linalg.solve(env_fisher, grad_env)
    nat_layer1 = np.linalg.solve(g1_factor, grad_layer1)
    nat_layer1 = np.linalg.solve(a0_factor.T, nat_layer1.T).T
    nat_out = np.linalg.solve(h_factor, grad_out)

    return np.concatenate(
        [
            nat_env,
            nat_layer1[:, :3].reshape(-1),
            nat_layer1[:, 3],
            nat_out[:-1],
            np.array([nat_out[-1]]),
        ]
    )


def grpo_objective_and_grad(theta, theta_old, configs, advantages, cfg, log_prob_old):
    stats = batch_stats(theta, configs, cfg)
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
    acceptances = []
    cumulative_time = 0.0
    total_grad_steps = 0
    total_el_evals = 0

    for _ in range(cfg.n_iterations):
        t0 = time.perf_counter()
        configs, acceptance = sample_configs(theta, cfg, return_acceptance=True)
        stats = batch_stats(theta, configs, cfg)
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
        acceptances.append(acceptance)

    return make_result_dict(
        energies,
        grad_steps,
        el_evals,
        wall_times,
        thetas,
        extras={"acceptances": np.asarray(acceptances)},
    )


def run_sr_exact(cfg):
    np.random.seed(cfg.seed)
    theta = init_theta(cfg)

    energies = []
    grad_steps = []
    el_evals = []
    wall_times = []
    thetas = []
    acceptances = []
    cumulative_time = 0.0
    total_grad_steps = 0
    total_el_evals = 0

    for _ in range(cfg.n_iterations):
        t0 = time.perf_counter()
        configs, acceptance = sample_configs(theta, cfg, return_acceptance=True)
        stats = batch_stats(theta, configs, cfg)
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
        acceptances.append(acceptance)

    return make_result_dict(
        energies,
        grad_steps,
        el_evals,
        wall_times,
        thetas,
        extras={"acceptances": np.asarray(acceptances)},
    )


def run_sr_kfac(cfg):
    np.random.seed(cfg.seed)
    theta = init_theta(cfg)

    energies = []
    grad_steps = []
    el_evals = []
    wall_times = []
    thetas = []
    acceptances = []
    cumulative_time = 0.0
    total_grad_steps = 0
    total_el_evals = 0

    for _ in range(cfg.n_iterations):
        t0 = time.perf_counter()
        configs, acceptance = sample_configs(theta, cfg, return_acceptance=True)
        stats = batch_stats(theta, configs, cfg)
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
        acceptances.append(acceptance)

    return make_result_dict(
        energies,
        grad_steps,
        el_evals,
        wall_times,
        thetas,
        extras={"acceptances": np.asarray(acceptances)},
    )


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
    acceptances = []

    for outer_idx in range(cfg.n_iterations):
        t0 = time.perf_counter()
        theta_old = theta.copy()
        configs, acceptance = sample_configs(theta_old, cfg, return_acceptance=True)
        old_stats = batch_stats(theta_old, configs, cfg)
        local_energies = old_stats["local_energies"]
        advantages = -(
            (local_energies - np.mean(local_energies))
            / (np.std(local_energies) + 1e-8)
        )

        for inner_idx in range(cfg.K_inner):
            _, grad, frac_clipped = grpo_objective_and_grad(
                theta=theta,
                theta_old=theta_old,
                configs=configs,
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
        acceptances.append(acceptance)

    extras = {
        "mean_frac_clipped": np.mean(frac_clipped_history, axis=0),
        "max_frac_clipped": np.max(frac_clipped_history, axis=0),
        "final_eta_inner": float(eta_inner),
        "num_eta_halvings": len(halving_events),
        "halving_events": halving_events,
        "acceptances": np.asarray(acceptances),
    }
    return make_result_dict(
        energies, grad_steps, el_evals, wall_times, thetas, extras=extras
    )


def first_hit(result, threshold):
    smoothed = rolling_mean(result["energies"], window=10)
    hits = np.flatnonzero(smoothed < threshold)

    if hits.size == 0:
        return "n/a", "n/a", "n/a"

    idx = int(hits[0])
    return (
        str(int(result["grad_steps"][idx])),
        str(int(result["el_evals"][idx])),
        f"{result['wall_times'][idx]:.3f}",
    )


def evaluate_theta(theta, cfg):
    means = []
    ses = []
    acceptances = []

    for rep in range(cfg.n_eval_repeats):
        np.random.seed(cfg.seed + 1000 + rep)
        configs, acceptance = sample_configs(
            theta,
            cfg,
            n_samples=cfg.n_eval_samples,
            n_burn=cfg.eval_burn,
            return_acceptance=True,
        )
        energies = batch_stats(theta, configs, cfg)["local_energies"]
        means.append(float(np.mean(energies)))
        ses.append(float(np.std(energies, ddof=1) / np.sqrt(len(energies))))
        acceptances.append(float(acceptance))

    means = np.asarray(means)
    ses = np.asarray(ses)
    acceptances = np.asarray(acceptances)
    return {
        "mean_energy": float(np.mean(means)),
        "std_across_repeats": float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
        "mean_chain_se": float(np.mean(ses)),
        "mean_acceptance": float(np.mean(acceptances)),
        "repeat_means": means,
    }


def attach_final_evaluations(results, cfg):
    for result in results.values():
        result["final_eval"] = evaluate_theta(result["thetas"][-1], cfg)
    return results


def print_summary(results, cfg):
    print(
        "Optimizer | Train Final E | Reeval Final E | Grad steps to smoothed E<thr | E_L evals to smoothed E<thr | Wall sec to smoothed E<thr\n"
        "----------|---------------|----------------|------------------------------|-----------------------------|-----------------------------"
    )
    for name, result in results.items():
        grad_hit, eval_hit, wall_hit = first_hit(result, cfg.threshold_energy)
        final_eval = result["final_eval"]
        reeval = (
            f"{final_eval['mean_energy']:.5f} +/- {final_eval['std_across_repeats']:.5f}"
        )
        print(
            f"{name:<10}| {result['energies'][-1]:>13.5f} | {reeval:>14} |"
            f" {grad_hit:>19} | {eval_hit:>18} | {wall_hit:>16}"
        )

    print("\nIndependent final-state evaluation diagnostics:")
    for name, result in results.items():
        final_eval = result["final_eval"]
        mean_accept = float(np.mean(result["acceptances"]))
        print(
            f"  {name}: mean_acceptance={mean_accept:.3f},"
            f" final_eval_acceptance={final_eval['mean_acceptance']:.3f},"
            f" chain_SE~{final_eval['mean_chain_se']:.5f}"
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


def make_plots(results, cfg, output_path="neural_helium_vmc_benchmark.png"):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    colors = {
        "SGD": "tab:blue",
        "SR-exact": "tab:orange",
        "SR-KFAC": "tab:red",
        "GRPO-VMC": "tab:green",
    }

    for name, result in results.items():
        smoothed = rolling_mean(result["energies"], window=10)
        axes[0].plot(
            result["wall_times"], smoothed, label=name, color=colors[name], lw=2
        )
        axes[1].plot(
            result["grad_steps"], smoothed, label=name, color=colors[name], lw=2
        )
        axes[2].plot(
            result["el_evals"], smoothed, label=name, color=colors[name], lw=2
        )

    for ax in axes:
        ax.axhline(
            HELIUM_GROUND_STATE_ENERGY,
            color="black",
            linestyle="--",
            linewidth=1.25,
            label="Exact He ground state",
        )
        ax.set_ylabel(r"$\langle E_L \rangle$")
        ax.grid(alpha=0.25)

    axes[0].set_xlabel("Cumulative wall-clock time (s)")
    axes[0].set_title("Helium: Energy vs. Wall Time")

    axes[1].set_xlabel("Cumulative gradient steps")
    axes[1].set_title("Helium: Energy vs. Gradient Steps")

    axes[2].set_xlabel(r"Cumulative $E_L$ evaluations")
    axes[2].set_title(r"Helium: Energy vs. $E_L$ Evaluations")

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    for ax in axes:
        ax.legend(unique.values(), unique.keys(), frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")


def run_width_sweep(cfg):
    rows = []
    for hidden_dim in cfg.width_sweep_hidden_dims:
        sweep_cfg = HeliumBenchmarkConfig(
            n_iterations=cfg.width_sweep_iterations,
            G=cfg.G,
            n_burn=cfg.n_burn,
            mh_step_size=cfg.mh_step_size,
            hidden_dim=hidden_dim,
            log_z_init=cfg.log_z_init,
            log_beta_init=cfg.log_beta_init,
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
            coordinate_eps=cfg.coordinate_eps,
        )

        sr_kfac = run_sr_kfac(sweep_cfg)
        grpo = run_grpo(sweep_cfg)
        _, _, sr_wall = first_hit(sr_kfac, sweep_cfg.threshold_energy)
        _, _, grpo_wall = first_hit(grpo, sweep_cfg.threshold_energy)

        rows.append(
            {
                "hidden_dim": hidden_dim,
                "num_params": 3 + 5 * hidden_dim,
                "sr_kfac_final": float(sr_kfac["energies"][-1]),
                "sr_kfac_wall_hit": sr_wall,
                "grpo_final": float(grpo["energies"][-1]),
                "grpo_wall_hit": grpo_wall,
            }
        )
    return rows


def print_width_sweep(rows, cfg):
    print(
        "\nWidth sweep: SR-KFAC vs GRPO-VMC on helium\n"
        f"Threshold energy = {cfg.threshold_energy:.2f}\n"
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
    cfg = HeliumBenchmarkConfig()

    results = {
        "SGD": run_sgd(cfg),
        "SR-exact": run_sr_exact(cfg),
        "SR-KFAC": run_sr_kfac(cfg),
        "GRPO-VMC": run_grpo(cfg),
    }
    results = attach_final_evaluations(results, cfg)

    make_plots(results, cfg)
    width_rows = run_width_sweep(cfg)
    print(
        "Neural helium benchmark with two-electron 3D coordinates,"
        f" hidden_dim={cfg.hidden_dim}, P={3 + 5 * cfg.hidden_dim},"
        f" G={cfg.G}, threshold={cfg.threshold_energy:.2f}"
    )
    print_summary(results, cfg)
    print_grpo_diagnostics(results["GRPO-VMC"], cfg)
    print_width_sweep(width_rows, cfg)
    print("\nSaved plot to neural_helium_vmc_benchmark.png")


if __name__ == "__main__":
    main()
