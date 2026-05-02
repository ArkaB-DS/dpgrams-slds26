# experiment_gauss_vs_order4_common_silverman.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import seaborn as sns
import time
import sys
import os
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main_scripts.dp_grams import dp_grams
from main_scripts.dp_grams_beta import dp_grams_beta
from main_scripts.merge import merge_modes
from main_scripts.mode_matching_mse import mode_matching_mse


sns.set_context("talk")
sns.set_style("white")
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
    }
)

# ---------------------------------------------------------------------
# Explicit experiment-wide constants
# ---------------------------------------------------------------------
# These are passed explicitly to both Gaussian DP-GRAMS and Order-4
# DP-GRAMS-beta, so the comparison never silently inherits defaults from
# main_scripts/dp_grams.py or main_scripts/dp_grams_beta.py.
DEFAULT_KAPPA_INIT = 5.0
DEFAULT_INIT_EPSILON_FRAC = 0.1
DEFAULT_BETA_DAP = 3.0
DEFAULT_C_RHO = 2.0
DEFAULT_DAP_SCORE_MULTIPLIER = 3.0
DEFAULT_R = None

DEFAULT_GAUSS_CLIP_MULT = 1.0
DEFAULT_ORDER4_CLIP_MULT = 1.0
DEFAULT_ORDER4_ETA_MULT = 1.0
DEFAULT_ORDER4_P_FLOOR_MULTIPLIER = 1.0


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def silverman_scalar_bandwidth(X):
    """
    Common scalar Silverman bandwidth used for BOTH dp_grams and dp_grams_beta.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    std = np.std(X, axis=0, ddof=1)
    iqr = (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)) / 1.349
    robust_scale = np.minimum(std, iqr)

    sigma = np.mean(
        np.where(
            np.isfinite(robust_scale) & (robust_scale > 1e-12),
            robust_scale,
            std,
        )
    )
    if not np.isfinite(sigma) or sigma <= 1e-12:
        sigma = float(np.mean(std))

    const = (4.0 / (d + 2.0)) ** (1.0 / (d + 4.0))
    h = const * (n ** (-1.0 / (d + 4.0))) * sigma
    return float(max(1e-6, h))


def generate_4corners(n_samples, seed=None):
    rng = np.random.default_rng(seed)
    means = np.array([[3, 3], [3, -3], [-3, 3], [-3, -3]], dtype=float)
    cov = np.eye(2)

    base = n_samples // 4
    rem = n_samples % 4
    counts = [base] * 4
    for j in range(rem):
        counts[j] += 1

    pts = [rng.multivariate_normal(means[j], cov, counts[j]) for j in range(4)]
    return np.vstack(pts), means


def sample_oracle_initial_modes_4corners(k, seed=None):
    rng = np.random.default_rng(seed)
    means = np.array([[3, 3], [3, -3], [-3, 3], [-3, -3]], dtype=float)
    comp = rng.integers(0, 4, size=int(k))
    return means[comp] + rng.normal(size=(int(k), 2))


def _run_dp_many(
    method_name,
    X,
    true_modes,
    eps,
    delta,
    beta,
    n_runs,
    h_common,
    run_seeds,
    shared_initial_modes_list,
    gauss_clip_mult=DEFAULT_GAUSS_CLIP_MULT,
    order4_clip_mult=DEFAULT_ORDER4_CLIP_MULT,
    order4_eta_mult=DEFAULT_ORDER4_ETA_MULT,
    order4_p_floor_multiplier=DEFAULT_ORDER4_P_FLOOR_MULTIPLIER,
    init_epsilon_frac=DEFAULT_INIT_EPSILON_FRAC,
    beta_dap=DEFAULT_BETA_DAP,
    c_rho=DEFAULT_C_RHO,
    dap_score_multiplier=DEFAULT_DAP_SCORE_MULTIPLIER,
    R=DEFAULT_R,
):
    mses = []
    times = []
    h_used_vals = []
    floor_active_rates = []

    n, _ = X.shape
    T = int(np.ceil(np.log(max(3, n))))
    m = None  # use each method's default minibatch rule

    for r in range(n_runs):
        seed = run_seeds[r]
        rng = np.random.default_rng(seed)
        init_modes = shared_initial_modes_list[r].copy()
        t0 = time.perf_counter()

        if method_name == "gaussian":
            h_used, raw = dp_grams(
                X=X,
                epsilon=eps,
                delta=delta,
                initial_modes=init_modes,
                T=T,
                m=m,
                h=h_common,  # same common Silverman h
                rng=rng,
                R=R,
                clip_multiplier=gauss_clip_mult,
                init_epsilon_frac=init_epsilon_frac,
                beta_dap=beta_dap,
                c_rho=c_rho,
                dap_score_multiplier=dap_score_multiplier,
            )
            est = merge_modes(raw, bandwidth=h_common, k=1)
            rt = time.perf_counter() - t0
            mse = mode_matching_mse(true_modes, est)

            mses.append(mse)
            times.append(rt)
            h_used_vals.append(h_used)

        elif method_name == "order4":
            h_used, raw, diag = dp_grams_beta(
                X=X,
                epsilon=eps,
                delta=delta,
                beta=beta,
                initial_modes=init_modes,
                T=T,
                m=m,
                h=h_common,  # same common Silverman h
                rng=rng,
                R=R,
                clip_multiplier=order4_clip_mult,
                init_epsilon_frac=init_epsilon_frac,
                beta_dap=beta_dap,
                c_rho=c_rho,
                dap_score_multiplier=dap_score_multiplier,
                eta_multiplier=order4_eta_mult,
                p_floor_multiplier=order4_p_floor_multiplier,
                return_diagnostics=True,
            )
            est = merge_modes(raw, bandwidth=h_common, k=1)
            rt = time.perf_counter() - t0
            mse = mode_matching_mse(true_modes, est)

            mses.append(mse)
            times.append(rt)
            h_used_vals.append(h_used)
            floor_active_rates.append(diag.get("floor_active_rate", np.nan))

        else:
            raise ValueError("method_name must be 'gaussian' or 'order4'")

    mses = np.asarray(mses, dtype=float)
    times = np.asarray(times, dtype=float)
    h_used_vals = np.asarray(h_used_vals, dtype=float)

    mse_mean = float(np.nanmean(mses))
    mse_sd = float(np.nanstd(mses, ddof=1)) if len(mses) > 1 else 0.0
    t_mean = float(np.nanmean(times))
    t_sd = float(np.nanstd(times, ddof=1)) if len(times) > 1 else 0.0
    h_mean = float(np.nanmean(h_used_vals))

    diag_out = None
    if method_name == "order4":
        diag_out = {
            "floor_active_rate_mean": float(np.nanmean(floor_active_rates))
            if floor_active_rates
            else np.nan,
        }

    return mse_mean, mse_sd, t_mean, t_sd, h_mean, diag_out


def run_experiment(
    n_list,
    eps_list,
    delta=1e-6,
    beta=4,
    n_runs=20,
    kappa_init=DEFAULT_KAPPA_INIT,
    gauss_clip_mult=DEFAULT_GAUSS_CLIP_MULT,
    order4_clip_mult=DEFAULT_ORDER4_CLIP_MULT,
    order4_eta_mult=DEFAULT_ORDER4_ETA_MULT,
    order4_p_floor_multiplier=DEFAULT_ORDER4_P_FLOOR_MULTIPLIER,
    init_epsilon_frac=DEFAULT_INIT_EPSILON_FRAC,
    beta_dap=DEFAULT_BETA_DAP,
    c_rho=DEFAULT_C_RHO,
    dap_score_multiplier=DEFAULT_DAP_SCORE_MULTIPLIER,
    R=DEFAULT_R,
    base_seed=123,
    results_dir="results/bivariate_4mix_kernel_order4_dp_common_silverman_same_inits",
):
    ensure_dir(results_dir)

    print("\n[config] Gaussian vs Order-4 comparison")
    print(f"[config] kappa_init={kappa_init}, init_epsilon_frac={init_epsilon_frac}")
    print(f"[config] beta_dap={beta_dap}, c_rho={c_rho}, dap_score_multiplier={dap_score_multiplier}, R={R}")
    print(f"[config] gauss_clip_mult={gauss_clip_mult}, order4_clip_mult={order4_clip_mult}")
    print(f"[config] order4_eta_mult={order4_eta_mult}, order4_p_floor_multiplier={order4_p_floor_multiplier}\n")

    out_csv = os.path.join(
        results_dir,
        "dp_kernel_compare_gauss_vs_order4_common_silverman_same_inits.csv",
    )
    out_pdf = os.path.join(
        results_dir,
        "dp_kernel_compare_2x2_eps_grid_common_silverman_same_inits.pdf",
    )

    series = {
        eps: {"n": [], "g_mean": [], "g_sd": [], "o_mean": [], "o_sd": []}
        for eps in eps_list
    }

    rows = []
    for eps_idx, eps in enumerate(eps_list):
        for idx, n in enumerate(n_list):
            X, true_modes = generate_4corners(n, seed=base_seed + idx)

            # ---------------------------------------------------------
            # Common Silverman bandwidth ONCE on the data.
            # Use the SAME h for Gaussian and Order-4.
            # ---------------------------------------------------------
            h_common = silverman_scalar_bandwidth(X)

            k_init = max(1, int(np.ceil(kappa_init * np.log(max(3, n)))))

            run_seeds = [
                base_seed + 1_000_000 * eps_idx + 10_000 * idx + r
                for r in range(n_runs)
            ]

            # ---------------------------------------------------------
            # Same initializations for BOTH methods.
            # ---------------------------------------------------------
            shared_initial_modes_list = [
                sample_oracle_initial_modes_4corners(
                    k_init,
                    seed=run_seeds[r] + 777_777,
                )
                for r in range(n_runs)
            ]

            g_mean, g_sd, gt_mean, gt_sd, g_h_mean, _ = _run_dp_many(
                method_name="gaussian",
                X=X,
                true_modes=true_modes,
                eps=eps,
                delta=delta,
                beta=beta,
                n_runs=n_runs,
                h_common=h_common,
                run_seeds=run_seeds,
                shared_initial_modes_list=shared_initial_modes_list,
                gauss_clip_mult=gauss_clip_mult,
                order4_clip_mult=order4_clip_mult,
                order4_eta_mult=order4_eta_mult,
                order4_p_floor_multiplier=order4_p_floor_multiplier,
                init_epsilon_frac=init_epsilon_frac,
                beta_dap=beta_dap,
                c_rho=c_rho,
                dap_score_multiplier=dap_score_multiplier,
                R=R,
            )

            o_mean, o_sd, ot_mean, ot_sd, o_h_mean, o_diag = _run_dp_many(
                method_name="order4",
                X=X,
                true_modes=true_modes,
                eps=eps,
                delta=delta,
                beta=beta,
                n_runs=n_runs,
                h_common=h_common,
                run_seeds=run_seeds,
                shared_initial_modes_list=shared_initial_modes_list,
                gauss_clip_mult=gauss_clip_mult,
                order4_clip_mult=order4_clip_mult,
                order4_eta_mult=order4_eta_mult,
                order4_p_floor_multiplier=order4_p_floor_multiplier,
                init_epsilon_frac=init_epsilon_frac,
                beta_dap=beta_dap,
                c_rho=c_rho,
                dap_score_multiplier=dap_score_multiplier,
                R=R,
            )

            series[eps]["n"].append(n)
            series[eps]["g_mean"].append(g_mean)
            series[eps]["g_sd"].append(g_sd)
            series[eps]["o_mean"].append(o_mean)
            series[eps]["o_sd"].append(o_sd)

            rows.append(
                [
                    n,
                    eps,
                    delta,
                    beta,
                    n_runs,
                    kappa_init,
                    k_init,
                    h_common,
                    gauss_clip_mult,
                    order4_clip_mult,
                    order4_eta_mult,
                    order4_p_floor_multiplier,
                    init_epsilon_frac,
                    beta_dap,
                    c_rho,
                    dap_score_multiplier,
                    R if R is not None else "None",
                    g_h_mean,
                    o_h_mean,
                    g_mean,
                    g_sd,
                    gt_mean,
                    gt_sd,
                    o_mean,
                    o_sd,
                    ot_mean,
                    ot_sd,
                    o_diag["floor_active_rate_mean"] if o_diag else np.nan,
                ]
            )

            print(
                f"[eps={eps:.3g}][n={n}] h_common={h_common:.4g}, k_init={k_init}, "
                f"init_frac={init_epsilon_frac:g}, c_rho={c_rho:g}, "
                f"score_mult={dap_score_multiplier:g} :: "
                f"Gauss MSE={g_mean:.4e} (SD={g_sd:.2e}) | "
                f"Order4 MSE={o_mean:.4e} (SD={o_sd:.2e}) | "
                f"same_inits=True"
            )

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "n",
                "epsilon",
                "delta",
                "beta",
                "n_runs",
                "kappa_init",
                "k_init",
                "h_common_silverman",
                "gauss_clip_mult",
                "order4_clip_mult",
                "order4_eta_mult",
                "order4_p_floor_multiplier",
                "init_epsilon_frac",
                "beta_dap",
                "c_rho",
                "dap_score_multiplier",
                "R",
                "gauss_h_used_mean",
                "order4_h_used_mean",
                "gauss_mse_mean",
                "gauss_mse_sd",
                "gauss_time_mean",
                "gauss_time_sd",
                "order4_mse_mean",
                "order4_mse_sd",
                "order4_time_mean",
                "order4_time_sd",
                "order4_floor_active_rate_mean",
            ]
        )
        for r in rows:
            w.writerow(r)

    print(f"[saved] CSV -> {out_csv}")

    _plot_2x2(series, eps_list, out_pdf, n_list)

    print(f"[saved] Plot -> {out_pdf}")
    return out_csv, out_pdf


def _plot_2x2(series, eps_list, out_pdf, n_list):
    ensure_dir(os.path.dirname(out_pdf))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    g_kwargs = dict(
        marker="o",
        linestyle="-",
        linewidth=2.0,
        markersize=6,
        capsize=4,
        elinewidth=1.3,
        capthick=1.1,
    )
    o_kwargs = dict(
        marker="s",
        linestyle="-",
        linewidth=2.0,
        markersize=6,
        capsize=4,
        elinewidth=1.3,
        capthick=1.1,
    )

    for k, eps in enumerate(eps_list):
        ax = axes[k]
        n_vals = np.asarray(series[eps]["n"], dtype=float)

        g_mean = np.asarray(series[eps]["g_mean"], dtype=float)
        g_sd = np.asarray(series[eps]["g_sd"], dtype=float)

        o_mean = np.asarray(series[eps]["o_mean"], dtype=float)
        o_sd = np.asarray(series[eps]["o_sd"], dtype=float)

        label_g = "Gaussian DP-GRAMS" if k == 0 else None
        label_o = r"Order-4 DP-GRAMS-$\beta$" if k == 0 else None

        ax.errorbar(n_vals, g_mean, yerr=g_sd, label=label_g, **g_kwargs)
        ax.errorbar(n_vals, o_mean, yerr=o_sd, label=label_o, **o_kwargs)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(min(n_list) * 0.9, max(n_list) * 1.1)
        ax.set_xticks(n_list, minor=False)
        ax.set_xticklabels([str(n) for n in n_list])
        ax.xaxis.set_minor_formatter(NullFormatter())

        ax.grid(True, which="both", alpha=0.35)
        ax.set_title(rf"$\varepsilon = {eps}$, $\delta = 10^{{-6}}$")

        if k in (2, 3):
            ax.set_xlabel("Sample size $n$")
        if k in (0, 2):
            ax.set_ylabel("MSE")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True)

    fig.suptitle(
        "Gaussian vs Order-4 DP-GRAMS on 4-Modal Bivariate Gaussian Mixture",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    fig.savefig(out_pdf, dpi=120)
    plt.show()
    plt.close(fig)


def main():
    n_list = [800, 1200, 1500, 2500]
    eps_list = [0.8, 1.0, 1.2, 1.5]

    run_experiment(
        n_list=n_list,
        eps_list=eps_list,
        delta=1e-6,
        beta=4,
        n_runs=20,
        kappa_init=DEFAULT_KAPPA_INIT,
        gauss_clip_mult=DEFAULT_GAUSS_CLIP_MULT,
        order4_clip_mult=DEFAULT_ORDER4_CLIP_MULT,
        order4_eta_mult=DEFAULT_ORDER4_ETA_MULT,
        order4_p_floor_multiplier=DEFAULT_ORDER4_P_FLOOR_MULTIPLIER,
        init_epsilon_frac=DEFAULT_INIT_EPSILON_FRAC,
        beta_dap=DEFAULT_BETA_DAP,
        c_rho=DEFAULT_C_RHO,
        dap_score_multiplier=DEFAULT_DAP_SCORE_MULTIPLIER,
        R=DEFAULT_R,
        base_seed=123,
        results_dir="results/bivariate_4mix_kernel_order4_dp_common_silverman_same_inits",
    )


if __name__ == "__main__":
    main()