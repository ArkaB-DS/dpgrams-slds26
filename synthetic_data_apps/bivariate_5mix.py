# bivariate_5mix.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
import os
import math
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main_scripts.ms import mean_shift
from main_scripts.dp_grams import dp_grams
from main_scripts.merge import merge_modes
from main_scripts.bandwidth import silverman_bandwidth
from main_scripts.mode_matching_mse import mode_matching_mse

# ---------------------------------------------------------------------
# Experiment-wide DP-GRAMS defaults
# ---------------------------------------------------------------------

DEFAULT_CLIP_MULTIPLIER = 0.5
DEFAULT_KAPPA_INIT = 6
DEFAULT_INIT_EPSILON_FRAC = 0.1
DEFAULT_ETA = 2

# ---------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------
sns.set_context("talk")
sns.set_style("white")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11
})

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_text(path, text):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(text)

# ---------------------------------------------------------------------
# Data generation: 5-component bivariate t mixture
# ---------------------------------------------------------------------

def generate_5mix(n_samples, seed=None):
    """
    Generate a 5-component bivariate t-mixture:
      - Means: (0,0), (6,0), (-6,0), (0,6), (0,-6)
      - df: [15, 6, 10, 8, 20]
      - equal weights 0.2
      - component-specific scales
    """
    rng = np.random.default_rng(seed)
    means = np.array([
        [0, 0],
        [6, 0],
        [-6, 0],
        [0, 6],
        [0, -6]
    ])
    dfs = [15, 6, 10, 8, 20]
    probs = np.array([0.2] * 5)
    scales = np.array([0.1, 0.9, 1.3, 1.0, 0.4])

    samples_per_mode = (probs * n_samples).astype(int)
    pts = []
    for m, df, n_pts, scale in zip(means, dfs, samples_per_mode, scales):
        t_samples = rng.standard_t(df, size=(n_pts, 2)) * scale + m
        pts.append(t_samples)

    return np.vstack(pts), means

def make_data(n_samples, seed):
    """Generate data, true modes, and Silverman bandwidth."""
    data, true_modes = generate_5mix(n_samples, seed=seed)
    h = silverman_bandwidth(data)
    return data, true_modes, h

# ---------------------------------------------------------------------
# Full-data MS init helper + DP-GRAMS DAP helper
# ---------------------------------------------------------------------

def run_ms_and_dp_with_full_data_ms_init(
    data,
    epsilon,
    delta,
    h,
    ms_T,
    dp_T=None,
    seed=0,
    clip_multiplier=DEFAULT_CLIP_MULTIPLIER,
    m=None,
    eta=DEFAULT_ETA,
    kappa_init=DEFAULT_KAPPA_INIT,
    init_epsilon_frac=DEFAULT_INIT_EPSILON_FRAC,
    suppression_radius=None,
):
    """
    Run DP-GRAMS with its private DAP initialization, and run non-private
    mean shift from the full dataset as the initial modes.

    MS is therefore independent of epsilon and does not use DAP anchors.

    Returns
    -------
    ms_est, dp_est, ms_time, dp_time, init_info, h_used
    """
    rng_dp = np.random.default_rng(seed)

    t0 = time.perf_counter()
    h_used, dp_raw, init_info = dp_grams(
        X=data,
        epsilon=epsilon,
        delta=delta,
        T=dp_T,
        h=h,
        rng=rng_dp,
        m=m,
        clip_multiplier=clip_multiplier,
        eta=eta,
        kappa_init=kappa_init,
        init_epsilon_frac=init_epsilon_frac,
        suppression_radius=suppression_radius,
        return_init_info=True,
    )
    dp_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    ms_raw = mean_shift(
        data,
        initial_modes=data.copy(),
        T=ms_T,
        bandwidth=h,
    )
    ms_time = time.perf_counter() - t0

    ms_est = merge_modes(ms_raw)
    dp_est = merge_modes(dp_raw)

    return ms_est, dp_est, ms_time, dp_time, init_info, h_used


# ---------------------------------------------------------------------
# Visualization: DAP initialization
# ---------------------------------------------------------------------

def plot_dap_initialization(data, init_info, results_dir):
    ensure_dir(results_dir)

    u_scores = init_info.u
    probs = init_info.probs
    init_modes = init_info.init_modes
    candidates = init_info.candidates

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        data[:, 0], data[:, 1],
        s=16,
        c="0.2",
        alpha=0.35,
        linewidths=0,
        label="Data"
    )

    if candidates is not None and candidates.size > 0:
        sc = ax.scatter(
            candidates[:, 0], candidates[:, 1],
            c=u_scores,
            s=20,
            alpha=0.85,
            linewidths=0,
            cmap="viridis",
            label="Candidate grid"
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.ax.set_title("u scores", pad=10)

    if init_modes is not None and init_modes.size > 0:
        ax.scatter(
            init_modes[:, 0], init_modes[:, 1],
            s=90,
            marker="o",
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            label="DAP Initializations"
        )

    ax.set_aspect("equal", "box")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("DAP Initialization")
    ax.legend(
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        fontsize=10
    )

    plt.tight_layout()

    out_pdf = os.path.join(results_dir, "dap_initialization_scatter.pdf")
    fig.savefig(out_pdf, dpi=120)
    plt.show()
    plt.close(fig)

    print(f"[saved] DAP Initialization scatter -> {out_pdf}")

    npz_path = os.path.join(results_dir, "dap_init_info.npz")
    np.savez(
        npz_path,
        u=u_scores,
        probs=probs,
        init_modes=init_modes,
        candidates=candidates,
        anchor_idx=init_info.anchor_idx,
        k=init_info.k,
        epsilon_init=init_info.epsilon_init,
        eps_draw=init_info.eps_draw
    )
    print(f"[saved] DAP init raw data -> {npz_path}")

    csv_path = os.path.join(results_dir, "dap_init_probs.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["z1", "z2", "u_score", "avg_selection_prob"])
        for i in range(len(candidates)):
            writer.writerow([candidates[i, 0], candidates[i, 1], u_scores[i], probs[i]])
    print(f"[saved] DAP init CSV -> {csv_path}")

    txt_path = os.path.join(results_dir, "dap_init_summary.txt")
    summary = (
        f"DAP Initialization Summary\n"
        f"---------------------------\n"
        f"Num data points: {len(data)}\n"
        f"Num candidate points: {len(candidates)}\n"
        f"k (anchors): {init_info.k}\n"
        f"Epsilon init: {init_info.epsilon_init:.6f}\n"
        f"Epsilon per draw: {init_info.eps_draw:.6f}\n"
        f"Init score: local empirical mass score\n"
        f"u-score min/max: {u_scores.min():.6g} / {u_scores.max():.6g}\n"
        f"avg selection prob min/max: {probs.min():.6g} / {probs.max():.6g}\n"
    )
    save_text(txt_path, summary)
    print(f"[saved] DAP init summary -> {txt_path}")

# ---------------------------------------------------------------------
# Visualization: KDE surface
# ---------------------------------------------------------------------

def kde_surface_plot(data, true_modes, results_dir):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from scipy.stats import gaussian_kde

    ensure_dir(results_dir)

    kde = gaussian_kde(data.T)
    x_min, x_max = data[:, 0].min() - 2, data[:, 0].max() + 2
    y_min, y_max = data[:, 1].min() - 2, data[:, 1].max() + 2
    xgrid, ygrid = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    positions = np.vstack([xgrid.ravel(), ygrid.ravel()])
    Z = np.reshape(kde(positions), xgrid.shape)

    fig3d = plt.figure(figsize=(8, 6))
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.plot_surface(xgrid, ygrid, Z, cmap='Blues', alpha=0.85, linewidth=0)

    true_modes = np.atleast_2d(true_modes)
    if true_modes.size > 0:
        z_true = kde(true_modes.T)
        ax3d.scatter(
            true_modes[:, 0],
            true_modes[:, 1],
            z_true,
            color="#2ca02c",
            edgecolor="k",
            s=180,
            marker="*",
            label="True modes"
        )

    ax3d.set_xlabel(r"$x_1$")
    ax3d.set_ylabel(r"$x_2$")
    ax3d.set_zlabel("Density", labelpad=10)
    ax3d.set_title("Estimated KDE Surface for 5-Modal Bivariate t-Mixture")
    if true_modes.size > 0:
        ax3d.legend(loc="upper right")

    plt.tight_layout()
    outpath3d = os.path.join(results_dir, "kde_surface_3d.pdf")
    fig3d.savefig(outpath3d, dpi=120)
    plt.show()
    print(f"[saved] 3D KDE surface plot -> {outpath3d}")
    plt.close(fig3d)

# ---------------------------------------------------------------------
# Visualization: single-run contour + modes
# ---------------------------------------------------------------------

def contour_plot_single(data, true_modes, ms_modes, dp_modes, results_dir):
    from scipy.stats import gaussian_kde

    ensure_dir(results_dir)

    kde = gaussian_kde(data.T)
    x_min, x_max = data[:, 0].min() - 2, data[:, 0].max() + 2
    y_min, y_max = data[:, 1].min() - 2, data[:, 1].max() + 2
    xgrid, ygrid = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    positions = np.vstack([xgrid.ravel(), ygrid.ravel()])
    Z = np.reshape(kde(positions), xgrid.shape)

    fig2d, ax2d = plt.subplots(figsize=(7, 6))
    ax2d.contourf(xgrid, ygrid, Z, levels=50, cmap="Blues", alpha=0.25)
    ax2d.contour(xgrid, ygrid, Z, levels=8, colors='k', linewidths=0.5, alpha=0.5)
    ax2d.scatter(
        data[:, 0], data[:, 1],
        s=20, c="0.2", alpha=0.5, edgecolors='none'
    )

    def plot_modes(modes, color, marker, label=None):
        modes = np.atleast_2d(modes)
        if modes.size == 0:
            return
        ax2d.scatter(
            modes[:, 0], modes[:, 1],
            s=250, facecolor='white', edgecolor='none', alpha=0.9
        )
        ax2d.scatter(
            modes[:, 0], modes[:, 1],
            s=120, facecolor=color, edgecolor='k',
            linewidth=1, marker=marker, label=label
        )

    plot_modes(true_modes, "#2ca02c", "*", label="True modes")
    plot_modes(ms_modes, "#1f77b4", "X", label="MS")
    plot_modes(dp_modes, "#ff7f0e", "o", label="DP-GRAMS")

    ax2d.set_xlabel(r"$x_1$")
    ax2d.set_ylabel(r"$x_2$")
    ax2d.set_title("Mode Estimation for 5-Modal Bivariate t-Mixture")
    ax2d.set_aspect('equal', 'box')
    ax2d.tick_params(axis='both', which='both', length=0)
    ax2d.legend(frameon=True, loc="upper right", fontsize=11)

    plt.tight_layout()
    outpath2d = os.path.join(results_dir, "contour_modes_2d.pdf")
    fig2d.savefig(outpath2d, dpi=120)
    plt.show()
    print(f"[saved] 2D contour + modes plot -> {outpath2d}")
    plt.close(fig2d)

# ---------------------------------------------------------------------
# Visualization: grid of runs
# ---------------------------------------------------------------------

def contour_plot_grid(data, true_modes, ms_modes_all, dp_modes_all,
                      results_dir, nrows=4, ncols=5,
                      figsize_per_subplot=(4.5, 4.5)):
    from scipy.stats import gaussian_kde
    from matplotlib import gridspec
    from matplotlib.lines import Line2D

    ensure_dir(results_dir)
    total_slots = nrows * ncols
    nplots = len(ms_modes_all)

    kde = gaussian_kde(data.T)
    x_min, x_max = data[:, 0].min() - 2, data[:, 0].max() + 2
    y_min, y_max = data[:, 1].min() - 2, data[:, 1].max() + 2
    xgrid, ygrid = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    positions = np.vstack([xgrid.ravel(), ygrid.ravel()])
    Z = np.reshape(kde(positions), xgrid.shape)

    fig_w = figsize_per_subplot[0] * ncols
    fig_h = figsize_per_subplot[1] * nrows
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(nrows, ncols, wspace=0.15, hspace=0.18)

    for slot in range(total_slots):
        r = slot // ncols
        c = slot % ncols
        ax = fig.add_subplot(gs[r, c])

        if slot >= nplots:
            ax.axis("off")
            continue

        ax.contourf(xgrid, ygrid, Z, levels=50, cmap="Blues", alpha=0.2, zorder=0)
        ax.contour(xgrid, ygrid, Z, levels=6, colors='k', linewidths=0.5, alpha=0.4, zorder=1)
        ax.scatter(data[:, 0], data[:, 1],
                   s=15, c="0.2", alpha=0.4, edgecolors='none', zorder=2)

        def plot_modes_subplot(modes, color, marker, zorder_base=3):
            modes = np.atleast_2d(modes)
            if modes.size == 0:
                return
            ax.scatter(modes[:, 0], modes[:, 1],
                       s=200, facecolor='white', edgecolor='none',
                       alpha=0.9, zorder=zorder_base)
            ax.scatter(modes[:, 0], modes[:, 1],
                       s=90, facecolor=color, edgecolor='k',
                       linewidth=0.8, marker=marker, zorder=zorder_base + 1)

        plot_modes_subplot(true_modes, "#2ca02c", "*")
        plot_modes_subplot(ms_modes_all[slot], "#1f77b4", "X")
        plot_modes_subplot(dp_modes_all[slot], "#ff7f0e", "o")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', 'box')
        ax.set_title(f"Run {slot + 1}", fontsize=10)

    legend_handles = [
        Line2D([0], [0], marker="*", color="#2ca02c",
               label="True modes", markeredgecolor="k",
               markersize=12, linestyle=""),
        Line2D([0], [0], marker="X", color="#1f77b4",
               label="MS", markersize=10, linestyle=""),
        Line2D([0], [0], marker="o", color="#ff7f0e",
               label="DP-GRAMS", markeredgecolor="k",
               markersize=10, linestyle="")
    ]
    fig.legend(handles=legend_handles,
               loc="lower center",
               bbox_to_anchor=(0.5, 0.04),
               ncol=3, frameon=True, fontsize=11)
    fig.suptitle(
        "Mode Estimation Across Runs for 5-Modal Bivariate t-Mixture",
        fontsize=14, y=0.94
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    outpath = os.path.join(results_dir, "contour_modes_grid.pdf")
    fig.savefig(outpath, dpi=120)
    print(f"[saved] Contour grid plot -> {outpath}")
    plt.close(fig)

# ---------------------------------------------------------------------
# Simulation: MSE vs n and vs epsilon
# ---------------------------------------------------------------------

def mse_vs_n_samples(n_samples_list, epsilon, delta, p, n_runs,
                     base_seed=42,
                     results_dir="results/bivariate_5mix",
                     clip_multiplier=DEFAULT_CLIP_MULTIPLIER,
                     kappa_init=DEFAULT_KAPPA_INIT,
                     init_epsilon_frac=DEFAULT_INIT_EPSILON_FRAC,
                     eta=DEFAULT_ETA):
    ms_mse_means, ms_mse_ses = [], []
    dp_mse_means, dp_mse_ses = [], []
    ms_time_means, ms_time_ses = [], []
    dp_time_means, dp_time_ses = [], []

    for idx, n_samples in enumerate(n_samples_list):
        ms_mses, dp_mses = [], []
        ms_times, dp_times = [], []

        data, true_modes, h = make_data(n_samples, base_seed + idx)
        ms_T = int(np.log(n_samples))

        for run in range(n_runs):
            run_seed = base_seed + idx * 100 + run

            ms_est, dp_est, ms_time, dp_time, _, _ = run_ms_and_dp_with_full_data_ms_init(
                data=data,
                epsilon=epsilon,
                delta=delta,
                h=h,
                ms_T=ms_T,
                dp_T=None,
                seed=run_seed,
                clip_multiplier=clip_multiplier,
                eta=eta,
                kappa_init=kappa_init,
                init_epsilon_frac=init_epsilon_frac,
            )

            ms_mses.append(mode_matching_mse(true_modes, ms_est))
            dp_mses.append(mode_matching_mse(true_modes, dp_est))
            ms_times.append(ms_time)
            dp_times.append(dp_time)

        ms_mse_means.append(float(np.mean(ms_mses)))
        ms_mse_ses.append(float(np.std(ms_mses, ddof=1) / math.sqrt(n_runs)))
        dp_mse_means.append(float(np.mean(dp_mses)))
        dp_mse_ses.append(float(np.std(dp_mses, ddof=1) / math.sqrt(n_runs)))
        ms_time_means.append(float(np.mean(ms_times)))
        ms_time_ses.append(float(np.std(ms_times, ddof=1) / math.sqrt(n_runs)))
        dp_time_means.append(float(np.mean(dp_times)))
        dp_time_ses.append(float(np.std(dp_times, ddof=1) / math.sqrt(n_runs)))

    ensure_dir(results_dir)
    csv_path = os.path.join(results_dir, "privacy_vs_n.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n_samples", "MS_mean", "MS_se",
            "DP_mean", "DP_se",
            "MS_time_mean", "MS_time_se",
            "DP_time_mean", "DP_time_se"
        ])
        for i, n in enumerate(n_samples_list):
            writer.writerow([
                n,
                ms_mse_means[i], ms_mse_ses[i],
                dp_mse_means[i], dp_mse_ses[i],
                ms_time_means[i], ms_time_ses[i],
                dp_time_means[i], dp_time_ses[i]
            ])
    print(f"[saved] MSE vs n results -> {csv_path}")
    return ms_mse_means, ms_mse_ses, dp_mse_means, dp_mse_ses


def mse_vs_epsilon_for_n_samples(n_samples_list, epsilon_values,
                                 delta, p, n_runs,
                                 base_seed=42,
                                 results_dir="results/bivariate_5mix",
                                 clip_multiplier=DEFAULT_CLIP_MULTIPLIER,
                                 kappa_init=DEFAULT_KAPPA_INIT,
                                 init_epsilon_frac=DEFAULT_INIT_EPSILON_FRAC,
                                 eta=DEFAULT_ETA):
    mse_dict = {}

    for idx, n_samples in enumerate(n_samples_list):
        data, true_modes, h = make_data(n_samples, base_seed + idx)
        ms_T = int(np.log(n_samples))

        ms_means, ms_ses = [], []
        dp_means, dp_ses = [], []
        ms_time_means, ms_time_ses = [], []
        dp_time_means, dp_time_ses = [], []

        for eps_idx, eps in enumerate(epsilon_values):
            ms_mses, dp_mses = [], []
            ms_times, dp_times = [], []

            for run in range(n_runs):
                run_seed = base_seed + idx * 100 + run + eps_idx * 1000

                ms_est, dp_est, ms_time, dp_time, _, _ = run_ms_and_dp_with_full_data_ms_init(
                    data=data,
                    epsilon=eps,
                    delta=delta,
                    h=h,
                    ms_T=ms_T,
                    dp_T=None,
                    seed=run_seed,
                    clip_multiplier=clip_multiplier,
                    eta=eta,
                    kappa_init=kappa_init,
                    init_epsilon_frac=init_epsilon_frac,
                )

                ms_mses.append(mode_matching_mse(true_modes, ms_est))
                dp_mses.append(mode_matching_mse(true_modes, dp_est))
                ms_times.append(ms_time)
                dp_times.append(dp_time)

            ms_means.append(float(np.mean(ms_mses)))
            ms_ses.append(float(np.std(ms_mses, ddof=1) / math.sqrt(n_runs)))
            dp_means.append(float(np.mean(dp_mses)))
            dp_ses.append(float(np.std(dp_mses, ddof=1) / math.sqrt(n_runs)))
            ms_time_means.append(float(np.mean(ms_times)))
            ms_time_ses.append(float(np.std(ms_times, ddof=1) / math.sqrt(n_runs)))
            dp_time_means.append(float(np.mean(dp_times)))
            dp_time_ses.append(float(np.std(dp_times, ddof=1) / math.sqrt(n_runs)))

        mse_dict[n_samples] = {
            "ms_means": ms_means,
            "ms_ses": ms_ses,
            "dp_means": dp_means,
            "dp_ses": dp_ses,
            "ms_time_means": ms_time_means,
            "ms_time_ses": ms_time_ses,
            "dp_time_means": dp_time_means,
            "dp_time_ses": dp_time_ses,
        }

    return mse_dict

# ---------------------------------------------------------------------
# Grid for LaTeX table: MSE + runtime over (n, epsilon)
# ---------------------------------------------------------------------

def mse_runtime_table(n_samples_list, epsilon_values,
                      delta, p, n_runs,
                      base_seed=42,
                      results_dir="results/bivariate_5mix",
                      clip_multiplier=DEFAULT_CLIP_MULTIPLIER,
                      kappa_init=DEFAULT_KAPPA_INIT,
                      init_epsilon_frac=DEFAULT_INIT_EPSILON_FRAC,
                      eta=DEFAULT_ETA):
    ensure_dir(results_dir)
    csv_path = os.path.join(results_dir, "mse_runtime_grid.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n", "epsilon",
            "DP_MSE_mean", "DP_MSE_se",
            "MS_MSE_mean", "MS_MSE_se",
            "DP_time_mean", "DP_time_se",
            "MS_time_mean", "MS_time_se"
        ])

        for idx, n_samples in enumerate(n_samples_list):
            data, true_modes, h = make_data(n_samples, base_seed + idx)
            ms_T = int(np.log(n_samples))

            for eps_idx, eps in enumerate(epsilon_values):
                ms_mses, dp_mses = [], []
                ms_times, dp_times = [], []

                for run in range(n_runs):
                    run_seed = base_seed + idx * 100 + run + eps_idx * 1000

                    ms_est, dp_est, ms_time, dp_time, _, _ = run_ms_and_dp_with_full_data_ms_init(
                        data=data,
                        epsilon=eps,
                        delta=delta,
                        h=h,
                        ms_T=ms_T,
                        dp_T=None,
                        seed=run_seed,
                        clip_multiplier=clip_multiplier,
                        eta=eta,
                        kappa_init=kappa_init,
                        init_epsilon_frac=init_epsilon_frac,
                    )

                    ms_mses.append(mode_matching_mse(true_modes, ms_est))
                    dp_mses.append(mode_matching_mse(true_modes, dp_est))
                    ms_times.append(ms_time)
                    dp_times.append(dp_time)

                ms_mse_mean = float(np.mean(ms_mses))
                ms_mse_se = float(np.std(ms_mses, ddof=1) / math.sqrt(n_runs))
                ms_time_mean = float(np.mean(ms_times))
                ms_time_se = float(np.std(ms_times, ddof=1) / math.sqrt(n_runs))

                dp_mse_mean = float(np.mean(dp_mses))
                dp_mse_se = float(np.std(dp_mses, ddof=1) / math.sqrt(n_runs))
                dp_time_mean = float(np.mean(dp_times))
                dp_time_se = float(np.std(dp_times, ddof=1) / math.sqrt(n_runs))

                writer.writerow([
                    n_samples, eps,
                    dp_mse_mean, dp_mse_se,
                    ms_mse_mean, ms_mse_se,
                    dp_time_mean, dp_time_se,
                    ms_time_mean, ms_time_se
                ])

    print(f"[saved] MSE/runtime grid -> {csv_path}")
    return csv_path

# ---------------------------------------------------------------------
# Privacy-utility curve plotting
# ---------------------------------------------------------------------

def privacy_utility_plot(n_samples_list, epsilon_values, mse_eps_dict, results_dir):
    ensure_dir(results_dir)
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("Set2", len(n_samples_list))

    for i, n_samples in enumerate(n_samples_list):
        stats = mse_eps_dict[n_samples]
        ms_means = stats["ms_means"]
        ms_ses = stats["ms_ses"]
        dp_means = stats["dp_means"]
        dp_ses = stats["dp_ses"]

        ax.errorbar(
            epsilon_values, dp_means, yerr=dp_ses,
            marker='o', linestyle='-',
            linewidth=1.8, markersize=6,
            label=f"DP-GRAMS n={n_samples}",
            color=palette[i], capsize=3
        )

        ax.errorbar(
            epsilon_values, ms_means, yerr=ms_ses,
            marker='s', linestyle='--',
            linewidth=1.4, markersize=5,
            label=f"MS n={n_samples}",
            color=palette[i], capsize=3, alpha=0.95
        )

    ax.set_xlabel("Privacy budget $\epsilon$")
    ax.set_ylabel("MSE")
    ax.set_title("Privacy-Utility Tradeoff for 5-Modal Bivariate t-Mixture")
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc="upper right", frameon=True)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.show()

    outpath = os.path.join(results_dir, "privacy_utility_pretty.pdf")
    fig.savefig(outpath, dpi=120)
    print(f"[saved] Privacy-utility plot -> {outpath}")
    plt.close(fig)

# ---------------------------------------------------------------------
# Helpers for hyperparameter sweeps
# ---------------------------------------------------------------------

def _run_dp_grams_mse_single(
    data,
    true_modes,
    epsilon,
    delta,
    h,
    T=None,
    m=None,
    clip_multiplier=DEFAULT_CLIP_MULTIPLIER,
    eta=DEFAULT_ETA,
    kappa_init=DEFAULT_KAPPA_INIT,
    init_epsilon_frac=DEFAULT_INIT_EPSILON_FRAC,
    seed=0,
):
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    _, dp_raw = dp_grams(
        X=data,
        epsilon=epsilon,
        delta=delta,
        T=T,
        h=h,
        rng=rng,
        m=m,
        clip_multiplier=clip_multiplier,
        eta=eta,
        kappa_init=kappa_init,
        init_epsilon_frac=init_epsilon_frac,
    )
    runtime = time.perf_counter() - t0

    if dp_raw.size == 0:
        mse = mode_matching_mse(true_modes, dp_raw)
        return mse, runtime

    dp_est = merge_modes(dp_raw)
    mse = mode_matching_mse(true_modes, dp_est)
    return mse, runtime


def sweep_clip_multiplier(
    data,
    true_modes,
    epsilon,
    delta,
    h,
    T,
    m_fixed,
    clip_multipliers,
    n_reps=20,
    base_seed=12345,
    kappa_init_fixed=DEFAULT_KAPPA_INIT,
    init_epsilon_frac_fixed=DEFAULT_INIT_EPSILON_FRAC,
    eta_fixed=DEFAULT_ETA,
):
    results = []
    rng = np.random.default_rng(base_seed)

    for cm in clip_multipliers:
        mses, times = [], []
        for _ in range(n_reps):
            seed = int(rng.integers(0, 2**31 - 1))
            mse, rt = _run_dp_grams_mse_single(
                data=data,
                true_modes=true_modes,
                epsilon=epsilon,
                delta=delta,
                h=h,
                T=T,
                m=m_fixed,
                clip_multiplier=cm,
                eta=eta_fixed,
                kappa_init=kappa_init_fixed,
                init_epsilon_frac=init_epsilon_frac_fixed,
                seed=seed,
            )
            mses.append(mse)
            times.append(rt)

        mses = np.array(mses, dtype=float)
        times = np.array(times, dtype=float)
        results.append({
            "clip_multiplier": cm,
            "mean_mse": float(np.nanmean(mses)),
            "std_mse": float(np.nanstd(mses)),
            "min_mse": float(np.nanmin(mses)),
            "max_mse": float(np.nanmax(mses)),
            "mean_time": float(np.nanmean(times)),
            "std_time": float(np.nanstd(times)),
        })

    results.sort(key=lambda r: r["clip_multiplier"])
    return results


def sweep_minibatch_size(
    data,
    true_modes,
    epsilon,
    delta,
    h,
    T,
    m_values,
    clip_multiplier_fixed=DEFAULT_CLIP_MULTIPLIER,
    n_reps=20,
    base_seed=54321,
    kappa_init_fixed=DEFAULT_KAPPA_INIT,
    init_epsilon_frac_fixed=DEFAULT_INIT_EPSILON_FRAC,
    eta_fixed=DEFAULT_ETA,
):
    results = []
    rng = np.random.default_rng(base_seed)

    for m in m_values:
        mses, times = [], []
        for _ in range(n_reps):
            seed = int(rng.integers(0, 2**31 - 1))
            mse, rt = _run_dp_grams_mse_single(
                data=data,
                true_modes=true_modes,
                epsilon=epsilon,
                delta=delta,
                h=h,
                T=T,
                m=m,
                clip_multiplier=clip_multiplier_fixed,
                eta=eta_fixed,
                kappa_init=kappa_init_fixed,
                init_epsilon_frac=init_epsilon_frac_fixed,
                seed=seed,
            )
            mses.append(mse)
            times.append(rt)

        mses = np.array(mses, dtype=float)
        times = np.array(times, dtype=float)
        results.append({
            "m": int(m),
            "mean_mse": float(np.nanmean(mses)),
            "std_mse": float(np.nanstd(mses)),
            "min_mse": float(np.nanmin(mses)),
            "max_mse": float(np.nanmax(mses)),
            "mean_time": float(np.nanmean(times)),
            "std_time": float(np.nanstd(times)),
        })

    results.sort(key=lambda r: r["m"])
    return results


def sweep_eta(
    data,
    true_modes,
    epsilon,
    delta,
    h,
    T,
    m_fixed,
    eta_values,
    clip_multiplier_fixed=DEFAULT_CLIP_MULTIPLIER,
    n_reps=20,
    base_seed=24680,
    kappa_init_fixed=DEFAULT_KAPPA_INIT,
    init_epsilon_frac_fixed=DEFAULT_INIT_EPSILON_FRAC,
):
    results = []
    rng = np.random.default_rng(base_seed)

    for eta in eta_values:
        mses, times = [], []
        for _ in range(n_reps):
            seed = int(rng.integers(0, 2**31 - 1))
            mse, rt = _run_dp_grams_mse_single(
                data=data,
                true_modes=true_modes,
                epsilon=epsilon,
                delta=delta,
                h=h,
                T=T,
                m=m_fixed,
                clip_multiplier=clip_multiplier_fixed,
                eta=eta,
                kappa_init=kappa_init_fixed,
                init_epsilon_frac=init_epsilon_frac_fixed,
                seed=seed,
            )
            mses.append(mse)
            times.append(rt)

        mses = np.array(mses, dtype=float)
        times = np.array(times, dtype=float)
        results.append({
            "eta": float(eta),
            "mean_mse": float(np.nanmean(mses)),
            "std_mse": float(np.nanstd(mses)),
            "min_mse": float(np.nanmin(mses)),
            "max_mse": float(np.nanmax(mses)),
            "mean_time": float(np.nanmean(times)),
            "std_time": float(np.nanstd(times)),
        })

    results.sort(key=lambda r: r["eta"])
    return results


def save_clip_multiplier_results_txt(results, path):
    ensure_dir(os.path.dirname(path))
    header = (
        f"{'clip_mult':>10} | {'mean_mse':>10} | {'std_mse':>10} | "
        f"{'min_mse':>10} | {'max_mse':>10} | {'mean_t':>10} | {'std_t':>10}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        line = (
            f"{r['clip_multiplier']:10.3f} | "
            f"{r['mean_mse']:10.4f} | "
            f"{r['std_mse']:10.4f} | "
            f"{r['min_mse']:10.4f} | "
            f"{r['max_mse']:10.4f} | "
            f"{r['mean_time']:10.4f} | "
            f"{r['std_time']:10.4f}"
        )
        lines.append(line)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[saved] Clip-multiplier sweep results -> {path}")


def save_minibatch_results_txt(results, path):
    ensure_dir(os.path.dirname(path))
    header = (
        f"{'m':>10} | {'mean_mse':>10} | {'std_mse':>10} | "
        f"{'min_mse':>10} | {'max_mse':>10} | {'mean_t':>10} | {'std_t':>10}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        line = (
            f"{r['m']:10d} | "
            f"{r['mean_mse']:10.4f} | "
            f"{r['std_mse']:10.4f} | "
            f"{r['min_mse']:10.4f} | "
            f"{r['max_mse']:10.4f} | "
            f"{r['mean_time']:10.4f} | "
            f"{r['std_time']:10.4f}"
        )
        lines.append(line)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[saved] Minibatch sweep results -> {path}")


def save_eta_results_txt(results, path):
    ensure_dir(os.path.dirname(path))
    header = (
        f"{'eta':>10} | {'mean_mse':>10} | {'std_mse':>10} | "
        f"{'min_mse':>10} | {'max_mse':>10} | {'mean_t':>10} | {'std_t':>10}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        line = (
            f"{r['eta']:10.4f} | "
            f"{r['mean_mse']:10.4f} | "
            f"{r['std_mse']:10.4f} | "
            f"{r['min_mse']:10.4f} | "
            f"{r['max_mse']:10.4f} | "
            f"{r['mean_time']:10.4f} | "
            f"{r['std_time']:10.4f}"
        )
        lines.append(line)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[saved] Eta sweep results -> {path}")


def plot_clip_multiplier_results_multi_n(clip_results_by_n, clip_grid, n_list, out_path):
    ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("Set2", len(n_list))

    for idx, n in enumerate(n_list):
        results = clip_results_by_n[n]
        stats_by_cm = {r["clip_multiplier"]: r for r in results}

        xs, ys, yerr = [], [], []
        for cm in clip_grid:
            if cm in stats_by_cm:
                xs.append(cm)
                ys.append(stats_by_cm[cm]["mean_mse"])
                yerr.append(stats_by_cm[cm]["std_mse"])

        if not xs:
            continue

        plt.errorbar(
            xs, ys, yerr=yerr,
            marker="o", linestyle="-",
            label=f"n={n}",
            color=palette[idx],
            capsize=3
        )

    plt.xlabel("Clip Multiplier ($C^*$ scaling)")
    plt.ylabel("MSE")
    plt.title("MSE vs $C^*$ across n for 5-Modal Bivariate t-Mixture")
    plt.grid(True, alpha=0.4)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"[saved] Multi-n clip-multiplier MSE plot -> {out_path}")
    plt.show()
    plt.close()


def plot_minibatch_results_grid(m_results_by_n, n_list, out_path):
    ensure_dir(os.path.dirname(out_path))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    axes = axes.flatten()
    palette = sns.color_palette("Set2", len(n_list))

    for idx, n in enumerate(n_list):
        ax = axes[idx]
        results = m_results_by_n[n]
        m_vals = [r["m"] for r in results]
        mean_mses = [r["mean_mse"] for r in results]
        std_mses = [r["std_mse"] for r in results]

        ax.errorbar(
            m_vals, mean_mses, yerr=std_mses,
            marker="o", linestyle="-",
            color=palette[idx],
            capsize=3
        )
        ax.set_title(f"n = {n}")
        ax.grid(True, alpha=0.4)
        ax.set_xscale("log")

    axes[0].set_ylabel("MSE")
    axes[2].set_ylabel("MSE")
    axes[2].set_xlabel("Minibatch size m")
    axes[3].set_xlabel("Minibatch size m")

    fig.suptitle("MSE vs m across n for 5-Modal Bivariate t-Mixture", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=120)
    print(f"[saved] Minibatch-size MSE grid plot -> {out_path}")
    plt.show()
    plt.close()


def plot_eta_results_grid(eta_results_by_n, n_list, out_path):
    ensure_dir(os.path.dirname(out_path))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    axes = axes.flatten()
    palette = sns.color_palette("Set2", len(n_list))

    for idx, n in enumerate(n_list):
        ax = axes[idx]
        results = eta_results_by_n[n]
        eta_vals = [r["eta"] for r in results]
        mean_mses = [r["mean_mse"] for r in results]
        std_mses = [r["std_mse"] for r in results]

        ax.errorbar(
            eta_vals, mean_mses, yerr=std_mses,
            marker="o", linestyle="-",
            color=palette[idx],
            capsize=3
        )
        ax.set_title(f"n = {n}")
        ax.grid(True, alpha=0.4)
        ax.set_xscale("log")

    axes[0].set_ylabel("MSE")
    axes[2].set_ylabel("MSE")
    axes[2].set_xlabel(r"Step size $\eta$")
    axes[3].set_xlabel(r"Step size $\eta$")

    fig.suptitle(r"MSE vs $\eta$ across n for 5-Modal Bivariate t-Mixture", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=120)
    print(f"[saved] Eta MSE grid plot -> {out_path}")
    plt.show()
    plt.close()

# ---------------------------------------------------------------------
# Main experiment entrypoint
# ---------------------------------------------------------------------

def main():
    epsilon, delta = 1.0, 1e-6
    p = 0.1
    n_runs = 20
    results_dir = "results/bivariate_5mix"
    ensure_dir(results_dir)

    base_seed = 42
    n_samples_base = 1200

    n_samples_list = [700, 1000, 2000, 5000]
    epsilon_values = [0.1, 0.2, 0.5, 1, 5]

    clip_multiplier = DEFAULT_CLIP_MULTIPLIER
    kappa_init = DEFAULT_KAPPA_INIT
    init_epsilon_frac = DEFAULT_INIT_EPSILON_FRAC
    eta = DEFAULT_ETA

    data, true_modes, h = make_data(n_samples_base, base_seed)
    T_ms = int(math.ceil(math.log(max(2, n_samples_base))))

    ms_modes_all, dp_modes_all = [], []
    ms_mses, dp_mses = [], []
    ms_times, dp_times = [], []

    dap_init_info = None

    for run in range(n_runs):
        run_seed = base_seed + run

        ms_est, dp_est, ms_time, dp_time, init_info_run, h_used = run_ms_and_dp_with_full_data_ms_init(
            data=data,
            epsilon=epsilon,
            delta=delta,
            h=h,
            ms_T=T_ms,
            dp_T=None,
            seed=run_seed,
            clip_multiplier=clip_multiplier,
            eta=eta,
            kappa_init=kappa_init,
            init_epsilon_frac=init_epsilon_frac,
        )

        if run == 0:
            dap_init_info = init_info_run

        ms_times.append(ms_time)
        dp_times.append(dp_time)
        ms_modes_all.append(ms_est)
        dp_modes_all.append(dp_est)

        ms_mses.append(mode_matching_mse(true_modes, ms_est))
        dp_mses.append(mode_matching_mse(true_modes, dp_est))

    stats = (
        f"n_samples: {n_samples_base}\n"
        f"dimension d: {data.shape[1]}\n"
        f"T (MS iterations): {T_ms}\n"
        f"T (DP iterations): internal via dp_grams (cT * base_T)\n"
        f"privacy epsilon: {epsilon}\n"
        f"privacy delta: {delta}\n"
        f"bandwidth h (Silverman on data): {h:.6f}\n"
        f"clip_multiplier: {clip_multiplier}\n"
        f"kappa_init: {kappa_init}\n"
        f"init_epsilon_frac: {init_epsilon_frac}\n"
        f"eta: {eta}\n\n"
        f"MS MSE mean±std: {np.mean(ms_mses):.6f} ± {np.std(ms_mses):.6f}\n"
        f"DP-GRAMS MSE mean±std: {np.mean(dp_mses):.6f} ± {np.std(dp_mses):.6f}\n"
        f"MS time mean±std: {np.mean(ms_times):.4f} ± {np.std(ms_times):.4f}\n"
        f"DP-GRAMS time mean±std: {np.mean(dp_times):.4f} ± {np.std(dp_times):.4f}\n"
    )
    stats_path = os.path.join(results_dir, "stats.txt")
    save_text(stats_path, stats)
    print(f"[saved] Stats summary -> {stats_path}")

    all_modes_text = ""
    for i in range(n_runs):
        all_modes_text += (
            f"Run {i + 1}\n"
            f"MS Modes:\n{np.array2string(np.atleast_2d(ms_modes_all[i]), precision=6)}\n"
            f"MS MSE: {ms_mses[i]:.6f}\n"
            f"DP-GRAMS Modes:\n{np.array2string(np.atleast_2d(dp_modes_all[i]), precision=6)}\n"
            f"DP MSE: {dp_mses[i]:.6f}\n\n"
        )
    all_modes_path = os.path.join(results_dir, "all_modes.txt")
    save_text(all_modes_path, all_modes_text)
    print(f"[saved] All modes listing -> {all_modes_path}")

    # 1) DAP
    print("\n[info] Running DAP Initialization visualization...")
    if dap_init_info is not None:
        plot_dap_initialization(data, dap_init_info, results_dir)
    else:
        print("[warning] DAP init info was not captured!")

    # 2) KDE surface
    kde_surface_plot(data, true_modes, results_dir)

    # 3) Single plot of MS, truth, DP-GRAMS
    contour_plot_single(
        data, true_modes,
        ms_modes_all[0], dp_modes_all[0],
        results_dir
    )

    # 4) 20-grid
    contour_plot_grid(
        data, true_modes,
        ms_modes_all, dp_modes_all,
        results_dir, ncols=5
    )

    # 5) Privacy-utility curve
    mse_eps_dict = mse_vs_epsilon_for_n_samples(
        n_samples_list,
        epsilon_values,
        delta,
        p,
        n_runs,
        base_seed=base_seed,
        results_dir=results_dir,
        clip_multiplier=clip_multiplier,
        kappa_init=kappa_init,
        init_epsilon_frac=init_epsilon_frac,
        eta=eta,
    )
    privacy_utility_plot(
        n_samples_list,
        epsilon_values,
        mse_eps_dict,
        results_dir
    )

    # 6) Hyperparameter sweeps
    clip_grid = [0.01, 0.1, 0.5, 1.0, 2.0]
    eta_grid = [0.25, 0.5, 1.0, 2.0, 4.0]

    print("\n[hyperparam] Running clip_multiplier sweeps across n...")
    clip_results_by_n = {}
    for idx, n_sweep in enumerate(n_samples_list):
        seed_n = base_seed + 10000 * (idx + 1)
        data_sweep, true_modes_sweep, h_sweep = make_data(n_sweep, seed_n)
        m_default = int(n_sweep / math.log(n_sweep))

        print(
            f"\n[hyperparam][clip] n={n_sweep}, "
            f"h={h_sweep:.4f}, m_default={m_default}"
        )

        clip_results = sweep_clip_multiplier(
            data=data_sweep,
            true_modes=true_modes_sweep,
            epsilon=epsilon,
            delta=delta,
            h=h_sweep,
            T=None,
            m_fixed=m_default,
            clip_multipliers=clip_grid,
            n_reps=20,
            base_seed=12345,
            kappa_init_fixed=kappa_init,
            init_epsilon_frac_fixed=init_epsilon_frac,
            eta_fixed=eta,
        )
        clip_results_by_n[n_sweep] = clip_results

        clip_txt_path_n = os.path.join(
            results_dir,
            f"mse_vs_clip_multiplier_n{n_sweep}.txt"
        )
        save_clip_multiplier_results_txt(clip_results, clip_txt_path_n)

    clip_plot_multi_path = os.path.join(results_dir, "mse_vs_clip_multiplier_multi_n.pdf")
    plot_clip_multiplier_results_multi_n(
        clip_results_by_n,
        clip_grid,
        n_samples_list,
        clip_plot_multi_path
    )

    m_grid_frac = [0.01, 0.05, 0.1, 0.2, 1.0]
    print("\n[hyperparam] Running minibatch size sweeps across n...")
    m_results_by_n = {}
    for idx, n_sweep in enumerate(n_samples_list):
        seed_n = base_seed + 20000 * (idx + 1)
        data_sweep, true_modes_sweep, h_sweep = make_data(n_sweep, seed_n)

        m_grid = sorted(set(max(1, int(frac * n_sweep)) for frac in m_grid_frac))
        print(
            f"\n[hyperparam][minibatch] n={n_sweep}, "
            f"h={h_sweep:.4f}, m_grid={m_grid}"
        )

        m_results = sweep_minibatch_size(
            data=data_sweep,
            true_modes=true_modes_sweep,
            epsilon=epsilon,
            delta=delta,
            h=h_sweep,
            T=None,
            m_values=m_grid,
            clip_multiplier_fixed=clip_multiplier,
            n_reps=20,
            base_seed=54321,
            kappa_init_fixed=kappa_init,
            init_epsilon_frac_fixed=init_epsilon_frac,
            eta_fixed=eta,
        )
        m_results_by_n[n_sweep] = m_results

        m_txt_path_n = os.path.join(
            results_dir,
            f"mse_vs_minibatch_n{n_sweep}.txt"
        )
        save_minibatch_results_txt(m_results, m_txt_path_n)

    m_plot_grid_path = os.path.join(results_dir, "mse_vs_minibatch_grid.pdf")
    plot_minibatch_results_grid(
        m_results_by_n,
        n_samples_list,
        m_plot_grid_path
    )

    print("\n[hyperparam] Running eta sweeps across n...")
    eta_results_by_n = {}
    for idx, n_sweep in enumerate(n_samples_list):
        seed_n = base_seed + 30000 * (idx + 1)
        data_sweep, true_modes_sweep, h_sweep = make_data(n_sweep, seed_n)
        m_default = int(n_sweep / math.log(n_sweep))

        print(
            f"\n[hyperparam][eta] n={n_sweep}, "
            f"h={h_sweep:.4f}, m_default={m_default}, eta_grid={eta_grid}"
        )

        eta_results = sweep_eta(
            data=data_sweep,
            true_modes=true_modes_sweep,
            epsilon=epsilon,
            delta=delta,
            h=h_sweep,
            T=None,
            m_fixed=m_default,
            eta_values=eta_grid,
            clip_multiplier_fixed=clip_multiplier,
            n_reps=20,
            base_seed=24680,
            kappa_init_fixed=kappa_init,
            init_epsilon_frac_fixed=init_epsilon_frac,
        )
        eta_results_by_n[n_sweep] = eta_results

        eta_txt_path_n = os.path.join(
            results_dir,
            f"mse_vs_eta_n{n_sweep}.txt"
        )
        save_eta_results_txt(eta_results, eta_txt_path_n)

    eta_plot_grid_path = os.path.join(results_dir, "mse_vs_eta_grid.pdf")
    plot_eta_results_grid(
        eta_results_by_n,
        n_samples_list,
        eta_plot_grid_path
    )

    # Optional summary tables
    mse_vs_n_samples(
        n_samples_list,
        epsilon,
        delta,
        p,
        n_runs,
        base_seed=base_seed,
        results_dir=results_dir,
        clip_multiplier=clip_multiplier,
        kappa_init=kappa_init,
        init_epsilon_frac=init_epsilon_frac,
        eta=eta,
    )

    mse_runtime_table(
        n_samples_list,
        epsilon_values,
        delta,
        p,
        n_runs,
        base_seed=base_seed,
        results_dir=results_dir,
        clip_multiplier=clip_multiplier,
        kappa_init=kappa_init,
        init_epsilon_frac=init_epsilon_frac,
        eta=eta,
    )

if __name__ == "__main__":
    main()
