import numpy as np
import os
import sys
import time
import csv
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from diffprivlib.models import KMeans as DPKMeans

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from real_data_scripts.dp_grams_c import dpms_private
from main_scripts.mode_matching_mse import mode_matching_mse
from main_scripts.ms import mean_shift
from main_scripts.merge import merge_modes_agglomerative
from main_scripts.bandwidth import silverman_bandwidth

# ---------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------
results_dir = "results/blobs_centroid_comparison"
os.makedirs(results_dir, exist_ok=True)

sns.set(style="whitegrid", context="talk")

base_n_samples = 1000
n_centers = 4
cluster_std = 1.2

X_raw, y_true = make_blobs(
    n_samples=base_n_samples,
    centers=n_centers,
    cluster_std=cluster_std,
    random_state=1,
)

n, d = X_raw.shape

merge_n_clusters = len(np.unique(y_true))
ms_T = int(np.ceil(np.log(max(2, n))))
ms_p = 0.1
delta = 1e-6
base_rng_seed = 41
n_runs = 20

eps_modes_list = [0.1, 0.2, 0.5, 1.0, 5.0]
n_samples_list_privacy = [700, 1000, 2000, 5000]

clip_grid_subsample = [0.01, 0.1, 0.5, 1.0, 2.0]
m_frac_grid_subsample = [0.01, 0.05, 0.1, 0.2, 1.0]
n_runs_subsample_effect = 20
epsilon_subsample_effect = 1.0

# ---------------------------------------------------------------------
# Explicit DP-GRAMS-C defaults
# ---------------------------------------------------------------------
# Keep these explicit so this script never silently inherits dp_grams / dp_grams_c
# defaults.  In particular, candidate_points stays None, so DP-GRAMS-C uses its
# default public-box DAP grid and does NOT use data points as candidates.
DEFAULT_CLIP_MULTIPLIER = 0.1
DEFAULT_KAPPA_INIT = 5.0
DEFAULT_INIT_EPSILON_FRAC = 0.1
DEFAULT_ETA = 1.0
DEFAULT_BETA_DAP = 3.0
DEFAULT_C_RHO = 2.0
DEFAULT_DAP_SCORE_MULTIPLIER = 3.0
DEFAULT_R = None
DEFAULT_CANDIDATE_POINTS = None

# ---------------------------------------------------------------------
# Standardized clustering space
# ---------------------------------------------------------------------
# Run all clustering algorithms in standardized space.  This avoids scale/offset
# pathologies in the public DAP grid and makes centroid MSE directly comparable.
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
true_means = np.array([X[y_true == c].mean(axis=0) for c in np.unique(y_true)])
true_means_scaled = true_means.copy()

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def compute_metrics(y_true, labels):
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    return ari, nmi


def timed(func, *args, **kwargs):
    start = time.time()
    out = func(*args, **kwargs)
    return out, time.time() - start


def relabel_clusters(y_true, y_pred, n_clusters):
    cm = confusion_matrix(y_true, y_pred, labels=range(n_clusters))
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping.get(label, label) for label in y_pred], dtype=int)


def fit_dpkmeans(X_input, eps, seed, n_clusters):
    model = DPKMeans(
        n_clusters=n_clusters,
        epsilon=eps,
        random_state=seed,
    )
    try:
        model.fit(X_input)
        return model
    except Exception:
        mins_local = X_input.min(axis=0)
        maxs_local = X_input.max(axis=0)
        model = DPKMeans(
            n_clusters=n_clusters,
            epsilon=eps,
            random_state=seed,
            bounds=(mins_local, maxs_local),
        )
        model.fit(X_input)
        return model


def run_dpgrams_c_blobs(
    data,
    epsilon_modes,
    h,
    rng,
    *,
    k_est,
    clip_multiplier=DEFAULT_CLIP_MULTIPLIER,
    m=None,
):
    """
    Single DP-GRAMS-C call with all experiment defaults explicit.

    candidate_points is intentionally None: the wrapper calls dp_grams with the
    default public candidate grid.  No data points are passed as DAP candidates.
    """
    return dpms_private(
        data=data,
        epsilon_modes=float(epsilon_modes),
        delta=delta,
        h=h,
        rng=rng,
        k_est=k_est,
        clip_multiplier=float(clip_multiplier),
        m=m,
        kappa_init=DEFAULT_KAPPA_INIT,
        init_epsilon_frac=DEFAULT_INIT_EPSILON_FRAC,
        eta=DEFAULT_ETA,
        beta_dap=DEFAULT_BETA_DAP,
        c_rho=DEFAULT_C_RHO,
        dap_score_multiplier=DEFAULT_DAP_SCORE_MULTIPLIER,
        R=DEFAULT_R,
        candidate_points=DEFAULT_CANDIDATE_POINTS,
    )


def plot_subsample_effect_grid(
    results_by_n,
    subsample_sizes,
    x_key,
    metric_key,
    x_label,
    y_label,
    title_prefix,
    out_path,
    log_x=True,
):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    axes = axes.flatten()

    for ax, n_sub in zip(axes, subsample_sizes):
        stats_list = results_by_n.get(n_sub, [])
        if not stats_list:
            ax.set_visible(False)
            continue

        xs = [r[x_key] for r in stats_list]
        ys = [r[f"mean_{metric_key}"] for r in stats_list]
        yerr = [r[f"std_{metric_key}"] for r in stats_list]

        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=6,
            capsize=3,
        )
        if log_x:
            ax.set_xscale("log")
        ax.set_title(f"n = {n_sub}")
        ax.grid(True, alpha=0.4)

    axes[0].set_ylabel(y_label)
    axes[2].set_ylabel(y_label)
    axes[2].set_xlabel(x_label)
    axes[3].set_xlabel(x_label)

    fig.suptitle(title_prefix, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=120)
    plt.show()
    print(f"[saved] {title_prefix} -> {out_path}")


# ---------------------------------------------------------------------
# Base bandwidth
# ---------------------------------------------------------------------
h_base = silverman_bandwidth(X)

# ---------------------------------------------------------------------
# 1) Mean Shift baseline
# ---------------------------------------------------------------------
rng_ms = np.random.default_rng(base_rng_seed)
(raw_ms_modes, ms_time) = timed(
    mean_shift,
    X,
    T=ms_T,
    bandwidth=h_base,
    p=ms_p,
    seed=rng_ms,
)
ms_merged_modes = merge_modes_agglomerative(
    raw_ms_modes,
    n_clusters=merge_n_clusters,
    random_state=base_rng_seed,
)

dists_ms = np.linalg.norm(X[:, None, :] - ms_merged_modes[None, :, :], axis=2)
labels_ms = np.argmin(dists_ms, axis=1)
labels_ms = relabel_clusters(y_true, labels_ms, merge_n_clusters)

# ---------------------------------------------------------------------
# 2) DP-GRAMS-C
# ---------------------------------------------------------------------
rng_dpms = np.random.default_rng(base_rng_seed)
(modes_dpms, labels_dpms), dpms_time = timed(
    run_dpgrams_c_blobs,
    data=X,
    epsilon_modes=1.0,
    h=h_base,
    rng=rng_dpms,
    k_est=merge_n_clusters,
    clip_multiplier=DEFAULT_CLIP_MULTIPLIER,
)
labels_dpms = relabel_clusters(y_true, labels_dpms, merge_n_clusters)

# ---------------------------------------------------------------------
# 3) Non-private KMeans
# ---------------------------------------------------------------------
kmeans = KMeans(
    n_clusters=merge_n_clusters,
    random_state=base_rng_seed,
    n_init=10,
)
(labels_km, km_time) = timed(kmeans.fit_predict, X)
modes_km = kmeans.cluster_centers_
labels_km = relabel_clusters(y_true, labels_km, merge_n_clusters)

# ---------------------------------------------------------------------
# 4) DP-KMeans
# ---------------------------------------------------------------------
(dp_kmeans, dpkm_time) = timed(
    fit_dpkmeans,
    X,
    1.0,
    base_rng_seed,
    merge_n_clusters,
)
labels_dpkm = (
    dp_kmeans.labels_.astype(int)
    if hasattr(dp_kmeans, "labels_")
    else dp_kmeans.predict(X)
)
modes_dpkm = dp_kmeans.cluster_centers_
labels_dpkm = relabel_clusters(y_true, labels_dpkm, merge_n_clusters)

# ---------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------
algorithms = ["MS Clustering", "DP-GRAMS-C", "KMeans", "DP-KMeans"]
labels_list = [labels_ms, labels_dpms, labels_km, labels_dpkm]
modes_list = [ms_merged_modes, modes_dpms, modes_km, modes_dpkm]
runtimes = [ms_time, dpms_time, km_time, dpkm_time]

metrics = {}
mse_centroids = {}

for alg, lbls, modes in zip(algorithms, labels_list, modes_list):
    ari, nmi = compute_metrics(y_true, lbls)
    metrics[alg] = (ari, nmi)
    mse_centroids[alg] = mode_matching_mse(true_means_scaled.copy(), modes.copy())

# ---------------------------------------------------------------------
# Save base metrics
# ---------------------------------------------------------------------
base_metrics_csv = os.path.join(results_dir, f"clustering_metrics_base_n{base_n_samples}.csv")
with open(base_metrics_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Algorithm", "ARI", "NMI", "MSE", "Runtime(s)"])
    for alg, rt in zip(algorithms, runtimes):
        ari, nmi = metrics[alg]
        writer.writerow([alg, ari, nmi, mse_centroids[alg], rt])

base_metrics_txt = os.path.join(results_dir, f"clustering_metrics_base_n{base_n_samples}.txt")
with open(base_metrics_txt, "w") as f:
    f.write(f"Silverman bandwidth h_base = {float(h_base):.8f}\n")
    f.write("All clustering and MSE are in standardized feature space.\n")
    f.write("DP-GRAMS-C candidate_points = None; default public DAP grid is used.\n")
    f.write(f"DP-GRAMS-C clip_multiplier = {DEFAULT_CLIP_MULTIPLIER}\n")
    f.write(f"DP-GRAMS-C kappa_init = {DEFAULT_KAPPA_INIT}\n")
    f.write(f"DP-GRAMS-C init_epsilon_frac = {DEFAULT_INIT_EPSILON_FRAC}\n")
    f.write(f"DP-GRAMS-C c_rho = {DEFAULT_C_RHO}\n")
    f.write(f"DP-GRAMS-C R = {DEFAULT_R}\n\n")
    for alg, rt in zip(algorithms, runtimes):
        ari, nmi = metrics[alg]
        f.write(
            f"{alg}: ARI={ari:.4f}, NMI={nmi:.4f}, "
            f"MSE={mse_centroids[alg]:.6f}, Runtime={rt:.4f}s\n"
        )

print(f"\n[Blobs, base n={base_n_samples}] Metrics (MSE on scaled features):")
print(f"{'Alg':<14} {'ARI':>6} {'NMI':>6} {'MSE_centroids':>14} {'Runtime(s)':>12}")
for alg, rt in zip(algorithms, runtimes):
    ari, nmi = metrics[alg]
    print(f"{alg:<14} {ari:6.3f} {nmi:6.3f} {mse_centroids[alg]:14.6f} {rt:12.4f}")

# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------
X_2d = X
true_means_2d = true_means
modes_2d_list = modes_list

fig, axes = plt.subplots(1, 4, figsize=(20, 6))
palette = sns.color_palette("tab10", merge_n_clusters)

global_handles = []
global_labels = []

for ax, alg, labels_pred, modes_2d in zip(axes, algorithms, labels_list, modes_2d_list):
    for i, color in enumerate(palette):
        sc = ax.scatter(
            X_2d[labels_pred == i, 0],
            X_2d[labels_pred == i, 1],
            c=[color],
            s=20,
            alpha=0.7,
        )
        if len(global_handles) < merge_n_clusters:
            global_handles.append(sc)
            global_labels.append(f"Cluster {i}")

    true_sc = ax.scatter(
        true_means_2d[:, 0],
        true_means_2d[:, 1],
        marker="X",
        c="magenta",
        s=120,
        linewidths=2,
    )
    if "True means" not in global_labels:
        global_handles.append(true_sc)
        global_labels.append("True means")

    modes_sc = ax.scatter(
        modes_2d[:, 0],
        modes_2d[:, 1],
        marker="X",
        c="blue",
        s=90,
        linewidths=2,
    )
    if "Estimated modes" not in global_labels:
        global_handles.append(modes_sc)
        global_labels.append("Estimated modes")

    ax.set_title(alg, fontsize=14)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

legend = fig.legend(
    global_handles,
    global_labels,
    fontsize=9,
    loc="lower center",
    ncol=6,
    bbox_to_anchor=(0.5, 0.0),
    title="Cluster Assignments & Centroids",
)
plt.setp(legend.get_title(), fontsize=11, fontweight="bold")
fig.suptitle("Clustering Comparison on Blobs Dataset", fontsize=18, y=0.95)

save_path_clusters = os.path.join(
    results_dir, f"blobs_clustering_comparison_base_n{base_n_samples}.pdf"
)
plt.tight_layout(rect=[0, 0.05, 1, 0.98])
plt.savefig(save_path_clusters, dpi=120)
plt.show()
print("Clustering comparison plot saved to:", save_path_clusters)

# ---------------------------------------------------------------------
# Privacy-utility across n
# ---------------------------------------------------------------------
metrics_names = ["ARI", "NMI", "MSE"]

dpms_metrics_multi = {n_samp: {m: [] for m in metrics_names} for n_samp in n_samples_list_privacy}
dpkm_metrics_multi = {n_samp: {m: [] for m in metrics_names} for n_samp in n_samples_list_privacy}
dpms_err_multi = {n_samp: {m: [] for m in metrics_names} for n_samp in n_samples_list_privacy}
dpkm_err_multi = {n_samp: {m: [] for m in metrics_names} for n_samp in n_samples_list_privacy}
dpms_time_mean_multi = {n_samp: [] for n_samp in n_samples_list_privacy}
dpkm_time_mean_multi = {n_samp: [] for n_samp in n_samples_list_privacy}
dpms_time_std_multi = {n_samp: [] for n_samp in n_samples_list_privacy}
dpkm_time_std_multi = {n_samp: [] for n_samp in n_samples_list_privacy}

privacy_csv = os.path.join(results_dir, "privacy_utility_metrics_multi_n.csv")
with open(privacy_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "n_samples",
            "Eps_modes",
            "Algorithm",
            "ARI",
            "Std_ARI",
            "NMI",
            "Std_NMI",
            "MSE",
            "Std_MSE",
            "Mean_Runtime(s)",
            "Std_Runtime(s)",
            "Silverman_h",
            "standardized_features",
            "dpgrams_c_clip_multiplier",
            "dpgrams_c_kappa_init",
            "dpgrams_c_init_epsilon_frac",
            "dpgrams_c_c_rho",
            "dpgrams_c_candidate_points",
        ]
    )

    for idx_n, n_samp in enumerate(n_samples_list_privacy):
        print(
            f"\n[Privacy-Utility][blobs] n={n_samp}, "
            f"eps grid = {eps_modes_list}, runs per config = {n_runs}"
        )

        seed_n = base_rng_seed + 1000 * (idx_n + 1)
        Xn_raw, yn = make_blobs(
            n_samples=n_samp,
            centers=n_centers,
            cluster_std=cluster_std,
            random_state=seed_n,
        )

        scaler_n = StandardScaler()
        Xn = scaler_n.fit_transform(Xn_raw)
        h_n = silverman_bandwidth(Xn)
        true_means_n = np.array([Xn[yn == c].mean(axis=0) for c in np.unique(yn)])

        for eps in eps_modes_list:
            # DP-GRAMS-C
            ari_list_dpms, nmi_list_dpms, mse_list_dpms, time_list_dpms = [], [], [], []

            for run in range(n_runs):
                rng_run = np.random.default_rng(
                    base_rng_seed + idx_n * 10000 + int(100 * eps) * 100 + run
                )
                t0 = time.time()
                modes_dpms_run, labels_dpms_run = run_dpgrams_c_blobs(
                    data=Xn,
                    epsilon_modes=eps,
                    h=h_n,
                    rng=rng_run,
                    k_est=merge_n_clusters,
                    clip_multiplier=DEFAULT_CLIP_MULTIPLIER,
                )
                t_dpms = time.time() - t0
                time_list_dpms.append(t_dpms)

                labels_dpms_run = relabel_clusters(yn, labels_dpms_run, merge_n_clusters)
                run_ari, run_nmi = compute_metrics(yn, labels_dpms_run)
                run_mse = mode_matching_mse(true_means_n.copy(), modes_dpms_run.copy())

                ari_list_dpms.append(run_ari)
                nmi_list_dpms.append(run_nmi)
                mse_list_dpms.append(run_mse)

            ari_arr = np.array(ari_list_dpms, dtype=float)
            nmi_arr = np.array(nmi_list_dpms, dtype=float)
            mse_arr = np.array(mse_list_dpms, dtype=float)
            t_arr = np.array(time_list_dpms, dtype=float)

            dpms_metrics_multi[n_samp]["ARI"].append(float(ari_arr.mean()))
            dpms_metrics_multi[n_samp]["NMI"].append(float(nmi_arr.mean()))
            dpms_metrics_multi[n_samp]["MSE"].append(float(mse_arr.mean()))
            dpms_err_multi[n_samp]["ARI"].append(float(ari_arr.std(ddof=1)))
            dpms_err_multi[n_samp]["NMI"].append(float(nmi_arr.std(ddof=1)))
            dpms_err_multi[n_samp]["MSE"].append(float(mse_arr.std(ddof=1)))
            dpms_time_mean_multi[n_samp].append(float(t_arr.mean()))
            dpms_time_std_multi[n_samp].append(float(t_arr.std(ddof=1)))

            writer.writerow(
                [
                    n_samp, eps, "DP-GRAMS-C",
                    float(ari_arr.mean()), float(ari_arr.std(ddof=1)),
                    float(nmi_arr.mean()), float(nmi_arr.std(ddof=1)),
                    float(mse_arr.mean()), float(mse_arr.std(ddof=1)),
                    float(t_arr.mean()), float(t_arr.std(ddof=1)),
                    float(h_n),
                    True,
                    DEFAULT_CLIP_MULTIPLIER,
                    DEFAULT_KAPPA_INIT,
                    DEFAULT_INIT_EPSILON_FRAC,
                    DEFAULT_C_RHO,
                    "None",
                ]
            )

            # DP-KMeans
            ari_list_dpkm, nmi_list_dpkm, mse_list_dpkm, time_list_dpkm = [], [], [], []

            for run in range(n_runs):
                seed = base_rng_seed + idx_n * 10000 + int(100 * eps) * 100 + run
                t0 = time.time()
                dp_kmeans_run = fit_dpkmeans(
                    Xn,
                    eps,
                    seed,
                    merge_n_clusters,
                )
                t_dpkm = time.time() - t0
                time_list_dpkm.append(t_dpkm)

                labels_dpkm_run = (
                    dp_kmeans_run.labels_.astype(int)
                    if hasattr(dp_kmeans_run, "labels_")
                    else dp_kmeans_run.predict(Xn)
                )
                labels_dpkm_run = relabel_clusters(yn, labels_dpkm_run, merge_n_clusters)
                modes_dpkm_run = dp_kmeans_run.cluster_centers_

                run_ari, run_nmi = compute_metrics(yn, labels_dpkm_run)
                run_mse = mode_matching_mse(true_means_n.copy(), modes_dpkm_run.copy())

                ari_list_dpkm.append(run_ari)
                nmi_list_dpkm.append(run_nmi)
                mse_list_dpkm.append(run_mse)

            ari_arr = np.array(ari_list_dpkm, dtype=float)
            nmi_arr = np.array(nmi_list_dpkm, dtype=float)
            mse_arr = np.array(mse_list_dpkm, dtype=float)
            t_arr = np.array(time_list_dpkm, dtype=float)

            dpkm_metrics_multi[n_samp]["ARI"].append(float(ari_arr.mean()))
            dpkm_metrics_multi[n_samp]["NMI"].append(float(nmi_arr.mean()))
            dpkm_metrics_multi[n_samp]["MSE"].append(float(mse_arr.mean()))
            dpkm_err_multi[n_samp]["ARI"].append(float(ari_arr.std(ddof=1)))
            dpkm_err_multi[n_samp]["NMI"].append(float(nmi_arr.std(ddof=1)))
            dpkm_err_multi[n_samp]["MSE"].append(float(mse_arr.std(ddof=1)))
            dpkm_time_mean_multi[n_samp].append(float(t_arr.mean()))
            dpkm_time_std_multi[n_samp].append(float(t_arr.std(ddof=1)))

            writer.writerow(
                [
                    n_samp, eps, "DP-KMeans",
                    float(ari_arr.mean()), float(ari_arr.std(ddof=1)),
                    float(nmi_arr.mean()), float(nmi_arr.std(ddof=1)),
                    float(mse_arr.mean()), float(mse_arr.std(ddof=1)),
                    float(t_arr.mean()), float(t_arr.std(ddof=1)),
                    float(h_n),
                    True,
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )

print(
    "\nAll blobs privacy-utility experiments across n completed. "
    f"Metrics saved to: {privacy_csv}"
)

for metric in metrics_names:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, n_samp in enumerate(n_samples_list_privacy):
        ax = axes_flat[idx]

        ax.errorbar(
            eps_modes_list,
            dpms_metrics_multi[n_samp][metric],
            yerr=dpms_err_multi[n_samp][metric],
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=6,
            capsize=3,
            label="DP-GRAMS-C",
        )

        ax.errorbar(
            eps_modes_list,
            dpkm_metrics_multi[n_samp][metric],
            yerr=dpkm_err_multi[n_samp][metric],
            marker="s",
            linestyle="--",
            linewidth=2,
            markersize=6,
            capsize=3,
            label="DP-KMeans",
        )

        ax.set_xscale("log")
        ax.set_title(f"n = {n_samp}")
        ax.grid(True, linestyle="--", alpha=0.6)

        if idx % 2 == 0:
            ax.set_ylabel(metric)
        if idx >= 2:
            ax.set_xlabel(r"$\epsilon_{\mathrm{modes}}$")

        ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"Privacy-Utility ({metric}) Tradeoff across n for Blobs", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(results_dir, f"blobs_privacy_utility_{metric.lower()}_grid_multi_n.pdf")
    plt.savefig(out_path, dpi=120)
    plt.show()
    print(f"[saved] Privacy-utility 2x2 grid ({metric}) figure -> {out_path}")

# ---------------------------------------------------------------------
# Subsampling-effect for DP-GRAMS-C
# ---------------------------------------------------------------------
def dpms_hparam_single_blobs(
    n_samples,
    epsilon_modes,
    clip_multiplier,
    m,
    run,
    base_seed=12345,
):
    rs_data = base_seed + run
    X_raw_run, y_run = make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        cluster_std=cluster_std,
        random_state=rs_data,
    )

    scaler_run = StandardScaler()
    X_run = scaler_run.fit_transform(X_raw_run)
    _h_run = silverman_bandwidth(X_run)
    true_means_run = np.array([X_run[y_run == c].mean(axis=0) for c in np.unique(y_run)])

    rng_dp = np.random.default_rng(base_seed + 1000 + run)
    t0 = time.time()
    modes_dpms_run, labels_dpms_run = run_dpgrams_c_blobs(
        data=X_run,
        epsilon_modes=epsilon_modes,
        h=_h_run,
        rng=rng_dp,
        k_est=merge_n_clusters,
        clip_multiplier=clip_multiplier,
        m=m,
    )
    runtime = time.time() - t0

    if modes_dpms_run.size == 0:
        return np.nan, np.nan, np.nan, float(runtime)

    labels_dpms_run = relabel_clusters(y_run, labels_dpms_run, merge_n_clusters)
    ari, nmi = compute_metrics(y_run, labels_dpms_run)

    mse = mode_matching_mse(true_means_run.copy(), modes_dpms_run.copy())

    return float(ari), float(nmi), float(mse), float(runtime)


def subsampling_effect_clip_grid():
    results_by_n = {}

    for n_sub in n_samples_list_privacy:
        print(f"\n[subsample-effect][blobs][C*] n={n_sub}")
        stats_list = []
        for cm in clip_grid_subsample:
            ari_vals, nmi_vals, mse_vals, time_vals = [], [], [], []
            for run in range(n_runs_subsample_effect):
                ari, nmi, mse, rt = dpms_hparam_single_blobs(
                    n_samples=n_sub,
                    epsilon_modes=epsilon_subsample_effect,
                    clip_multiplier=cm,
                    m=None,
                    run=run + int(1000 * cm),
                )
                ari_vals.append(ari)
                nmi_vals.append(nmi)
                mse_vals.append(mse)
                time_vals.append(rt)

            ari_vals = np.array(ari_vals, dtype=float)
            nmi_vals = np.array(nmi_vals, dtype=float)
            mse_vals = np.array(mse_vals, dtype=float)
            time_vals = np.array(time_vals, dtype=float)

            stats_list.append({
                "clip_multiplier": float(cm),
                "mean_ari": float(np.nanmean(ari_vals)),
                "std_ari": float(np.nanstd(ari_vals, ddof=1)),
                "mean_nmi": float(np.nanmean(nmi_vals)),
                "std_nmi": float(np.nanstd(nmi_vals, ddof=1)),
                "mean_mse": float(np.nanmean(mse_vals)),
                "std_mse": float(np.nanstd(mse_vals, ddof=1)),
                "mean_time": float(np.nanmean(time_vals)),
                "std_time": float(np.nanstd(time_vals, ddof=1)),
            })

        stats_list.sort(key=lambda r: r["clip_multiplier"])
        results_by_n[n_sub] = stats_list

    return results_by_n


def subsampling_effect_m_grid():
    results_by_n = {}

    for n_sub in n_samples_list_privacy:
        print(f"\n[subsample-effect][blobs][m] n={n_sub}")
        m_grid = sorted(set(max(1, int(frac * n_sub)) for frac in m_frac_grid_subsample))
        stats_list = []

        for m_val in m_grid:
            ari_vals, nmi_vals, mse_vals, time_vals = [], [], [], []
            for run in range(n_runs_subsample_effect):
                ari, nmi, mse, rt = dpms_hparam_single_blobs(
                    n_samples=n_sub,
                    epsilon_modes=epsilon_subsample_effect,
                    clip_multiplier=DEFAULT_CLIP_MULTIPLIER,
                    m=int(m_val),
                    run=run + int(10 * m_val),
                )
                ari_vals.append(ari)
                nmi_vals.append(nmi)
                mse_vals.append(mse)
                time_vals.append(rt)

            ari_vals = np.array(ari_vals, dtype=float)
            nmi_vals = np.array(nmi_vals, dtype=float)
            mse_vals = np.array(mse_vals, dtype=float)
            time_vals = np.array(time_vals, dtype=float)

            stats_list.append({
                "m": int(m_val),
                "mean_ari": float(np.nanmean(ari_vals)),
                "std_ari": float(np.nanstd(ari_vals, ddof=1)),
                "mean_nmi": float(np.nanmean(nmi_vals)),
                "std_nmi": float(np.nanstd(nmi_vals, ddof=1)),
                "mean_mse": float(np.nanmean(mse_vals)),
                "std_mse": float(np.nanstd(mse_vals, ddof=1)),
                "mean_time": float(np.nanmean(time_vals)),
                "std_time": float(np.nanstd(time_vals, ddof=1)),
            })

        stats_list.sort(key=lambda r: r["m"])
        results_by_n[n_sub] = stats_list

    return results_by_n


clip_results_by_n = subsampling_effect_clip_grid()
clip_txt_path = os.path.join(results_dir, "blobs_dpgrams_c_subsample_effect_clip_stats.txt")
with open(clip_txt_path, "w") as f:
    for n_sub in n_samples_list_privacy:
        f.write(f"=== n = {n_sub} ===\n")
        for r in clip_results_by_n[n_sub]:
            f.write(
                f"C*={r['clip_multiplier']:.3f}, "
                f"ARI={r['mean_ari']:.4f}±{r['std_ari']:.4f}, "
                f"NMI={r['mean_nmi']:.4f}±{r['std_nmi']:.4f}, "
                f"MSE={r['mean_mse']:.4f}±{r['std_mse']:.4f}, "
                f"time={r['mean_time']:.4f}±{r['std_time']:.4f}\n"
            )
        f.write("\n")
print(f"[saved] Clip subsampling-effect stats -> {clip_txt_path}")

plot_subsample_effect_grid(
    clip_results_by_n,
    n_samples_list_privacy,
    x_key="clip_multiplier",
    metric_key="ari",
    x_label=r"Clip Multiplier ($C^*$)",
    y_label="ARI",
    title_prefix="Blobs: ARI vs $C^*$ across n",
    out_path=os.path.join(results_dir, "blobs_dpgrams_c_subsample_effect_clip_ari_grid.pdf"),
    log_x=True,
)
plot_subsample_effect_grid(
    clip_results_by_n,
    n_samples_list_privacy,
    x_key="clip_multiplier",
    metric_key="nmi",
    x_label=r"Clip Multiplier ($C^*$)",
    y_label="NMI",
    title_prefix="Blobs: NMI vs $C^*$ across n",
    out_path=os.path.join(results_dir, "blobs_dpgrams_c_subsample_effect_clip_nmi_grid.pdf"),
    log_x=True,
)
plot_subsample_effect_grid(
    clip_results_by_n,
    n_samples_list_privacy,
    x_key="clip_multiplier",
    metric_key="mse",
    x_label=r"Clip Multiplier ($C^*$)",
    y_label="Centroid MSE",
    title_prefix="Blobs: MSE vs $C^*$ across n",
    out_path=os.path.join(results_dir, "blobs_dpgrams_c_subsample_effect_clip_mse_grid.pdf"),
    log_x=True,
)

m_results_by_n = subsampling_effect_m_grid()
m_txt_path = os.path.join(results_dir, "blobs_dpgrams_c_subsample_effect_minibatch_stats.txt")
with open(m_txt_path, "w") as f:
    for n_sub in n_samples_list_privacy:
        f.write(f"=== n = {n_sub} ===\n")
        for r in m_results_by_n[n_sub]:
            f.write(
                f"m={r['m']}, "
                f"ARI={r['mean_ari']:.4f}±{r['std_ari']:.4f}, "
                f"NMI={r['mean_nmi']:.4f}±{r['std_nmi']:.4f}, "
                f"MSE={r['mean_mse']:.4f}±{r['std_mse']:.4f}, "
                f"time={r['mean_time']:.4f}±{r['std_time']:.4f}\n"
            )
        f.write("\n")
print(f"[saved] Minibatch subsampling-effect stats -> {m_txt_path}")

plot_subsample_effect_grid(
    m_results_by_n,
    n_samples_list_privacy,
    x_key="m",
    metric_key="ari",
    x_label="Minibatch size m",
    y_label="ARI",
    title_prefix="Blobs: ARI vs m across n",
    out_path=os.path.join(results_dir, "blobs_dpgrams_c_subsample_effect_m_ari_grid.pdf"),
    log_x=True,
)
plot_subsample_effect_grid(
    m_results_by_n,
    n_samples_list_privacy,
    x_key="m",
    metric_key="nmi",
    x_label="Minibatch size m",
    y_label="NMI",
    title_prefix="Blobs: NMI vs m across n",
    out_path=os.path.join(results_dir, "blobs_dpgrams_c_subsample_effect_m_nmi_grid.pdf"),
    log_x=True,
)
plot_subsample_effect_grid(
    m_results_by_n,
    n_samples_list_privacy,
    x_key="m",
    metric_key="mse",
    x_label="Minibatch size m",
    y_label="Centroid MSE",
    title_prefix="Blobs: MSE vs m across n",
    out_path=os.path.join(results_dir, "blobs_dpgrams_c_subsample_effect_m_mse_grid.pdf"),
    log_x=True,
)
