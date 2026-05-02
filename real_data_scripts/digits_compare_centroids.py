# import csv
# import os
# import sys
# import time

# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# from diffprivlib.models import KMeans as DPKMeans
# from scipy.optimize import linear_sum_assignment
# from sklearn.cluster import KMeans
# from sklearn.datasets import load_digits
# from sklearn.decomposition import PCA
# from sklearn.metrics import (
#     adjusted_rand_score,
#     confusion_matrix,
#     normalized_mutual_info_score,
# )
# from sklearn.preprocessing import StandardScaler

# # ---------------------------------------------------------------------
# # Local imports
# # ---------------------------------------------------------------------
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from main_scripts.bandwidth import silverman_bandwidth
# from main_scripts.mode_matching_mse import mode_matching_mse
# from main_scripts.ms import mean_shift
# from main_scripts.merge import merge_modes_agglomerative
# from real_data_scripts.dp_grams_c import dpms_private

# # ---------------------------------------------------------------------
# # Setup
# # ---------------------------------------------------------------------
# results_dir = "results/digits_centroid_comparison"
# os.makedirs(results_dir, exist_ok=True)

# sns.set(style="whitegrid", context="talk")

# digits = load_digits()
# X_raw = digits.data
# y_true = digits.target

# n, d = X_raw.shape

# merge_n_clusters = 10
# ms_T = int(np.ceil(np.log(max(2, n))))
# ms_p = 0.1
# delta = 1e-5
# base_rng_seed = 41
# n_runs = 20

# clip_grid = [0.001, 0.005, 0.01, 0.05, 0.1]
# m_frac_grid = [0.05, 0.1, 0.2, 0.5, 1.0]
# n_reps_hparam = 20
# epsilon_hparam = 1.0

# # ---------------------------------------------------------------------
# # Explicit DP-GRAMS-C defaults
# # ---------------------------------------------------------------------
# # Keep these explicit so this script never silently inherits dp_grams / dp_grams_c
# # defaults. In particular, candidate_points stays None, so DP-GRAMS-C uses its
# # default public-box DAP grid and does NOT use data points as candidates.
# DEFAULT_CLIP_MULTIPLIER = 1
# DEFAULT_KAPPA_INIT = 6.0
# DEFAULT_INIT_EPSILON_FRAC = 0.5
# DEFAULT_ETA = 1
# DEFAULT_BETA_DAP = 3.0
# DEFAULT_C_RHO = 2.0
# DEFAULT_DAP_SCORE_MULTIPLIER = 10.0
# DEFAULT_R = None
# DEFAULT_CANDIDATE_POINTS = None
# DEFAULT_DAP_INIT_STRATEGY = "factorized"

# # PCA dimension selection. The value of r is computed from the PCA spectrum; it
# # is not hard-coded. The capped broken-stick rule is the default for digits.
# PCA_DIM_SELECTION_RULE = "broken_stick_capped"
# PCA_R_MIN = 2
# PCA_R_MAX = 8

# # ---------------------------------------------------------------------
# # Data preprocessing and automatic PCA dimension selection
# # ---------------------------------------------------------------------
# def _safe_rank_from_count(count, *, min_rank=1, max_rank=None):
#     if max_rank is None:
#         max_rank = int(count)
#     return int(max(min_rank, min(int(count), int(max_rank))))


# def select_pca_dimension(
#     X_scaled,
#     *,
#     rule="broken_stick_capped",
#     r_min=2,
#     r_max=10,
# ):
#     """
#     Compute several PCA dimension rules and return the selected dimension.

#     The selected dimension is computed from the spectrum. It is not hard-coded.
#     """
#     X_scaled = np.asarray(X_scaled, dtype=float)
#     pca_probe = PCA(random_state=base_rng_seed)
#     pca_probe.fit(X_scaled)

#     eigvals = np.asarray(pca_probe.explained_variance_, dtype=float)
#     evr = np.asarray(pca_probe.explained_variance_ratio_, dtype=float)
#     p = int(eigvals.size)

#     if p == 0:
#         raise ValueError("PCA spectrum is empty.")

#     # Ahn-Horenstein eigenvalue ratio: argmax lambda_j / lambda_{j+1}.
#     if p >= 2:
#         ratios = eigvals[:-1] / np.maximum(eigvals[1:], 1e-12)
#         ahn_horenstein = int(np.argmax(ratios) + 1)
#     else:
#         ahn_horenstein = 1

#     # Broken-stick: keep components whose explained-variance share exceeds the
#     # broken-stick expected share.
#     harmonic_tail = np.array([
#         np.sum(1.0 / np.arange(j, p + 1, dtype=float)) / float(p)
#         for j in range(1, p + 1)
#     ])
#     broken_stick = int(np.sum(evr > harmonic_tail))
#     broken_stick = max(1, broken_stick)

#     # Kaiser rule for standardized variables: eigenvalue above average.
#     kaiser = int(np.sum(eigvals > np.mean(eigvals)))
#     kaiser = max(1, kaiser)

#     # Effective-rank style summaries.
#     eig_sum = float(np.sum(eigvals))
#     if eig_sum <= 0.0:
#         participation_rank_float = 1.0
#         entropy_rank_float = 1.0
#     else:
#         participation_rank_float = float((eig_sum ** 2) / np.sum(eigvals ** 2))
#         probs = eigvals / eig_sum
#         probs_pos = probs[probs > 0]
#         entropy_rank_float = float(np.exp(-np.sum(probs_pos * np.log(probs_pos))))

#     participation_rank = int(np.ceil(participation_rank_float))
#     entropy_rank = int(np.ceil(entropy_rank_float))

#     cum = np.cumsum(evr)
#     r80 = int(np.searchsorted(cum, 0.80) + 1)
#     r85 = int(np.searchsorted(cum, 0.85) + 1)
#     r90 = int(np.searchsorted(cum, 0.90) + 1)

#     candidates = {
#         "ahn_horenstein": ahn_horenstein,
#         "broken_stick": broken_stick,
#         "kaiser": kaiser,
#         "participation_rank": participation_rank,
#         "entropy_rank": entropy_rank,
#         "var80": r80,
#         "var85": r85,
#         "var90": r90,
#     }

#     if rule == "broken_stick_capped":
#         raw_selected = candidates["broken_stick"]
#     elif rule == "ahn_horenstein_capped":
#         raw_selected = candidates["ahn_horenstein"]
#     elif rule == "participation_capped":
#         raw_selected = candidates["participation_rank"]
#     elif rule == "kaiser_capped":
#         raw_selected = candidates["kaiser"]
#     elif rule == "var80_capped":
#         raw_selected = candidates["var80"]
#     elif rule == "var85_capped":
#         raw_selected = candidates["var85"]
#     elif rule == "var90_capped":
#         raw_selected = candidates["var90"]
#     else:
#         raise ValueError(f"Unknown PCA_DIM_SELECTION_RULE={rule!r}.")

#     selected = _safe_rank_from_count(
#         raw_selected,
#         min_rank=int(r_min),
#         max_rank=min(int(r_max), p),
#     )

#     diagnostics = {
#         "rule": str(rule),
#         "r_min": int(r_min),
#         "r_max": int(r_max),
#         "selected_raw": int(raw_selected),
#         "selected": int(selected),
#         "ahn_horenstein": int(ahn_horenstein),
#         "broken_stick": int(broken_stick),
#         "kaiser": int(kaiser),
#         "participation_rank_float": float(participation_rank_float),
#         "participation_rank": int(participation_rank),
#         "entropy_rank_float": float(entropy_rank_float),
#         "entropy_rank": int(entropy_rank),
#         "var80": int(r80),
#         "var85": int(r85),
#         "var90": int(r90),
#         "cumvar_selected": float(cum[selected - 1]),
#         "top_eigenvalues": eigvals[: min(10, p)].copy(),
#         "top_explained_variance_ratio": evr[: min(10, p)].copy(),
#     }
#     return selected, diagnostics, pca_probe


# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_raw)

# true_means_scaled = np.array([
#     X_scaled[y_true == c].mean(axis=0)
#     for c in np.unique(y_true)
# ])

# pca_dim, pca_dim_info, _pca_probe = select_pca_dimension(
#     X_scaled,
#     rule=PCA_DIM_SELECTION_RULE,
#     r_min=PCA_R_MIN,
#     r_max=PCA_R_MAX,
# )

# print("[pca] PCA dimension diagnostics:")
# print(f"[pca] Ahn-Horenstein eigenvalue-ratio r = {pca_dim_info['ahn_horenstein']}")
# print(f"[pca] Broken-stick r = {pca_dim_info['broken_stick']}")
# print(f"[pca] Kaiser r = {pca_dim_info['kaiser']}")
# print(
#     "[pca] Participation effective rank = "
#     f"{pca_dim_info['participation_rank_float']:.3f} "
#     f"-> {pca_dim_info['participation_rank']}"
# )
# print(
#     "[pca] Entropy effective rank = "
#     f"{pca_dim_info['entropy_rank_float']:.3f} "
#     f"-> {pca_dim_info['entropy_rank']}"
# )
# print(f"[pca] 80% variance r = {pca_dim_info['var80']}")
# print(f"[pca] 85% variance r = {pca_dim_info['var85']}")
# print(f"[pca] 90% variance r = {pca_dim_info['var90']}")
# print(
#     f"[pca] selected rule = {pca_dim_info['rule']}, "
#     f"raw r = {pca_dim_info['selected_raw']}, "
#     f"capped selected r = {pca_dim_info['selected']}, "
#     f"cumvar = {pca_dim_info['cumvar_selected']:.4f}"
# )

# pca_shared = PCA(n_components=pca_dim, random_state=base_rng_seed)
# X = pca_shared.fit_transform(X_scaled)

# true_means = np.array([
#     X[y_true == c].mean(axis=0)
#     for c in np.unique(y_true)
# ])


# def pca_to_scaled(centers):
#     centers = np.asarray(centers, dtype=float)
#     if centers.size == 0:
#         return np.empty((0, X_scaled.shape[1]), dtype=float)
#     centers = np.atleast_2d(centers)
#     return pca_shared.inverse_transform(centers)


# # ---------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------
# def compute_metrics(y_true, labels):
#     ari = adjusted_rand_score(y_true, labels)
#     nmi = normalized_mutual_info_score(y_true, labels)
#     return ari, nmi


# def timed(func, *args, **kwargs):
#     start = time.time()
#     out = func(*args, **kwargs)
#     return out, time.time() - start


# def relabel_clusters(y_true, y_pred, n_clusters=10):
#     cm = confusion_matrix(y_true, y_pred, labels=range(n_clusters))
#     row_ind, col_ind = linear_sum_assignment(-cm)
#     mapping = {col: row for row, col in zip(row_ind, col_ind)}
#     return np.array([mapping.get(label, label) for label in y_pred], dtype=int)


# def fit_dpkmeans(X_input, n_clusters, epsilon, random_state):
#     model = DPKMeans(
#         n_clusters=n_clusters,
#         epsilon=epsilon,
#         random_state=random_state,
#     )
#     try:
#         model.fit(X_input)
#         return model
#     except Exception:
#         mins_local = X_input.min(axis=0)
#         maxs_local = X_input.max(axis=0)
#         model = DPKMeans(
#             n_clusters=n_clusters,
#             epsilon=epsilon,
#             bounds=(mins_local, maxs_local),
#             random_state=random_state,
#         )
#         model.fit(X_input)
#         return model


# def run_dpgrams_c_digits(
#     data,
#     epsilon_modes,
#     h,
#     rng,
#     *,
#     k_est,
#     clip_multiplier=DEFAULT_CLIP_MULTIPLIER,
#     m=None,
#     return_info=False,
# ):
#     """
#     Single DP-GRAMS-C call with all experiment defaults explicit.

#     candidate_points is intentionally None: the wrapper calls dp_grams with the
#     default public candidate grid. No data points are passed as DAP candidates.
#     """
#     return dpms_private(
#         data=data,
#         epsilon_modes=float(epsilon_modes),
#         delta=delta,
#         h=h,
#         rng=rng,
#         k_est=k_est,
#         clip_multiplier=float(clip_multiplier),
#         m=m,
#         kappa_init=DEFAULT_KAPPA_INIT,
#         init_epsilon_frac=DEFAULT_INIT_EPSILON_FRAC,
#         eta=DEFAULT_ETA,
#         beta_dap=DEFAULT_BETA_DAP,
#         c_rho=DEFAULT_C_RHO,
#         dap_score_multiplier=DEFAULT_DAP_SCORE_MULTIPLIER,
#         dap_init_strategy=DEFAULT_DAP_INIT_STRATEGY,
#         R=DEFAULT_R,
#         candidate_points=DEFAULT_CANDIDATE_POINTS,
#         return_info=return_info,
#     )


# def make_2d_view(X_in, centers_list, true_means_in, seed):
#     X_in = np.asarray(X_in, dtype=float)
#     true_means_in = np.asarray(true_means_in, dtype=float)

#     if X_in.shape[1] >= 2:
#         pca2 = PCA(n_components=2, random_state=seed)
#         X_2d = pca2.fit_transform(X_in)
#         true_means_2d = pca2.transform(true_means_in)
#         centers_2d_list = [pca2.transform(np.asarray(m, dtype=float)) for m in centers_list]
#     else:
#         X_2d = np.column_stack([X_in[:, 0], np.zeros(X_in.shape[0])])
#         true_means_2d = np.column_stack([true_means_in[:, 0], np.zeros(true_means_in.shape[0])])
#         centers_2d_list = [
#             np.column_stack([np.asarray(m, dtype=float)[:, 0], np.zeros(np.asarray(m).shape[0])])
#             for m in centers_list
#         ]

#     return X_2d, true_means_2d, centers_2d_list


# # ---------------------------------------------------------------------
# # Bandwidth
# # ---------------------------------------------------------------------
# h_base = silverman_bandwidth(X)

# # ---------------------------------------------------------------------
# # 1) Mean Shift baseline
# # ---------------------------------------------------------------------
# rng_ms = np.random.default_rng(base_rng_seed)

# (raw_ms_modes, ms_time) = timed(
#     mean_shift,
#     X,
#     T=ms_T,
#     bandwidth=h_base,
#     p=ms_p,
#     seed=rng_ms,
# )

# ms_merged_modes = merge_modes_agglomerative(
#     raw_ms_modes,
#     n_clusters=merge_n_clusters,
#     random_state=base_rng_seed,
# )

# dists_ms = np.linalg.norm(X[:, None, :] - ms_merged_modes[None, :, :], axis=2)
# labels_ms = np.argmin(dists_ms, axis=1)
# labels_ms = relabel_clusters(y_true, labels_ms, merge_n_clusters)

# # ---------------------------------------------------------------------
# # 2) DP-GRAMS-C
# # ---------------------------------------------------------------------
# rng_dp = np.random.default_rng(base_rng_seed)
# (((modes_dp, labels_dp), h_used, init_info), dp_time) = timed(
#     run_dpgrams_c_digits,
#     data=X,
#     epsilon_modes=1.0,
#     h=h_base,
#     rng=rng_dp,
#     k_est=merge_n_clusters,
#     clip_multiplier=DEFAULT_CLIP_MULTIPLIER,
#     return_info=True,
# )
# labels_dp = relabel_clusters(y_true, labels_dp, merge_n_clusters)

# # ---------------------------------------------------------------------
# # 3) Non-private KMeans
# # ---------------------------------------------------------------------
# kmeans = KMeans(
#     n_clusters=merge_n_clusters,
#     random_state=base_rng_seed,
#     n_init=10,
# )
# (labels_km, km_time) = timed(kmeans.fit_predict, X)
# modes_km = kmeans.cluster_centers_
# labels_km = relabel_clusters(y_true, labels_km, merge_n_clusters)

# # ---------------------------------------------------------------------
# # 4) DP-KMeans
# # ---------------------------------------------------------------------
# (dp_kmeans, dpkm_time) = timed(
#     fit_dpkmeans,
#     X,
#     merge_n_clusters,
#     1.0,
#     base_rng_seed,
# )

# labels_dpkm = (
#     dp_kmeans.labels_.astype(int)
#     if hasattr(dp_kmeans, "labels_")
#     else dp_kmeans.predict(X)
# )
# modes_dpkm = dp_kmeans.cluster_centers_
# labels_dpkm = relabel_clusters(y_true, labels_dpkm, merge_n_clusters)

# # ---------------------------------------------------------------------
# # Aggregate metrics
# # ---------------------------------------------------------------------
# algorithms = ["MS Clustering", "DP-GRAMS-C", "KMeans", "DP-KMeans"]
# labels_list = [labels_ms, labels_dp, labels_km, labels_dpkm]
# modes_list = [ms_merged_modes, modes_dp, modes_km, modes_dpkm]
# runtimes = [ms_time, dp_time, km_time, dpkm_time]

# metrics = {}
# mse_centroids = {}

# for alg, labels, modes in zip(algorithms, labels_list, modes_list):
#     metrics[alg] = compute_metrics(y_true, labels)
#     mse_centroids[alg] = mode_matching_mse(
#         true_means_scaled.copy(),
#         pca_to_scaled(modes).copy(),
#     )

# # ---------------------------------------------------------------------
# # Save metrics
# # ---------------------------------------------------------------------
# csv_file = os.path.join(results_dir, "clustering_metrics.csv")
# with open(csv_file, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Algorithm", "ARI", "NMI", "MSE", "Runtime(s)"])
#     for alg, rt in zip(algorithms, runtimes):
#         ari, nmi = metrics[alg]
#         writer.writerow([alg, ari, nmi, mse_centroids[alg], rt])

# txt_file = os.path.join(results_dir, "clustering_metrics.txt")
# with open(txt_file, "w") as f:
#     f.write("Digits comparison with standardized raw features and shared PCA clustering space\n")
#     f.write(f"n = {n}, raw d = {d}, selected PCA r = {pca_dim}\n")
#     f.write(f"PCA dimension rule = {PCA_DIM_SELECTION_RULE}\n")
#     f.write(f"PCA selected raw r = {pca_dim_info['selected_raw']}\n")
#     f.write(f"PCA selected capped r = {pca_dim_info['selected']}\n")
#     f.write(f"PCA selected cumulative variance = {pca_dim_info['cumvar_selected']:.6f}\n")
#     f.write(f"Ahn-Horenstein r = {pca_dim_info['ahn_horenstein']}\n")
#     f.write(f"Broken-stick r = {pca_dim_info['broken_stick']}\n")
#     f.write(f"Kaiser r = {pca_dim_info['kaiser']}\n")
#     f.write(f"Participation effective rank = {pca_dim_info['participation_rank_float']:.6f} -> {pca_dim_info['participation_rank']}\n")
#     f.write(f"Entropy effective rank = {pca_dim_info['entropy_rank_float']:.6f} -> {pca_dim_info['entropy_rank']}\n")
#     f.write(f"Variance ranks: r80={pca_dim_info['var80']}, r85={pca_dim_info['var85']}, r90={pca_dim_info['var90']}\n")
#     f.write("All methods clustered on the shared PCA representation.\n")
#     f.write("Centroid MSE is computed after inverse PCA back to standardized raw feature space.\n")
#     f.write("DP-GRAMS-C candidate_points = None; factorized public DAP grids are used.\n")
#     f.write(f"DP-GRAMS-C internal d_ambient = {int(init_info.d_ambient)}\n")
#     f.write(f"Silverman bandwidth h_base = {float(h_base):.8f}\n")
#     f.write(f"DP-GRAMS h_used = {float(h_used):.8f}\n")
#     f.write(
#         "DP-GRAMS-C settings: indicator utility, logarithmic suppression, "
#         f"kappa_init={DEFAULT_KAPPA_INIT}, "
#         f"init_epsilon_frac={DEFAULT_INIT_EPSILON_FRAC}, "
#         f"clip_multiplier={DEFAULT_CLIP_MULTIPLIER}, "
#         f"c_rho={DEFAULT_C_RHO}, "
#         f"dap_score_multiplier={DEFAULT_DAP_SCORE_MULTIPLIER}, "
#         f"R={DEFAULT_R}, "
#         f"dap_init_strategy={DEFAULT_DAP_INIT_STRATEGY}\n"
#     )
#     for alg, rt in zip(algorithms, runtimes):
#         ari, nmi = metrics[alg]
#         f.write(
#             f"{alg}: ARI={ari:.4f}, NMI={nmi:.4f}, "
#             f"MSE={mse_centroids[alg]:.6f}, Runtime={rt:.4f}s\n"
#         )

# print("\nMetrics (clustering on shared PCA representation; MSE in standardized raw space):")
# print(f"{'Alg':<14} {'ARI':>6} {'NMI':>6} {'MSE_centroids':>14} {'Runtime(s)':>12}")
# for alg, rt in zip(algorithms, runtimes):
#     ari, nmi = metrics[alg]
#     print(
#         f"{alg:<14} {ari:6.3f} {nmi:6.3f} "
#         f"{mse_centroids[alg]:14.6f} {rt:12.4f}"
#     )

# # ---------------------------------------------------------------------
# # 2D visualization
# # ---------------------------------------------------------------------
# X_2d, true_means_2d, modes_2d_list = make_2d_view(
#     X,
#     modes_list,
#     true_means,
#     seed=base_rng_seed,
# )

# fig, axes = plt.subplots(1, 4, figsize=(20, 6))
# palette = sns.color_palette("tab10", merge_n_clusters)

# global_handles = []
# global_labels = []

# for ax, alg, labels_pred, modes_2d in zip(axes, algorithms, labels_list, modes_2d_list):
#     for i, color in enumerate(palette):
#         sc = ax.scatter(
#             X_2d[labels_pred == i, 0],
#             X_2d[labels_pred == i, 1],
#             c=[color],
#             s=15,
#             alpha=0.7,
#         )
#         if len(global_handles) < merge_n_clusters:
#             global_handles.append(sc)
#             global_labels.append(f"Cluster {i}")

#     true_sc = ax.scatter(
#         true_means_2d[:, 0],
#         true_means_2d[:, 1],
#         marker="X",
#         c="magenta",
#         s=140,
#         linewidths=2,
#     )
#     if "True means" not in global_labels:
#         global_handles.append(true_sc)
#         global_labels.append("True means")

#     modes_sc = ax.scatter(
#         modes_2d[:, 0],
#         modes_2d[:, 1],
#         marker="X",
#         c="blue",
#         s=100,
#         linewidths=2,
#     )
#     if "Estimated modes" not in global_labels:
#         global_handles.append(modes_sc)
#         global_labels.append("Estimated modes")

#     ax.set_title(alg, fontsize=14)
#     ax.set_xlabel("PC1")
#     ax.set_ylabel("PC2")

# legend = fig.legend(
#     global_handles,
#     global_labels,
#     fontsize=9,
#     loc="lower center",
#     ncol=6,
#     bbox_to_anchor=(0.5, 0.01),
#     title="Cluster Assignments & Centroids",
# )
# plt.setp(legend.get_title(), fontsize=11, fontweight="bold")
# fig.suptitle("Clustering Comparison on Digits Dataset", fontsize=18, y=0.92)

# save_path_clusters = os.path.join(results_dir, "digits_clustering_comparison.pdf")
# plt.tight_layout(rect=[0, 0.05, 1, 0.93])
# plt.savefig(save_path_clusters, dpi=120)
# plt.show()
# print("Clustering comparison plot saved to:", save_path_clusters)

# # ---------------------------------------------------------------------
# # Privacy-utility curves
# # ---------------------------------------------------------------------
# epsilon_list = [0.25, 0.5, 1.0, 2.5, 5.0]

# dp_metrics = {m: [] for m in ["ARI", "NMI", "MSE"]}
# dpkm_metrics = {m: [] for m in ["ARI", "NMI", "MSE"]}
# dp_err = {m: [] for m in ["ARI", "NMI", "MSE"]}
# dpkm_err = {m: [] for m in ["ARI", "NMI", "MSE"]}
# dp_times_mean, dpkm_times_mean = [], []
# dp_times_std, dpkm_times_std = [], []

# for eps in epsilon_list:
#     ari_list, nmi_list, mse_list = [], [], []
#     dp_times = []

#     for run in range(n_runs):
#         rng_run = np.random.default_rng(base_rng_seed + run)

#         t0 = time.time()
#         modes_run, labels_run = run_dpgrams_c_digits(
#             data=X,
#             epsilon_modes=eps,
#             h=h_base,
#             rng=rng_run,
#             k_est=merge_n_clusters,
#             clip_multiplier=DEFAULT_CLIP_MULTIPLIER,
#         )
#         t_dp = time.time() - t0
#         dp_times.append(t_dp)

#         labels_run = relabel_clusters(y_true, labels_run, merge_n_clusters)

#         run_ari, run_nmi = compute_metrics(y_true, labels_run)
#         run_mse = mode_matching_mse(
#             true_means_scaled.copy(),
#             pca_to_scaled(modes_run).copy(),
#         )

#         ari_list.append(run_ari)
#         nmi_list.append(run_nmi)
#         mse_list.append(run_mse)

#     dp_metrics["ARI"].append(float(np.mean(ari_list)))
#     dp_metrics["NMI"].append(float(np.mean(nmi_list)))
#     dp_metrics["MSE"].append(float(np.mean(mse_list)))
#     dp_err["ARI"].append(float(np.std(ari_list, ddof=1)))
#     dp_err["NMI"].append(float(np.std(nmi_list, ddof=1)))
#     dp_err["MSE"].append(float(np.std(mse_list, ddof=1)))
#     dp_times_mean.append(float(np.mean(dp_times)))
#     dp_times_std.append(float(np.std(dp_times, ddof=1)))

#     ari_list, nmi_list, mse_list = [], [], []
#     dpkm_times = []

#     for run in range(n_runs):
#         t0 = time.time()
#         dp_kmeans_run = fit_dpkmeans(
#             X,
#             merge_n_clusters,
#             eps,
#             base_rng_seed + run,
#         )
#         t_dpkm = time.time() - t0
#         dpkm_times.append(t_dpkm)

#         labels_dpkm_run = (
#             dp_kmeans_run.labels_.astype(int)
#             if hasattr(dp_kmeans_run, "labels_")
#             else dp_kmeans_run.predict(X)
#         )
#         labels_dpkm_run = relabel_clusters(y_true, labels_dpkm_run, merge_n_clusters)
#         modes_dpkm_run = dp_kmeans_run.cluster_centers_

#         run_ari, run_nmi = compute_metrics(y_true, labels_dpkm_run)
#         run_mse = mode_matching_mse(
#             true_means_scaled.copy(),
#             pca_to_scaled(modes_dpkm_run).copy(),
#         )

#         ari_list.append(run_ari)
#         nmi_list.append(run_nmi)
#         mse_list.append(run_mse)

#     dpkm_metrics["ARI"].append(float(np.mean(ari_list)))
#     dpkm_metrics["NMI"].append(float(np.mean(nmi_list)))
#     dpkm_metrics["MSE"].append(float(np.mean(mse_list)))
#     dpkm_err["ARI"].append(float(np.std(ari_list, ddof=1)))
#     dpkm_err["NMI"].append(float(np.std(nmi_list, ddof=1)))
#     dpkm_err["MSE"].append(float(np.std(mse_list, ddof=1)))
#     dpkm_times_mean.append(float(np.mean(dpkm_times)))
#     dpkm_times_std.append(float(np.std(dpkm_times, ddof=1)))

# # ---------------------------------------------------------------------
# # Save privacy-utility metrics
# # ---------------------------------------------------------------------
# csv_file = os.path.join(results_dir, "privacy_utility_metrics.csv")
# with open(csv_file, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow([
#         "Epsilon", "Algorithm",
#         "ARI", "NMI", "MSE",
#         "Std_ARI", "Std_NMI", "Std_MSE",
#         "Mean_Runtime(s)", "Std_Runtime(s)",
#         "dpgrams_c_clip_multiplier", "dpgrams_c_kappa_init",
#         "dpgrams_c_init_epsilon_frac", "dpgrams_c_c_rho",
#         "dpgrams_c_candidate_points",
#         "dpgrams_c_dap_init_strategy",
#         "pca_dim",
#         "pca_dim_rule",
#     ])
#     for i, eps in enumerate(epsilon_list):
#         writer.writerow([
#             eps, "DP-GRAMS-C",
#             dp_metrics["ARI"][i],
#             dp_metrics["NMI"][i],
#             dp_metrics["MSE"][i],
#             dp_err["ARI"][i],
#             dp_err["NMI"][i],
#             dp_err["MSE"][i],
#             dp_times_mean[i],
#             dp_times_std[i],
#             DEFAULT_CLIP_MULTIPLIER,
#             DEFAULT_KAPPA_INIT,
#             DEFAULT_INIT_EPSILON_FRAC,
#             DEFAULT_C_RHO,
#             "None",
#             DEFAULT_DAP_INIT_STRATEGY,
#             pca_dim,
#             PCA_DIM_SELECTION_RULE,
#         ])
#         writer.writerow([
#             eps, "DP-KMeans",
#             dpkm_metrics["ARI"][i],
#             dpkm_metrics["NMI"][i],
#             dpkm_metrics["MSE"][i],
#             dpkm_err["ARI"][i],
#             dpkm_err["NMI"][i],
#             dpkm_err["MSE"][i],
#             dpkm_times_mean[i],
#             dpkm_times_std[i],
#             "", "", "", "", "", "", pca_dim, PCA_DIM_SELECTION_RULE,
#         ])

# print("All Digits experiments completed. Results saved in:", results_dir)

# # ---------------------------------------------------------------------
# # Privacy-utility plots
# # ---------------------------------------------------------------------
# metrics_names = ["ARI", "NMI", "MSE"]

# for metric in metrics_names:
#     fig, ax = plt.subplots(figsize=(7, 5))

#     ax.errorbar(
#         epsilon_list,
#         dp_metrics[metric],
#         yerr=dp_err[metric],
#         marker="o",
#         linestyle="-",
#         linewidth=2,
#         markersize=7,
#         capsize=3,
#         label="DP-GRAMS-C",
#     )

#     ax.errorbar(
#         epsilon_list,
#         dpkm_metrics[metric],
#         yerr=dpkm_err[metric],
#         marker="s",
#         linestyle="-",
#         linewidth=2,
#         markersize=7,
#         capsize=3,
#         label="DP-KMeans",
#     )

#     ax.set_xscale("log")
#     ax.set_xlabel(r"$\epsilon$")
#     ax.set_ylabel(metric)
#     ax.set_title(f"Privacy-Utility: {metric} (Digits)")
#     ax.grid(True, linestyle="--", alpha=0.6)
#     ax.legend(loc="best")

#     plt.tight_layout()
#     out_path = os.path.join(
#         results_dir,
#         f"digits_privacy_utility_{metric.lower()}.pdf"
#     )
#     plt.savefig(out_path, dpi=120)
#     plt.show()
#     print(f"Privacy-utility ({metric}) figure saved to:", out_path)

# # ---------------------------------------------------------------------
# # Hyperparameter sweeps
# # ---------------------------------------------------------------------
# def _run_dpgrams_c_single(
#     epsilon_modes,
#     clip_multiplier,
#     m,
#     seed,
# ):
#     rng = np.random.default_rng(int(seed))
#     t0 = time.time()
#     modes_run, labels_run = run_dpgrams_c_digits(
#         data=X,
#         epsilon_modes=epsilon_modes,
#         h=h_base,
#         rng=rng,
#         m=m,
#         k_est=merge_n_clusters,
#         clip_multiplier=clip_multiplier,
#     )
#     runtime = time.time() - t0

#     if modes_run.size == 0:
#         return float("nan"), float("nan"), float("nan"), float(runtime)

#     labels_run = relabel_clusters(y_true, labels_run, merge_n_clusters)
#     ari, nmi = compute_metrics(y_true, labels_run)
#     mse = mode_matching_mse(
#         true_means_scaled.copy(),
#         np.asarray(modes_run, dtype=float).copy(),
#     )
#     return float(mse), float(ari), float(nmi), float(runtime)


# def sweep_clip_multiplier_digits(
#     clip_values,
#     epsilon,
#     n_reps=20,
#     base_seed=12345,
# ):
#     rng = np.random.default_rng(base_seed)
#     results = []

#     m_default = None

#     for cm in clip_values:
#         mses, aris, nmis, times = [], [], [], []
#         for _ in range(n_reps):
#             seed_run = int(rng.integers(0, 2**31 - 1))
#             mse, ari, nmi, rt = _run_dpgrams_c_single(
#                 epsilon_modes=epsilon,
#                 clip_multiplier=cm,
#                 m=m_default,
#                 seed=seed_run,
#             )
#             mses.append(mse)
#             aris.append(ari)
#             nmis.append(nmi)
#             times.append(rt)

#         mses = np.array(mses, dtype=float)
#         aris = np.array(aris, dtype=float)
#         nmis = np.array(nmis, dtype=float)
#         times = np.array(times, dtype=float)

#         results.append({
#             "clip_multiplier": float(cm),
#             "mean_mse": float(np.nanmean(mses)),
#             "std_mse": float(np.nanstd(mses)),
#             "mean_ari": float(np.nanmean(aris)),
#             "std_ari": float(np.nanstd(aris)),
#             "mean_nmi": float(np.nanmean(nmis)),
#             "std_nmi": float(np.nanstd(nmis)),
#             "mean_time": float(np.nanmean(times)),
#             "std_time": float(np.nanstd(times)),
#         })

#     results.sort(key=lambda r: r["clip_multiplier"])
#     return results


# def sweep_minibatch_digits(
#     m_frac_grid,
#     epsilon,
#     clip_multiplier_fixed=DEFAULT_CLIP_MULTIPLIER,
#     n_reps=20,
#     base_seed=54321,
# ):
#     rng = np.random.default_rng(base_seed)
#     results = []

#     n_samples = X.shape[0]
#     m_grid = sorted(set(max(1, int(frac * n_samples)) for frac in m_frac_grid))

#     for m_val in m_grid:
#         mses, aris, nmis, times = [], [], [], []
#         for _ in range(n_reps):
#             seed_run = int(rng.integers(0, 2**31 - 1))
#             mse, ari, nmi, rt = _run_dpgrams_c_single(
#                 epsilon_modes=epsilon,
#                 clip_multiplier=clip_multiplier_fixed,
#                 m=m_val,
#                 seed=seed_run,
#             )
#             mses.append(mse)
#             aris.append(ari)
#             nmis.append(nmi)
#             times.append(rt)

#         mses = np.array(mses, dtype=float)
#         aris = np.array(aris, dtype=float)
#         nmis = np.array(nmis, dtype=float)
#         times = np.array(times, dtype=float)

#         results.append({
#             "m": int(m_val),
#             "mean_mse": float(np.nanmean(mses)),
#             "std_mse": float(np.nanstd(mses)),
#             "mean_ari": float(np.nanmean(aris)),
#             "std_ari": float(np.nanstd(aris)),
#             "mean_nmi": float(np.nanmean(nmis)),
#             "std_nmi": float(np.nanstd(nmis)),
#             "mean_time": float(np.nanmean(times)),
#             "std_time": float(np.nanstd(times)),
#         })

#     results.sort(key=lambda r: r["m"])
#     return results


# print("\n[hyperparam] Running DP-GRAMS-C hyperparameter sweeps on Digits...")

# clip_results = sweep_clip_multiplier_digits(
#     clip_values=clip_grid,
#     epsilon=epsilon_hparam,
#     n_reps=n_reps_hparam,
#     base_seed=12345,
# )

# m_results = sweep_minibatch_digits(
#     m_frac_grid=m_frac_grid,
#     epsilon=epsilon_hparam,
#     clip_multiplier_fixed=DEFAULT_CLIP_MULTIPLIER,
#     n_reps=n_reps_hparam,
#     base_seed=54321,
# )

# # ---------------------------------------------------------------------
# # Save hyperparameter sweep results
# # ---------------------------------------------------------------------
# clip_txt_path = os.path.join(results_dir, "dpgrams_c_vs_clip_multiplier_digits.txt")
# with open(clip_txt_path, "w") as f:
#     header = (
#         f"{'clip_mult':>10} | "
#         f"{'mean_mse':>10} | {'std_mse':>10} | "
#         f"{'mean_ari':>10} | {'std_ari':>10} | "
#         f"{'mean_nmi':>10} | {'std_nmi':>10} | "
#         f"{'mean_t':>10} | {'std_t':>10}"
#     )
#     f.write(header + "\n")
#     f.write("-" * len(header) + "\n")
#     for r in clip_results:
#         line = (
#             f"{r['clip_multiplier']:10.3f} | "
#             f"{r['mean_mse']:10.4f} | {r['std_mse']:10.4f} | "
#             f"{r['mean_ari']:10.4f} | {r['std_ari']:10.4f} | "
#             f"{r['mean_nmi']:10.4f} | {r['std_nmi']:10.4f} | "
#             f"{r['mean_time']:10.4f} | {r['std_time']:10.4f}"
#         )
#         f.write(line + "\n")
# print(f"[saved] Clip-multiplier sweep results -> {clip_txt_path}")

# m_txt_path = os.path.join(results_dir, "dpgrams_c_vs_minibatch_digits.txt")
# with open(m_txt_path, "w") as f:
#     header = (
#         f"{'m':>10} | "
#         f"{'mean_mse':>10} | {'std_mse':>10} | "
#         f"{'mean_ari':>10} | {'std_ari':>10} | "
#         f"{'mean_nmi':>10} | {'std_nmi':>10} | "
#         f"{'mean_t':>10} | {'std_t':>10}"
#     )
#     f.write(header + "\n")
#     f.write("-" * len(header) + "\n")
#     for r in m_results:
#         line = (
#             f"{r['m']:10d} | "
#             f"{r['mean_mse']:10.4f} | {r['std_mse']:10.4f} | "
#             f"{r['mean_ari']:10.4f} | {r['std_ari']:10.4f} | "
#             f"{r['mean_nmi']:10.4f} | {r['std_nmi']:10.4f} | "
#             f"{r['mean_time']:10.4f} | {r['std_time']:10.4f}"
#         )
#         f.write(line + "\n")
# print(f"[saved] Minibatch sweep results -> {m_txt_path}")

# # ---------------------------------------------------------------------
# # Plots: MSE / ARI / NMI vs C^*
# # ---------------------------------------------------------------------
# clip_x = [r["clip_multiplier"] for r in clip_results]

# fig, ax = plt.subplots(figsize=(7, 5))
# ax.errorbar(
#     clip_x,
#     [r["mean_mse"] for r in clip_results],
#     yerr=[r["std_mse"] for r in clip_results],
#     marker="o",
#     linestyle="-",
#     linewidth=2,
#     markersize=7,
#     capsize=3,
# )
# ax.set_xlabel(r"Clip Multiplier ($C^*$)")
# ax.set_ylabel("Centroid MSE")
# ax.set_title("DP-GRAMS-C on Digits: MSE vs $C^*$")
# ax.grid(True, alpha=0.4)
# plt.tight_layout()
# clip_plot_path_mse = os.path.join(results_dir, "digits_dpgrams_c_mse_vs_clip_multiplier.pdf")
# plt.savefig(clip_plot_path_mse, dpi=120)
# plt.show()
# print(f"[saved] MSE vs C^* plot -> {clip_plot_path_mse}")

# fig, ax = plt.subplots(figsize=(7, 5))
# ax.errorbar(
#     clip_x,
#     [r["mean_ari"] for r in clip_results],
#     yerr=[r["std_ari"] for r in clip_results],
#     marker="o",
#     linestyle="-",
#     linewidth=2,
#     markersize=7,
#     capsize=3,
# )
# ax.set_xlabel(r"Clip Multiplier ($C^*$)")
# ax.set_ylabel("ARI")
# ax.set_title("DP-GRAMS-C on Digits: ARI vs $C^*$")
# ax.grid(True, alpha=0.4)
# plt.tight_layout()
# clip_plot_path_ari = os.path.join(results_dir, "digits_dpgrams_c_ari_vs_clip_multiplier.pdf")
# plt.savefig(clip_plot_path_ari, dpi=120)
# plt.show()
# print(f"[saved] ARI vs C^* plot -> {clip_plot_path_ari}")

# fig, ax = plt.subplots(figsize=(7, 5))
# ax.errorbar(
#     clip_x,
#     [r["mean_nmi"] for r in clip_results],
#     yerr=[r["std_nmi"] for r in clip_results],
#     marker="o",
#     linestyle="-",
#     linewidth=2,
#     markersize=7,
#     capsize=3,
# )
# ax.set_xlabel(r"Clip Multiplier ($C^*$)")
# ax.set_ylabel("NMI")
# ax.set_title("DP-GRAMS-C on Digits: NMI vs $C^*$")
# ax.grid(True, alpha=0.4)
# plt.tight_layout()
# clip_plot_path_nmi = os.path.join(results_dir, "digits_dpgrams_c_nmi_vs_clip_multiplier.pdf")
# plt.savefig(clip_plot_path_nmi, dpi=120)
# plt.show()
# print(f"[saved] NMI vs C^* plot -> {clip_plot_path_nmi}")

# # ---------------------------------------------------------------------
# # Plots: MSE / ARI / NMI vs m
# # ---------------------------------------------------------------------
# m_x = [r["m"] for r in m_results]

# fig, ax = plt.subplots(figsize=(7, 5))
# ax.errorbar(
#     m_x,
#     [r["mean_mse"] for r in m_results],
#     yerr=[r["std_mse"] for r in m_results],
#     marker="o",
#     linestyle="-",
#     linewidth=2,
#     markersize=7,
#     capsize=3,
# )
# ax.set_xscale("log")
# ax.set_xlabel("Minibatch size m")
# ax.set_ylabel("Centroid MSE")
# ax.set_title("DP-GRAMS-C on Digits: MSE vs m")
# ax.grid(True, alpha=0.4)
# plt.tight_layout()
# m_plot_path_mse = os.path.join(results_dir, "digits_dpgrams_c_mse_vs_minibatch.pdf")
# plt.savefig(m_plot_path_mse, dpi=120)
# plt.show()
# print(f"[saved] MSE vs m plot -> {m_plot_path_mse}")

# fig, ax = plt.subplots(figsize=(7, 5))
# ax.errorbar(
#     m_x,
#     [r["mean_ari"] for r in m_results],
#     yerr=[r["std_ari"] for r in m_results],
#     marker="o",
#     linestyle="-",
#     linewidth=2,
#     markersize=7,
#     capsize=3,
# )
# ax.set_xscale("log")
# ax.set_xlabel("Minibatch size m")
# ax.set_ylabel("ARI")
# ax.set_title("DP-GRAMS-C on Digits: ARI vs m")
# ax.grid(True, alpha=0.4)
# plt.tight_layout()
# m_plot_path_ari = os.path.join(results_dir, "digits_dpgrams_c_ari_vs_minibatch.pdf")
# plt.savefig(m_plot_path_ari, dpi=120)
# plt.show()
# print(f"[saved] ARI vs m plot -> {m_plot_path_ari}")

# fig, ax = plt.subplots(figsize=(7, 5))
# ax.errorbar(
#     m_x,
#     [r["mean_nmi"] for r in m_results],
#     yerr=[r["std_nmi"] for r in m_results],
#     marker="o",
#     linestyle="-",
#     linewidth=2,
#     markersize=7,
#     capsize=3,
# )
# ax.set_xscale("log")
# ax.set_xlabel("Minibatch size m")
# ax.set_ylabel("NMI")
# ax.set_title("DP-GRAMS-C on Digits: NMI vs m")
# ax.grid(True, alpha=0.4)
# plt.tight_layout()
# m_plot_path_nmi = os.path.join(results_dir, "digits_dpgrams_c_nmi_vs_minibatch.pdf")
# plt.savefig(m_plot_path_nmi, dpi=120)
# plt.show()
# print(f"[saved] NMI vs m plot -> {m_plot_path_nmi}")

import csv
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from diffprivlib.models import KMeans as DPKMeans
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main_scripts.bandwidth import silverman_bandwidth
from main_scripts.mode_matching_mse import mode_matching_mse
from main_scripts.ms import mean_shift
from main_scripts.merge import merge_modes_agglomerative
from real_data_scripts.dp_grams_c import dpms_private

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
results_dir = "results/digits_centroid_comparison"
os.makedirs(results_dir, exist_ok=True)

sns.set(style="whitegrid", context="talk")

digits = load_digits()
X_raw = digits.data
y_true = digits.target

n, d = X_raw.shape

merge_n_clusters = 10
ms_T = int(np.ceil(np.log(max(2, n))))
ms_p = 0.1
delta = 1e-5
base_rng_seed = 41
n_runs = 20

# Include the main default value in the sweep grid.
clip_grid = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
m_frac_grid = [0.05, 0.1, 0.2, 0.5, 1.0]
n_reps_hparam = 20
epsilon_hparam = 1.0

# ---------------------------------------------------------------------
# Explicit DP-GRAMS-C defaults
# ---------------------------------------------------------------------
# Keep these explicit so this script never silently inherits dp_grams / dp_grams_c
# defaults. In particular, candidate_points stays None, so DP-GRAMS-C uses its
# default public-box DAP grid and does NOT use data points as candidates.
#
# Fast/simple digits fix:
#   standardize -> PCA dimension selected by spectrum -> cap at 3 -> grid DAP.
DEFAULT_CLIP_MULTIPLIER = 1
DEFAULT_KAPPA_INIT = 3
DEFAULT_INIT_EPSILON_FRAC = 0.3
DEFAULT_ETA = 1.0
DEFAULT_BETA_DAP = 3.0
DEFAULT_C_RHO = 2.0
DEFAULT_DAP_SCORE_MULTIPLIER = 10.0
DEFAULT_R = None
DEFAULT_CANDIDATE_POINTS = None
DEFAULT_DAP_INIT_STRATEGY = "grid"

# PCA dimension selection. The value of r is computed from the PCA spectrum;
# it is not hard-coded. The capped broken-stick rule is the default for digits.
PCA_DIM_SELECTION_RULE = "broken_stick_dp_capped"
PCA_R_MIN = 2
PCA_R_MAX = 3

# ---------------------------------------------------------------------
# Data preprocessing and automatic PCA dimension selection
# ---------------------------------------------------------------------
def _safe_rank_from_count(count, *, min_rank=1, max_rank=None):
    if max_rank is None:
        max_rank = int(count)
    return int(max(min_rank, min(int(count), int(max_rank))))


def select_pca_dimension(
    X_scaled,
    *,
    rule="broken_stick_dp_capped",
    r_min=2,
    r_max=3,
):
    """
    Compute several PCA dimension rules and return the selected dimension.

    The selected dimension is computed from the spectrum. It is not hard-coded.
    The default rule uses broken-stick as a data-adaptive spectral estimate and
    then applies a DP-GRAMS-C cap to avoid public grid explosion.
    """
    X_scaled = np.asarray(X_scaled, dtype=float)
    pca_probe = PCA(random_state=base_rng_seed)
    pca_probe.fit(X_scaled)

    eigvals = np.asarray(pca_probe.explained_variance_, dtype=float)
    evr = np.asarray(pca_probe.explained_variance_ratio_, dtype=float)
    p = int(eigvals.size)

    if p == 0:
        raise ValueError("PCA spectrum is empty.")

    # Ahn-Horenstein eigenvalue ratio: argmax lambda_j / lambda_{j+1}.
    if p >= 2:
        ratios = eigvals[:-1] / np.maximum(eigvals[1:], 1e-12)
        ahn_horenstein = int(np.argmax(ratios) + 1)
    else:
        ahn_horenstein = 1

    # Broken-stick: keep components whose explained-variance share exceeds the
    # broken-stick expected share.
    harmonic_tail = np.array([
        np.sum(1.0 / np.arange(j, p + 1, dtype=float)) / float(p)
        for j in range(1, p + 1)
    ])
    broken_stick = int(np.sum(evr > harmonic_tail))
    broken_stick = max(1, broken_stick)

    # Kaiser rule for standardized variables: eigenvalue above average.
    kaiser = int(np.sum(eigvals > np.mean(eigvals)))
    kaiser = max(1, kaiser)

    # Effective-rank style summaries.
    eig_sum = float(np.sum(eigvals))
    if eig_sum <= 0.0:
        participation_rank_float = 1.0
        entropy_rank_float = 1.0
    else:
        participation_rank_float = float((eig_sum ** 2) / np.sum(eigvals ** 2))
        probs = eigvals / eig_sum
        probs_pos = probs[probs > 0]
        entropy_rank_float = float(np.exp(-np.sum(probs_pos * np.log(probs_pos))))

    participation_rank = int(np.ceil(participation_rank_float))
    entropy_rank = int(np.ceil(entropy_rank_float))

    cum = np.cumsum(evr)
    r80 = int(np.searchsorted(cum, 0.80) + 1)
    r85 = int(np.searchsorted(cum, 0.85) + 1)
    r90 = int(np.searchsorted(cum, 0.90) + 1)

    candidates = {
        "ahn_horenstein": ahn_horenstein,
        "broken_stick": broken_stick,
        "kaiser": kaiser,
        "participation_rank": participation_rank,
        "entropy_rank": entropy_rank,
        "var80": r80,
        "var85": r85,
        "var90": r90,
    }

    if rule in {"broken_stick_capped", "broken_stick_dp_capped"}:
        raw_selected = candidates["broken_stick"]
    elif rule == "ahn_horenstein_capped":
        raw_selected = candidates["ahn_horenstein"]
    elif rule == "participation_capped":
        raw_selected = candidates["participation_rank"]
    elif rule == "kaiser_capped":
        raw_selected = candidates["kaiser"]
    elif rule == "var80_capped":
        raw_selected = candidates["var80"]
    elif rule == "var85_capped":
        raw_selected = candidates["var85"]
    elif rule == "var90_capped":
        raw_selected = candidates["var90"]
    else:
        raise ValueError(f"Unknown PCA_DIM_SELECTION_RULE={rule!r}.")

    selected = _safe_rank_from_count(
        raw_selected,
        min_rank=int(r_min),
        max_rank=min(int(r_max), p),
    )

    diagnostics = {
        "rule": str(rule),
        "r_min": int(r_min),
        "r_max": int(r_max),
        "selected_raw": int(raw_selected),
        "selected": int(selected),
        "ahn_horenstein": int(ahn_horenstein),
        "broken_stick": int(broken_stick),
        "kaiser": int(kaiser),
        "participation_rank_float": float(participation_rank_float),
        "participation_rank": int(participation_rank),
        "entropy_rank_float": float(entropy_rank_float),
        "entropy_rank": int(entropy_rank),
        "var80": int(r80),
        "var85": int(r85),
        "var90": int(r90),
        "cumvar_selected": float(cum[selected - 1]),
        "top_eigenvalues": eigvals[: min(10, p)].copy(),
        "top_explained_variance_ratio": evr[: min(10, p)].copy(),
    }
    return selected, diagnostics, pca_probe


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

true_means_scaled = np.array([
    X_scaled[y_true == c].mean(axis=0)
    for c in np.unique(y_true)
])

pca_dim, pca_dim_info, _pca_probe = select_pca_dimension(
    X_scaled,
    rule=PCA_DIM_SELECTION_RULE,
    r_min=PCA_R_MIN,
    r_max=PCA_R_MAX,
)

print("[pca] PCA dimension diagnostics:")
print(f"[pca] Ahn-Horenstein eigenvalue-ratio r = {pca_dim_info['ahn_horenstein']}")
print(f"[pca] Broken-stick r = {pca_dim_info['broken_stick']}")
print(f"[pca] Kaiser r = {pca_dim_info['kaiser']}")
print(
    "[pca] Participation effective rank = "
    f"{pca_dim_info['participation_rank_float']:.3f} "
    f"-> {pca_dim_info['participation_rank']}"
)
print(
    "[pca] Entropy effective rank = "
    f"{pca_dim_info['entropy_rank_float']:.3f} "
    f"-> {pca_dim_info['entropy_rank']}"
)
print(f"[pca] 80% variance r = {pca_dim_info['var80']}")
print(f"[pca] 85% variance r = {pca_dim_info['var85']}")
print(f"[pca] 90% variance r = {pca_dim_info['var90']}")
print(
    f"[pca] selected rule = {pca_dim_info['rule']}, "
    f"raw r = {pca_dim_info['selected_raw']}, "
    f"capped selected r = {pca_dim_info['selected']}, "
    f"cumvar = {pca_dim_info['cumvar_selected']:.4f}"
)

pca_shared = PCA(n_components=pca_dim, random_state=base_rng_seed)
X = pca_shared.fit_transform(X_scaled)

true_means = np.array([
    X[y_true == c].mean(axis=0)
    for c in np.unique(y_true)
])


def pca_to_scaled(centers):
    centers = np.asarray(centers, dtype=float)
    if centers.size == 0:
        return np.empty((0, X_scaled.shape[1]), dtype=float)
    centers = np.atleast_2d(centers)
    return pca_shared.inverse_transform(centers)


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


def relabel_clusters(y_true, y_pred, n_clusters=10):
    cm = confusion_matrix(y_true, y_pred, labels=range(n_clusters))
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping.get(label, label) for label in y_pred], dtype=int)


def fit_dpkmeans(X_input, n_clusters, epsilon, random_state):
    model = DPKMeans(
        n_clusters=n_clusters,
        epsilon=epsilon,
        random_state=random_state,
    )
    try:
        model.fit(X_input)
        return model
    except Exception:
        mins_local = X_input.min(axis=0)
        maxs_local = X_input.max(axis=0)
        model = DPKMeans(
            n_clusters=n_clusters,
            epsilon=epsilon,
            bounds=(mins_local, maxs_local),
            random_state=random_state,
        )
        model.fit(X_input)
        return model


def run_dpgrams_c_digits(
    data,
    epsilon_modes,
    h,
    rng,
    *,
    k_est,
    clip_multiplier=DEFAULT_CLIP_MULTIPLIER,
    m=None,
    return_info=False,
):
    """
    Single DP-GRAMS-C call with all experiment defaults explicit.

    candidate_points is intentionally None: the wrapper calls dp_grams with the
    default public candidate grid. No data points are passed as DAP candidates.
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
        dap_init_strategy=DEFAULT_DAP_INIT_STRATEGY,
        R=DEFAULT_R,
        candidate_points=DEFAULT_CANDIDATE_POINTS,
        return_info=return_info,
    )


def make_2d_view(X_in, centers_list, true_means_in, seed):
    X_in = np.asarray(X_in, dtype=float)
    true_means_in = np.asarray(true_means_in, dtype=float)

    if X_in.shape[1] >= 2:
        pca2 = PCA(n_components=2, random_state=seed)
        X_2d = pca2.fit_transform(X_in)
        true_means_2d = pca2.transform(true_means_in)
        centers_2d_list = [pca2.transform(np.asarray(m, dtype=float)) for m in centers_list]
    else:
        X_2d = np.column_stack([X_in[:, 0], np.zeros(X_in.shape[0])])
        true_means_2d = np.column_stack([true_means_in[:, 0], np.zeros(true_means_in.shape[0])])
        centers_2d_list = [
            np.column_stack([np.asarray(m, dtype=float)[:, 0], np.zeros(np.asarray(m).shape[0])])
            for m in centers_list
        ]

    return X_2d, true_means_2d, centers_2d_list


# ---------------------------------------------------------------------
# Bandwidth
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
rng_dp = np.random.default_rng(base_rng_seed)
(((modes_dp, labels_dp), h_used, init_info), dp_time) = timed(
    run_dpgrams_c_digits,
    data=X,
    epsilon_modes=1.0,
    h=h_base,
    rng=rng_dp,
    k_est=merge_n_clusters,
    clip_multiplier=DEFAULT_CLIP_MULTIPLIER,
    return_info=True,
)
labels_dp = relabel_clusters(y_true, labels_dp, merge_n_clusters)

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
    merge_n_clusters,
    1.0,
    base_rng_seed,
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
labels_list = [labels_ms, labels_dp, labels_km, labels_dpkm]
modes_list = [ms_merged_modes, modes_dp, modes_km, modes_dpkm]
runtimes = [ms_time, dp_time, km_time, dpkm_time]

metrics = {}
mse_centroids = {}

for alg, labels, modes in zip(algorithms, labels_list, modes_list):
    metrics[alg] = compute_metrics(y_true, labels)
    mse_centroids[alg] = mode_matching_mse(
        true_means_scaled.copy(),
        pca_to_scaled(modes).copy(),
    )

# ---------------------------------------------------------------------
# Save metrics
# ---------------------------------------------------------------------
csv_file = os.path.join(results_dir, "clustering_metrics.csv")
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Algorithm", "ARI", "NMI", "MSE", "Runtime(s)"])
    for alg, rt in zip(algorithms, runtimes):
        ari, nmi = metrics[alg]
        writer.writerow([alg, ari, nmi, mse_centroids[alg], rt])

txt_file = os.path.join(results_dir, "clustering_metrics.txt")
with open(txt_file, "w") as f:
    f.write("Digits comparison with standardized raw features and shared PCA clustering space\n")
    f.write(f"n = {n}, raw d = {d}, selected PCA r = {pca_dim}\n")
    f.write(f"PCA dimension rule = {PCA_DIM_SELECTION_RULE}\n")
    f.write(f"PCA selected raw r = {pca_dim_info['selected_raw']}\n")
    f.write(f"PCA selected capped r = {pca_dim_info['selected']}\n")
    f.write(f"PCA selected cumulative variance = {pca_dim_info['cumvar_selected']:.6f}\n")
    f.write(f"Ahn-Horenstein r = {pca_dim_info['ahn_horenstein']}\n")
    f.write(f"Broken-stick r = {pca_dim_info['broken_stick']}\n")
    f.write(f"Kaiser r = {pca_dim_info['kaiser']}\n")
    f.write(f"Participation effective rank = {pca_dim_info['participation_rank_float']:.6f} -> {pca_dim_info['participation_rank']}\n")
    f.write(f"Entropy effective rank = {pca_dim_info['entropy_rank_float']:.6f} -> {pca_dim_info['entropy_rank']}\n")
    f.write(f"Variance ranks: r80={pca_dim_info['var80']}, r85={pca_dim_info['var85']}, r90={pca_dim_info['var90']}\n")
    f.write("All methods clustered on the shared PCA representation.\n")
    f.write("Centroid MSE is computed after inverse PCA back to standardized raw feature space.\n")
    f.write("DP-GRAMS-C candidate_points = None; default public-box grid DAP is used.\n")
    f.write(f"DP-GRAMS-C internal d_ambient = {int(init_info.d_ambient)}\n")
    f.write(f"DP-GRAMS-C init strategy = {getattr(init_info, 'dap_init_strategy', DEFAULT_DAP_INIT_STRATEGY)}\n")
    f.write(f"Silverman bandwidth h_base = {float(h_base):.8f}\n")
    f.write(f"DP-GRAMS h_used = {float(h_used):.8f}\n")
    f.write(
        "DP-GRAMS-C settings: indicator utility, logarithmic suppression, "
        f"kappa_init={DEFAULT_KAPPA_INIT}, "
        f"init_epsilon_frac={DEFAULT_INIT_EPSILON_FRAC}, "
        f"clip_multiplier={DEFAULT_CLIP_MULTIPLIER}, "
        f"c_rho={DEFAULT_C_RHO}, "
        f"dap_score_multiplier={DEFAULT_DAP_SCORE_MULTIPLIER}, "
        f"R={DEFAULT_R}, "
        f"dap_init_strategy={DEFAULT_DAP_INIT_STRATEGY}\n"
    )
    for alg, rt in zip(algorithms, runtimes):
        ari, nmi = metrics[alg]
        f.write(
            f"{alg}: ARI={ari:.4f}, NMI={nmi:.4f}, "
            f"MSE={mse_centroids[alg]:.6f}, Runtime={rt:.4f}s\n"
        )

print("\nMetrics (clustering on shared PCA representation; MSE in standardized raw space):")
print(f"{'Alg':<14} {'ARI':>6} {'NMI':>6} {'MSE_centroids':>14} {'Runtime(s)':>12}")
for alg, rt in zip(algorithms, runtimes):
    ari, nmi = metrics[alg]
    print(
        f"{alg:<14} {ari:6.3f} {nmi:6.3f} "
        f"{mse_centroids[alg]:14.6f} {rt:12.4f}"
    )

# ---------------------------------------------------------------------
# 2D visualization
# ---------------------------------------------------------------------
X_2d, true_means_2d, modes_2d_list = make_2d_view(
    X,
    modes_list,
    true_means,
    seed=base_rng_seed,
)

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
            s=15,
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
        s=140,
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
        s=100,
        linewidths=2,
    )
    if "Estimated modes" not in global_labels:
        global_handles.append(modes_sc)
        global_labels.append("Estimated modes")

    ax.set_title(alg, fontsize=14)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

legend = fig.legend(
    global_handles,
    global_labels,
    fontsize=9,
    loc="lower center",
    ncol=6,
    bbox_to_anchor=(0.5, 0.01),
    title="Cluster Assignments & Centroids",
)
plt.setp(legend.get_title(), fontsize=11, fontweight="bold")
fig.suptitle("Clustering Comparison on Digits Dataset", fontsize=18, y=0.92)

save_path_clusters = os.path.join(results_dir, "digits_clustering_comparison.pdf")
plt.tight_layout(rect=[0, 0.05, 1, 0.93])
plt.savefig(save_path_clusters, dpi=120)
plt.show()
print("Clustering comparison plot saved to:", save_path_clusters)

# ---------------------------------------------------------------------
# Privacy-utility curves
# ---------------------------------------------------------------------
epsilon_list = [0.25, 0.5, 1.0, 2.5, 5.0]

dp_metrics = {m: [] for m in ["ARI", "NMI", "MSE"]}
dpkm_metrics = {m: [] for m in ["ARI", "NMI", "MSE"]}
dp_err = {m: [] for m in ["ARI", "NMI", "MSE"]}
dpkm_err = {m: [] for m in ["ARI", "NMI", "MSE"]}
dp_times_mean, dpkm_times_mean = [], []
dp_times_std, dpkm_times_std = [], []

for eps in epsilon_list:
    ari_list, nmi_list, mse_list = [], [], []
    dp_times = []

    for run in range(n_runs):
        rng_run = np.random.default_rng(base_rng_seed + run)

        t0 = time.time()
        modes_run, labels_run = run_dpgrams_c_digits(
            data=X,
            epsilon_modes=eps,
            h=h_base,
            rng=rng_run,
            k_est=merge_n_clusters,
            clip_multiplier=DEFAULT_CLIP_MULTIPLIER,
        )
        t_dp = time.time() - t0
        dp_times.append(t_dp)

        labels_run = relabel_clusters(y_true, labels_run, merge_n_clusters)

        run_ari, run_nmi = compute_metrics(y_true, labels_run)
        run_mse = mode_matching_mse(
            true_means_scaled.copy(),
            pca_to_scaled(modes_run).copy(),
        )

        ari_list.append(run_ari)
        nmi_list.append(run_nmi)
        mse_list.append(run_mse)

    dp_metrics["ARI"].append(float(np.mean(ari_list)))
    dp_metrics["NMI"].append(float(np.mean(nmi_list)))
    dp_metrics["MSE"].append(float(np.mean(mse_list)))
    dp_err["ARI"].append(float(np.std(ari_list, ddof=1)))
    dp_err["NMI"].append(float(np.std(nmi_list, ddof=1)))
    dp_err["MSE"].append(float(np.std(mse_list, ddof=1)))
    dp_times_mean.append(float(np.mean(dp_times)))
    dp_times_std.append(float(np.std(dp_times, ddof=1)))

    ari_list, nmi_list, mse_list = [], [], []
    dpkm_times = []

    for run in range(n_runs):
        t0 = time.time()
        dp_kmeans_run = fit_dpkmeans(
            X,
            merge_n_clusters,
            eps,
            base_rng_seed + run,
        )
        t_dpkm = time.time() - t0
        dpkm_times.append(t_dpkm)

        labels_dpkm_run = (
            dp_kmeans_run.labels_.astype(int)
            if hasattr(dp_kmeans_run, "labels_")
            else dp_kmeans_run.predict(X)
        )
        labels_dpkm_run = relabel_clusters(y_true, labels_dpkm_run, merge_n_clusters)
        modes_dpkm_run = dp_kmeans_run.cluster_centers_

        run_ari, run_nmi = compute_metrics(y_true, labels_dpkm_run)
        run_mse = mode_matching_mse(
            true_means_scaled.copy(),
            pca_to_scaled(modes_dpkm_run).copy(),
        )

        ari_list.append(run_ari)
        nmi_list.append(run_nmi)
        mse_list.append(run_mse)

    dpkm_metrics["ARI"].append(float(np.mean(ari_list)))
    dpkm_metrics["NMI"].append(float(np.mean(nmi_list)))
    dpkm_metrics["MSE"].append(float(np.mean(mse_list)))
    dpkm_err["ARI"].append(float(np.std(ari_list, ddof=1)))
    dpkm_err["NMI"].append(float(np.std(nmi_list, ddof=1)))
    dpkm_err["MSE"].append(float(np.std(mse_list, ddof=1)))
    dpkm_times_mean.append(float(np.mean(dpkm_times)))
    dpkm_times_std.append(float(np.std(dpkm_times, ddof=1)))

# ---------------------------------------------------------------------
# Save privacy-utility metrics
# ---------------------------------------------------------------------
csv_file = os.path.join(results_dir, "privacy_utility_metrics.csv")
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Epsilon", "Algorithm",
        "ARI", "NMI", "MSE",
        "Std_ARI", "Std_NMI", "Std_MSE",
        "Mean_Runtime(s)", "Std_Runtime(s)",
        "dpgrams_c_clip_multiplier", "dpgrams_c_kappa_init",
        "dpgrams_c_init_epsilon_frac", "dpgrams_c_c_rho",
        "dpgrams_c_candidate_points",
        "dpgrams_c_dap_init_strategy",
        "pca_dim",
        "pca_dim_rule",
    ])
    for i, eps in enumerate(epsilon_list):
        writer.writerow([
            eps, "DP-GRAMS-C",
            dp_metrics["ARI"][i],
            dp_metrics["NMI"][i],
            dp_metrics["MSE"][i],
            dp_err["ARI"][i],
            dp_err["NMI"][i],
            dp_err["MSE"][i],
            dp_times_mean[i],
            dp_times_std[i],
            DEFAULT_CLIP_MULTIPLIER,
            DEFAULT_KAPPA_INIT,
            DEFAULT_INIT_EPSILON_FRAC,
            DEFAULT_C_RHO,
            "None",
            DEFAULT_DAP_INIT_STRATEGY,
            pca_dim,
            PCA_DIM_SELECTION_RULE,
        ])
        writer.writerow([
            eps, "DP-KMeans",
            dpkm_metrics["ARI"][i],
            dpkm_metrics["NMI"][i],
            dpkm_metrics["MSE"][i],
            dpkm_err["ARI"][i],
            dpkm_err["NMI"][i],
            dpkm_err["MSE"][i],
            dpkm_times_mean[i],
            dpkm_times_std[i],
            "", "", "", "", "", "", pca_dim, PCA_DIM_SELECTION_RULE,
        ])

print("All Digits experiments completed. Results saved in:", results_dir)

# ---------------------------------------------------------------------
# Privacy-utility plots
# ---------------------------------------------------------------------
metrics_names = ["ARI", "NMI", "MSE"]

for metric in metrics_names:
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(
        epsilon_list,
        dp_metrics[metric],
        yerr=dp_err[metric],
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=7,
        capsize=3,
        label="DP-GRAMS-C",
    )

    ax.errorbar(
        epsilon_list,
        dpkm_metrics[metric],
        yerr=dpkm_err[metric],
        marker="s",
        linestyle="-",
        linewidth=2,
        markersize=7,
        capsize=3,
        label="DP-KMeans",
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(metric)
    ax.set_title(f"Privacy-Utility: {metric} (Digits)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")

    plt.tight_layout()
    out_path = os.path.join(
        results_dir,
        f"digits_privacy_utility_{metric.lower()}.pdf"
    )
    plt.savefig(out_path, dpi=120)
    plt.show()
    print(f"Privacy-utility ({metric}) figure saved to:", out_path)

# ---------------------------------------------------------------------
# Hyperparameter sweeps
# ---------------------------------------------------------------------
def _run_dpgrams_c_single(
    epsilon_modes,
    clip_multiplier,
    m,
    seed,
):
    rng = np.random.default_rng(int(seed))
    t0 = time.time()
    modes_run, labels_run = run_dpgrams_c_digits(
        data=X,
        epsilon_modes=epsilon_modes,
        h=h_base,
        rng=rng,
        m=m,
        k_est=merge_n_clusters,
        clip_multiplier=clip_multiplier,
    )
    runtime = time.time() - t0

    if modes_run.size == 0:
        return float("nan"), float("nan"), float("nan"), float(runtime)

    labels_run = relabel_clusters(y_true, labels_run, merge_n_clusters)
    ari, nmi = compute_metrics(y_true, labels_run)
    mse = mode_matching_mse(
        true_means_scaled.copy(),
        pca_to_scaled(modes_run).copy(),
    )
    return float(mse), float(ari), float(nmi), float(runtime)


def sweep_clip_multiplier_digits(
    clip_values,
    epsilon,
    n_reps=20,
    base_seed=12345,
):
    rng = np.random.default_rng(base_seed)
    results = []

    m_default = None

    for cm in clip_values:
        mses, aris, nmis, times = [], [], [], []
        for _ in range(n_reps):
            seed_run = int(rng.integers(0, 2**31 - 1))
            mse, ari, nmi, rt = _run_dpgrams_c_single(
                epsilon_modes=epsilon,
                clip_multiplier=cm,
                m=m_default,
                seed=seed_run,
            )
            mses.append(mse)
            aris.append(ari)
            nmis.append(nmi)
            times.append(rt)

        mses = np.array(mses, dtype=float)
        aris = np.array(aris, dtype=float)
        nmis = np.array(nmis, dtype=float)
        times = np.array(times, dtype=float)

        results.append({
            "clip_multiplier": float(cm),
            "mean_mse": float(np.nanmean(mses)),
            "std_mse": float(np.nanstd(mses)),
            "mean_ari": float(np.nanmean(aris)),
            "std_ari": float(np.nanstd(aris)),
            "mean_nmi": float(np.nanmean(nmis)),
            "std_nmi": float(np.nanstd(nmis)),
            "mean_time": float(np.nanmean(times)),
            "std_time": float(np.nanstd(times)),
        })

    results.sort(key=lambda r: r["clip_multiplier"])
    return results


def sweep_minibatch_digits(
    m_frac_grid,
    epsilon,
    clip_multiplier_fixed=DEFAULT_CLIP_MULTIPLIER,
    n_reps=20,
    base_seed=54321,
):
    rng = np.random.default_rng(base_seed)
    results = []

    n_samples = X.shape[0]
    m_grid = sorted(set(max(1, int(frac * n_samples)) for frac in m_frac_grid))

    for m_val in m_grid:
        mses, aris, nmis, times = [], [], [], []
        for _ in range(n_reps):
            seed_run = int(rng.integers(0, 2**31 - 1))
            mse, ari, nmi, rt = _run_dpgrams_c_single(
                epsilon_modes=epsilon,
                clip_multiplier=clip_multiplier_fixed,
                m=m_val,
                seed=seed_run,
            )
            mses.append(mse)
            aris.append(ari)
            nmis.append(nmi)
            times.append(rt)

        mses = np.array(mses, dtype=float)
        aris = np.array(aris, dtype=float)
        nmis = np.array(nmis, dtype=float)
        times = np.array(times, dtype=float)

        results.append({
            "m": int(m_val),
            "mean_mse": float(np.nanmean(mses)),
            "std_mse": float(np.nanstd(mses)),
            "mean_ari": float(np.nanmean(aris)),
            "std_ari": float(np.nanstd(aris)),
            "mean_nmi": float(np.nanmean(nmis)),
            "std_nmi": float(np.nanstd(nmis)),
            "mean_time": float(np.nanmean(times)),
            "std_time": float(np.nanstd(times)),
        })

    results.sort(key=lambda r: r["m"])
    return results


print("\n[hyperparam] Running DP-GRAMS-C hyperparameter sweeps on Digits...")

clip_results = sweep_clip_multiplier_digits(
    clip_values=clip_grid,
    epsilon=epsilon_hparam,
    n_reps=n_reps_hparam,
    base_seed=12345,
)

m_results = sweep_minibatch_digits(
    m_frac_grid=m_frac_grid,
    epsilon=epsilon_hparam,
    clip_multiplier_fixed=DEFAULT_CLIP_MULTIPLIER,
    n_reps=n_reps_hparam,
    base_seed=54321,
)

# ---------------------------------------------------------------------
# Save hyperparameter sweep results
# ---------------------------------------------------------------------
clip_txt_path = os.path.join(results_dir, "dpgrams_c_vs_clip_multiplier_digits.txt")
with open(clip_txt_path, "w") as f:
    header = (
        f"{'clip_mult':>10} | "
        f"{'mean_mse':>10} | {'std_mse':>10} | "
        f"{'mean_ari':>10} | {'std_ari':>10} | "
        f"{'mean_nmi':>10} | {'std_nmi':>10} | "
        f"{'mean_t':>10} | {'std_t':>10}"
    )
    f.write(header + "\n")
    f.write("-" * len(header) + "\n")
    for r in clip_results:
        line = (
            f"{r['clip_multiplier']:10.3f} | "
            f"{r['mean_mse']:10.4f} | {r['std_mse']:10.4f} | "
            f"{r['mean_ari']:10.4f} | {r['std_ari']:10.4f} | "
            f"{r['mean_nmi']:10.4f} | {r['std_nmi']:10.4f} | "
            f"{r['mean_time']:10.4f} | {r['std_time']:10.4f}"
        )
        f.write(line + "\n")
print(f"[saved] Clip-multiplier sweep results -> {clip_txt_path}")

m_txt_path = os.path.join(results_dir, "dpgrams_c_vs_minibatch_digits.txt")
with open(m_txt_path, "w") as f:
    header = (
        f"{'m':>10} | "
        f"{'mean_mse':>10} | {'std_mse':>10} | "
        f"{'mean_ari':>10} | {'std_ari':>10} | "
        f"{'mean_nmi':>10} | {'std_nmi':>10} | "
        f"{'mean_t':>10} | {'std_t':>10}"
    )
    f.write(header + "\n")
    f.write("-" * len(header) + "\n")
    for r in m_results:
        line = (
            f"{r['m']:10d} | "
            f"{r['mean_mse']:10.4f} | {r['std_mse']:10.4f} | "
            f"{r['mean_ari']:10.4f} | {r['std_ari']:10.4f} | "
            f"{r['mean_nmi']:10.4f} | {r['std_nmi']:10.4f} | "
            f"{r['mean_time']:10.4f} | {r['std_time']:10.4f}"
        )
        f.write(line + "\n")
print(f"[saved] Minibatch sweep results -> {m_txt_path}")

# ---------------------------------------------------------------------
# Plots: MSE / ARI / NMI vs C^*
# ---------------------------------------------------------------------
clip_x = [r["clip_multiplier"] for r in clip_results]

fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(
    clip_x,
    [r["mean_mse"] for r in clip_results],
    yerr=[r["std_mse"] for r in clip_results],
    marker="o",
    linestyle="-",
    linewidth=2,
    markersize=7,
    capsize=3,
)
ax.set_xlabel(r"Clip Multiplier ($C^*$)")
ax.set_ylabel("Centroid MSE")
ax.set_title("DP-GRAMS-C on Digits: MSE vs $C^*$")
ax.grid(True, alpha=0.4)
plt.tight_layout()
clip_plot_path_mse = os.path.join(results_dir, "digits_dpgrams_c_mse_vs_clip_multiplier.pdf")
plt.savefig(clip_plot_path_mse, dpi=120)
plt.show()
print(f"[saved] MSE vs C^* plot -> {clip_plot_path_mse}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(
    clip_x,
    [r["mean_ari"] for r in clip_results],
    yerr=[r["std_ari"] for r in clip_results],
    marker="o",
    linestyle="-",
    linewidth=2,
    markersize=7,
    capsize=3,
)
ax.set_xlabel(r"Clip Multiplier ($C^*$)")
ax.set_ylabel("ARI")
ax.set_title("DP-GRAMS-C on Digits: ARI vs $C^*$")
ax.grid(True, alpha=0.4)
plt.tight_layout()
clip_plot_path_ari = os.path.join(results_dir, "digits_dpgrams_c_ari_vs_clip_multiplier.pdf")
plt.savefig(clip_plot_path_ari, dpi=120)
plt.show()
print(f"[saved] ARI vs C^* plot -> {clip_plot_path_ari}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(
    clip_x,
    [r["mean_nmi"] for r in clip_results],
    yerr=[r["std_nmi"] for r in clip_results],
    marker="o",
    linestyle="-",
    linewidth=2,
    markersize=7,
    capsize=3,
)
ax.set_xlabel(r"Clip Multiplier ($C^*$)")
ax.set_ylabel("NMI")
ax.set_title("DP-GRAMS-C on Digits: NMI vs $C^*$")
ax.grid(True, alpha=0.4)
plt.tight_layout()
clip_plot_path_nmi = os.path.join(results_dir, "digits_dpgrams_c_nmi_vs_clip_multiplier.pdf")
plt.savefig(clip_plot_path_nmi, dpi=120)
plt.show()
print(f"[saved] NMI vs C^* plot -> {clip_plot_path_nmi}")

# ---------------------------------------------------------------------
# Plots: MSE / ARI / NMI vs m
# ---------------------------------------------------------------------
m_x = [r["m"] for r in m_results]

fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(
    m_x,
    [r["mean_mse"] for r in m_results],
    yerr=[r["std_mse"] for r in m_results],
    marker="o",
    linestyle="-",
    linewidth=2,
    markersize=7,
    capsize=3,
)
ax.set_xscale("log")
ax.set_xlabel("Minibatch size m")
ax.set_ylabel("Centroid MSE")
ax.set_title("DP-GRAMS-C on Digits: MSE vs m")
ax.grid(True, alpha=0.4)
plt.tight_layout()
m_plot_path_mse = os.path.join(results_dir, "digits_dpgrams_c_mse_vs_minibatch.pdf")
plt.savefig(m_plot_path_mse, dpi=120)
plt.show()
print(f"[saved] MSE vs m plot -> {m_plot_path_mse}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(
    m_x,
    [r["mean_ari"] for r in m_results],
    yerr=[r["std_ari"] for r in m_results],
    marker="o",
    linestyle="-",
    linewidth=2,
    markersize=7,
    capsize=3,
)
ax.set_xscale("log")
ax.set_xlabel("Minibatch size m")
ax.set_ylabel("ARI")
ax.set_title("DP-GRAMS-C on Digits: ARI vs m")
ax.grid(True, alpha=0.4)
plt.tight_layout()
m_plot_path_ari = os.path.join(results_dir, "digits_dpgrams_c_ari_vs_minibatch.pdf")
plt.savefig(m_plot_path_ari, dpi=120)
plt.show()
print(f"[saved] ARI vs m plot -> {m_plot_path_ari}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(
    m_x,
    [r["mean_nmi"] for r in m_results],
    yerr=[r["std_nmi"] for r in m_results],
    marker="o",
    linestyle="-",
    linewidth=2,
    markersize=7,
    capsize=3,
)
ax.set_xscale("log")
ax.set_xlabel("Minibatch size m")
ax.set_ylabel("NMI")
ax.set_title("DP-GRAMS-C on Digits: NMI vs m")
ax.grid(True, alpha=0.4)
plt.tight_layout()
m_plot_path_nmi = os.path.join(results_dir, "digits_dpgrams_c_nmi_vs_minibatch.pdf")
plt.savefig(m_plot_path_nmi, dpi=120)
plt.show()
print(f"[saved] NMI vs m plot -> {m_plot_path_nmi}")
