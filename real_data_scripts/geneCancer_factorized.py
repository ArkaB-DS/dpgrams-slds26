
import csv
import os
import sys
import time
import zipfile
import tarfile
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from diffprivlib.models import KMeans as DPKMeans
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main_scripts.mode_matching_mse import mode_matching_mse
from main_scripts.ms import mean_shift
from main_scripts.merge import merge_modes_agglomerative
from main_scripts.bandwidth import silverman_bandwidth
from real_data_scripts.dp_grams_c import dpms_private

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
SCRIPT_TAG = "factorized"
results_dir = f"results/gene50_centroid_comparison_{SCRIPT_TAG}_pca5"
os.makedirs(results_dir, exist_ok=True)

sns.set(style="whitegrid", context="talk")

DATA_ROOT = Path("data/uci_gene_expression_rnaseq").resolve()
DATA_ROOT.mkdir(parents=True, exist_ok=True)
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/401/gene+expression+cancer+rna+seq.zip"
UCI_ZIP_PATH = DATA_ROOT / "gene+expression+cancer+rna+seq.zip"
EXTRACT_DIR = DATA_ROOT / "extracted"
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

merge_n_clusters = 5
delta = 1e-5
base_rng_seed = 12
n_runs = 20
ms_p = 0.1
eps_modes_list = [0.25, 0.5, 1.0, 2.5, 5.0]
clip_grid = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
m_frac_grid = [0.05, 0.1, 0.2, 0.5, 1.0]
n_reps_hparam = 20
epsilon_hparam = 1.0

# ---------------------------------------------------------------------
# Explicit DP-GRAMS-C defaults
# ---------------------------------------------------------------------
# candidate_points stays None, so no data points are used as DAP candidates.
DEFAULT_DAP_INIT_STRATEGY = "factorized"
DEFAULT_CLIP_MULTIPLIER = 0.1
DEFAULT_KAPPA_INIT = 3.0
DEFAULT_INIT_EPSILON_FRAC = 0.1
DEFAULT_ETA = 1.0
DEFAULT_BETA_DAP = 3.0
DEFAULT_C_RHO = 2.0
DEFAULT_DAP_SCORE_MULTIPLIER = 5.0
DEFAULT_R = None
DEFAULT_CANDIDATE_POINTS = None

# PCA dimension selection. The raw dimension is computed from the spectrum and
# then capped. Edit PCA_R_MAX to test 3, 5, 8, etc.
PCA_DIM_SELECTION_RULE = "broken_stick_dp_capped"
PCA_R_MIN = 2
PCA_R_MAX = 5
PCA_WHITEN = True

# ---------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------
def download_if_needed(url: str, out_path: Path):
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    print(f"[download] Fetching: {url}")
    urllib.request.urlretrieve(url, out_path)
    print(f"[download] Saved -> {out_path}")


def unzip_if_needed(zip_path: Path, extract_dir: Path):
    marker = extract_dir / ".unzipped_ok"
    if marker.exists():
        return
    print(f"[extract] Unzipping -> {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    marker.write_text("ok")
    print("[extract] Unzip done.")


def untar_all_archives(extract_dir: Path):
    marker = extract_dir / ".untar_ok"
    if marker.exists():
        return
    archives = list(extract_dir.rglob("*.tar.gz")) + list(extract_dir.rglob("*.tgz")) + list(extract_dir.rglob("*.tar"))
    for arch in archives:
        print(f"[extract] Untarring: {arch.name}")
        mode = "r:gz" if arch.suffixes[-2:] == [".tar", ".gz"] or arch.suffix == ".tgz" else "r"
        with tarfile.open(arch, mode) as tf:
            tf.extractall(arch.parent)
    marker.write_text("ok")
    print("[extract] Tar extraction done.")


def find_data_and_labels_csv(extract_dir: Path):
    data_csvs = list(extract_dir.rglob("data.csv"))
    label_csvs = list(extract_dir.rglob("labels.csv"))
    if data_csvs and label_csvs:
        return data_csvs[0], label_csvs[0]
    raise FileNotFoundError("Could not find data.csv and labels.csv after extraction.")


def load_gene_dataset(extract_dir: Path):
    data_csv, labels_csv = find_data_and_labels_csv(extract_dir)
    print(f"[data] Using data file:   {data_csv}")
    print(f"[data] Using labels file: {labels_csv}")

    raw = np.genfromtxt(data_csv, delimiter=",", dtype=None, encoding="utf-8", skip_header=1)
    if raw.dtype.names is not None:
        fields = list(raw.dtype.names)
        numeric_fields = fields[1:]
        X = np.vstack([raw[name].astype(float) for name in numeric_fields]).T
    else:
        raw = np.asarray(raw)
        X = raw[:, 1:].astype(float)

    y_raw = np.genfromtxt(labels_csv, delimiter=",", dtype=str, encoding="utf-8", skip_header=1)
    if y_raw.ndim == 0:
        y_raw = np.array([str(y_raw)])
    if y_raw.ndim > 1:
        y_raw = y_raw[:, -1]
    y_raw = np.array([s.strip() for s in y_raw], dtype=str)

    uniq = sorted(set(y_raw.tolist()))
    name_to_int = {name: i for i, name in enumerate(uniq)}
    y = np.array([name_to_int[name] for name in y_raw], dtype=int)
    return X, y, uniq


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------
def compute_metrics(y_true, labels):
    return adjusted_rand_score(y_true, labels), normalized_mutual_info_score(y_true, labels)


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
    model = DPKMeans(n_clusters=n_clusters, epsilon=eps, random_state=seed)
    try:
        model.fit(X_input)
        return model
    except Exception:
        mins_local = X_input.min(axis=0)
        maxs_local = X_input.max(axis=0)
        model = DPKMeans(n_clusters=n_clusters, epsilon=eps, random_state=seed, bounds=(mins_local, maxs_local))
        model.fit(X_input)
        return model


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
        centers_2d_list = [np.column_stack([np.asarray(m, dtype=float)[:, 0], np.zeros(np.asarray(m).shape[0])]) for m in centers_list]
    return X_2d, true_means_2d, centers_2d_list


def _safe_rank_from_count(count, *, min_rank=1, max_rank=None):
    if max_rank is None:
        max_rank = int(count)
    return int(max(min_rank, min(int(count), int(max_rank))))


def select_pca_dimension(X_scaled, *, rule="broken_stick_dp_capped", r_min=2, r_max=5):
    X_scaled = np.asarray(X_scaled, dtype=float)
    pca_probe = PCA(random_state=base_rng_seed)
    pca_probe.fit(X_scaled)

    eigvals = np.asarray(pca_probe.explained_variance_, dtype=float)
    evr = np.asarray(pca_probe.explained_variance_ratio_, dtype=float)
    p = int(eigvals.size)
    if p == 0:
        raise ValueError("PCA spectrum is empty.")

    if p >= 2:
        ratios = eigvals[:-1] / np.maximum(eigvals[1:], 1e-12)
        ahn_horenstein = int(np.argmax(ratios) + 1)
    else:
        ahn_horenstein = 1

    harmonic_tail = np.array([np.sum(1.0 / np.arange(j, p + 1, dtype=float)) / float(p) for j in range(1, p + 1)])
    broken_stick = int(np.sum(evr > harmonic_tail))
    broken_stick = max(1, broken_stick)

    kaiser = int(np.sum(eigvals > np.mean(eigvals)))
    kaiser = max(1, kaiser)

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

    selected = _safe_rank_from_count(raw_selected, min_rank=int(r_min), max_rank=min(int(r_max), p))
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
    }
    return selected, diagnostics


def run_dpgrams_c_gene(data, epsilon_modes, h, rng, *, k_est, clip_multiplier=DEFAULT_CLIP_MULTIPLIER, m=None, return_info=False):
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


def estimate_grid_size(d, h_dap_used, R_used):
    d = int(d)
    h_dap_used = float(max(1e-12, h_dap_used))
    R_used = float(R_used)
    lo_idx = int(np.floor(-R_used / h_dap_used))
    hi_idx = int(np.ceil(R_used / h_dap_used))
    axis_count = int(hi_idx - lo_idx + 1)
    return axis_count, int(axis_count ** d)


# ---------------------------------------------------------------------
# Download + extract
# ---------------------------------------------------------------------
download_if_needed(UCI_ZIP_URL, UCI_ZIP_PATH)
unzip_if_needed(UCI_ZIP_PATH, EXTRACT_DIR)
untar_all_archives(EXTRACT_DIR)

X_raw, y_true, class_names = load_gene_dataset(EXTRACT_DIR)
n, d = X_raw.shape
print(f"[data] X shape = {X_raw.shape} (n={n}, d={d})")
print(f"[data] classes ({len(class_names)}): {class_names}")

ms_T = int(np.ceil(np.log(max(2, n))))

# ---------------------------------------------------------------------
# Standardize raw gene space and select PCA dimension
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
true_means_scaled = np.array([X_scaled[y_true == c].mean(axis=0) for c in range(merge_n_clusters)])

pca_dim, pca_dim_info = select_pca_dimension(X_scaled, rule=PCA_DIM_SELECTION_RULE, r_min=PCA_R_MIN, r_max=PCA_R_MAX)
print("[pca] PCA dimension diagnostics:")
print(f"[pca] Ahn-Horenstein eigenvalue-ratio r = {pca_dim_info['ahn_horenstein']}")
print(f"[pca] Broken-stick r = {pca_dim_info['broken_stick']}")
print(f"[pca] Kaiser r = {pca_dim_info['kaiser']}")
print(f"[pca] Participation effective rank = {pca_dim_info['participation_rank_float']:.3f} -> {pca_dim_info['participation_rank']}")
print(f"[pca] Entropy effective rank = {pca_dim_info['entropy_rank_float']:.3f} -> {pca_dim_info['entropy_rank']}")
print(f"[pca] 80% variance r = {pca_dim_info['var80']}")
print(f"[pca] 85% variance r = {pca_dim_info['var85']}")
print(f"[pca] 90% variance r = {pca_dim_info['var90']}")
print(
    f"[pca] selected rule = {pca_dim_info['rule']}, raw r = {pca_dim_info['selected_raw']}, "
    f"capped selected r = {pca_dim_info['selected']}, cumvar = {pca_dim_info['cumvar_selected']:.4f}, whiten = {PCA_WHITEN}"
)

pca_shared = PCA(n_components=pca_dim, random_state=base_rng_seed, whiten=PCA_WHITEN)
X = pca_shared.fit_transform(X_scaled)
true_means = np.array([X[y_true == c].mean(axis=0) for c in range(merge_n_clusters)])


def shared_pca_to_scaled(centers):
    centers = np.asarray(centers, dtype=float)
    if centers.size == 0:
        return np.empty((0, X_scaled.shape[1]), dtype=float)
    return pca_shared.inverse_transform(np.atleast_2d(centers))


# ---------------------------------------------------------------------
# Bandwidth and public-grid diagnostic
# ---------------------------------------------------------------------
h = silverman_bandwidth(X)
R_diag = float(max(np.max(np.abs(X)), h))
h_dap_diag = (np.log(max(3, n)) / n) ** (1.0 / (pca_dim + 2.0 * DEFAULT_BETA_DAP))
axis_count_diag, grid_count_diag = estimate_grid_size(pca_dim, h_dap_diag, R_diag)
print(
    f"[dap-grid-diagnostic] strategy={DEFAULT_DAP_INIT_STRATEGY}, r={pca_dim}, "
    f"h_dap≈{h_dap_diag:.4g}, R≈{R_diag:.4g}, axis_count≈{axis_count_diag}, "
    f"full_grid_candidates≈{grid_count_diag}"
)
if DEFAULT_DAP_INIT_STRATEGY == "grid" and grid_count_diag > 5_000_000:
    print("[warning] The public grid is large. If runtime is too high, reduce PCA_R_MAX or use the factorized script.")

# ---------------------------------------------------------------------
# 1) Mean Shift baseline
# ---------------------------------------------------------------------
rng_ms = np.random.default_rng(base_rng_seed)
(raw_ms_modes, ms_time) = timed(mean_shift, X, T=ms_T, bandwidth=h, p=ms_p, seed=rng_ms)
ms_merged_modes = merge_modes_agglomerative(raw_ms_modes, n_clusters=merge_n_clusters, random_state=base_rng_seed)
dists_ms = np.linalg.norm(X[:, None, :] - ms_merged_modes[None, :, :], axis=2)
labels_ms = np.argmin(dists_ms, axis=1).astype(int)
labels_ms = relabel_clusters(y_true, labels_ms, merge_n_clusters)

# ---------------------------------------------------------------------
# 2) DP-GRAMS-C
# ---------------------------------------------------------------------
rng_dp = np.random.default_rng(base_rng_seed)
(((modes_dp, labels_dp), h_used, init_info), dp_time) = timed(
    run_dpgrams_c_gene,
    data=X,
    epsilon_modes=1.0,
    h=h,
    rng=rng_dp,
    k_est=merge_n_clusters,
    clip_multiplier=DEFAULT_CLIP_MULTIPLIER,
    return_info=True,
)
labels_dp = relabel_clusters(y_true, labels_dp, merge_n_clusters)

# ---------------------------------------------------------------------
# 3) Non-private KMeans
# ---------------------------------------------------------------------
kmeans = KMeans(n_clusters=merge_n_clusters, random_state=base_rng_seed, n_init=10)
(labels_km, km_time) = timed(kmeans.fit_predict, X)
modes_km = kmeans.cluster_centers_
labels_km = relabel_clusters(y_true, labels_km.astype(int), merge_n_clusters)

# ---------------------------------------------------------------------
# 4) DP-KMeans
# ---------------------------------------------------------------------
(dp_kmeans, dpkm_time) = timed(fit_dpkmeans, X, 1.0, base_rng_seed, merge_n_clusters)
labels_dpkm = dp_kmeans.labels_.astype(int) if hasattr(dp_kmeans, "labels_") else dp_kmeans.predict(X).astype(int)
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
    modes_scaled = shared_pca_to_scaled(modes)
    mse_centroids[alg] = mode_matching_mse(true_means_scaled.copy(), modes_scaled.copy())

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
    f.write(f"Classes: {class_names}\n")
    f.write(f"Raw standardized shape: n={n}, d={d}\n")
    f.write(f"PCA dimension rule: {PCA_DIM_SELECTION_RULE}\n")
    f.write(f"PCA selected raw r: {pca_dim_info['selected_raw']}\n")
    f.write(f"PCA selected capped r: {pca_dim_info['selected']}\n")
    f.write(f"PCA selected cumulative variance: {pca_dim_info['cumvar_selected']:.6f}\n")
    f.write(f"Ahn-Horenstein r: {pca_dim_info['ahn_horenstein']}\n")
    f.write(f"Broken-stick r: {pca_dim_info['broken_stick']}\n")
    f.write(f"Kaiser r: {pca_dim_info['kaiser']}\n")
    f.write(f"Participation effective rank: {pca_dim_info['participation_rank_float']:.6f} -> {pca_dim_info['participation_rank']}\n")
    f.write(f"Entropy effective rank: {pca_dim_info['entropy_rank_float']:.6f} -> {pca_dim_info['entropy_rank']}\n")
    f.write(f"Variance ranks: r80={pca_dim_info['var80']}, r85={pca_dim_info['var85']}, r90={pca_dim_info['var90']}\n")
    f.write(f"Shared PCA dimension for all methods: {pca_dim}\n")
    f.write(f"PCA whitening: {PCA_WHITEN}\n")
    f.write("All methods clustered on the shared PCA representation.\n")
    f.write("Centroid MSE is computed after inverse PCA back to standardized gene space.\n")
    f.write("DP-GRAMS-C candidate_points = None; no data points are used as DAP candidates.\n")
    f.write(f"DP-GRAMS-C internal d_ambient: {int(getattr(init_info, 'd_ambient', pca_dim))}\n")
    f.write(f"DP-GRAMS-C init strategy: {getattr(init_info, 'dap_init_strategy', DEFAULT_DAP_INIT_STRATEGY)}\n")
    f.write(f"DAP diagnostic axis_count≈{axis_count_diag}, full_grid_candidates≈{grid_count_diag}\n")
    f.write(f"Silverman bandwidth h = {float(h):.8f}\n")
    f.write(f"DP-GRAMS h_used = {float(h_used):.8f}\n")
    f.write(
        "DP-GRAMS-C settings: indicator utility, logarithmic suppression, "
        f"kappa_init={DEFAULT_KAPPA_INIT}, init_epsilon_frac={DEFAULT_INIT_EPSILON_FRAC}, "
        f"clip_multiplier={DEFAULT_CLIP_MULTIPLIER}, c_rho={DEFAULT_C_RHO}, "
        f"dap_score_multiplier={DEFAULT_DAP_SCORE_MULTIPLIER}, R={DEFAULT_R}, "
        f"dap_init_strategy={DEFAULT_DAP_INIT_STRATEGY}\n"
    )
    for alg, rt in zip(algorithms, runtimes):
        ari, nmi = metrics[alg]
        f.write(f"{alg}: ARI={ari:.4f}, NMI={nmi:.4f}, MSE={mse_centroids[alg]:.6f}, Runtime={rt:.4f}s\n")

print("\nMetrics (clustering on shared PCA, MSE on standardized gene space):")
print(f"{'Alg':<12} {'ARI':>6} {'NMI':>6} {'MSE_centroids':>14} {'Runtime(s)':>12}")
for alg, rt in zip(algorithms, runtimes):
    ari, nmi = metrics[alg]
    print(f"{alg:<12} {ari:6.3f} {nmi:6.3f} {mse_centroids[alg]:14.6f} {rt:12.4f}")

# ---------------------------------------------------------------------
# 2D visualization
# ---------------------------------------------------------------------
X_2d, true_means_2d, modes_2d_list = make_2d_view(X, modes_list, true_means, seed=base_rng_seed)
fig, axes = plt.subplots(1, 4, figsize=(20, 6))
palette = sns.color_palette("tab10", merge_n_clusters)
global_handles = []
global_labels = []
for ax, alg, labels_pred, modes_2d in zip(axes, algorithms, labels_list, modes_2d_list):
    for i, color in enumerate(palette[:merge_n_clusters]):
        sc = ax.scatter(X_2d[labels_pred == i, 0], X_2d[labels_pred == i, 1], c=[color], s=15, alpha=0.7)
        if len(global_handles) < merge_n_clusters:
            global_handles.append(sc)
            global_labels.append(f"Cluster {i}")
    true_sc = ax.scatter(true_means_2d[:, 0], true_means_2d[:, 1], marker="X", c="magenta", s=140, linewidths=2)
    if "True means" not in global_labels:
        global_handles.append(true_sc)
        global_labels.append("True means")
    modes_sc = ax.scatter(modes_2d[:, 0], modes_2d[:, 1], marker="X", c="blue", s=100, linewidths=2)
    if "Estimated modes" not in global_labels:
        global_handles.append(modes_sc)
        global_labels.append("Estimated modes")
    ax.set_title(alg, fontsize=14)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
legend = fig.legend(global_handles, global_labels, fontsize=9, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.01), title="Cluster Assignments & Centroids")
plt.setp(legend.get_title(), fontsize=11, fontweight="bold")
fig.suptitle("Clustering Comparison on Cancer RNA-Seq Data", fontsize=18, y=0.92)
save_path_clusters = os.path.join(results_dir, "gene50_clustering_comparison.pdf")
plt.tight_layout(rect=[0, 0.05, 1, 0.93])
plt.savefig(save_path_clusters, dpi=120)
plt.show()
print("Clustering comparison plot saved to:", save_path_clusters)

# ---------------------------------------------------------------------
# Privacy-utility curves
# ---------------------------------------------------------------------
dp_metrics = {m: [] for m in ["ARI", "NMI", "MSE"]}
dpkm_metrics = {m: [] for m in ["ARI", "NMI", "MSE"]}
dp_err = {m: [] for m in ["ARI", "NMI", "MSE"]}
dpkm_err = {m: [] for m in ["ARI", "NMI", "MSE"]}
dp_times_mean, dpkm_times_mean = [], []
dp_times_std, dpkm_times_std = [], []

for eps in eps_modes_list:
    ari_list, nmi_list, mse_list, dp_times = [], [], [], []
    for run in range(n_runs):
        rng_run = np.random.default_rng(base_rng_seed + run)
        t0 = time.time()
        modes_run, labels_run = run_dpgrams_c_gene(data=X, epsilon_modes=eps, h=h, rng=rng_run, k_est=merge_n_clusters, clip_multiplier=DEFAULT_CLIP_MULTIPLIER)
        t_dp = time.time() - t0
        dp_times.append(t_dp)
        labels_run = relabel_clusters(y_true, labels_run, merge_n_clusters)
        run_ari, run_nmi = compute_metrics(y_true, labels_run)
        modes_scaled = shared_pca_to_scaled(modes_run)
        run_mse = mode_matching_mse(true_means_scaled.copy(), modes_scaled.copy())
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

    ari_list, nmi_list, mse_list, dpkm_times = [], [], [], []
    for run in range(n_runs):
        t0 = time.time()
        dp_kmeans_run = fit_dpkmeans(X, eps, base_rng_seed + run, merge_n_clusters)
        t_dpkm = time.time() - t0
        dpkm_times.append(t_dpkm)
        labels_dpkm_run = dp_kmeans_run.labels_.astype(int) if hasattr(dp_kmeans_run, "labels_") else dp_kmeans_run.predict(X).astype(int)
        labels_dpkm_run = relabel_clusters(y_true, labels_dpkm_run, merge_n_clusters)
        modes_dpkm_run = dp_kmeans_run.cluster_centers_
        run_ari, run_nmi = compute_metrics(y_true, labels_dpkm_run)
        modes_scaled = shared_pca_to_scaled(modes_dpkm_run)
        run_mse = mode_matching_mse(true_means_scaled.copy(), modes_scaled.copy())
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

csv_file = os.path.join(results_dir, "privacy_utility_metrics.csv")
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epsilon", "Algorithm", "ARI", "NMI", "MSE", "Std_ARI", "Std_NMI", "Std_MSE", "Mean_Runtime(s)", "Std_Runtime(s)", "dpgrams_c_clip_multiplier", "dpgrams_c_kappa_init", "dpgrams_c_init_epsilon_frac", "dpgrams_c_c_rho", "dpgrams_c_candidate_points", "dpgrams_c_dap_init_strategy", "pca_dim", "pca_dim_rule", "pca_whiten"])
    for i, eps in enumerate(eps_modes_list):
        writer.writerow([eps, "DP-GRAMS-C", dp_metrics["ARI"][i], dp_metrics["NMI"][i], dp_metrics["MSE"][i], dp_err["ARI"][i], dp_err["NMI"][i], dp_err["MSE"][i], dp_times_mean[i], dp_times_std[i], DEFAULT_CLIP_MULTIPLIER, DEFAULT_KAPPA_INIT, DEFAULT_INIT_EPSILON_FRAC, DEFAULT_C_RHO, "None", DEFAULT_DAP_INIT_STRATEGY, pca_dim, PCA_DIM_SELECTION_RULE, PCA_WHITEN])
        writer.writerow([eps, "DP-KMeans", dpkm_metrics["ARI"][i], dpkm_metrics["NMI"][i], dpkm_metrics["MSE"][i], dpkm_err["ARI"][i], dpkm_err["NMI"][i], dpkm_err["MSE"][i], dpkm_times_mean[i], dpkm_times_std[i], "", "", "", "", "", "", pca_dim, PCA_DIM_SELECTION_RULE, PCA_WHITEN])
print("All Gene experiments completed. Results saved in:", results_dir)

for metric in ["ARI", "NMI", "MSE"]:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(eps_modes_list, dp_metrics[metric], yerr=dp_err[metric], marker="o", linestyle="-", linewidth=2, markersize=7, capsize=3, label="DP-GRAMS-C")
    ax.errorbar(eps_modes_list, dpkm_metrics[metric], yerr=dpkm_err[metric], marker="s", linestyle="-", linewidth=2, markersize=7, capsize=3, label="DP-KMeans")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(metric)
    ax.set_title(f"Privacy-Utility: {metric} (Cancer RNA-Seq)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")
    plt.tight_layout()
    out_path = os.path.join(results_dir, f"gene50_privacy_utility_{metric.lower()}.pdf")
    plt.savefig(out_path, dpi=120)
    plt.show()
    print(f"Privacy-utility ({metric}) figure saved to:", out_path)

# ---------------------------------------------------------------------
# Hyperparameter sweeps
# ---------------------------------------------------------------------
def _run_dpgrams_c_single(epsilon_modes, clip_multiplier, m, seed):
    rng = np.random.default_rng(int(seed))
    t0 = time.time()
    modes_run, labels_run = run_dpgrams_c_gene(data=X, epsilon_modes=epsilon_modes, h=h, rng=rng, m=m, k_est=merge_n_clusters, clip_multiplier=clip_multiplier)
    runtime = time.time() - t0
    if modes_run.size == 0:
        return float("nan"), float("nan"), float("nan"), float(runtime)
    labels_run = relabel_clusters(y_true, labels_run, merge_n_clusters)
    ari, nmi = compute_metrics(y_true, labels_run)
    modes_scaled = shared_pca_to_scaled(modes_run)
    mse = mode_matching_mse(true_means_scaled.copy(), modes_scaled.copy())
    return float(mse), float(ari), float(nmi), float(runtime)


def sweep_clip_multiplier_gene(clip_values, epsilon, n_reps=20, base_seed=12345):
    rng = np.random.default_rng(base_seed)
    results = []
    m_default = None
    for cm in clip_values:
        mses, aris, nmis, times = [], [], [], []
        for _ in range(n_reps):
            seed_run = int(rng.integers(0, 2**31 - 1))
            mse, ari, nmi, rt = _run_dpgrams_c_single(epsilon_modes=epsilon, clip_multiplier=cm, m=m_default, seed=seed_run)
            mses.append(mse); aris.append(ari); nmis.append(nmi); times.append(rt)
        mses = np.array(mses, dtype=float); aris = np.array(aris, dtype=float); nmis = np.array(nmis, dtype=float); times = np.array(times, dtype=float)
        results.append({"clip_multiplier": float(cm), "mean_mse": float(np.nanmean(mses)), "std_mse": float(np.nanstd(mses)), "mean_ari": float(np.nanmean(aris)), "std_ari": float(np.nanstd(aris)), "mean_nmi": float(np.nanmean(nmis)), "std_nmi": float(np.nanstd(nmis)), "mean_time": float(np.nanmean(times)), "std_time": float(np.nanstd(times))})
    results.sort(key=lambda r: r["clip_multiplier"])
    return results


def sweep_minibatch_gene(m_frac_grid, epsilon, clip_multiplier_fixed=DEFAULT_CLIP_MULTIPLIER, n_reps=20, base_seed=54321):
    rng = np.random.default_rng(base_seed)
    results = []
    n_samples = X.shape[0]
    m_grid = sorted(set(max(1, int(frac * n_samples)) for frac in m_frac_grid))
    for m_val in m_grid:
        mses, aris, nmis, times = [], [], [], []
        for _ in range(n_reps):
            seed_run = int(rng.integers(0, 2**31 - 1))
            mse, ari, nmi, rt = _run_dpgrams_c_single(epsilon_modes=epsilon, clip_multiplier=clip_multiplier_fixed, m=m_val, seed=seed_run)
            mses.append(mse); aris.append(ari); nmis.append(nmi); times.append(rt)
        mses = np.array(mses, dtype=float); aris = np.array(aris, dtype=float); nmis = np.array(nmis, dtype=float); times = np.array(times, dtype=float)
        results.append({"m": int(m_val), "mean_mse": float(np.nanmean(mses)), "std_mse": float(np.nanstd(mses)), "mean_ari": float(np.nanmean(aris)), "std_ari": float(np.nanstd(aris)), "mean_nmi": float(np.nanmean(nmis)), "std_nmi": float(np.nanstd(nmis)), "mean_time": float(np.nanmean(times)), "std_time": float(np.nanstd(times))})
    results.sort(key=lambda r: r["m"])
    return results


print("\n[hyperparam] Running DP-GRAMS-C hyperparameter sweeps on Gene...")
clip_results = sweep_clip_multiplier_gene(clip_values=clip_grid, epsilon=epsilon_hparam, n_reps=n_reps_hparam, base_seed=12345)
m_results = sweep_minibatch_gene(m_frac_grid=m_frac_grid, epsilon=epsilon_hparam, clip_multiplier_fixed=DEFAULT_CLIP_MULTIPLIER, n_reps=n_reps_hparam, base_seed=54321)

clip_txt_path = os.path.join(results_dir, "dpgrams_c_vs_clip_multiplier_gene.txt")
with open(clip_txt_path, "w") as f:
    header = f"{'clip_mult':>10} | {'mean_mse':>10} | {'std_mse':>10} | {'mean_ari':>10} | {'std_ari':>10} | {'mean_nmi':>10} | {'std_nmi':>10} | {'mean_t':>10} | {'std_t':>10}"
    f.write(header + "\n"); f.write("-" * len(header) + "\n")
    for r in clip_results:
        f.write(f"{r['clip_multiplier']:10.3f} | {r['mean_mse']:10.4f} | {r['std_mse']:10.4f} | {r['mean_ari']:10.4f} | {r['std_ari']:10.4f} | {r['mean_nmi']:10.4f} | {r['std_nmi']:10.4f} | {r['mean_time']:10.4f} | {r['std_time']:10.4f}\n")
print(f"[saved] Clip-multiplier sweep results -> {clip_txt_path}")

m_txt_path = os.path.join(results_dir, "dpgrams_c_vs_minibatch_gene.txt")
with open(m_txt_path, "w") as f:
    header = f"{'m':>10} | {'mean_mse':>10} | {'std_mse':>10} | {'mean_ari':>10} | {'std_ari':>10} | {'mean_nmi':>10} | {'std_nmi':>10} | {'mean_t':>10} | {'std_t':>10}"
    f.write(header + "\n"); f.write("-" * len(header) + "\n")
    for r in m_results:
        f.write(f"{r['m']:10d} | {r['mean_mse']:10.4f} | {r['std_mse']:10.4f} | {r['mean_ari']:10.4f} | {r['std_ari']:10.4f} | {r['mean_nmi']:10.4f} | {r['std_nmi']:10.4f} | {r['mean_time']:10.4f} | {r['std_time']:10.4f}\n")
print(f"[saved] Minibatch sweep results -> {m_txt_path}")

# Plots: MSE / ARI / NMI vs C^*
clip_x = [r["clip_multiplier"] for r in clip_results]
for metric, ylab, out_suffix in [("mse", "Centroid MSE (std gene space)", "mse"), ("ari", "ARI", "ari"), ("nmi", "NMI", "nmi")]:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(clip_x, [r[f"mean_{metric}"] for r in clip_results], yerr=[r[f"std_{metric}"] for r in clip_results], marker="o", linestyle="-", linewidth=2, markersize=7, capsize=3)
    ax.set_xlabel(r"Clip Multiplier ($C^*$)")
    ax.set_ylabel(ylab)
    ax.set_title(f"DP-GRAMS-C on Gene: {ylab.split()[0]} vs $C^*$")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    out_path = os.path.join(results_dir, f"gene_dpgrams_c_{out_suffix}_vs_clip_multiplier.pdf")
    plt.savefig(out_path, dpi=120)
    plt.show()
    print(f"[saved] {ylab} vs C^* plot -> {out_path}")

# Plots: MSE / ARI / NMI vs m
m_x = [r["m"] for r in m_results]
for metric, ylab, out_suffix in [("mse", "Centroid MSE (std gene space)", "mse"), ("ari", "ARI", "ari"), ("nmi", "NMI", "nmi")]:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(m_x, [r[f"mean_{metric}"] for r in m_results], yerr=[r[f"std_{metric}"] for r in m_results], marker="o", linestyle="-", linewidth=2, markersize=7, capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("Minibatch size m")
    ax.set_ylabel(ylab)
    ax.set_title(f"DP-GRAMS-C on Gene: {ylab.split()[0]} vs m")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    out_path = os.path.join(results_dir, f"gene_dpgrams_c_{out_suffix}_vs_minibatch.pdf")
    plt.savefig(out_path, dpi=120)
    plt.show()
    print(f"[saved] {ylab} vs m plot -> {out_path}")
