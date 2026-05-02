# dp_grams_beta.py

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
from scipy.spatial.distance import cdist
from numpy.linalg import cholesky, eigh


# ============================================================
# Orthonormal Legendre basis phi_m on [-1,1]
# phi_m(u) = sqrt((2m+1)/2) * P_m(u)
# ============================================================
def _phi_m(u, m):
    from numpy.polynomial.legendre import Legendre
    Pm = Legendre.basis(m)(u)
    return math.sqrt((2 * m + 1) / 2.0) * Pm


def _phi_m_deriv(u, m):
    from numpy.polynomial.legendre import Legendre
    dPm = Legendre.basis(m).deriv()(u)
    return math.sqrt((2 * m + 1) / 2.0) * dPm


# ============================================================
# Tsybakov kernel (1D)
# K(u) = sum_{m=0}^{beta-1} phi_m(0) phi_m(u) I(|u|<=1)
# ============================================================
def _tsybakov_K1(u, beta=4):
    u = np.asarray(u, dtype=float)
    out = np.zeros_like(u, dtype=float)
    mask = (np.abs(u) <= 1.0)
    if not np.any(mask):
        return out

    um = u[mask]
    s = np.zeros_like(um, dtype=float)
    for m in range(beta):
        s += _phi_m(0.0, m) * _phi_m(um, m)

    out[mask] = s
    return out


def _tsybakov_K1_deriv(u, beta=4):
    u = np.asarray(u, dtype=float)
    out = np.zeros_like(u, dtype=float)
    mask = (np.abs(u) <= 1.0)
    if not np.any(mask):
        return out

    um = u[mask]
    s = np.zeros_like(um, dtype=float)
    for m in range(beta):
        s += _phi_m(0.0, m) * _phi_m_deriv(um, m)

    out[mask] = s
    return out


# ============================================================
# Stable softmax
# ============================================================
def _stable_softmax_from_logits(logits):
    logits = np.asarray(logits, dtype=float)
    if logits.size == 0:
        return np.array([], dtype=float)

    finite = np.isfinite(logits)
    if not np.any(finite):
        return np.ones_like(logits, dtype=float) / float(len(logits))

    out = np.zeros_like(logits, dtype=float)
    lf = logits[finite]
    lf = lf - np.max(lf)
    w = np.exp(lf)
    w_sum = float(np.sum(w))

    if (not np.isfinite(w_sum)) or (w_sum <= 0.0):
        out[finite] = 1.0 / float(np.sum(finite))
        return out

    out[finite] = w / w_sum
    return out


# ============================================================
# DAP public-box helpers and suppression radius
# rho_init = c_rho * (log n)^(-1/d)
# ============================================================
def _estimate_box_radius_nonprivate(
    X: np.ndarray,
    h_dap_used: float,
) -> float:
    """
    Non-private fallback estimate of a public box radius.

    If the caller supplies R to dp_grams_beta, this function is not used. When
    R is omitted, we estimate an axis-aligned box [-R, R]^d from the data.
    This is intentionally non-private and should be used only when acceptable.
    """
    X = np.asarray(X, dtype=float)
    max_abs = float(np.max(np.abs(X)))
    if not np.isfinite(max_abs):
        raise ValueError("Cannot estimate R from non-finite data.")

    return float(max(max_abs, h_dap_used))


def _default_dap_suppression_radius(
    n: int,
    d: int,
    c_rho: float = 2.0,
) -> float:
    if not (np.isfinite(c_rho) and c_rho > 0.0):
        raise ValueError("c_rho must be finite and > 0.")

    log_term = np.log(max(3, int(n)))
    rho_log = float(c_rho) * log_term ** (-1.0 / max(1, int(d)))
    return float(max(1e-12, float(rho_log)))


# ============================================================
# Baseline-style local empirical-mass DAP utility
# ============================================================
def _compute_local_mass_dap_utility(
    X: np.ndarray,
    Z: np.ndarray,
    h_score_used: float,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)

    if X.ndim != 2 or Z.ndim != 2:
        raise ValueError("X and Z must both be 2D arrays.")

    n, d = X.shape
    if Z.shape[1] != d:
        raise ValueError(
            f"X and Z must have the same ambient dimension. Got {X.shape} and {Z.shape}."
        )

    h_score_used = float(max(1e-12, h_score_used))
    D = cdist(Z, X, metric="euclidean")
    u = np.mean(D <= h_score_used, axis=1)
    return u.astype(float, copy=False)


# ============================================================
# Public-box DAP candidate grid
# Z_n = [-R,R]^d \cap h_DAP Z^d
# ============================================================
def _build_public_box_candidate_grid(
    d: int,
    h_dap_used: float,
    R: float,
) -> np.ndarray:
    d = int(d)
    if d <= 0:
        raise ValueError("d must be positive.")

    h_dap_used = float(max(1e-12, h_dap_used))
    R = float(R)
    if not (np.isfinite(R) and R > 0.0):
        raise ValueError("R must be finite and > 0.")

    lo_idx = int(np.floor(-R / h_dap_used))
    hi_idx = int(np.ceil(R / h_dap_used))

    axis = h_dap_used * np.arange(lo_idx, hi_idx + 1, dtype=float)
    axes = [axis for _ in range(d)]

    mesh = np.meshgrid(*axes, indexing="ij")
    Z = np.stack([m.ravel() for m in mesh], axis=1)
    return Z.astype(float, copy=False)


# ============================================================
# Sequential diversity-aware DAP initialization
# ============================================================
def _sequential_dap_initialize(
    X: np.ndarray,
    Z: np.ndarray,
    h_score_used: float,
    epsilon_init: float,
    k: int,
    rng: np.random.Generator,
    suppression_radius: Optional[float] = None,
    c_rho: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)

    n, d = X.shape
    N_cand = int(Z.shape[0])

    if suppression_radius is None:
        suppression_radius = _default_dap_suppression_radius(
            n=n,
            d=d,
            c_rho=c_rho,
        )
    suppression_radius = float(max(0.0, suppression_radius))

    u = _compute_local_mass_dap_utility(
        X=X,
        Z=Z,
        h_score_used=h_score_used,
    )

    eps_draw = float(epsilon_init / max(1, k))
    base_logits = 0.5 * n * eps_draw * u

    anchor_idx = np.empty(k, dtype=int)
    anchors = np.empty((k, d), dtype=float)

    probs_accum = np.zeros(N_cand, dtype=float)
    active = np.ones(N_cand, dtype=bool)
    seen_once = np.zeros(N_cand, dtype=bool)

    radius2 = suppression_radius * suppression_radius

    for ell in range(k):
        if not np.any(active):
            remaining = ~seen_once
            if np.any(remaining):
                active = remaining.copy()
            else:
                active = np.ones(N_cand, dtype=bool)

        active_idx = np.flatnonzero(active)
        active_probs = _stable_softmax_from_logits(base_logits[active_idx])

        probs_round = np.zeros(N_cand, dtype=float)
        probs_round[active_idx] = active_probs
        probs_accum += probs_round

        chosen = int(rng.choice(active_idx, p=active_probs))
        anchor_idx[ell] = chosen
        anchors[ell] = Z[chosen]
        seen_once[chosen] = True

        if suppression_radius > 0.0:
            diff = Z - Z[chosen]
            dist2 = np.sum(diff * diff, axis=1)
            active[dist2 <= radius2] = False
        else:
            active[chosen] = False

    probs_avg = probs_accum / float(max(1, k))
    return u, probs_avg, anchor_idx, anchors, eps_draw


# ============================================================
# Correlation kernel for DP noise across initializations
# Matérn nu=0.5 = Laplace kernel
# ============================================================
def _matern_half_kernel_matrix(A, B=None, hloc=1.0):
    A = np.asarray(A, dtype=float)
    if B is None:
        B = A
    else:
        B = np.asarray(B, dtype=float)

    D = cdist(A, B, metric="euclidean")
    return np.exp(-D / float(hloc))


def _factor_psd_matrix(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=float)
    try:
        return cholesky(K)
    except Exception:
        vals, vecs = eigh(K)
        vals = np.maximum(vals, 0.0)
        return (vecs * np.sqrt(vals)).astype(float)


# ============================================================
# Beta-kernel KDE and gradient contributions
# For x in R^d and batch B:
#   K_i(x)   = h^{-d} prod_j K((x_j - B_ij)/h)
#   grad_i,r = h^{-d-1} K'((x_r - B_ir)/h) prod_{j != r} K((x_j - B_ij)/h)
# ============================================================
def _beta_kde_and_grad_contribs(
    x: np.ndarray,
    B: np.ndarray,
    h_used: float,
    beta: int,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    B = np.asarray(B, dtype=float)

    m, d = B.shape
    u = (x[None, :] - B) / float(h_used)

    K1_all = np.zeros_like(u, dtype=float)
    K1p_all = np.zeros_like(u, dtype=float)

    for j in range(d):
        K1_all[:, j] = _tsybakov_K1(u[:, j], beta=beta)
        K1p_all[:, j] = _tsybakov_K1_deriv(u[:, j], beta=beta)

    prodK = np.prod(K1_all, axis=1)
    K_vals = prodK / (float(h_used) ** d)

    grad_vals = np.zeros((m, d), dtype=float)
    hpow = float(h_used) ** (d + 1)

    for r in range(d):
        prod_others = np.ones(m, dtype=float)
        for j in range(d):
            if j != r:
                prod_others *= K1_all[:, j]
        grad_vals[:, r] = (K1p_all[:, r] * prod_others) / hpow

    p_hat = float(np.mean(K_vals))
    grad_hat = np.mean(grad_vals, axis=0)

    return p_hat, grad_hat, K_vals, grad_vals


# ============================================================
# DP-GRAMS-beta main
# ============================================================
def dp_grams_beta(
    X,
    epsilon,
    delta,
    beta=4,
    R: Optional[float] = None,
    initial_modes=None,
    candidate_points=None,
    T=None,
    m=None,
    h=None,
    h_dap=None,
    beta_dap=3.0,
    kappa_init=4.0,
    suppression_radius=None,
    c_rho: float = 2.0,
    rng=None,
    clip_multiplier=1,
    noise_matern_h=None,
    init_epsilon_frac=0.5,
    eta=None,
    dap_score_multiplier: float = 3.0,
    eta_multiplier=0.5,
    p_floor=None,
    p_floor_multiplier=1.0,
    return_diagnostics=False,
):
    if rng is None:
        rng = np.random.default_rng()

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n,d). Got {X.shape}.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values.")

    n, d = X.shape

    if n < 2:
        raise ValueError("X must contain at least 2 samples.")
    if not (epsilon > 0):
        raise ValueError("epsilon must be > 0.")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1).")
    if not (0 < init_epsilon_frac < 1):
        raise ValueError("init_epsilon_frac must be in (0,1).")
    if not (np.isfinite(kappa_init) and kappa_init > 0.0):
        raise ValueError("kappa_init must be finite and > 0.")
    if not (isinstance(beta, (int, np.integer)) and beta >= 2):
        raise ValueError("beta must be an integer >= 2.")
    if not (np.isfinite(beta_dap) and beta_dap > 0.0):
        raise ValueError("beta_dap must be finite and > 0.")
    if not (np.isfinite(c_rho) and c_rho > 0.0):
        raise ValueError("c_rho must be finite and > 0.")
    if not (np.isfinite(dap_score_multiplier) and dap_score_multiplier > 0.0):
        raise ValueError("dap_score_multiplier must be finite and > 0.")
    if not (np.isfinite(p_floor_multiplier) and p_floor_multiplier > 0.0):
        raise ValueError("p_floor_multiplier must be finite and > 0.")

    epsilon_init = float(init_epsilon_frac * epsilon)
    epsilon_alg = float(max(1e-12, epsilon - epsilon_init))

    if m is None:
        m = max(1, int(n / np.log(max(3, n))))
    m = int(max(1, min(n, m)))

    if T is None:
        T = int(np.ceil(np.log(max(3, n))))
    T = int(max(1, T))

    # ------------------------------------------------------------
    # Ascent bandwidth selection
    # Same privacy skeleton as your baseline beta file
    # ------------------------------------------------------------
    if h is None:
        polylog = np.log(2.0 / delta) * np.log(2.5 * m * max(T, 1) / (n * delta))
        Kbundle = max(1.0, float(T * d * polylog))
        h_non_dp = (np.log(max(3, n)) / n) ** (1.0 / (d + 2.0 * beta + 2.0))
        h_dp = (Kbundle / (n * n * epsilon_alg * epsilon_alg)) ** (
            1.0 / (2.0 * d + 2.0 * beta + 2.0)
        )
        h_used = max(h_non_dp, h_dp)
    else:
        h_used = float(h)

    if not (np.isfinite(h_used) and h_used > 0.0):
        raise ValueError(f"Invalid bandwidth h={h_used}.")

    # ------------------------------------------------------------
    # Separate DAP bandwidth (restored from baseline-style design)
    # ------------------------------------------------------------
    if h_dap is None:
        c_h = 1.0
        h_dap_used = c_h * (np.log(max(3, n)) / n) ** (1.0 / (d + 2.0 * beta_dap))
    else:
        h_dap_used = float(h_dap)

    if not (np.isfinite(h_dap_used) and h_dap_used > 0.0):
        raise ValueError(f"Invalid DAP bandwidth h_dap_used={h_dap_used}.")

    if R is None:
        R_used = _estimate_box_radius_nonprivate(
            X=X,
            h_dap_used=h_dap_used,
        )
    else:
        R_used = float(R)
        if not (np.isfinite(R_used) and R_used > 0.0):
            raise ValueError("R must be finite and > 0 when supplied.")

    h_score_used = float(dap_score_multiplier * h_dap_used)

    if noise_matern_h is None:
        noise_matern_h = h_used
    noise_matern_h = float(noise_matern_h)

    if p_floor is None:
        p_floor_used = float(p_floor_multiplier) / max(1.0, n * (h_used ** d))
    else:
        p_floor_used = float(p_floor)

    if not (np.isfinite(p_floor_used) and p_floor_used > 0.0):
        raise ValueError(f"Invalid p_floor={p_floor_used}.")

    # ------------------------------------------------------------
    # Initialization (restored baseline-style DAP design)
    # ------------------------------------------------------------
    init_u = None
    init_probs = None
    init_anchor_idx = None
    init_candidates = None
    eps_draw = None

    if initial_modes is None:
        k = max(1, int(np.ceil(float(kappa_init) * np.log(max(3, n)))))

        if candidate_points is None:
            Z = _build_public_box_candidate_grid(
                d=d,
                h_dap_used=float(h_dap_used),
                R=R_used,
            )
        else:
            Z = np.asarray(candidate_points, dtype=float)
            if Z.ndim != 2 or Z.shape[1] != d:
                raise ValueError(
                    f"candidate_points must have shape (N_cand,{d}). Got {Z.shape}."
                )
            if not np.all(np.isfinite(Z)):
                raise ValueError("candidate_points must contain only finite values.")

        if Z.shape[0] == 0:
            raise ValueError("candidate_points must contain at least one point.")

        if suppression_radius is None:
            suppression_radius_used = _default_dap_suppression_radius(
                n=n,
                d=d,
                c_rho=c_rho,
            )
        else:
            suppression_radius_used = float(max(0.0, suppression_radius))

        init_u, init_probs, init_anchor_idx, anchors, eps_draw = _sequential_dap_initialize(
            X=X,
            Z=Z,
            h_score_used=h_score_used,
            epsilon_init=epsilon_init,
            k=k,
            rng=rng,
            suppression_radius=suppression_radius_used,
            c_rho=c_rho,
        )

        init_candidates = Z.copy()
        modes = anchors.copy()
    else:
        modes = np.asarray(initial_modes, dtype=float).copy()
        if modes.ndim != 2 or modes.shape[1] != d:
            raise ValueError(
                f"initial_modes must have shape (k,{d}). Got {modes.shape}."
            )
        k = int(modes.shape[0])
        suppression_radius_used = float(
            _default_dap_suppression_radius(n=n, d=d, c_rho=c_rho)
            if suppression_radius is None
            else max(0.0, suppression_radius)
        )

    k = int(modes.shape[0])
    if k == 0:
        empty = np.empty((0, d), dtype=float)
        if return_diagnostics:
            return h_used, empty, {
                "sigma": np.nan,
                "C": np.nan,
                "eta": np.nan,
                "T": int(T),
                "m": int(m),
                "k_inits": 0,
                "beta": int(beta),
                "beta_dap": float(beta_dap),
                "h_used": float(h_used),
                "h_dap_used": float(h_dap_used),
                "h_score_used": float(h_score_used),
                "R_used": float(R_used),
                "c_rho": float(c_rho),
                "dap_score_multiplier": float(dap_score_multiplier),
                "p_floor_used": float(p_floor_used),
            }
        return h_used, empty

    # ------------------------------------------------------------
    # Privacy skeleton
    # ------------------------------------------------------------
    epsilon_iter = epsilon_alg / (2.0 * np.sqrt(2.0 * T * np.log(2.0 / delta)))

    # Keep the same clipping scale as in your uploaded beta file
    C = (h_used ** (-1.0 - d)) * float(clip_multiplier)

    def compute_sigma(C_local: float) -> float:
        q = m / n
        denom = np.log(1.0 + (1.0 / max(1e-12, q)) * (np.exp(epsilon_iter) - 1.0))
        denom = max(1e-12, denom)

        sigma_exact = (2.0 * C_local / m) / denom * np.sqrt(
            2.0 * np.log(2.5 * m * T / (n * delta))
        )

        if epsilon_alg <= 1.0:
            sigma_approx = (8.0 * C_local / (n * epsilon_alg)) * np.sqrt(
                T * np.log(2.0 / delta) * np.log(2.5 * m * T / (n * delta))
            )
            return float(sigma_approx)

        return float(sigma_exact)

    sigma = compute_sigma(C)

    modes_curr = modes.copy()

    if eta is None:
        eta = float(eta_multiplier) * (h_used * h_used)
    else:
        eta = float(eta)

    floor_active_count = 0
    total_score_evals = 0

    # ------------------------------------------------------------
    # Main DP-GRAMS-beta loop
    # Uses truncated KDE denominator rather than positive-part truncation
    # of raw higher-order kernel weights
    # ------------------------------------------------------------
    for _t in range(T):
        # Unfreeze correlation: recompute at CURRENT iterates
        K_modes = _matern_half_kernel_matrix(
            modes_curr,
            modes_curr,
            hloc=noise_matern_h,
        ) + 1e-8 * np.eye(k)
        L_curr = _factor_psd_matrix(K_modes)

        avg_steps = np.zeros_like(modes_curr)

        batch_idx = rng.choice(n, m, replace=False)
        B = X[batch_idx]

        for i in range(k):
            x = modes_curr[i]

            p_hat, _grad_hat, _k_vals, grad_vals = _beta_kde_and_grad_contribs(
                x=x,
                B=B,
                h_used=h_used,
                beta=beta,
            )

            total_score_evals += 1
            if p_hat <= p_floor_used:
                floor_active_count += 1

            denom = max(p_hat, p_floor_used)

            # Per-sample stabilized score contributions
            q_i = grad_vals / denom

            norms = np.linalg.norm(q_i, axis=1, keepdims=True)
            scales = np.minimum(1.0, C / (norms + 1e-12))
            q_i_clipped = q_i * scales

            avg_steps[i] = np.mean(q_i_clipped, axis=0)

        G = rng.normal(size=(k, d))
        noise = sigma * (L_curr @ G)

        modes_curr += eta * (avg_steps + noise)

    final_modes = modes_curr.copy()

    if not return_diagnostics:
        return h_used, final_modes

    diag = {
        "sigma": float(sigma),
        "C": float(C),
        "eta": float(eta),
        "T": int(T),
        "m": int(m),
        "k_inits": int(k),
        "beta": int(beta),
        "beta_dap": float(beta_dap),
        "h_used": float(h_used),
        "h_dap_used": float(h_dap_used),
        "h_score_used": float(h_score_used),
        "R_used": float(R_used),
        "c_rho": float(c_rho),
        "dap_score_multiplier": float(dap_score_multiplier),
        "p_floor_used": float(p_floor_used),
        "suppression_radius_used": float(suppression_radius_used),
        "floor_active_rate": float(floor_active_count / max(1, total_score_evals)),
        "epsilon_init": float(epsilon_init),
        "epsilon_alg": float(epsilon_alg),
        "init_u": init_u,
        "init_probs": init_probs,
        "init_anchor_idx": init_anchor_idx,
        "init_candidates": init_candidates,
        "eps_draw": eps_draw,
    }
    return h_used, final_modes, diag