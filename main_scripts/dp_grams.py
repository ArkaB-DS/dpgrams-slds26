from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from scipy.spatial.distance import cdist
from numpy.linalg import cholesky, eigh


@dataclass(frozen=True)
class DPGramsInitInfo:
    """Initialization diagnostics."""
    u: np.ndarray
    probs: np.ndarray
    anchor_idx: np.ndarray
    anchors: np.ndarray
    init_modes: np.ndarray
    candidates: np.ndarray
    k: int
    epsilon_init: float
    eps_draw: float
    h_used: float
    h_dap_used: float
    h_score_used: float
    R_used: float
    c_rho: float
    d_ambient: int
    dap_init_strategy: str = "grid"


def _stable_softmax_from_logits(logits: np.ndarray) -> np.ndarray:
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


def _estimate_box_radius_nonprivate(
    X: np.ndarray,
    h_dap_used: float,
) -> float:
    """
    Non-private fallback estimate of a public box radius.

    If the caller supplies R to dp_grams, this function is not used. When R is
    omitted, we estimate an axis-aligned box [-R, R]^d from the data. This is
    intentionally non-private and should be used only when that is acceptable.
    """
    X = np.asarray(X, dtype=float)
    max_abs = float(np.max(np.abs(X)))
    if not np.isfinite(max_abs):
        raise ValueError("Cannot estimate R from non-finite data.")

    # Avoid a degenerate empty/zero-radius candidate grid when all data are zero.
    return float(max(max_abs, h_dap_used))


def _default_dap_suppression_radius(
    n: int,
    d: int,
    c_rho: float = 2.0,
) -> float:
    """
    Default DAP suppression radius: c_rho * log(n)^(-1/d).
    """
    if not (np.isfinite(c_rho) and c_rho > 0.0):
        raise ValueError("c_rho must be finite and > 0.")

    log_term = np.log(max(3, int(n)))
    rho = float(c_rho) * log_term ** (-1.0 / max(1, int(d)))
    return float(max(1e-12, rho))


def _compute_indicator_dap_utility(
    X: np.ndarray,
    Z: np.ndarray,
    h_score_used: float,
) -> np.ndarray:
    """
    Indicator local-mass utility.
    """
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


def _build_public_box_candidate_grid(
    d: int,
    h_dap_used: float,
    R: float,
) -> np.ndarray:
    """
    Build a public lattice candidate set over the public box [-R, R]^d.

    The lattice spacing is h_dap_used. The endpoints are rounded outward to the
    nearest lattice points, so the candidate set is public once d, h_dap_used,
    and R are fixed.
    """
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


def _build_public_axis_candidate_grids(
    d: int,
    h_dap_used: float,
    R: float,
) -> list[np.ndarray]:
    """
    Build public one-dimensional candidate grids for factorized DAP.

    This avoids the full product lattice.  Each coordinate uses the same public
    axis over [-R, R] with spacing h_dap_used.
    """
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

    if axis.size == 0:
        raise ValueError("Factorized DAP axis grid is empty.")

    return [axis.copy() for _ in range(d)]


def _compute_1d_indicator_dap_utility(
    x: np.ndarray,
    z_axis: np.ndarray,
    h_score_used: float,
) -> np.ndarray:
    """
    One-dimensional local-mass utility for one PCA coordinate.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    z_axis = np.asarray(z_axis, dtype=float).reshape(-1)

    if x.size == 0:
        raise ValueError("x must be non-empty.")
    if z_axis.size == 0:
        raise ValueError("z_axis must be non-empty.")

    h_score_used = float(max(1e-12, h_score_used))
    u = np.mean(np.abs(z_axis[:, None] - x[None, :]) <= h_score_used, axis=1)
    return u.astype(float, copy=False)


def _axis_candidates_as_embedded_matrix(axis_grids: list[np.ndarray]) -> np.ndarray:
    """
    Diagnostic-only candidate matrix for factorized DAP.

    The true factorized candidate space is the collection of 1D public grids,
    not their Cartesian product.  For diagnostics, we embed each 1D axis grid
    into R^d with all other coordinates set to zero.
    """
    d = len(axis_grids)
    rows = []
    for j, axis in enumerate(axis_grids):
        axis = np.asarray(axis, dtype=float).reshape(-1)
        Zj = np.zeros((axis.size, d), dtype=float)
        Zj[:, j] = axis
        rows.append(Zj)
    if not rows:
        return np.empty((0, 0), dtype=float)
    return np.vstack(rows)


def _sequential_factorized_dap_initialize(
    X: np.ndarray,
    axis_grids: list[np.ndarray],
    h_score_used: float,
    epsilon_init: float,
    k: int,
    rng: np.random.Generator,
    suppression_radius: Optional[float] = None,
    c_rho: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Factorized DAP initialization for moderate/high-dimensional PCA spaces.

    Assumption used only for initialization: PCA coordinates are treated as
    approximately independent, so each coordinate is privately sampled from its
    own one-dimensional public grid.  The final DP-GRAMS ascent still runs in
    the full joint ambient space and can correct the factorized anchors.

    Privacy accounting uses basic composition over k * d coordinate draws.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")

    n, d = X.shape
    if len(axis_grids) != d:
        raise ValueError("axis_grids length must equal ambient dimension.")

    k = int(k)
    if k <= 0:
        raise ValueError("k must be positive.")

    if suppression_radius is None:
        suppression_radius = _default_dap_suppression_radius(
            n=n,
            d=d,
            c_rho=c_rho,
        )
    suppression_radius = float(max(0.0, suppression_radius))

    eps_draw = float(epsilon_init) / float(max(1, k * d))

    anchors = np.empty((k, d), dtype=float)
    anchor_idx = np.empty((k, d), dtype=int)

    u_all = []
    probs_all = []

    for j in range(d):
        axis = np.asarray(axis_grids[j], dtype=float).reshape(-1)
        if axis.size == 0:
            raise ValueError("Each axis grid must contain at least one point.")

        u_j = _compute_1d_indicator_dap_utility(
            x=X[:, j],
            z_axis=axis,
            h_score_used=h_score_used,
        )
        base_logits = 0.5 * n * eps_draw * u_j

        active = np.ones(axis.size, dtype=bool)
        seen_once = np.zeros(axis.size, dtype=bool)
        probs_accum_j = np.zeros(axis.size, dtype=float)

        for ell in range(k):
            if not np.any(active):
                remaining = ~seen_once
                if np.any(remaining):
                    active = remaining.copy()
                else:
                    active = np.ones(axis.size, dtype=bool)

            active_idx = np.flatnonzero(active)
            active_probs = _stable_softmax_from_logits(base_logits[active_idx])

            probs_round = np.zeros(axis.size, dtype=float)
            probs_round[active_idx] = active_probs
            probs_accum_j += probs_round

            chosen = int(rng.choice(active_idx, p=active_probs))
            anchor_idx[ell, j] = chosen
            anchors[ell, j] = axis[chosen]
            seen_once[chosen] = True

            if suppression_radius > 0.0:
                active[np.abs(axis - axis[chosen]) <= suppression_radius] = False
            else:
                active[chosen] = False

        u_all.append(u_j)
        probs_all.append(probs_accum_j / float(max(1, k)))

    u = np.concatenate(u_all) if u_all else np.array([], dtype=float)
    probs = np.concatenate(probs_all) if probs_all else np.array([], dtype=float)
    candidates = _axis_candidates_as_embedded_matrix(axis_grids)
    return u, probs, anchor_idx, anchors, eps_draw, candidates


def _basic_composition_eps_draw(
    epsilon_init: float,
    k: int,
) -> float:
    """
    Per-draw epsilon under basic composition.
    """
    epsilon_init = float(epsilon_init)
    k = int(k)

    if not (epsilon_init > 0.0):
        return 0.0
    if k <= 0:
        raise ValueError("k must be positive.")

    return float(epsilon_init / k)


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
    """
    Sequential DAP initialization using indicator utility and hard suppression.
    """
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

    u = _compute_indicator_dap_utility(
        X=X,
        Z=Z,
        h_score_used=h_score_used,
    )

    eps_draw = _basic_composition_eps_draw(
        epsilon_init=epsilon_init,
        k=k,
    )
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


def _matern_half_kernel_matrix(
    A: np.ndarray,
    B: Optional[np.ndarray] = None,
    length_scale: float = 1.0,
) -> np.ndarray:
    """
    Matérn kernel with nu = 0.5.
    """
    A = np.asarray(A, dtype=float)
    if B is None:
        B = A
    else:
        B = np.asarray(B, dtype=float)

    D = cdist(A, B, metric="euclidean")
    return np.exp(-D / float(length_scale))


def _factor_psd_matrix(K: np.ndarray) -> np.ndarray:
    """
    Return a factor L such that L L^T approximates K.
    """
    K = np.asarray(K, dtype=float)
    try:
        return cholesky(K)
    except Exception:
        vals, vecs = eigh(K)
        vals = np.maximum(vals, 0.0)
        return (vecs * np.sqrt(vals)).astype(float)


def dp_grams(
    X: np.ndarray,
    epsilon: float,
    delta: float,
    *,
    R: Optional[float] = None,
    initial_modes: Optional[np.ndarray] = None,
    candidate_points: Optional[np.ndarray] = None,
    T: Optional[int] = None,
    cT: float = 1.0,
    m: Optional[int] = None,
    h: Optional[float] = None,
    h_dap: Optional[float] = None,
    beta_dap: float = 3.0,
    kappa_init: float = 4.0,
    suppression_radius: Optional[float] = None,
    c_rho: float = 2.0,
    rng: Optional[np.random.Generator] = None,
    clip_multiplier: float = 1,
    init_epsilon_frac: float = 0.5, # CHANGE THIS!!!
    eta: float = 1.0,
    dap_score_multiplier: float = 3.0,
    dap_init_strategy: str = "grid",
    return_init_info: bool = False,
) -> Union[
    Tuple[float, np.ndarray],
    Tuple[float, np.ndarray, DPGramsInitInfo],
]:
    """
    Run DP-GRAMS to obtain private mode estimates.

    Parameters
    ----------
    X:
        Data array with shape (n, d).
    epsilon, delta:
        Privacy parameters for the full DP-GRAMS procedure.
    R:
        Public data bound for the DAP candidate grid. When candidate_points is
        not supplied, the default candidate grid is built over [-R, R]^d. If R
        is None, R is estimated non-privately from X.
    candidate_points:
        Optional public candidate set. If supplied, it overrides the default
        box-grid construction and R is only recorded diagnostically.
    c_rho:
        Suppression-radius multiplier: rho_init = c_rho * log(n)^(-1/d).
    """
    if rng is None:
        rng = np.random.default_rng()

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n,d). Got shape {X.shape}.")
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
    if not np.isfinite(eta):
        raise ValueError("eta must be finite.")
    if not (np.isfinite(kappa_init) and kappa_init > 0.0):
        raise ValueError("kappa_init must be finite and > 0.")
    if not (np.isfinite(cT) and cT > 0.0):
        raise ValueError("cT must be finite and > 0.")
    if not (np.isfinite(beta_dap) and beta_dap > 0.0):
        raise ValueError("beta_dap must be finite and > 0.")
    if not (np.isfinite(c_rho) and c_rho > 0.0):
        raise ValueError("c_rho must be finite and > 0.")
    if not (np.isfinite(dap_score_multiplier) and dap_score_multiplier > 0.0):
        raise ValueError("dap_score_multiplier must be finite and > 0.")

    dap_init_strategy = str(dap_init_strategy).lower()
    if dap_init_strategy not in {"grid", "factorized"}:
        raise ValueError(
            "dap_init_strategy must be either 'grid' or 'factorized'."
        )

    if T is None:
        T_base = np.log(max(2, n))
    else:
        T_base = float(T)
    if not (np.isfinite(T_base) and T_base > 0.0):
        raise ValueError("Base T must be finite and > 0.")

    T = max(1, int(np.ceil(float(cT) * T_base)))

    if m is None:
        denom = np.log(max(3, n))
        m = int(n / denom)
    m = int(max(1, min(n, m)))

    polylog = np.log(2.0 / delta) * np.log(2.5 * m * max(T, 1) / (n * delta))
    Kconst = T * d * polylog

    h_non_dp = (np.log(max(3, n)) / n) ** (1.0 / (d + 6))
    eps_non = np.sqrt(Kconst) / (
        n ** (3.0 / (d + 6)) * np.log(max(3, n))
    )

    epsilon_init = float(init_epsilon_frac * epsilon)
    epsilon_alg = float(max(1e-12, epsilon - epsilon_init))

    if h is None:
        if epsilon_alg <= eps_non:
            h_used = (Kconst / (n ** 2 * epsilon_alg ** 2)) ** (
                1.0 / (2.0 * d + 6.0)
            )
        else:
            h_used = h_non_dp
    else:
        h_used = float(h)

    if not (h_used > 0) or not np.isfinite(h_used):
        raise ValueError(f"Invalid ascent bandwidth h_used={h_used}.")

    if h_dap is None:
        c_h = 1.0
        h_dap_used = c_h * (
            np.log(max(3, n)) / n
        ) ** (1.0 / (d + 2.0 * beta_dap))
    else:
        h_dap_used = float(h_dap)

    if not (h_dap_used > 0) or not np.isfinite(h_dap_used):
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
    h2 = h_used * h_used

    init_info: Optional[DPGramsInitInfo] = None

    if initial_modes is None:
        k = max(1, int(np.ceil(float(kappa_init) * np.log(max(3, n)))))

        if suppression_radius is None:
            suppression_radius_used = _default_dap_suppression_radius(
                n=n,
                d=d,
                c_rho=c_rho,
            )
        else:
            suppression_radius_used = float(max(0.0, suppression_radius))

        if dap_init_strategy == "grid":
            if candidate_points is None:
                Z = _build_public_box_candidate_grid(
                    d=d,
                    h_dap_used=h_dap_used,
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

            u, probs, anchor_idx, anchors, eps_draw = _sequential_dap_initialize(
                X=X,
                Z=Z,
                h_score_used=h_score_used,
                epsilon_init=epsilon_init,
                k=k,
                rng=rng,
                suppression_radius=suppression_radius_used,
                c_rho=c_rho,
            )
            candidates_for_info = Z

        else:  # dap_init_strategy == "factorized"
            if candidate_points is not None:
                raise ValueError(
                    "candidate_points is not supported with "
                    "dap_init_strategy='factorized'. Factorized DAP builds "
                    "public one-dimensional grids internally and never uses "
                    "data points as candidates."
                )

            axis_grids = _build_public_axis_candidate_grids(
                d=d,
                h_dap_used=h_dap_used,
                R=R_used,
            )
            (
                u,
                probs,
                anchor_idx,
                anchors,
                eps_draw,
                candidates_for_info,
            ) = _sequential_factorized_dap_initialize(
                X=X,
                axis_grids=axis_grids,
                h_score_used=h_score_used,
                epsilon_init=epsilon_init,
                k=k,
                rng=rng,
                suppression_radius=suppression_radius_used,
                c_rho=c_rho,
            )

        # DAP anchors are used directly as initial modes.
        modes = anchors.copy()

        if return_init_info:
            init_info = DPGramsInitInfo(
                u=u.copy(),
                probs=probs.copy(),
                anchor_idx=anchor_idx.copy(),
                anchors=anchors.copy(),
                init_modes=modes.copy(),
                candidates=candidates_for_info.copy(),
                k=int(k),
                epsilon_init=float(epsilon_init),
                eps_draw=float(eps_draw),
                h_used=float(h_used),
                h_dap_used=float(h_dap_used),
                h_score_used=float(h_score_used),
                R_used=float(R_used),
                c_rho=float(c_rho),
                d_ambient=int(d),
                dap_init_strategy=str(dap_init_strategy),
            )
    else:
        modes = np.asarray(initial_modes, dtype=float).copy()
        if modes.ndim != 2 or modes.shape[1] != d:
            raise ValueError(
                f"initial_modes must have shape (k,{d}). Got {modes.shape}."
            )
        if not np.all(np.isfinite(modes)):
            raise ValueError("initial_modes must contain only finite values.")

        if return_init_info:
            init_info = DPGramsInitInfo(
                u=np.array([], dtype=float),
                probs=np.array([], dtype=float),
                anchor_idx=np.array([], dtype=int),
                anchors=np.empty((0, d), dtype=float),
                init_modes=modes.copy(),
                candidates=np.empty((0, d), dtype=float),
                k=int(modes.shape[0]),
                epsilon_init=float(epsilon_init),
                eps_draw=0.0,
                h_used=float(h_used),
                h_dap_used=float(h_dap_used),
                h_score_used=float(h_score_used),
                R_used=float(R_used),
                c_rho=float(c_rho),
                d_ambient=int(d),
                dap_init_strategy="provided_initial_modes",
            )

    k = int(modes.shape[0])
    if k == 0:
        empty = np.empty((0, d), dtype=float)
        if return_init_info and init_info is not None:
            return h_used, empty, init_info
        return h_used, empty

    epsilon_iter = epsilon_alg / (
        2.0 * np.sqrt(2.0 * T * np.log(2.0 / delta))
    )

    def compute_sigma(C: float) -> float:
        q = m / n
        denom = np.log(
            1.0 + (1.0 / max(1e-12, q)) * (np.exp(epsilon_iter) - 1.0)
        )
        denom = max(1e-12, denom)

        sigma_exact = (2.0 * C / m) / denom * np.sqrt(
            2.0 * np.log(2.5 * m * T / (n * delta))
        )

        if epsilon_alg <= 1.0:
            sigma_approx = (8.0 * C / (n * epsilon_alg)) * np.sqrt(
                T
                * np.log(2.0 / delta)
                * np.log(2.5 * m * T / (n * delta))
            )
            return float(sigma_approx)

        return float(sigma_exact)

    C = (h_used ** (1.0 - d)) * float(clip_multiplier)
    sigma = compute_sigma(C)

    modes_curr = modes.copy()

    for _t in range(T):
        K_modes = _matern_half_kernel_matrix(
            modes_curr,
            modes_curr,
            length_scale=h_used,
        ) + 1e-8 * np.eye(k)
        L_curr = _factor_psd_matrix(K_modes)

        avg_grads = np.zeros_like(modes_curr)

        batch_indices = rng.choice(n, m, replace=False)
        batch = X[batch_indices]

        for i in range(k):
            x = modes_curr[i]

            diffs = batch - x
            weights = np.exp(-np.sum(diffs ** 2, axis=1) / (2.0 * h2))
            weights_sum = float(np.sum(weights) + 1e-12)

            q_i = (weights[:, None] * diffs) / weights_sum

            norms = np.linalg.norm(q_i, axis=1, keepdims=True)
            scales = np.minimum(1.0, C / (norms + 1e-12))
            q_i_clipped = q_i * scales

            avg_grads[i] = np.sum(q_i_clipped, axis=0)

        G = rng.normal(size=(k, d))
        noise = sigma * (L_curr @ G)
        modes_curr += float(eta) * (avg_grads + noise)

    final_modes = modes_curr.copy()

    if return_init_info and init_info is not None:
        return h_used, final_modes, init_info
    return h_used, final_modes
