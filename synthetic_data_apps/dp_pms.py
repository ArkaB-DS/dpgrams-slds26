# dp_pms.py

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from numpy.linalg import cholesky, eigh

from main_scripts.bandwidth import silverman_bandwidth


def gaussian_kernel_1d(u: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * (u ** 2))


def _stable_softmax_from_logits(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    if logits.size == 0:
        return np.array([], dtype=float)

    logits = logits - np.max(logits)
    w = np.exp(logits)
    w_sum = float(np.sum(w))

    if (not np.isfinite(w_sum)) or (w_sum <= 0.0):
        return np.ones_like(w, dtype=float) / float(len(w))

    return w / w_sum


def _resolve_grid_domain(values, public_domain, *, pad):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("values must be non-empty.")

    if public_domain is None:
        lo = float(np.min(values)) - float(pad)
        hi = float(np.max(values)) + float(pad)
    else:
        lo = float(public_domain[0])
        hi = float(public_domain[1])

    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError("domain must contain finite values.")
    if hi < lo:
        raise ValueError("domain must satisfy lo <= hi.")

    if hi == lo:
        hi = lo + max(1e-12, 1e-6 * max(1.0, abs(lo)))

    return lo, hi


def _default_dap_suppression_radius(n: int, d_joint: int, h_dap_used: float) -> float:
    """
    Baseline-style DAP suppression radius:
        rho_init ~ c_rho * (1 / log n)^{1/d_joint}
    """
    del h_dap_used
    log_term = np.log(max(3, int(n)))
    c_rho = 2.5
    rho_log = c_rho * log_term ** (-1.0 / max(1, int(d_joint)))
    return float(max(1e-12, float(rho_log)))


def _build_public_xy_modal_candidate_grid(
    X,
    Y,
    h_dap_used,
    *,
    x_domain=None,
    y_domain=None,
):
    """
    Candidate lattice in joint (x,y)-space.
    """
    X = np.asarray(X, dtype=float).reshape(-1)
    Y = np.asarray(Y, dtype=float).reshape(-1)

    ax, bx = _resolve_grid_domain(X, x_domain, pad=0.0)
    ay, by = _resolve_grid_domain(Y, y_domain, pad=0.0)

    h_dap_used = float(max(1e-12, h_dap_used))

    xs = np.arange(ax, bx + 1e-12, h_dap_used, dtype=float)
    ys = np.arange(ay, by + 1e-12, h_dap_used, dtype=float)

    mesh_x, mesh_y = np.meshgrid(xs, ys, indexing="ij")
    Z = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
    return Z.astype(float, copy=False)


def _compute_conditional_mass_dap_utility(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    h_dap_used: float,
) -> np.ndarray:
    """
    Conditional DAP score:
        u(x,y) = #{|X_j-x|<=h_dap, |Y_j-y|<=h_dap} / #{|X_j-x|<=h_dap}
    """
    X = np.asarray(X, dtype=float).reshape(-1)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    Z = np.asarray(Z, dtype=float)

    if Z.ndim != 2 or Z.shape[1] != 2:
        raise ValueError(f"Z must have shape (N,2). Got {Z.shape}.")

    u = np.zeros(len(Z), dtype=float)

    for i, (x0, y0) in enumerate(Z):
        xmask = np.abs(X - x0) <= h_dap_used
        denom = int(np.sum(xmask))
        if denom <= 0:
            u[i] = 0.0
        else:
            num = int(np.sum(xmask & (np.abs(Y - y0) <= h_dap_used)))
            u[i] = float(num) / float(denom)

    return u


def _sequential_conditional_dap_initialize_on_support(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    h_dap_used: float,
    epsilon_init: float,
    k: int,
    rng: np.random.Generator,
    suppression_radius: float | None = None,
):
    """
    Score a public grid conditionally, but sample anchors on observed support.
    This preserves the good old PMS behavior while making init density-aware.
    """
    X = np.asarray(X, dtype=float).reshape(-1)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    XY = np.column_stack([X, Y])

    n = len(X)

    if suppression_radius is None:
        suppression_radius = _default_dap_suppression_radius(
            n=n,
            d_joint=2,
            h_dap_used=h_dap_used,
        )
    suppression_radius = float(max(0.0, suppression_radius))

    u_grid = _compute_conditional_mass_dap_utility(
        X=X,
        Y=Y,
        Z=Z,
        h_dap_used=h_dap_used,
    )

    # Transfer grid scores back to observed support
    nn_idx = np.argmin(cdist(XY, Z, metric="euclidean"), axis=1)
    score_obs = u_grid[nn_idx]

    eps_draw = float(epsilon_init / max(1, k))
    base_logits = 0.5 * n * eps_draw * score_obs

    anchor_idx = np.empty(k, dtype=int)
    anchors = np.empty((k, 2), dtype=float)

    probs_accum = np.zeros(n, dtype=float)
    active = np.ones(n, dtype=bool)
    seen_once = np.zeros(n, dtype=bool)

    radius2 = suppression_radius * suppression_radius

    for ell in range(k):
        if not np.any(active):
            remaining = ~seen_once
            if np.any(remaining):
                active = remaining.copy()
            else:
                active = np.ones(n, dtype=bool)

        active_idx = np.flatnonzero(active)
        active_probs = _stable_softmax_from_logits(base_logits[active_idx])

        probs_round = np.zeros(n, dtype=float)
        probs_round[active_idx] = active_probs
        probs_accum += probs_round

        chosen = int(rng.choice(active_idx, p=active_probs))
        anchor_idx[ell] = chosen
        anchors[ell] = XY[chosen]
        seen_once[chosen] = True

        if suppression_radius > 0.0:
            diff = XY - XY[chosen]
            dist2 = np.sum(diff ** 2, axis=1)
            active[dist2 <= radius2] = False
        else:
            active[chosen] = False

    probs_avg = probs_accum / float(max(1, k))
    return u_grid, score_obs, probs_avg, anchor_idx, anchors, eps_draw, suppression_radius


def _matern_half_kernel_matrix(A, B=None, hloc=1.0):
    """
    DP-GRAMS-style exponential/Laplace correlation kernel:
        K(a,b) = exp(-|a-b| / hloc)
    """
    A = np.asarray(A, dtype=float).reshape(-1, 1)
    if B is None:
        B = A
    else:
        B = np.asarray(B, dtype=float).reshape(-1, 1)

    D = cdist(A, B, metric="euclidean")
    return np.exp(-D / float(hloc))


def _factor_psd_matrix(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=float)
    try:
        return cholesky(K)
    except Exception:
        vals, vecs = eigh(K)
        vals[vals < 0.0] = 0.0
        return (vecs * np.sqrt(vals)).astype(float)


def _resolve_initial_state(
    X,
    Y,
    mesh_points,
    epsilon_init,
    rng,
    *,
    h_dap_used,
    kappa_init,
    suppression_radius,
    x_domain,
    y_domain,
    candidate_points,
):
    """
    Resolve sparse DP-PMS state (x_mesh, y_mesh).

    If mesh_points is None:
      - run conditional DAP in joint (x,y)-space,
      - sample sparse anchors on observed support,
      - no jitter.

    If mesh_points is provided:
      - if tuple/list length 2, interpret as (x_mesh, y_mesh),
      - if 1D array length n, interpret as y_mesh with x_mesh = X.
    """
    X = np.asarray(X, dtype=float).reshape(-1)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    n = len(X)

    init_info = {
        "u_grid": None,
        "score_obs": None,
        "probs": None,
        "anchor_idx": None,
        "anchors": None,
        "candidates": None,
        "eps_draw": None,
        "k_init": None,
        "suppression_radius_used": None,
    }

    if mesh_points is None:
        k_init = max(1, int(np.ceil(float(kappa_init) * np.log(max(3, n)))))

        if candidate_points is None:
            Z = _build_public_xy_modal_candidate_grid(
                X,
                Y,
                h_dap_used,
                x_domain=x_domain,
                y_domain=y_domain,
            )
        else:
            Z = np.asarray(candidate_points, dtype=float)
            if Z.ndim != 2 or Z.shape[1] != 2:
                raise ValueError(
                    f"candidate_points must have shape (N_cand,2). Got {Z.shape}."
                )

        if Z.shape[0] == 0:
            raise ValueError("candidate_points must contain at least one point.")

        u_grid, score_obs, probs, anchor_idx, anchors, eps_draw, sr = (
            _sequential_conditional_dap_initialize_on_support(
                X=X,
                Y=Y,
                Z=Z,
                h_dap_used=h_dap_used,
                epsilon_init=epsilon_init,
                k=k_init,
                rng=rng,
                suppression_radius=suppression_radius,
            )
        )

        x_mesh = anchors[:, 0].copy()
        y_mesh = anchors[:, 1].copy()

        init_info.update(
            {
                "u_grid": u_grid,
                "score_obs": score_obs,
                "probs": probs,
                "anchor_idx": anchor_idx,
                "anchors": anchors,
                "candidates": Z,
                "eps_draw": eps_draw,
                "k_init": int(k_init),
                "suppression_radius_used": float(sr),
            }
        )
        return x_mesh, y_mesh, init_info

    if isinstance(mesh_points, (tuple, list)) and len(mesh_points) == 2:
        x_mesh = np.asarray(mesh_points[0], dtype=float).reshape(-1)
        y_mesh = np.asarray(mesh_points[1], dtype=float).reshape(-1)
        if len(x_mesh) != len(y_mesh):
            raise ValueError("x_mesh and y_mesh must have the same length.")
        return x_mesh, y_mesh, init_info

    y_mesh = np.asarray(mesh_points, dtype=float).reshape(-1)
    if len(y_mesh) != n:
        raise ValueError(
            "If mesh_points is 1D, it must have length n. "
            "Otherwise pass mesh_points=(x_mesh, y_mesh)."
        )
    x_mesh = X.copy()
    return x_mesh, y_mesh, init_info


def dp_pms(
    X,
    Y,
    mesh_points=None,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    T: int | None = None,
    m: int | None = None,
    clip_multiplier: float = 0.01,
    rng=None,
    bandwidth=None,
    verbose: bool = False,
    init_epsilon_frac: float = 0.1,
    x_domain=None,
    y_domain=None,
    return_x_positions: bool = False,
    h_dap: float | None = None,
    beta_dap: float = 3.0,
    kappa_init: float = 20.0,
    suppression_radius: float | None = None,
    candidate_points=None,
    eta: float | None = None,
    eta_multiplier: float = 1.0,
    noise_matern_h: float | None = None,
    return_init_info: bool = False,
):
    """
    Differentially Private Partial Mean Shift (DP-PMS) for 1D modal regression.

    Mechanism:
      - conditional DAP init in joint (x,y)-space,
      - anchors sampled on observed support,
      - old PMS ascent drift kept unchanged,
      - DP-GRAMS-style exponential correlation refreshed each iteration.
    """
    if rng is None:
        rng = np.random.default_rng()

    X = np.asarray(X, dtype=float).reshape(-1)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    n = len(X)

    if len(Y) != n:
        raise ValueError("X and Y must have the same length.")
    if n < 2:
        raise ValueError("X and Y must contain at least 2 samples.")
    if not (epsilon > 0):
        raise ValueError("epsilon must be > 0.")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1).")
    if not (0 < init_epsilon_frac < 1):
        raise ValueError("init_epsilon_frac must be in (0,1).")
    if not (np.isfinite(kappa_init) and kappa_init > 0.0):
        raise ValueError("kappa_init must be finite and > 0.")
    if not (np.isfinite(beta_dap) and beta_dap > 0.0):
        raise ValueError("beta_dap must be finite and > 0.")
    if not np.isfinite(eta_multiplier):
        raise ValueError("eta_multiplier must be finite.")

    # ------------------------------------------------------------
    # Ascent bandwidth
    # ------------------------------------------------------------
    if bandwidth is None:
        bandwidth = float(silverman_bandwidth(Y.reshape(-1, 1)))
    h = max(1e-3, float(bandwidth))
    h2 = h ** 2

    # ------------------------------------------------------------
    # Default T and m
    # ------------------------------------------------------------
    if T is None:
        T = int(np.ceil(np.log(max(n, 2))))

    if m is None:
        m = int(n / max(np.log(max(n, 2)), 1.0))
    m = max(1, min(int(m), n))

    # ------------------------------------------------------------
    # Conditional DAP bandwidth
    # Default: tie to ascent bandwidth. This is what worked.
    # ------------------------------------------------------------
    if h_dap is None:
        h_dap_used = float(h)
    else:
        h_dap_used = float(h_dap)

    if not (np.isfinite(h_dap_used) and h_dap_used > 0.0):
        raise ValueError(f"Invalid h_dap={h_dap_used}.")

    # ------------------------------------------------------------
    # Split privacy
    # ------------------------------------------------------------
    eps_global = float(epsilon)
    delta_global = float(delta)

    epsilon_init = float(init_epsilon_frac) * eps_global
    epsilon_alg = max(1e-12, eps_global - epsilon_init)

    # ------------------------------------------------------------
    # Resolve sparse initial state
    # ------------------------------------------------------------
    x_mesh, y_mesh, init_info = _resolve_initial_state(
        X,
        Y,
        mesh_points,
        epsilon_init,
        rng,
        h_dap_used=h_dap_used,
        kappa_init=kappa_init,
        suppression_radius=suppression_radius,
        x_domain=x_domain,
        y_domain=y_domain,
        candidate_points=candidate_points,
    )

    k = len(y_mesh)
    if len(x_mesh) != k:
        raise ValueError("x_mesh and y_mesh must have the same length.")
    if k == 0:
        empty = np.empty((0,), dtype=float)
        if return_x_positions:
            if return_init_info:
                return empty, empty, init_info
            return empty, empty
        if return_init_info:
            return empty, init_info
        return empty

    # ------------------------------------------------------------
    # Clipping constant
    # ------------------------------------------------------------
    C = clip_multiplier * (1.0 / h)

    # ------------------------------------------------------------
    # Noise scale
    # ------------------------------------------------------------
    eps_safe = max(epsilon_alg, 1e-12)
    delta_safe = max(delta_global, 1e-12)
    q_samp = m / n

    eps_iter = eps_safe / (2.0 * np.sqrt(2.0 * T * np.log(2.0 / delta_safe)))

    def compute_sigma():
        denom = np.log(
            1.0 + (1.0 / max(1e-12, q_samp)) * (np.exp(eps_iter) - 1.0)
        )
        denom = max(1e-12, denom)

        sigma_exact = (2.0 * C / m) / denom * np.sqrt(
            2.0 * np.log(2.5 * m * max(T, 1) / (n * delta_safe))
        )

        if eps_safe <= 1.0:
            sigma_approx = (8.0 * C / (n * eps_safe)) * np.sqrt(
                T
                * np.log(2.0 / delta_safe)
                * np.log(2.5 * m * max(T, 1) / (n * delta_safe))
            )
            return sigma_approx

        return sigma_exact

    sigma = compute_sigma()

    if noise_matern_h is None:
        noise_matern_h = h
    noise_matern_h = float(noise_matern_h)

    if eta is None:
        eta = float(eta_multiplier)
    else:
        eta = float(eta)

    if verbose:
        print(
            f"[dp_pms] n={n}, k={k}, h={h:.6f}, h_dap={h_dap_used:.6f}, "
            f"C={C:.6e}, sigma={sigma:.6e}, T={T}, m={m}, eta={eta:.6f}, "
            f"eps_global={eps_global:.3g}, eps_init={epsilon_init:.3g}, "
            f"eps_alg={epsilon_alg:.3g}, delta={delta_global:.1e}"
        )
        if init_info["suppression_radius_used"] is not None:
            print(
                f"[dp_pms] suppression_radius={init_info['suppression_radius_used']:.6f}"
            )

    # ------------------------------------------------------------
    # DP-PMS ascent
    # Old working PMS drift, unchanged
    # ------------------------------------------------------------
    y_new = y_mesh.copy()

    for _ in range(T):
        delta_vec = np.zeros(k, dtype=float)

        for i in range(k):
            xi = x_mesh[i]
            yi = y_new[i]

            if m == n:
                batch_idx = np.arange(n)
            else:
                batch_idx = rng.choice(n, size=m, replace=False)

            Xb = X[batch_idx]
            Yb = Y[batch_idx]

            # old working PMS drift
            w = np.exp(-((Xb - xi) ** 2 + (Yb - yi) ** 2) / (2.0 * h2))
            w_sum = float(np.sum(w))

            if w_sum > 1e-12:
                q = (w * (Yb - yi)) / (w_sum + 1e-12)

                abs_q = np.abs(q)
                scales = np.minimum(1.0, C / (abs_q + 1e-12))
                q_clipped = q * scales

                delta_vec[i] = np.sum(q_clipped)
            else:
                delta_vec[i] = 0.0

        # dynamic DP-GRAMS-style exponential correlation
        K_y = _matern_half_kernel_matrix(
            y_new,
            y_new,
            hloc=noise_matern_h,
        ) + 1e-8 * np.eye(k)
        L_curr = _factor_psd_matrix(K_y)

        G = rng.normal(size=(k, 1))
        noise = (L_curr @ G).reshape(-1)

        y_new = y_new + float(eta) * delta_vec + sigma * noise

    if return_x_positions:
        if return_init_info:
            return x_mesh.copy(), y_new, init_info
        return x_mesh.copy(), y_new

    if return_init_info:
        return y_new, init_info
    return y_new