# dp_grams_c.py

import os
import sys
from typing import Optional, Tuple, Union

import numpy as np

# Ensure local imports work when run as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main_scripts.dp_grams import DPGramsInitInfo, dp_grams
from main_scripts.merge import merge_modes, merge_modes_agglomerative


def dpms_private(
    data: np.ndarray,
    epsilon_modes: float,
    delta: float = 1e-5,
    R: Optional[float] = None,
    h: Optional[float] = None,
    bandwidth_multiplier: Optional[float] = None,
    clip_multiplier: float = 1.0,
    m: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    k_est: Optional[int] = None,
    kappa_init: float = 5.0,
    eta: float = 1.0,
    cT: float = 1.0,
    suppression_radius: Optional[float] = None,
    c_rho: float = 2.0,
    h_dap: Optional[float] = None,
    beta_dap: float = 3.0,
    candidate_points: Optional[np.ndarray] = None,
    init_epsilon_frac: float = 0.1,
    dap_score_multiplier: float = 3.0,
    dap_init_strategy: str = "grid",
    return_info: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[Tuple[np.ndarray, np.ndarray], float, DPGramsInitInfo],
]:
    """
    Differentially private clustering via DP-GRAMS followed by deterministic
    nearest-center assignment.

    Pipeline
    --------
    1. Run DP-GRAMS on ``data`` to obtain private mode estimates.
    2. If ``k_est`` is provided, merge private modes by agglomerative clustering
       into at most ``k_est`` centers.
       Otherwise, use the default radius-based ``merge_modes`` routine.
    3. Assign each data point to the nearest merged private center.

    Notes
    -----
    ``R``, ``c_rho``, and ``dap_init_strategy`` are passed through to the
    current DAP initialization in ``dp_grams``. Anchor mean-shift smoothing
    arguments are intentionally not
    exposed here because the current cleaned ``dp_grams`` uses DAP anchors
    directly as initial modes.
    """
    if rng is None:
        rng = np.random.default_rng()

    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be a 2D array of shape (n, d).")
    if not np.all(np.isfinite(data)):
        raise ValueError("data must contain only finite values.")

    n, d = data.shape
    if n < 2:
        raise ValueError("data must contain at least 2 samples.")
    if not (epsilon_modes > 0):
        raise ValueError("epsilon_modes must be > 0.")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1).")
    if R is not None:
        if not (np.isfinite(R) and R > 0):
            raise ValueError("R must be None or a finite positive number.")
    if not (np.isfinite(clip_multiplier) and clip_multiplier > 0):
        raise ValueError("clip_multiplier must be finite and > 0.")
    if not (np.isfinite(kappa_init) and kappa_init > 0):
        raise ValueError("kappa_init must be finite and > 0.")
    if not np.isfinite(eta):
        raise ValueError("eta must be finite.")
    if not (np.isfinite(cT) and cT > 0):
        raise ValueError("cT must be finite and > 0.")
    if not (np.isfinite(c_rho) and c_rho > 0):
        raise ValueError("c_rho must be finite and > 0.")
    if not (np.isfinite(beta_dap) and beta_dap > 0):
        raise ValueError("beta_dap must be finite and > 0.")
    if not (0 < init_epsilon_frac < 1):
        raise ValueError("init_epsilon_frac must be in (0,1).")
    if bandwidth_multiplier is not None:
        if not (np.isfinite(bandwidth_multiplier) and bandwidth_multiplier > 0):
            raise ValueError(
                "bandwidth_multiplier must be None or a finite positive number."
            )
    if h is not None:
        if not (np.isfinite(h) and h > 0):
            raise ValueError("h must be None or a finite positive number.")
    if h is not None and bandwidth_multiplier is not None:
        raise ValueError("Provide at most one of h and bandwidth_multiplier.")
    if h_dap is not None:
        if not (np.isfinite(h_dap) and h_dap > 0):
            raise ValueError("h_dap must be None or a finite positive number.")
    if k_est is not None:
        if not (isinstance(k_est, (int, np.integer)) and k_est > 0):
            raise ValueError("k_est must be None or a positive integer.")
    if candidate_points is not None:
        candidate_points = np.asarray(candidate_points, dtype=float)
        if candidate_points.ndim != 2 or candidate_points.shape[1] != d:
            raise ValueError(
                f"candidate_points must have shape (N_cand, {d}). "
                f"Got {candidate_points.shape}."
            )
        if not np.all(np.isfinite(candidate_points)):
            raise ValueError("candidate_points must contain only finite values.")
    if not (np.isfinite(dap_score_multiplier) and dap_score_multiplier > 0):
        raise ValueError("dap_score_multiplier must be finite and > 0.")
    dap_init_strategy = str(dap_init_strategy).lower()
    if dap_init_strategy not in {"grid", "factorized"}:
        raise ValueError("dap_init_strategy must be either 'grid' or 'factorized'.")

    if h is None and bandwidth_multiplier is not None:
        base_h = (np.log(max(3, n)) / n) ** (1.0 / (d + 6.0))
        h_used_arg = float(base_h) * float(bandwidth_multiplier)
    else:
        h_used_arg = h

    dp_kwargs = dict(
        X=data,
        epsilon=epsilon_modes,
        delta=delta,
        R=R,
        rng=rng,
        T=None,
        cT=cT,
        h=h_used_arg,
        h_dap=h_dap,
        beta_dap=beta_dap,
        m=m,
        clip_multiplier=clip_multiplier,
        kappa_init=kappa_init,
        eta=eta,
        suppression_radius=suppression_radius,
        c_rho=c_rho,
        candidate_points=candidate_points,
        init_epsilon_frac=init_epsilon_frac,
        dap_score_multiplier=dap_score_multiplier,
        dap_init_strategy=dap_init_strategy,
    )

    if return_info:
        h_used, dp_modes, init_info = dp_grams(
            **dp_kwargs,
            return_init_info=True,
        )
    else:
        h_used, dp_modes = dp_grams(
            **dp_kwargs,
            return_init_info=False,
        )
        init_info = None

    if dp_modes.size == 0:
        empty_modes = np.empty((0, d), dtype=float)
        empty_labels = np.zeros(n, dtype=int)
        if return_info and init_info is not None:
            return (empty_modes, empty_labels), float(h_used), init_info
        return empty_modes, empty_labels

    dp_modes = np.atleast_2d(np.asarray(dp_modes, dtype=float))
    if dp_modes.ndim != 2 or dp_modes.shape[1] != d:
        raise ValueError(
            f"dp_grams returned modes with shape {dp_modes.shape}; expected (*, {d})."
        )

    if k_est is None:
        merged_modes = merge_modes(dp_modes)
    else:
        # Agglomerative merging cannot request more clusters than available
        # private mode estimates. Cap to avoid sklearn/agglomerative failures.
        n_private_modes = int(dp_modes.shape[0])
        n_clusters = min(int(k_est), n_private_modes)
        merged_modes = merge_modes_agglomerative(dp_modes, n_clusters=n_clusters)

    if merged_modes.size == 0:
        empty_modes = np.empty((0, d), dtype=float)
        empty_labels = np.zeros(n, dtype=int)
        if return_info and init_info is not None:
            return (empty_modes, empty_labels), float(h_used), init_info
        return empty_modes, empty_labels

    merged_modes = np.atleast_2d(np.asarray(merged_modes, dtype=float))
    if merged_modes.ndim != 2 or merged_modes.shape[1] != d:
        raise ValueError(
            f"merged modes have shape {merged_modes.shape}; expected (*, {d})."
        )

    diffs = data[:, None, :] - merged_modes[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    labels = np.argmin(dists, axis=1).astype(int)

    if return_info and init_info is not None:
        return (merged_modes, labels), float(h_used), init_info

    return merged_modes, labels
