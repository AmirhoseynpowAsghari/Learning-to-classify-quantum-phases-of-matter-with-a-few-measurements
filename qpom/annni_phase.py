"""Theoretical ANNNI phase diagram: the three transition-line formulas of
paper eq. (8), and the resulting 4-class phase label used as ground truth
when labeling training points (and for drawing figure 1(a)).

Phases (paper figure 1(a)):
    0 = Ferromagnet
    1 = Paramagnet
    2 = Antiphase (AP)
    3 = Floating phase (thin sliver between AP and paramagnet, k>0.5)
"""
from __future__ import annotations

import numpy as np

FERRO, PARA, AP, FLOATING = 0, 1, 2, 3
PHASE_NAMES = {FERRO: "Ferromagnet", PARA: "Paramagnet", AP: "Antiphase", FLOATING: "Floating"}
PHASE_COLORS = {FERRO: "#3f7f74", PARA: "#f4c430", AP: "#7b2d8e", FLOATING: "#ffffff"}


def h_I(k: np.ndarray) -> np.ndarray:
    """Eq. (8a): ferromagnet <-> paramagnet 2nd order transition, k<0.5."""
    k = np.asarray(k, dtype=float)
    k_safe = np.where(k <= 1e-9, 1e-9, k)
    val = (1 - k_safe) / k_safe * (1 - np.sqrt(np.clip((1 - 3 * k_safe + 4 * k_safe ** 2) / (1 - k_safe), 0, None)))
    return np.where(k <= 1e-9, 1.0, val)


def h_KT(k: np.ndarray) -> np.ndarray:
    """Eq. (8b): Kosterlitz-Thouless transition, floating <-> paramagnet."""
    k = np.asarray(k, dtype=float)
    return 1.05 * np.sqrt(np.clip((k - 0.5) * (k - 0.1), 0, None))


def h_AP(k: np.ndarray) -> np.ndarray:
    """Eq. (8c): ferromagnet/antiphase <-> floating/antiphase transition."""
    k = np.asarray(k, dtype=float)
    return 1.05 * (k - 0.5)


def phase_label(k: float, h: float) -> int:
    """Ground-truth 4-class phase label for a point (k, h), k,h >= 0,
    following the geometry of figure 1(a)."""
    if k <= 0.5:
        return FERRO if h < h_I(k) else PARA
    hap = h_AP(k)
    hkt = h_KT(k)
    if h < hap:
        return AP
    elif h < hkt:
        return FLOATING
    else:
        return PARA


def phase_label_grid(k_grid: np.ndarray, h_grid: np.ndarray) -> np.ndarray:
    """Vectorized ``phase_label`` over broadcastable arrays of k and h."""
    k_grid = np.asarray(k_grid, dtype=float)
    h_grid = np.asarray(h_grid, dtype=float)
    k_b, h_b = np.broadcast_arrays(k_grid, h_grid)
    out = np.empty(k_b.shape, dtype=int)
    it = np.nditer(k_b, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        out[idx] = phase_label(float(k_b[idx]), float(h_b[idx]))
    return out
