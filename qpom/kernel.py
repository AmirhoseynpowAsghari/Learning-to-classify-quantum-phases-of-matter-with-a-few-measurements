"""Squared-fidelity quantum kernel, k(x_n, x_m) = Tr[rho(x_n) rho(x_m)] =
|<psi(x_n)|psi(x_m)>|^2 (paper eq. (1)), for pure states.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import quimb.tensor as qtn


def overlap(a: qtn.MatrixProductState, b: qtn.MatrixProductState) -> complex:
    """<a|b> for two MPS (assumed normalized)."""
    return complex(a.H @ b)


def fidelity_kernel_matrix(
    states_a: Sequence[qtn.MatrixProductState],
    states_b: Sequence[qtn.MatrixProductState] | None = None,
) -> np.ndarray:
    """Dense matrix of squared-fidelity kernel entries |<a_i|b_j>|^2.

    If ``states_b`` is None, computes the (symmetric) Gram matrix of
    ``states_a`` with itself.
    """
    symmetric = states_b is None
    if symmetric:
        states_b = states_a

    na, nb = len(states_a), len(states_b)
    K = np.empty((na, nb), dtype=float)

    if symmetric:
        for i in range(na):
            K[i, i] = 1.0
            for j in range(i + 1, nb):
                ov = overlap(states_a[i], states_b[j])
                val = abs(ov) ** 2
                K[i, j] = val
                K[j, i] = val
    else:
        for i in range(na):
            for j in range(nb):
                K[i, j] = abs(overlap(states_a[i], states_b[j])) ** 2
    return K
