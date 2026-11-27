"""Appendix D: generalization-error bound B ~ Tr sqrt(K), estimated from m
random ground states of the ANNNI model, with

    K_jk = <psi(x_j)|psi(x_k)> / m          (note: NOT squared, unlike the
                                              classification kernel eq. (1))

For the small system sizes used here (N=4..16) we use exact diagonalization
(restricted to a single Z2-parity sector, exploiting that for the ANNNI
model parity = (-1)^(Hamming weight) is diagonal in the computational basis,
so the sector restriction is just a row/column gather) instead of DMRG, both
for speed and because it is numerically exact.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import quimb as qu
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

X = qu.pauli("X")
Z = qu.pauli("Z")


@dataclass
class ANNNIOperators:
    """Pre-built, N-dependent sparse pieces of the ANNNI Hamiltonian so that
    H(k, h) = -A + k*B - h*C can be assembled cheaply for many (k, h)."""

    N: int
    A: sp.spmatrix  # sum_j X_j X_{j+1}
    B: sp.spmatrix  # sum_j X_j X_{j+2}
    C: sp.spmatrix  # sum_j Z_j
    idx_even: np.ndarray
    idx_odd: np.ndarray


def _parity_indices(N: int):
    idx = np.arange(2 ** N, dtype=np.uint64)
    pc = np.zeros(2 ** N, dtype=np.int32)
    tmp = idx.copy()
    for _ in range(N):
        pc += (tmp & 1).astype(np.int32)
        tmp >>= 1
    even = np.flatnonzero(pc % 2 == 0)
    odd = np.flatnonzero(pc % 2 == 1)
    return even, odd


def build_annni_operators(N: int) -> ANNNIOperators:
    dims = [2] * N
    A = sum(qu.ikron([X, X], dims, [j, j + 1], sparse=True) for j in range(N - 1))
    if N >= 3:
        B = sum(qu.ikron([X, X], dims, [j, j + 2], sparse=True) for j in range(N - 2))
    else:
        B = 0 * A
    C = sum(qu.ikron([Z], dims, [j], sparse=True) for j in range(N))
    idx_even, idx_odd = _parity_indices(N)
    return ANNNIOperators(N=N, A=A.tocsr(), B=sp.csr_matrix(B).tocsr(), C=C.tocsr(),
                           idx_even=idx_even, idx_odd=idx_odd)


def sector_ground_state(ops: ANNNIOperators, k: float, h: float, sector: str) -> np.ndarray:
    """Exact ground state vector (length 2^(N-1)) restricted to the
    requested parity ``sector`` ('even' or 'odd'), expanded back to the full
    2^N-dimensional Hilbert space (zeros outside the sector)."""
    H = -ops.A + k * ops.B - h * ops.C
    idx = ops.idx_even if sector == "even" else ops.idx_odd
    Hsub = H[idx][:, idx]
    if Hsub.shape[0] <= 2:
        dense = Hsub.toarray()
        evals, evecs = np.linalg.eigh(dense)
        v_sub = evecs[:, 0]
    else:
        evals, evecs = eigsh(Hsub, k=1, which="SA")
        v_sub = evecs[:, 0]
    full = np.zeros(2 ** ops.N, dtype=float)
    full[idx] = v_sub
    return full


def generalization_bound_sq(
    N: int,
    sector: str,
    m: int,
    rng: np.random.Generator,
    ops: ANNNIOperators | None = None,
    h_range=(0.0, 2.0),
    k_range=(0.0, 1.0),
) -> float:
    """Estimate B^2 via eq. (D1) from m random uniform samples of (h, k)."""
    if ops is None:
        ops = build_annni_operators(N)
    idx = ops.idx_even if sector == "even" else ops.idx_odd
    dim_sub = len(idx)
    states = np.empty((m, dim_sub), dtype=float)
    hs = rng.uniform(*h_range, size=m)
    ks = rng.uniform(*k_range, size=m)
    for i in range(m):
        H = -ops.A + ks[i] * ops.B - hs[i] * ops.C
        Hsub = H[idx][:, idx]
        if Hsub.shape[0] <= 2:
            evals, evecs = np.linalg.eigh(Hsub.toarray())
            v = evecs[:, 0]
        else:
            evals, evecs = eigsh(Hsub, k=1, which="SA")
            v = evecs[:, 0]
        v = np.asarray(v).real
        # fix sign ambiguity for reproducibility (not needed numerically,
        # since we only use |overlap|, but keeps things tidy)
        if v[np.argmax(np.abs(v))] < 0:
            v = -v
        states[i] = v
    # K_jk = <psi_j|psi_k>/m  (real symmetric here since H is real)
    K = (states @ states.T) / m
    # B = Tr sqrt(K): use eigen-decomposition since K is symmetric PSD
    evals = np.linalg.eigvalsh(K)
    evals = np.clip(evals, 0, None)
    B = float(np.sum(np.sqrt(evals)))
    return B ** 2
