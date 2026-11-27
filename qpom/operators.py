"""Finite state-machine (FSM) MPO builders for the two Hamiltonians studied
in the paper, plus small helpers (identity / parity product operators).

Both Hamiltonians have only short-range interactions, so each can be written
exactly as a small finite-bond-dimension matrix product operator (MPO) using
a standard "finite state machine" construction: each site tensor is a matrix
W[a, b] of operators, where the row index `a` is "what has happened so far"
and column index `b` is "what the tail of the chain still owes". The chain
starts in a dedicated `start` state and must end in a dedicated `finish`
state. See e.g. Schollwoeck's DMRG review for the general method; the
explicit automata below were derived by hand for the two models of the
paper and validated against exact diagonalization in ``tests/``.
"""
from __future__ import annotations

import numpy as np
import quimb as qu
import quimb.tensor as qtn

# Pauli matrices (eigenvalues +-1), matching the paper's convention sigma^x,
# sigma^y, sigma^z (NOT the spin-1/2 operators S=sigma/2).
X = qu.pauli("X")
Y = qu.pauli("Y")
Z = qu.pauli("Z")
I2 = qu.eye(2)


def _fsm_to_mpo(N: int, W: np.ndarray, start: int, finish: int) -> qtn.MatrixProductOperator:
    """Build an open-boundary MPO of length N from a single, translationally
    invariant bulk FSM tensor ``W`` of shape (B, B, 2, 2).

    The left-boundary tensor is the `start` row of W, the right-boundary
    tensor is the `finish` column of W (quimb's default `shape='lrud'`
    convention: array axes are left-bond, right-bond, ket, bra).
    """
    left = W[start, :, :, :]
    right = W[:, finish, :, :]
    arrays = [left] + [W] * (N - 2) + [right]
    return qtn.MatrixProductOperator(arrays, shape="lrud")


def annni_mpo(N: int, k: float, h: float) -> qtn.MatrixProductOperator:
    """ANNNI Hamiltonian (paper eq. 7), open boundary conditions:

        H = -sum_{j=1}^{N-1} X_j X_{j+1} + k sum_{j=1}^{N-2} X_j X_{j+2}
            - h sum_{j=1}^{N} Z_j

    FSM states: 0=start, 1=(applied -X, awaiting NN partner),
    2,3=(applied k*X, awaiting NNN partner two sites away), 4=finish.
    """
    B = 5
    W = np.zeros((B, B, 2, 2), dtype=complex)
    W[0, 0] = I2
    W[0, 1] = -1.0 * X
    W[0, 2] = k * X
    W[0, 4] = -h * Z
    W[1, 4] = X
    W[2, 3] = I2
    W[3, 4] = X
    W[4, 4] = I2
    return _fsm_to_mpo(N, W, start=0, finish=4)


def haldane_mpo(N: int, k: float, h: float) -> qtn.MatrixProductOperator:
    """Cluster/Haldane SPT Hamiltonian (paper eq. 9), open boundary
    conditions:

        H = -sum_{j=1}^{N-2} X_j Z_{j+1} X_{j+2}
            - k sum_{j=1}^{N-1} Z_j Z_{j+1} - h sum_{j=1}^{N} Z_j

    FSM states: 0=start, 1=(applied -X, awaiting Z), 2=(applied Z,
    awaiting X two sites away), 3=(applied -k*Z, awaiting ZZ partner),
    4=finish.
    """
    B = 5
    W = np.zeros((B, B, 2, 2), dtype=complex)
    W[0, 0] = I2
    W[0, 1] = -1.0 * X
    W[0, 3] = -k * Z
    W[0, 4] = -h * Z
    W[1, 2] = Z
    W[2, 4] = X
    W[3, 4] = Z
    W[4, 4] = I2
    return _fsm_to_mpo(N, W, start=0, finish=4)


def parity_mpo_annni(N: int) -> qtn.MatrixProductOperator:
    """Z2 parity operator for the ANNNI model, P = (x)_n sigma^z_n."""
    return qtn.MPO_product_operator([Z] * N)


def parity_mpo_even_sites(N: int) -> qtn.MatrixProductOperator:
    """Z2xZ2 generator for the Haldane model acting on even sites (1-indexed
    in the paper): X_even = prod_{i even} sigma^z_i."""
    ops = [Z if ((i + 1) % 2 == 0) else I2 for i in range(N)]
    return qtn.MPO_product_operator(ops)


def parity_mpo_odd_sites(N: int) -> qtn.MatrixProductOperator:
    """Z2xZ2 generator for the Haldane model acting on odd sites (1-indexed
    in the paper): X_odd = prod_{i odd} sigma^z_i."""
    ops = [Z if ((i + 1) % 2 == 1) else I2 for i in range(N)]
    return qtn.MPO_product_operator(ops)
