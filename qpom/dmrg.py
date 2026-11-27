"""Thin convenience wrapper around quimb's two-site DMRG (``DMRG2``), used to
obtain approximate ground states as matrix product states (MPS) with a
bounded bond dimension -- exactly the "tensor network methods with a small
bond dimension" used throughout the paper.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Sequence

import quimb.tensor as qtn


def fresh_bond_indices(mps: qtn.MatrixProductState) -> qtn.MatrixProductState:
    """Return a copy of ``mps`` with its internal bond indices renamed to
    freshly generated, globally unique labels (keeping the physical site
    indices untouched).

    Needed because quimb's automatic internal-index names are drawn from a
    per-process counter, not a secure random source: two MPS built in
    different worker processes (e.g. via ``multiprocessing``) can end up
    with identical bond-index names, which silently corrupts any tensor
    network built by combining them (e.g. computing an overlap/kernel
    between a train-set state and a test-set state coming from different
    processes). Call this once on every state as it leaves a worker.
    """
    suffix = uuid.uuid4().hex
    mapping = {ix: f"{ix}__{suffix}" for ix in mps.inner_inds()}
    return mps.reindex(mapping)


@dataclass
class DMRGResult:
    mps: qtn.MatrixProductState
    energy: float
    max_bond: int
    converged: bool


def ground_state(
    mpo: qtn.MatrixProductOperator,
    max_bond: int,
    *,
    bond_schedule: Sequence[int] | None = None,
    max_sweeps: int = 12,
    tol: float = 1e-9,
    cutoff: float = 1e-10,
    seed: int | None = 0,
) -> DMRGResult:
    """Run DMRG2 on ``mpo`` and return its (approximate) ground state.

    Parameters
    ----------
    mpo : the Hamiltonian (or penalized Hamiltonian, for sector selection).
    max_bond : maximum MPS bond dimension to allow (matches the paper's
        "small bond dimension" D=20 for the ANNNI model and D=150 for the
        Haldane model).
    bond_schedule : an optional increasing sequence of bond dimensions to
        ramp up through during the sweeps (helps convergence); defaults to
        a short geometric ramp ending at ``max_bond``.
    """
    if bond_schedule is None:
        schedule = []
        b = min(10, max_bond)
        while b < max_bond:
            schedule.append(b)
            b *= 3
        schedule.append(max_bond)
        bond_schedule = schedule

    dmrg = qtn.DMRG2(mpo, bond_dims=list(bond_schedule), cutoffs=cutoff)
    if seed is not None:
        # quimb seeds its random initial MPS from numpy's global state
        import numpy as np

        np.random.seed(seed)
    converged = dmrg.solve(tol=tol, verbosity=0, max_sweeps=max_sweeps)
    psi = dmrg.state
    psi.normalize()
    psi = fresh_bond_indices(psi)
    return DMRGResult(
        mps=psi,
        energy=float(dmrg.energy.real if hasattr(dmrg.energy, "real") else dmrg.energy),
        max_bond=psi.max_bond(),
        converged=bool(converged),
    )
