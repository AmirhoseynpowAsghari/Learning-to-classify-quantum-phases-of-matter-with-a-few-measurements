"""Symmetry-sector ("parity") handling, following section 2.1 of the paper.

Degenerate ground states, labeled by a conserved Z2 (or Z2xZ2) quantum
number, would otherwise make kernel entries essentially random (the kernel
compares whichever representative a diagonalization routine happens to
return). The paper's fix is to *train* with both parity sectors explicitly
inserted into the training set (so the classifier sees both and learns an
observable that works for either), while test states are left as whatever
state the solver returns (possibly an arbitrary superposition).

We realize "give me the ground state restricted to symmetry sector s" by
adding a large energy penalty to the complementary sector(s) before running
DMRG: for a Z2 generator P (P^2=1, [H,P]=0),

    H_s = H + (lambda/2) * (I - s*P),   s = +-1

has, for large enough lambda, the same low-energy state in the s-sector as H,
but every state in the (-s)-sector is pushed up by lambda.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import quimb.tensor as qtn

from .dmrg import DMRGResult, ground_state


def penalized_hamiltonian(
    mpo: qtn.MatrixProductOperator,
    parity_generators: Sequence[qtn.MatrixProductOperator],
    sector_signs: Sequence[int],
    lam: float = 50.0,
) -> qtn.MatrixProductOperator:
    """Return H + (lambda/2) * sum_g (I - sign_g * P_g), penalizing every
    sector that is not the requested one for each commuting Z2 generator.
    """
    N = mpo.L
    H = mpo
    I_mpo = qtn.MPO_identity(N, phys_dim=2)
    for P, s in zip(parity_generators, sector_signs):
        H = H + (lam / 2) * (I_mpo - s * P)
    return H


def sector_ground_state(
    mpo: qtn.MatrixProductOperator,
    parity_generators: Sequence[qtn.MatrixProductOperator],
    sector_signs: Sequence[int],
    max_bond: int,
    lam: float = 50.0,
    **dmrg_kwargs,
) -> DMRGResult:
    """Ground state of ``mpo`` restricted to the symmetry sector specified by
    ``sector_signs`` (one +-1 per generator in ``parity_generators``)."""
    H_pen = penalized_hamiltonian(mpo, parity_generators, sector_signs, lam=lam)
    return ground_state(H_pen, max_bond, **dmrg_kwargs)


def expectation(mps: qtn.MatrixProductState, mpo: qtn.MatrixProductOperator) -> complex:
    """<psi| O |psi> for MPS ``mps`` and MPO ``mpo``."""
    return complex((mps.H @ mpo.apply(mps)))


def sector_pair_by_projection(
    psi0: qtn.MatrixProductState,
    parity_mpo: qtn.MatrixProductOperator,
    degeneracy_threshold: float = 0.98,
):
    """Fast, numerically robust alternative to ``sector_ground_state``.

    Rather than re-running DMRG on a penalized Hamiltonian for each sector
    (which we found can be numerically unstable: the extra penalty term
    makes the two-site DMRG local-update landscape harder to optimize and it
    can get stuck far from the true minimum), we exploit that for an
    *exactly* double-degenerate ground space {|g+>, |g->} (eigenvalues +-1
    of the commuting parity operator P), the plain/unconstrained ground
    state psi0 returned by DMRG is necessarily some linear combination
    a|g+> + b|g->. Applying the projectors (1 +- P)/2 to psi0 then exactly
    recovers (up to normalization) the pure-parity representatives:

        |g+> ~ (1 + P) |psi0>,     |g-> ~ (1 - P) |psi0>.

    If the ground state is instead unique (non-degenerate, e.g. in the
    paramagnetic phase), psi0 is already very close to a parity eigenstate
    (|<psi0|P|psi0>| ~ 1) and there is nothing else to project out; in that
    case we simply return psi0 twice.

    Returns
    -------
    (psi_plus, psi_minus) : the two (normalized) sector representatives.
    """
    pe = expectation(psi0, parity_mpo).real
    if abs(pe) > degeneracy_threshold:
        return psi0, psi0

    p_psi0 = parity_mpo.apply(psi0)
    psi_plus = psi0 + p_psi0
    psi_minus = psi0 - p_psi0
    psi_plus.normalize()
    psi_minus.normalize()
    return psi_plus, psi_minus
