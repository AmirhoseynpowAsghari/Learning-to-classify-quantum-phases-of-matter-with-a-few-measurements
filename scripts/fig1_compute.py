#!/usr/bin/env python3
"""Heavy computation for figure 1 (ANNNI phase diagram, in- and
out-of-distribution training) and the kernel-matrix panel of figure 2(b).

This reproduces, with a reduced grid/bond-dimension-ramp tuned for a small
(2-CPU, 3GB RAM) sandbox, the paper's numerical experiment of section 3.1:

  * N = 50 spins, DMRG with max bond dimension D = 20 (same as the paper).
  * Panel (b): training points uniformly scattered over (k, h) in-distribution.
  * Panel (c): training points along the line h = 0.25, out-of-distribution.
  * Both panels tested on the same (k, h) grid.
  * Degenerate Z2 sectors are both inserted in the training set (section 2.1);
    test states are left as whatever (possibly symmetry-mixed) state DMRG
    converges to.

Reduced vs the paper (documented, due to sandbox compute limits): the test
grid resolution (26x33 instead of what looks like ~60x60 in the published
figure) and the number of DMRG sweeps. The physics and method are otherwise
identical; increase GRID_NK/GRID_NH/MAX_SWEEPS in a more powerful environment
to approach the paper's resolution.

Usage: python3 scripts/fig1_compute.py
Output: data/fig1.pkl (training/test MPS + labels + kernels), data/fig1.json
(human-readable summary/metadata).
"""
from __future__ import annotations

import os

# Pin BLAS/OpenMP to a single thread *per process*, before numpy/quimb are
# imported. We parallelize across (k, h) grid points using a process pool
# instead, so each worker needs just one thread; without this, every worker
# process spawns its own multi-threaded BLAS pool and the resulting thread
# oversubscription on a small sandbox (e.g. 2 CPUs) can slow DMRG down by one
# to two orders of magnitude (observed: ~1s -> ~15s+ per ground state).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qpom.annni_phase import phase_label
from qpom.dmrg import ground_state
from qpom.kernel import fidelity_kernel_matrix
from qpom.operators import annni_mpo, parity_mpo_annni
from qpom.parity import sector_pair_by_projection

# ----------------------------- configuration ------------------------------
N = 50
MAX_BOND = 20
K_RANGE = (0.0, 1.0)
H_RANGE = (0.0, 2.0)
MAX_SWEEPS = 10
TOL = 1e-9

GRID_NK = 26
GRID_NH = 33

TRAIN_B_NK = 8
TRAIN_B_NH = 10

TRAIN_C_H = 0.25
TRAIN_C_NK = 40

N_WORKERS = 2

OUT_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR.mkdir(exist_ok=True)
PKL_PATH = OUT_DIR / "fig1.pkl"
JSON_PATH = OUT_DIR / "fig1.json"


# ----------------------------- worker functions ----------------------------
def _solve_plain(args):
    k, h = args
    H = annni_mpo(N, k, h)
    res = ground_state(H, MAX_BOND, max_sweeps=MAX_SWEEPS, tol=TOL)
    return (k, h), res.mps, res.energy, res.max_bond


def _run_parallel(jobs, fn, desc):
    results = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(fn, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            key, mps, energy, bond = fut.result()
            results[key] = (mps, energy, bond)
            done += 1
            if done % 10 == 0 or done == len(jobs):
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(jobs) - done) / rate if rate > 0 else float("nan")
                print(
                    f"[{desc}] {done}/{len(jobs)}  "
                    f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s",
                    flush=True,
                )
    return results


def main():
    # ---- training set (b): uniform grid, in-distribution --------------
    ks_b = np.linspace(K_RANGE[0], K_RANGE[1], TRAIN_B_NK + 2)[1:-1]
    hs_b = np.linspace(H_RANGE[0], H_RANGE[1], TRAIN_B_NH + 2)[1:-1]
    train_b_points = [(float(k), float(h)) for k in ks_b for h in hs_b]
    train_b_labels = [phase_label(k, h) for (k, h) in train_b_points]

    # ---- training set (c): along h = 0.25, out-of-distribution ----------
    ks_c = np.linspace(K_RANGE[0], K_RANGE[1], TRAIN_C_NK)
    train_c_points = [(float(k), TRAIN_C_H) for k in ks_c]
    train_c_labels = [phase_label(k, TRAIN_C_H) for (k, _) in train_c_points]

    # ---- test grid (shared by both panels) ------------------------------
    ks_t = np.linspace(K_RANGE[0], K_RANGE[1], GRID_NK)
    hs_t = np.linspace(H_RANGE[0], H_RANGE[1], GRID_NH)
    test_points = [(float(k), float(h)) for k in ks_t for h in hs_t]
    test_true_labels = [phase_label(k, h) for (k, h) in test_points]

    print(f"train_b: {len(train_b_points)} points -> {2*len(train_b_points)} states (both parities)")
    print(f"train_c: {len(train_c_points)} points -> {2*len(train_c_points)} states (both parities)")
    print(f"test grid: {len(test_points)} points (single state each)")

    # ---- compute training states: one plain ground state per point, then
    # split into the two Z2 sectors by projection (section 2.1; see
    # qpom.parity.sector_pair_by_projection for why this is both faster and
    # more numerically robust than re-running DMRG with a penalty term). ---
    train_unique_points = sorted(set(train_b_points) | set(train_c_points))
    train_plain_results = _run_parallel(train_unique_points, _solve_plain, "train ground states")

    parity_op = parity_mpo_annni(N)
    train_sector_states = {}
    for (k, h), (mps, energy, bond) in train_plain_results.items():
        psi_plus, psi_minus = sector_pair_by_projection(mps, parity_op)
        train_sector_states[(k, h)] = (psi_plus, psi_minus)

    def get_train_states(points):
        states, labels = [], []
        for (k, h) in points:
            lbl = phase_label(k, h)
            psi_plus, psi_minus = train_sector_states[(k, h)]
            states.append(psi_plus)
            labels.append(lbl)
            states.append(psi_minus)
            labels.append(lbl)
        return states, labels

    train_b_states, train_b_states_labels = get_train_states(train_b_points)
    train_c_states, train_c_states_labels = get_train_states(train_c_points)

    # ---- compute test states (plain, unconstrained ground state) --------
    test_results = _run_parallel(test_points, _solve_plain, "test grid")
    test_states = [test_results[(k, h)][0] for (k, h) in test_points]
    test_energies = [test_results[(k, h)][1] for (k, h) in test_points]
    test_bonds = [test_results[(k, h)][2] for (k, h) in test_points]

    # ---- kernels ----------------------------------------------------------
    print("computing kernels...", flush=True)
    K_train_b = fidelity_kernel_matrix(train_b_states)
    K_train_c = fidelity_kernel_matrix(train_c_states)
    K_test_train_b = fidelity_kernel_matrix(test_states, train_b_states)
    K_test_train_c = fidelity_kernel_matrix(test_states, train_c_states)

    out = dict(
        config=dict(
            N=N, MAX_BOND=MAX_BOND, K_RANGE=K_RANGE, H_RANGE=H_RANGE,
            GRID_NK=GRID_NK, GRID_NH=GRID_NH,
            TRAIN_B_NK=TRAIN_B_NK, TRAIN_B_NH=TRAIN_B_NH,
            TRAIN_C_H=TRAIN_C_H, TRAIN_C_NK=TRAIN_C_NK,
        ),
        ks_t=ks_t, hs_t=hs_t,
        test_points=test_points, test_true_labels=test_true_labels,
        test_energies=test_energies, test_bonds=test_bonds,
        train_b_points=train_b_points, train_b_labels=train_b_labels,
        train_b_states_labels=train_b_states_labels,
        train_c_points=train_c_points, train_c_labels=train_c_labels,
        train_c_states_labels=train_c_states_labels,
        K_train_b=K_train_b, K_train_c=K_train_c,
        K_test_train_b=K_test_train_b, K_test_train_c=K_test_train_c,
    )

    with open(PKL_PATH, "wb") as f:
        pickle.dump(out, f)

    summary = dict(
        config=out["config"],
        n_train_b_states=len(train_b_states),
        n_train_c_states=len(train_c_states),
        n_test=len(test_points),
    )
    with open(JSON_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("DONE. Wrote", PKL_PATH, "and", JSON_PATH)


if __name__ == "__main__":
    main()
