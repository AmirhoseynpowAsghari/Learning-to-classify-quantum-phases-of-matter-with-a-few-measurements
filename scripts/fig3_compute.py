#!/usr/bin/env python3
"""Heavy computation for figure 3 (Haldane/cluster SPT model phase diagram),
paper section 3.2.

  * N = 41 spins, DMRG with bond dimension D = 150 (same cap as the paper).
  * 50 training points along h = 1, labeled from the *known* phase boundaries
    given in the paper (SPT for -1.15<=k<=0, paramagnet for k>0,
    antiferromagnet for k<-1.15) -- no DMRG-independent ground truth is used,
    exactly mirroring the paper's "train with what we know" setup.
  * Tested over a grid spanning the full (h,k) plane.

The paper does not mention an explicit Z2xZ2 sector-projection step for this
model (unlike the ANNNI case in section 2.1/figure 1), so here we use plain,
unconstrained DMRG throughout; any ground-state degeneracy is simply
resolved however the DMRG sweeps happen to break it. This mirrors the
textual description of this experiment and keeps the SPT/antiferromagnetic
training representatives self-consistent with the test states.

Reduced vs the paper (documented, due to sandbox compute limits): the test
grid is 15x15 instead of 30x30, and DMRG sweeps/tolerance are relaxed
(max 8 sweeps, tol=1e-6 instead of full convergence) since bond dimension
150 on a 2-CPU sandbox is expensive per state.
"""
from __future__ import annotations

import os

# Pin BLAS/OpenMP to a single thread per worker process; see
# scripts/fig1_compute.py for why this matters a lot on a small sandbox.
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

from qpom.dmrg import ground_state
from qpom.kernel import fidelity_kernel_matrix
from qpom.operators import haldane_mpo

# ----------------------------- configuration ------------------------------
N = 41
MAX_BOND = 150
H_RANGE = (0.0, 1.6)
K_RANGE = (-1.6, 1.6)
MAX_SWEEPS = 8
TOL = 1e-6
BOND_SCHEDULE = [20, 60, 100, 150]

TRAIN_H = 1.0
TRAIN_NK = 50

GRID_NH = 15
GRID_NK = 15

N_WORKERS = 2

SPT, PARA, AFM = 0, 1, 2
PHASE_NAMES = {SPT: "SPT", PARA: "Paramagnet", AFM: "Antiferromagnet"}


def training_label(k: float) -> int:
    if k > 0:
        return PARA
    elif k >= -1.15:
        return SPT
    else:
        return AFM


OUT_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR.mkdir(exist_ok=True)
PKL_PATH = OUT_DIR / "fig3.pkl"
JSON_PATH = OUT_DIR / "fig3.json"


def _solve(args):
    h, k = args
    H = haldane_mpo(N, k, h)
    res = ground_state(
        H, MAX_BOND, bond_schedule=BOND_SCHEDULE, max_sweeps=MAX_SWEEPS, tol=TOL
    )
    return (h, k), res.mps, res.energy, res.max_bond


def _run_parallel(jobs, desc):
    results = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_solve, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            key, mps, energy, bond = fut.result()
            results[key] = (mps, energy, bond)
            done += 1
            if done % 5 == 0 or done == len(jobs):
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(jobs) - done) / rate if rate > 0 else float("nan")
                print(
                    f"[{desc}] {done}/{len(jobs)}  elapsed={elapsed:7.1f}s  eta={eta:7.1f}s",
                    flush=True,
                )
    return results


def main():
    ks_train = np.linspace(K_RANGE[0], K_RANGE[1], TRAIN_NK)
    train_points = [(TRAIN_H, float(k)) for k in ks_train]
    train_labels = [training_label(k) for k in ks_train]

    hs_t = np.linspace(H_RANGE[0], H_RANGE[1], GRID_NH)
    ks_t = np.linspace(K_RANGE[0], K_RANGE[1], GRID_NK)
    test_points = [(float(h), float(k)) for h in hs_t for k in ks_t]

    print(f"train: {len(train_points)} points along h={TRAIN_H}")
    print(f"test grid: {len(test_points)} points ({GRID_NH}x{GRID_NK})")

    train_results = _run_parallel(train_points, "train (h=1 line)")
    test_results = _run_parallel(test_points, "test grid")

    train_states = [train_results[p][0] for p in train_points]
    train_energies = [train_results[p][1] for p in train_points]
    train_bonds = [train_results[p][2] for p in train_points]

    test_states = [test_results[p][0] for p in test_points]
    test_energies = [test_results[p][1] for p in test_points]
    test_bonds = [test_results[p][2] for p in test_points]

    print("computing kernels...", flush=True)
    K_train = fidelity_kernel_matrix(train_states)
    K_test_train = fidelity_kernel_matrix(test_states, train_states)

    out = dict(
        config=dict(
            N=N, MAX_BOND=MAX_BOND, H_RANGE=H_RANGE, K_RANGE=K_RANGE,
            TRAIN_H=TRAIN_H, TRAIN_NK=TRAIN_NK, GRID_NH=GRID_NH, GRID_NK=GRID_NK,
        ),
        hs_t=hs_t, ks_t=ks_t,
        train_points=train_points, train_labels=train_labels,
        train_energies=train_energies, train_bonds=train_bonds,
        test_points=test_points,
        test_energies=test_energies, test_bonds=test_bonds,
        K_train=K_train, K_test_train=K_test_train,
        phase_names=PHASE_NAMES,
    )
    with open(PKL_PATH, "wb") as f:
        pickle.dump(out, f)

    summary = dict(config=out["config"], n_train=len(train_points), n_test=len(test_points))
    with open(JSON_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print("DONE. Wrote", PKL_PATH, "and", JSON_PATH)


if __name__ == "__main__":
    main()
