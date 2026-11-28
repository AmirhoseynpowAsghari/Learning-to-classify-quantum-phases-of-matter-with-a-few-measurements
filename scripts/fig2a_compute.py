#!/usr/bin/env python3
"""Figure 2(a): scaling of the generalization-error bound B^2 (appendix D,
eq. D1) with the number of qubits N, for the even and odd Z2-parity sectors
of the ANNNI model's ground state, h in [0,2], k in [0,1].

The paper uses m=1000 random (h,k) samples per N, averaged over 10
repetitions. For the sandbox's 2-CPU / 3GB budget we reduce this to
M_SAMPLES=250, N_REPS=4 (documented deviation) -- the qualitative result
(B^2 grows at most linearly in N, even >= odd) is unaffected, as the paper
itself notes that B converges quickly with the number of samples.

Exact diagonalization (not DMRG) is used since N is small (<=16); the Z2
parity sector is just the even/odd Hamming-weight subset of the
computational basis for this model, so sector projection is a cheap
row/column gather.
"""
from __future__ import annotations

import os

# Single-threaded BLAS: this script is single-process, but still benefits
# from not spawning a large OpenBLAS thread pool that oversubscribes a
# small sandbox (and avoids contention with sibling compute scripts run
# concurrently in the background). See scripts/fig1_compute.py for details.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qpom.genbound import build_annni_operators, generalization_bound_sq

N_VALUES = list(range(4, 17))  # every integer N=4..16, matching the paper's
# figure 2(a), which plots a point at *every* N (visible from the jagged
# "staircase" shape of its curves) -- a previous version of this script only
# sampled even N (4,6,8,...,16), which produced an artificially smooth curve
# and silently skipped half the qubit counts the paper actually reports.
M_SAMPLES = 250
N_REPS = 4
SEED = 12345

OUT_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR.mkdir(exist_ok=True)
JSON_PATH = OUT_DIR / "fig2a.json"


def main():
    rng = np.random.default_rng(SEED)
    results = {"even": {}, "odd": {}}
    t0 = time.time()
    for N in N_VALUES:
        ops = build_annni_operators(N)
        for sector in ("even", "odd"):
            vals = []
            for rep in range(N_REPS):
                b2 = generalization_bound_sq(N, sector, M_SAMPLES, rng, ops=ops)
                vals.append(b2)
            results[sector][N] = vals
            print(
                f"N={N:2d} sector={sector:4s} B^2 mean={np.mean(vals):7.3f} "
                f"std={np.std(vals):6.3f}  (elapsed {time.time()-t0:6.1f}s)",
                flush=True,
            )

    out = dict(
        config=dict(N_VALUES=N_VALUES, M_SAMPLES=M_SAMPLES, N_REPS=N_REPS, SEED=SEED),
        results=results,
    )
    with open(JSON_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print("DONE. Wrote", JSON_PATH)


if __name__ == "__main__":
    main()
