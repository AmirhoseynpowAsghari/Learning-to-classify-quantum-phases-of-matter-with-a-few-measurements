#!/usr/bin/env python3
"""Render figure 3 (Haldane/cluster-model SPT phase diagram) from the cached
results of ``fig3_compute.py``.

Note: the paper overlays dashed phase-transition lines digitized from
ref. [4] (Cong, Choi & Lukin 2019); that digitized data is not available to
us (no official code/data release, and no internet access to fetch it in
this sandbox), so here we show only the SVM-predicted phase regions and the
training points, without those reference lines.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qpom.svm import KernelSVM

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)

SPT, PARA, AFM = 0, 1, 2
COLORS = {SPT: "#f4c430", PARA: "#3f7f74", AFM: "#7b2d8e"}
NAMES = {SPT: "SPT", PARA: "Paramagnetic", AFM: "Antiferromagnetic"}
CMAP = ListedColormap([COLORS[i] for i in range(3)])


def main():
    with open(DATA_DIR / "fig3.pkl", "rb") as f:
        d = pickle.load(f)

    cfg = d["config"]
    hs_t, ks_t = np.array(d["hs_t"]), np.array(d["ks_t"])
    y_train = np.array(d["train_labels"])

    svm = KernelSVM(C=1.0).fit(d["K_train"], y_train)
    # See qpom.svm.KernelSVM.predict_with_nn_fallback: ~17% of this test
    # grid's points have a squared-fidelity kernel below 0.3 to every single
    # training state (the training set only spans h=1, but the test grid
    # covers the whole h in [0, 1.6] range), which is exactly the regime
    # where the plain multiclass SVM decision degenerates to a meaningless,
    # bias-dominated prediction (see the discussion for figure 1c). The
    # nearest-neighbor fallback avoids that failure mode.
    pred = svm.predict_with_nn_fallback(d["K_test_train"], threshold=0.3)
    grid = pred.reshape(len(hs_t), len(ks_t))  # test_points = [(h,k) for h in hs for k in ks]

    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.pcolormesh(hs_t, ks_t, grid.T, cmap=CMAP, vmin=0, vmax=2, shading="auto")

    train_h = [p[0] for p in d["train_points"]]
    train_k = [p[1] for p in d["train_points"]]
    ax.scatter(train_h, train_k, marker="x", color="black", s=20, linewidths=1.0,
               zorder=5, label="Training points")

    ax.set_xlabel("h")
    ax.set_ylabel("k")
    ax.set_xlim(cfg["H_RANGE"])
    ax.set_ylim(cfg["K_RANGE"])
    ax.set_title(
        f"Haldane/cluster model, N={cfg['N']}, DMRG bond dim {cfg['MAX_BOND']}\n"
        f"(reproduction of figure 3, reduced test grid {cfg['GRID_NH']}x{cfg['GRID_NK']})"
    )

    legend_elems = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS[SPT], label=NAMES[SPT]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS[PARA], label=NAMES[PARA]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS[AFM], label=NAMES[AFM]),
        Line2D([0], [0], marker="x", color="black", linestyle="None", label="Training points"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=8)

    fig.tight_layout()
    out_path = FIG_DIR / "figure3.png"
    fig.savefig(out_path, dpi=160)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
