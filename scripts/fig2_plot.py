#!/usr/bin/env python3
"""Render figure 2: (a) generalization-bound scaling B^2(N) for the even/odd
parity sectors (from fig2a_compute.py), (b) kernel-matrix heatmap for the
figure-1(b) training states (from fig1_compute.py, reused here).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)


def main():
    with open(DATA_DIR / "fig2a.json") as f:
        d2a = json.load(f)
    with open(DATA_DIR / "fig1.pkl", "rb") as f:
        d1 = pickle.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # ---- panel (a) ----
    ax = axes[0]
    for sector, color, label in (("even", "tab:blue", "even"), ("odd", "tab:red", "odd")):
        Ns = sorted(int(n) for n in d2a["results"][sector].keys())
        means = [np.mean(d2a["results"][sector][str(n)]) for n in Ns]
        stds = [np.std(d2a["results"][sector][str(n)]) for n in Ns]
        ax.errorbar(Ns, means, yerr=stds, marker="o", color=color, label=label, capsize=3)
    ax.set_xlabel("N")
    ax.set_ylabel(r"$\mathcal{B}^2$")
    ax.set_title(f"(a) Generalization bound scaling\n(m={d2a['config']['M_SAMPLES']}, "
                 f"{d2a['config']['N_REPS']} reps -- reduced from paper's m=1000, 10 reps)")
    ax.legend()
    ax.grid(alpha=0.3)

    # ---- panel (b): kernel matrix of the figure-1(b) training states -----
    ax = axes[1]
    K = d1["K_train_b"]
    im = ax.imshow(K, cmap="Greys", vmin=0, vmax=1, origin="upper")
    ax.set_title("(b) Kernel matrix, figure 1(b) training states\n"
                  "(tiled blocks from +/- parity pairs)")
    ax.set_xlabel("training state index")
    ax.set_ylabel("training state index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path = FIG_DIR / "figure2.png"
    fig.savefig(out_path, dpi=160)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
