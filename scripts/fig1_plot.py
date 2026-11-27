#!/usr/bin/env python3
"""Render figure 1 (ANNNI phase diagram) from the cached results of
``fig1_compute.py``: panel (a) the theoretical phase diagram, panels (b) and
(c) the SVM-predicted phase diagrams for in-distribution (uniform) and
out-of-distribution (h=0.25 line) training.
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

from qpom.annni_phase import (AP, FERRO, FLOATING, PARA, h_AP, h_I, h_KT,
                               phase_label_grid)
from qpom.svm import KernelSVM

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)

PHASE_COLORS = {FERRO: "#3f7f74", PARA: "#f4c430", AP: "#7b2d8e", FLOATING: "#ffffff"}
CMAP = ListedColormap([PHASE_COLORS[i] for i in range(4)])


def draw_transition_lines(ax, k_range=(0.0, 1.0)):
    k1 = np.linspace(max(k_range[0], 1e-4), 0.5, 200)
    ax.plot(k1, h_I(k1), color="black", lw=1.4, zorder=5)
    k2 = np.linspace(0.5, k_range[1], 200)
    ax.plot(k2, h_AP(k2), color="black", lw=1.4, zorder=5)
    ax.plot(k2, h_KT(k2), color="red", lw=1.4, zorder=5)


def panel_a(ax, k_range, h_range):
    nk, nh = 400, 400
    ks = np.linspace(k_range[0], k_range[1], nk)
    hs = np.linspace(h_range[0], h_range[1], nh)
    KK, HH = np.meshgrid(ks, hs)
    labels = phase_label_grid(KK, HH)
    ax.pcolormesh(ks, hs, labels, cmap=CMAP, vmin=0, vmax=3, shading="auto")
    draw_transition_lines(ax, k_range)
    ax.set_title("(a) Theoretical phase diagram")
    ax.set_xlabel("k")
    ax.set_ylabel("h")
    ax.text(0.08, 1.6, "Ferromagnet", fontsize=8, weight="bold")
    ax.text(0.65, 1.7, "Paramagnet", fontsize=8, weight="bold")
    ax.text(0.75, 0.15, "Antiphase", fontsize=8, weight="bold")


def predict_grid(K_train, y_train, K_test_train, ks_t, hs_t):
    svm = KernelSVM(C=1.0).fit(K_train, y_train)
    pred = svm.predict(K_test_train)
    grid = pred.reshape(len(ks_t), len(hs_t))  # points were [(k,h) for k in ks for h in hs]
    return grid, svm


def panel_bc(ax, title, ks_t, hs_t, grid, train_points, k_range, h_range):
    ax.pcolormesh(ks_t, hs_t, grid.T, cmap=CMAP, vmin=0, vmax=3, shading="auto")
    draw_transition_lines(ax, k_range)
    tk = [p[0] for p in train_points]
    th = [p[1] for p in train_points]
    ax.scatter(tk, th, marker="x", color="black", s=22, linewidths=1.0, zorder=6, label="Training points")
    ax.set_title(title)
    ax.set_xlabel("k")
    ax.set_ylabel("h")
    ax.set_xlim(k_range)
    ax.set_ylim(h_range)


def main():
    with open(DATA_DIR / "fig1.pkl", "rb") as f:
        d = pickle.load(f)

    cfg = d["config"]
    k_range, h_range = tuple(cfg["K_RANGE"]), tuple(cfg["H_RANGE"])
    ks_t, hs_t = np.array(d["ks_t"]), np.array(d["hs_t"])

    grid_b, svm_b = predict_grid(
        d["K_train_b"], np.array(d["train_b_states_labels"]), d["K_test_train_b"], ks_t, hs_t
    )
    grid_c, svm_c = predict_grid(
        d["K_train_c"], np.array(d["train_c_states_labels"]), d["K_test_train_c"], ks_t, hs_t
    )

    # Figure height budget (inches), top to bottom: suptitle, panel (a),
    # panels (b)/(c), legend. Using explicit gridspec top/bottom/hspace
    # (rather than tight_layout, which does not reliably avoid collisions
    # between a tall suptitle and the row-two titles) keeps every title and
    # the bottom legend fully inside the saved canvas.
    fig = plt.figure(figsize=(7.5, 12.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.32, wspace=0.3,
                           top=0.91, bottom=0.08, left=0.1, right=0.97)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    panel_a(ax_a, k_range, h_range)
    panel_bc(ax_b, "(b) Uniform (in-distribution)\ntraining", ks_t, hs_t, grid_b,
             d["train_b_points"], k_range, h_range)
    panel_bc(ax_c, "(c) Line h=0.25 (out-of-\ndistribution) training", ks_t, hs_t, grid_c,
             d["train_c_points"], k_range, h_range)

    legend_elems = [
        plt.Rectangle((0, 0), 1, 1, color=PHASE_COLORS[FERRO], label="Ferromagnet"),
        plt.Rectangle((0, 0), 1, 1, color=PHASE_COLORS[PARA], label="Paramagnet"),
        plt.Rectangle((0, 0), 1, 1, color=PHASE_COLORS[AP], label="Antiphase"),
        plt.Rectangle((0, 0), 1, 1, color=PHASE_COLORS[FLOATING], ec="black", label="Floating"),
        Line2D([0], [0], marker="x", color="black", linestyle="None", label="Training points"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005),
               frameon=True)

    fig.suptitle(
        f"ANNNI model, N={cfg['N']} spins, DMRG bond dim {cfg['MAX_BOND']}\n"
        f"(reproduction of figure 1, reduced grid {cfg['GRID_NK']}x{cfg['GRID_NH']})",
        fontsize=10, y=0.985,
    )
    out_path = FIG_DIR / "figure1.png"
    fig.savefig(out_path, dpi=160)
    print("wrote", out_path)

    # also store SVM accuracy summary vs theoretical ground truth
    y_true = np.array(d["test_true_labels"])
    acc_b = np.mean(svm_b.model.predict(d["K_test_train_b"]) == y_true)
    acc_c = np.mean(svm_c.model.predict(d["K_test_train_c"]) == y_true)
    print(f"panel (b) accuracy vs theoretical labels: {acc_b:.3f}")
    print(f"panel (c) accuracy vs theoretical labels: {acc_c:.3f}")


if __name__ == "__main__":
    main()
