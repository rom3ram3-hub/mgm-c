"""Redraw the paper figures (Figs. 2-5) from the CSV files written by the notebook.

Usage:  python make_figures.py --results results --out figures
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

GREY, BLUE, MID = "#9a9a9a", "#1f5fbf", "#6b6b6b"
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "savefig.dpi": 600})


def mean_of(cell):
    return float(str(cell).split("±")[0])


def fig_realism(results, out):
    t = pd.read_csv(f"{results}/table9_realism_2x2_auc.csv", index_col=0)
    models = list(t.columns)
    labels = ["B1 Gate-count RF", "B2 Wire-bigram RF", "B3 Wire-bigram HGB", "B4 Idempotence residual", "GIN graph head", "GIN-L2D"]
    rows = {k: [mean_of(t.loc[k, m]) for m in models] for k in t.index}
    prior, fixed_family = rows["Fixed compiler + random split"], rows["Fixed compiler + family split"]
    honest_random, realistic = rows["Honest variation + random split"], rows["Honest variation + family split"]
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    y = np.arange(len(models))[::-1]
    for yi, a, b, c, d in zip(y, prior, fixed_family, honest_random, realistic):
        ax.plot([min(b, c, d), a], [yi, yi], color=GREY, lw=0.8, zorder=1)
        for tick in (b, c):
            ax.plot([tick, tick], [yi - 0.18, yi + 0.18], color=MID, lw=1.0, zorder=2)
        ax.scatter(a, yi, s=18, color=GREY, zorder=3)
        ax.scatter(d, yi, s=18, color=BLUE, zorder=3)
        ax.text(0.958, yi, f"−{a - d:.2f}", va="center", ha="left", fontsize=7)
    ax.axvline(0.5, color="#555555", lw=0.6, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels[:len(models)])
    ax.set_xlim(0.45, 0.955)
    ax.set_xlabel("ROC-AUC (five-fold mean)")
    ax.scatter([], [], s=18, color=GREY, label="Fixed compiler, random split")
    ax.scatter([], [], s=18, color=BLUE, label="Honest variation, family split")
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, 1.24), ncol=2, frameon=False, fontsize=7.5)
    fig.savefig(f"{out}/Fig2_realism.png", bbox_inches="tight")


def fig_unseen_splits(results, out):
    auc = pd.read_csv(f"{results}/r1_unseen_auc_per_split.csv")
    split_mean = auc.groupby(["seed", "model"]).AUC.mean().unstack()
    order = [("GIN-L2D", "Proposed GIN-L2D"), ("B4", "B4 Idempotence-residual HGB"),
             ("U1 bigram", "U1 Wire-bigram surprisal (calibrated)"), ("MGM-C", "MGM-C (calibrated)")]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(3.5, 1.35))
    for k, (label, col) in enumerate(order):
        v = split_mean[col].values
        colour = BLUE if label == "MGM-C" else MID
        ax.scatter(v, k + rng.uniform(-0.12, 0.12, len(v)), s=12, color=colour, alpha=0.75, lw=0)
        ax.plot([v.mean()] * 2, [k - 0.32, k + 0.32], color=colour if label == "MGM-C" else "black", lw=1.4)
    ax.axvline(0.5, color="#555555", lw=0.6, ls="--")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([o[0] for o in order])
    ax.set_xlim(0.45, 0.75)
    ax.set_xlabel("Unseen-type ROC-AUC per family split")
    fig.savefig(f"{out}/Fig3_unseen_splits.png", bbox_inches="tight", pad_inches=0.02)


def fig_family(results, out, min_samples=80):
    e7 = pd.read_csv(f"{results}/e7_mgm_predictions.csv")
    rows = []
    for family, d in e7.groupby("family"):
        if len(d) >= min_samples:
            rows.append((f"{family} ({len(d)})", roc_auc_score(d.label, d["Proposed GIN-L2D"]),
                         roc_auc_score(d.label, d["MGM-C (calibrated)"])))
    rows.sort(key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(3.5, 2.15))
    for k, (label, gin, mgm) in enumerate(rows):
        ax.plot([gin, mgm], [k, k], color="#bbbbbb", lw=0.9)
        ax.scatter(gin, k, marker="s", s=16, color=MID)
        ax.scatter(mgm, k, marker="o", s=16, color=BLUE)
    ax.axvline(0.5, color="#555555", lw=0.6, ls="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xlabel("ROC-AUC within family (five-fold, pooled)")
    ax.scatter([], [], marker="s", s=16, color=MID, label="GIN-L2D (supervised)")
    ax.scatter([], [], marker="o", s=16, color=BLUE, label="MGM-C (self-supervised)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2, frameon=False, fontsize=7.5)
    fig.savefig(f"{out}/Fig4_family.png", bbox_inches="tight")


def fig_attack(results, out):
    t = pd.read_csv(f"{results}/table16_attack_budget_curve.csv")
    styles = {"B4 Idempotence-residual HGB": ("B4", dict(color=MID, ls="-", marker="s")),
              "Proposed GIN-L2D": ("GIN-L2D", dict(color=MID, ls="--", marker="^")),
              "MGM-C (calibrated)": ("MGM-C", dict(color=MID, ls=":", marker="o")),
              "Ensemble (B4 + GIN-L2D + MGM-C)": ("Ensemble", dict(color=BLUE, ls="-", marker="D", lw=1.4))}
    fig, ax = plt.subplots(figsize=(3.5, 1.8))
    for col, (label, style) in styles.items():
        ax.plot(t.budget, t[col], label=label, markersize=3, lw=style.pop("lw", 0.9), **style)
    ax.axhline(0.5, color="#555555", lw=0.6, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks(t.budget)
    ax.set_xticklabels([str(b) for b in t.budget])
    ax.set_xlabel("Adversary budget (candidate placements, log scale)")
    ax.set_ylabel("ROC-AUC, attacked vs clean")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper right")
    fig.savefig(f"{out}/Fig5_attack.png", bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_realism(args.results, args.out)
    fig_unseen_splits(args.results, args.out)
    fig_family(args.results, args.out)
    fig_attack(args.results, args.out)
    print(f"Figures written to {args.out}/")


if __name__ == "__main__":
    main()
