"""Figures 1-6 of the manuscript as vector PDF (submission) and 300-dpi PNG (preview).

Usage:
    python code/make_figures.py --res results --out results/figures
Needs results/analysis_fold6_v3 (with predictions.csv.gz from Zenodo) and
results/cluster_bootstrap/cluster_bootstrap.csv.
"""
import argparse
import importlib.util
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("pf", os.path.join(HERE, "pilot_lce_final.py"))
pf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pf)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
ALGS = ["PSO-w", "PSO-cf", "DE-rand1", "DE-best1", "DE-ctb1", "GWO", "WOA", "HHO"]
COL = {"LastValue": "#9e9e9e", "PowerLaw": "#c2a26b", "LC-PFN": "#d1495b", "ELA-Traj": "#2e86ab",
       "Meta-Curve": "#a5b4c4", "Curve+ELA": "#6c8ebf", "PA-LCE": "#66a182", "PA-LCE+ELA": "#1b4332"}
REF = "PA-LCE+ELA"


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".pdf"))
    fig.savefig(os.path.join(out, name + ".png"), dpi=300)
    plt.close(fig)


def fig1(out):
    fig, ax = plt.subplots(figsize=(7.6, 3.75))
    ax.set_xlim(0, 760); ax.set_ylim(375, 0); ax.axis("off")
    W, H = 232, 88
    c1, c2, c3, rA, rB, rC = 6, 264, 522, 14, 160, 280
    edge, acc, q = "#555555", "#1f5fa8", "#444444"

    def box(x, y, name, lines, main=False, dashed=False, fc=None):
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0,rounding_size=8",
                                    lw=2 if main else 1.25, ls="--" if dashed else "-",
                                    ec=acc if main else edge,
                                    fc=fc if fc else ((0.122, 0.373, 0.659, 0.12) if main else "white")))
        ax.text(x + 12, y + 24, name, fontsize=9.0, fontweight="bold", va="baseline")
        for i, t in enumerate(lines):
            ax.text(x + 12, y + 44 + 16 * i, t, fontsize=8.6, color="#111111" if main else q, va="baseline")

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=edge, lw=1.25, shrinkA=0, shrinkB=0, mutation_scale=9))

    box(c1, rA, "1 Stop runs early", ["Each run stops at 5% of its", "budget (also 10% and 20%)", "43,200 runs, 3 dimensions"])
    box(c2, rA, "2 Read the prefix", ["Curve: 15 features", "Population state: 15", "Trajectory ELA: 33"])
    box(c3, rA, "3 Quantile boosting", ["Trained on other functions", "Median, 5% and 95%", "quantiles of final error"])
    box(c3, rB, "4 Cross-conformal interval", ["Scores from held-out folds", "90% interval that holds", "on unseen functions"])
    box(c2, rB, "5 Early-stopped benchmark", ["Friedman mean ranks and", "28 Wilcoxon–Holm tests", "on extrapolated values"])
    box(c1, rB, "6 Compare with full budget", ["Kendall τ of mean ranks", "Share of conclusions kept", "Reversed, spurious, lost"], main=True)
    box(c2, rC, "Equal-cost alternative", ["Fewer full runs per problem", "(or fewer instances),", "same tests, no model"], dashed=True, fc="#f4f4f4")
    arrow(c1 + W, rA + H / 2, c2, rA + H / 2)
    arrow(c2 + W, rA + H / 2, c3, rA + H / 2)
    arrow(c3 + W / 2, rA + H, c3 + W / 2, rB)
    arrow(c3, rB + H / 2, c2 + W, rB + H / 2)
    arrow(c2, rB + H / 2, c1 + W, rB + H / 2)
    ax.plot([c2, c1 + W / 2, c1 + W / 2], [rC + H / 2, rC + H / 2, rB + H + 10], color=edge, lw=1.25, ls="--")
    arrow(c1 + W / 2, rB + H + 10, c1 + W / 2, rB + H)
    ax.text(c1 + W / 2 + 8, rC + H / 2 - 8, "Sec. 5.7", fontsize=8, color=q, style="italic")
    fig.subplots_adjust(0, 0, 1, 1)
    save(fig, out, "Figure_1")


def fig2(res, out):
    a = pd.read_csv(os.path.join(res, "analysis_fold6_v3", "accuracy.csv"))
    a = a[a.kind == "within"].groupby(["predictor", "fraction"])[["MAE", "Spearman_within_problem"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, m, yl, t in [(axes[0], "MAE", "MAE of final log$_{10}$ error", "(a) Point accuracy"),
                         (axes[1], "Spearman_within_problem", "Within-problem Spearman $\\rho$", "(b) Ranking of runs")]:
        for x, c in COL.items():
            g = a[a.predictor == x].sort_values("fraction")
            ax.plot(g.fraction * 100, g[m], marker="o", ms=4, color=c, lw=3 if x == REF else 1.3,
                    label=x + (" (proposed)" if x == REF else ""), zorder=3 if x == REF else 2)
        ax.set_xticks([5, 10, 20]); ax.set_xlabel("Observed budget (% of FEs)"); ax.set_ylabel(yl); ax.set_title(t)
        ax.spines[["top", "right"]].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    save(fig, out, "Figure_2")


def fig3(res, out):
    a = pd.read_csv(os.path.join(res, "analysis_fold6_v3", "accuracy.csv"))
    a = a[a.kind == "within"].groupby(["predictor", "fraction"])[["coverage_raw", "coverage_cv"]].mean()
    preds = ["LC-PFN", "ELA-Traj", "Meta-Curve", "Curve+ELA", "PA-LCE", "PA-LCE+ELA"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    for ax, f in zip(axes, [0.05, 0.10, 0.20]):
        x = np.arange(len(preds))
        ax.bar(x - 0.2, [a.loc[(p, f), "coverage_raw"] for p in preds], 0.4, color="#c8c8c8", label="Raw quantile interval")
        ax.bar(x + 0.2, [a.loc[(p, f), "coverage_cv"] for p in preds], 0.4, color=COL[REF], label="Cross-conformal")
        ax.axhline(0.9, color="black", ls="--", lw=1)
        ax.set_ylim(0.5, 1.0); ax.set_xticks(x); ax.set_xticklabels(preds, rotation=40, ha="right", fontsize=8)
        ax.set_title(f"{int(f * 100)}% FEs"); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Empirical coverage (nominal 0.90)", fontsize=8.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, out, "Figure_3")


GROUPS = {"ELA: dispersi": ("ELA: dispersion", "ela"), "ELA: distribusi": ("ELA: y-distribution", "ela"),
          "ELA: meta-model": ("ELA: meta-model", "ela"), "ELA: nearest-better": ("ELA: nearest-better clustering", "ela"),
          "Kurva: bentuk (trajektori)": ("Curve: trajectory shape", "curve"),
          "Kurva: level & laju": ("Curve: level and rate", "curve"), "Kurva: stagnasi": ("Curve: stagnation", "curve"),
          "Populasi: level & tren diversitas": ("Population: diversity level/trend", "pop"),
          "Populasi: sebaran fitness": ("Population: fitness spread", "pop"),
          "Populasi: trajektori diversitas": ("Population: diversity trajectory", "pop")}
GCOL = {"ela": "#2e86ab", "curve": "#9e9e9e", "pop": "#66a182"}


def fig4(res, out):
    f = pd.read_csv(os.path.join(res, "analysis_fold6_v3", "feature_group_importance.csv"))
    f = f[f.kind == "within"]
    order = f.groupby("group").delta_MAE.mean().sort_values(ascending=False).index.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.6), sharey=True)
    y = np.arange(len(order))
    for ax, d in zip(axes, [5, 10, 20]):
        for k, (fr, al) in enumerate([(0.05, 1.0), (0.10, 0.65), (0.20, 0.35)]):
            g = f[(f.dim == d) & (f.fraction == fr)].set_index("group").reindex(order)
            ax.barh(y - 0.27 + 0.27 * k, g.delta_MAE, 0.27, color=[GCOL[GROUPS[o][1]] for o in order], alpha=al)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"d = {d}"); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y); axes[0].set_yticklabels([GROUPS[o][0] for o in order], fontsize=8.5); axes[0].invert_yaxis()
    fig.supxlabel("Increase in MAE when the feature group is permuted", fontsize=9.5, y=0.1)
    h = [Patch(color="#555555", alpha=a_) for a_ in (1.0, 0.65, 0.35)]
    fig.legend(h, ["5% FEs", "10% FEs", "20% FEs"], loc="lower right", ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, out, "Figure_4")


def fig5(res, out):
    b = pd.read_csv(os.path.join(res, "cluster_bootstrap", "cluster_bootstrap.csv"))
    preds = ["LastValue", "LC-PFN", "ELA-Traj", "Curve+ELA", "PA-LCE", "PA-LCE+ELA"]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.8), sharex=True, sharey="row")
    for c, d in enumerate([5, 10, 20]):
        for r, (m, yl) in enumerate([("tau", "Kendall $\\tau$ of mean ranks"),
                                     ("agree", "Agreement of pairwise\nWilcoxon–Holm conclusions")]):
            ax = axes[r, c]
            for x in preds:
                g = b[(b.dim == d) & (b.predictor == x)].sort_values("fraction")
                if x == REF:
                    ax.fill_between(g.fraction * 100, g[m + "_lo"], g[m + "_hi"], color=COL[x], alpha=0.15, lw=0)
                ax.plot(g.fraction * 100, g[m], marker="o", ms=3.5, color=COL[x], lw=2.6 if x == REF else 1.2,
                        label=x + (" (proposed)" if x == REF else ""), zorder=3 if x == REF else 2)
            ax.set_xticks([5, 10, 20]); ax.spines[["top", "right"]].set_visible(False)
            if r == 0:
                ax.set_title(f"d = {d}")
            if c == 0:
                ax.set_ylabel(yl, fontsize=9)
            if r == 1:
                ax.set_xlabel("Observed budget (% of FEs)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, out, "Figure_5")


def fig6(res, out, dim=10, frac=0.05):
    p = pd.read_csv(os.path.join(res, "analysis_fold6_v3", "predictions.csv.gz"))
    s = p[(p.kind == "within") & (p.dim == dim) & (p.fraction == frac)]
    tab = lambda c: s.groupby(["func", "inst", "alg"])[c].mean().unstack("alg")[ALGS].values
    pairs = list(itertools.combinations(range(8), 2))
    ct = pf.conclusions(tab("y_true"))[1]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.2))
    cmap = {1: "#e59866", -1: "#7fb3d5", 0: "#f2f2f2"}
    for ax, (x, lab) in zip(axes, [("LastValue", "(a) LastValue"), ("LC-PFN", "(b) LC-PFN"), (REF, "(c) PA-LCE+ELA (proposed)")]):
        cp = pf.conclusions(tab("pred_" + x))[1]
        M = np.zeros((8, 8), int); P = np.zeros((8, 8), int)
        for q, (i, j) in enumerate(pairs):
            M[i, j], M[j, i] = ct[q], -ct[q]; P[i, j], P[j, i] = cp[q], -cp[q]
        n = dict(rev=0, spur=0, lost=0)
        for i in range(8):
            for j in range(8):
                if i == j:
                    continue
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=cmap[M[i, j]], ec="white", lw=1))
                t, pr = M[i, j], P[i, j]
                mark = "×" if (t != 0 and pr != 0 and t != pr) else "+" if (t == 0 and pr != 0) else "○" if (t != 0 and pr == 0) else ""
                if mark:
                    ax.text(j + 0.5, i + 0.52, mark, ha="center", va="center", fontsize=9)
        for q in range(len(pairs)):
            n["rev"] += int(ct[q] != 0 and cp[q] != 0 and ct[q] != cp[q]); n["spur"] += int(ct[q] == 0 and cp[q] != 0)
            n["lost"] += int(ct[q] != 0 and cp[q] == 0)
        ax.set_xlim(0, 8); ax.set_ylim(8, 0); ax.set_aspect("equal")
        ax.set_xticks(np.arange(8) + 0.5); ax.set_xticklabels(ALGS, rotation=90, fontsize=6.5)
        ax.set_yticks(np.arange(8) + 0.5); ax.set_yticklabels(ALGS if ax is axes[0] else [], fontsize=6.5)
        ax.tick_params(length=0); [sp.set_visible(False) for sp in ax.spines.values()]
        ax.set_title(f"{lab}\nreversed {n['rev']} · spurious {n['spur']} · lost {n['lost']}", fontsize=8)
    h = [Patch(color=cmap[-1]), Patch(color=cmap[1]), Patch(color=cmap[0], ec="#bbbbbb")]
    fig.legend(h, ["Row algorithm significantly better", "Row algorithm significantly worse", "No significant difference"],
               loc="lower center", ncol=3, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, out, "Figure_6")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig1(a.out); fig2(a.res, a.out); fig3(a.res, a.out); fig4(a.res, a.out); fig5(a.res, a.out); fig6(a.res, a.out)
    print("written to", a.out)


if __name__ == "__main__":
    main()
