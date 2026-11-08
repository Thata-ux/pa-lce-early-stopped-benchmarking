"""Additional analyses on the released predictions (no new runs needed).

1. Equal-cost comparison: benchmark built from k full-budget runs per problem
   (k = 1, 2, 3; cost k/15 of the full study) versus early stopping of all 15 runs
   at 5/10/20% of the budget (cost 0.05/0.10/0.20).
2. Protocol step (4): pairs whose conclusion changes when the extrapolated values
   are replaced by the ends of their calibrated intervals (or that are unstable
   under a function-level bootstrap of the predictions) are completed at full
   budget; reports the reversals left and the net saving.
3. Sensitivity of the decision measures to the unit of analysis: function-level
   means (n = 24) instead of function x instance problems (n = 120).
4. Effect size (rank-biserial, full budget) of the pairs on which the
   early-stopped benchmark disagrees, compared with all significant pairs.
5. Figure 5 with function-level (cluster) bootstrap bands.

Usage:
    python code/extra_analyses.py results/analysis_fold6_v3 --boot results/cluster_bootstrap/cluster_bootstrap.csv \
        --out results/extra_analyses
"""
import argparse
import importlib.util
import itertools
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("pf", os.path.join(HERE, "pilot_lce_final.py"))
pf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pf)

ALGS = ["PSO-w", "PSO-cf", "DE-rand1", "DE-best1", "DE-ctb1", "GWO", "WOA", "HHO"]
PRED = ["LastValue", "LC-PFN", "ELA-Traj", "Curve+ELA", "PA-LCE", "PA-LCE+ELA"]
REF = "PA-LCE+ELA"
PAIRS = list(itertools.combinations(range(len(ALGS)), 2))


def table(df, col, level=("func", "inst")):
    return df.groupby(list(level) + ["alg"])[col].mean().unstack("alg")[ALGS]


def equal_cost(p, rng, n_draw):
    rows = []
    for d, s in p[p.fraction == p.fraction.min()].groupby("dim"):
        truth = pf.conclusions(table(s, "y_true").values)
        runs = sorted(s.run.unique())
        piv = s.pivot_table(index=["func", "inst", "alg"], columns="run", values="y_true")[runs]
        for k in (1, 2, 3):
            res = []
            for _ in range(n_draw):
                pick = np.array([rng.choice(len(runs), k, replace=False) for _ in range(len(piv))])
                v = np.take_along_axis(piv.values, pick, axis=1).mean(1)
                tab = pd.Series(v, index=piv.index).unstack("alg")[ALGS].values
                res.append(pf.compare(truth, pf.conclusions(tab)))
            r = pd.DataFrame(res)
            rows.append(dict(dim=d, method=f"{k} full run(s)/problem", cost=k / len(runs),
                             tau=r.kendall_tau.mean(), agree=r.wilcoxon_agree.mean(),
                             rev=r.n_reversed.mean(), p_rev0=(r.n_reversed == 0).mean(),
                             same_best=r.same_best.mean()))
    return pd.DataFrame(rows)


def equal_cost_cluster(p, rng, n_boot):
    """Like Table 8: function-level bootstrap. For each resample of the 24 functions, the
    early-stopped benchmark (fixed predictions) and a benchmark of k full runs (fresh random
    draw) are compared with the full benchmark on the same resample. Variant 'held-out' builds
    the reference from the 15 - k runs not drawn, so the k runs are not part of their own
    reference (for early stopping the reference stays all 15 runs, as in the paper)."""
    rows = []
    for d, sd in p.groupby("dim"):
        s0 = sd[sd.fraction == sd.fraction.min()]
        runs = sorted(s0.run.unique())
        piv = s0.pivot_table(index=["func", "inst", "alg"], columns="run", values="y_true")[runs]
        keys = piv.index
        prob = keys.droplevel("alg").unique()
        funcs = prob.get_level_values(0).values
        uf = np.unique(funcs)
        idx = {u: np.where(funcs == u)[0] for u in uf}
        full = table(s0, "y_true").values
        es = {f: {x: table(sf, "pred_" + x).values for x in ("LastValue", "LC-PFN", REF)}
              for f, sf in sd.groupby("fraction")}
        res = []
        for _ in range(n_boot):
            ix = np.concatenate([idx[u] for u in rng.choice(uf, len(uf))])
            truth = pf.conclusions(full[ix])
            for f, dd in es.items():
                for x, tab in dd.items():
                    res.append(dict(method=f"early stop {int(f * 100)}% + {x}", cost=f,
                                    **pf.compare(truth, pf.conclusions(tab[ix]))))
            for k in (1, 2, 3):
                pick = np.array([rng.choice(len(runs), k, replace=False) for _ in range(len(piv))])
                mask = np.zeros(piv.shape, bool)
                np.put_along_axis(mask, pick, True, axis=1)
                v_in = np.where(mask, piv.values, np.nan)
                v_out = np.where(~mask, piv.values, np.nan)
                tab_k = pd.Series(np.nanmean(v_in, 1), index=keys).unstack("alg")[ALGS].values[ix]
                ref_out = pd.Series(np.nanmean(v_out, 1), index=keys).unstack("alg")[ALGS].values[ix]
                res.append(dict(method=f"{k} full run(s)/problem", cost=k / len(runs),
                                **pf.compare(truth, pf.conclusions(tab_k))))
                res.append(dict(method=f"{k} full run(s)/problem, held-out reference", cost=k / len(runs),
                                **pf.compare(pf.conclusions(ref_out), pf.conclusions(tab_k))))
        r = pd.DataFrame(res)
        for m, g in r.groupby("method", sort=False):
            rows.append(dict(dim=d, method=m, cost=g.cost.iloc[0],
                             tau_median=g.kendall_tau.median(), tau_lo=g.kendall_tau.quantile(.025),
                             tau_hi=g.kendall_tau.quantile(.975), agree_mean=g.wilcoxon_agree.mean(),
                             agree_lo=g.wilcoxon_agree.quantile(.025), agree_hi=g.wilcoxon_agree.quantile(.975),
                             rev_mean=g.n_reversed.mean(), p_rev0=(g.n_reversed == 0).mean(),
                             same_best=g.same_best.mean()))
    return pd.DataFrame(rows)


def compare_sets(true_c, pred_c):
    """pf.compare without per-problem winner agreement (problem sets may differ)."""
    return pf.compare((true_c[0], true_c[1], np.zeros(1)), (pred_c[0], pred_c[1], np.zeros(1)))


def low_budget(p, rng, n_boot):
    """Budgets below one full run per problem (< 6.7%). Early stopping at 5% using r of the 15 runs
    (cost 0.05 r / 15) versus one full run on m of the 5 instances of each function
    (cost m / 75, problems = 24 m). Function-level bootstrap as in Table 8."""
    rows = []
    for d, sd in p.groupby("dim"):
        s = sd[sd.fraction == 0.05]
        runs = sorted(s.run.unique())
        insts = sorted(s.inst.unique())
        full = table(s, "y_true")
        prob = full.index
        funcs = prob.get_level_values(0).values
        uf = np.unique(funcs)
        idx = {u: np.where(funcs == u)[0] for u in uf}
        piv = {c: s.pivot_table(index=["func", "inst", "alg"], columns="run", values=c)[runs]
               for c in ("y_true", "pred_" + REF, "pred_LastValue")}
        res = []
        for _ in range(n_boot):
            draw = rng.choice(uf, len(uf))
            ix = np.concatenate([idx[u] for u in draw])
            truth = pf.conclusions(full.values[ix])
            for r in (1, 3, 5):
                pick = np.array([rng.choice(len(runs), r, replace=False) for _ in range(len(piv["y_true"]))])
                for x in (REF, "LastValue"):
                    v = np.take_along_axis(piv["pred_" + x].values, pick, axis=1).mean(1)
                    tab = pd.Series(v, index=piv["y_true"].index).unstack("alg")[ALGS].values[ix]
                    res.append(dict(method=f"early stop 5% + {x}, {r} run(s)/problem", cost=0.05 * r / len(runs),
                                    **pf.compare(truth, pf.conclusions(tab))))
            for m in (1, 2, 3):
                one = piv["y_true"].values[np.arange(len(piv["y_true"])), rng.integers(0, len(runs), len(piv["y_true"]))]
                one = pd.Series(one, index=piv["y_true"].index).unstack("alg")[ALGS]
                keep = {u: rng.choice(insts, m, replace=False) for u in uf}
                rows_ix = np.concatenate([[k for k in idx[u] if prob[k][1] in keep[u]] for u in draw])
                res.append(dict(method=f"1 full run on {m} of 5 instances", cost=m / (len(insts) * len(runs)),
                                **compare_sets(truth, pf.conclusions(one.values[rows_ix]))))
        rr = pd.DataFrame(res)
        for m_, g in rr.groupby("method", sort=False):
            rows.append(dict(dim=d, method=m_, cost=g.cost.iloc[0], tau_median=g.kendall_tau.median(),
                             tau_lo=g.kendall_tau.quantile(.025), agree_mean=g.wilcoxon_agree.mean(),
                             rev_mean=g.n_reversed.mean(), p_rev0=(g.n_reversed == 0).mean(),
                             same_best=g.same_best.mean()))
    return pd.DataFrame(rows)


def interval_flags(s, pred_tab, lo_tab, hi_tab):
    """Pairs whose Wilcoxon-Holm conclusion changes when one algorithm takes its lower and the
    other its upper interval ends (both directions)."""
    c0 = pf.conclusions(pred_tab)[1]
    flags = np.zeros(len(PAIRS), bool)
    for q, (i, j) in enumerate(PAIRS):
        for a, b in ((lo_tab, hi_tab), (hi_tab, lo_tab)):
            t = pred_tab.copy()
            t[:, i], t[:, j] = a[:, i], b[:, j]
            if pf.conclusions(t)[1][q] != c0[q]:
                flags[q] = True
                break
    return flags


def bootstrap_flags(s, pred_tab, rng, n_boot, stable=0.95):
    """Pairs whose extrapolated conclusion is reproduced in fewer than `stable` of the
    function-level bootstrap resamples of the extrapolated benchmark itself (no truth used)."""
    funcs = s.groupby(["func", "inst"]).size().index.get_level_values(0).values
    uf = np.unique(funcs)
    idx = {u: np.where(funcs == u)[0] for u in uf}
    c0 = pf.conclusions(pred_tab)[1]
    same = np.zeros(len(PAIRS))
    for _ in range(n_boot):
        ix = np.concatenate([idx[u] for u in rng.choice(uf, len(uf))])
        same += pf.conclusions(pred_tab[ix])[1] == c0
    return same / n_boot < stable


def protocol(p, rng, n_boot):
    rows = []
    for (d, f), s in p.groupby(["dim", "fraction"]):
        truth_tab = table(s, "y_true").values
        truth = pf.conclusions(truth_tab)
        pred_tab = table(s, "pred_" + REF).values
        lo, hi = table(s, "lo_" + REF).values, table(s, "hi_" + REF).values
        for rule, flags in (("interval ends", interval_flags(s, pred_tab, lo, hi)),
                            ("bootstrap stability < 0.95", bootstrap_flags(s, pred_tab, rng, n_boot))):
            done = sorted({a for q, fl in enumerate(flags) if fl for a in PAIRS[q]})
            tab = pred_tab.copy()
            tab[:, done] = truth_tab[:, done]
            cmp = pf.compare(truth, pf.conclusions(tab))
            cost = f + (1 - f) * len(done) / len(ALGS)
            rows.append(dict(dim=d, fraction=f, rule=rule, pairs_flagged=int(flags.sum()),
                             algorithms_completed=len(done), tau=cmp["kendall_tau"],
                             agree=cmp["wilcoxon_agree"], rev=cmp["n_reversed"],
                             spurious=cmp["n_false_sig"], lost=cmp["n_lost_sig"],
                             cost_fraction=cost, saving_vs_full=1 - cost))
    return pd.DataFrame(rows)


def function_level(p):
    rows = []
    for (d, f), s in p.groupby(["dim", "fraction"]):
        truth = pf.conclusions(table(s, "y_true", level=("func",)).values)
        n_sig = int((truth[1] != 0).sum())
        for x in PRED:
            cmp = pf.compare(truth, pf.conclusions(table(s, "pred_" + x, level=("func",)).values))
            rows.append(dict(dim=d, fraction=f, predictor=x, truth_significant=n_sig, **cmp))
    return pd.DataFrame(rows)


def disagreement_effects(p):
    rows = []
    for (d, f), s in p.groupby(["dim", "fraction"]):
        tt = table(s, "y_true").values
        truth = pf.conclusions(tt)[1]
        r_full = np.array([abs(pf.wilcoxon_p(tt[:, i], tt[:, j])[1]) for i, j in PAIRS])
        sig_med = float(np.median(r_full[truth != 0]))
        for x in PRED:
            c = pf.conclusions(table(s, "pred_" + x).values)[1]
            for kind, m in (("reversed", (truth != 0) & (c != 0) & (truth != c)),
                            ("spurious", (truth == 0) & (c != 0)),
                            ("lost", (truth != 0) & (c == 0))):
                for q in np.where(m)[0]:
                    i, j = PAIRS[q]
                    rows.append(dict(dim=d, fraction=f, predictor=x, kind=kind,
                                     pair=f"{ALGS[i]} vs {ALGS[j]}", abs_r_full=r_full[q],
                                     median_abs_r_significant_pairs=sig_med))
    return pd.DataFrame(rows)


def figure5(boot_csv, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b = pd.read_csv(boot_csv)
    col = {"LastValue": "#9e9e9e", "LC-PFN": "#d1495b", "ELA-Traj": "#2e86ab",
           "Curve+ELA": "#6c8ebf", "PA-LCE": "#66a182", "PA-LCE+ELA": "#1b4332"}
    lab = {"PA-LCE+ELA": "PA-LCE+ELA (proposed)"}
    dims = sorted(b.dim.unique())
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(2, len(dims), figsize=(10.8, 6.9), sharex=True, sharey="row")
    for c, d in enumerate(dims):
        for r, (m, ylab) in enumerate([("tau", "Kendall $\\tau$ of mean ranks"),
                                       ("agree", "Agreement of pairwise\nWilcoxon–Holm conclusions")]):
            ax = axes[r, c]
            for x in PRED:
                g = b[(b.dim == d) & (b.predictor == x)].sort_values("fraction")
                if x == REF:
                    ax.fill_between(g.fraction * 100, g[m + "_lo"], g[m + "_hi"], color=col[x],
                                    alpha=0.15, lw=0)
                ax.plot(g.fraction * 100, g[m], marker="o", ms=4, color=col[x],
                        lw=3 if x == REF else 1.3, label=lab.get(x, x), zorder=3 if x == REF else 2)
            ax.set_xticks([5, 10, 20])
            ax.spines[["top", "right"]].set_visible(False)
            if r == 0:
                ax.set_title(f"d = {d}")
            if c == 0:
                ax.set_ylabel(ylab)
            if r == 1:
                ax.set_xlabel("Observed budget (% of FEs)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(out, "fig5_preservation_cluster_bootstrap.png"), dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis_dir")
    ap.add_argument("--boot", default="results/cluster_bootstrap/cluster_bootstrap.csv")
    ap.add_argument("--out", default="results/extra_analyses")
    ap.add_argument("--n_draw", type=int, default=200)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    p = pd.read_csv(os.path.join(a.analysis_dir, "predictions.csv.gz"))
    p = p[p.kind == "within"]
    equal_cost(p, rng, a.n_draw).to_csv(os.path.join(a.out, "equal_cost_subsampling.csv"), index=False)
    equal_cost_cluster(p, rng, a.n_boot).to_csv(os.path.join(a.out, "equal_cost_cluster_bootstrap.csv"), index=False)
    protocol(p, rng, a.n_boot).to_csv(os.path.join(a.out, "protocol_step4.csv"), index=False)
    low_budget(p, rng, a.n_boot).to_csv(os.path.join(a.out, "low_budget.csv"), index=False)
    function_level(p).to_csv(os.path.join(a.out, "function_level_sensitivity.csv"), index=False)
    disagreement_effects(p).to_csv(os.path.join(a.out, "disagreement_effect_sizes.csv"), index=False)
    figure5(a.boot, a.out)
    print("written to", a.out)


if __name__ == "__main__":
    main()
