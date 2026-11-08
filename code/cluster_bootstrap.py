"""Function-level (cluster) bootstrap of the decision-level measures.

Resamples the 24 BBOB functions with replacement (all instances of a drawn
function stay together), recomputes Friedman ranks, the 28 pairwise
Wilcoxon-Holm conclusions and their agreement with the full-budget benchmark,
and reports 95% intervals, P(0 reversals) and paired differences between
extrapolators. Produces the intervals of Table 8 and the paragraph
"Uncertainty of these conclusions" (Section 5.4).

Usage:
    python code/cluster_bootstrap.py results/analysis_fold6_v3 --out results/cluster_bootstrap
"""
import argparse
import importlib.util
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis_dir")
    ap.add_argument("--out", default="results/cluster_bootstrap")
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    p = pd.read_csv(os.path.join(a.analysis_dir, "predictions.csv.gz"))
    p = p[p.kind == "within"]
    rng = np.random.default_rng(a.seed)
    rows, diffs = [], []
    for (d, f), s in p.groupby(["dim", "fraction"]):
        g = s.groupby(["func", "inst", "alg"])
        tabs = {c: g[c].mean().unstack("alg")[ALGS] for c in ["y_true"] + ["pred_" + x for x in PRED]}
        funcs = tabs["y_true"].index.get_level_values(0).values
        uf = np.unique(funcs)
        idx = {u: np.where(funcs == u)[0] for u in uf}
        tt = tabs["y_true"].values
        tc = pf.conclusions(tt)
        boots = [np.concatenate([idx[u] for u in rng.choice(uf, len(uf))]) for _ in range(a.n_boot)]
        tcb = [pf.conclusions(tt[ix]) for ix in boots]
        res = {}
        for x in PRED:
            tp = tabs["pred_" + x].values
            point = pf.compare(tc, pf.conclusions(tp))
            bs = pd.DataFrame([pf.compare(t, pf.conclusions(tp[ix])) for t, ix in zip(tcb, boots)])
            res[x] = bs
            rows.append(dict(dim=d, fraction=f, predictor=x,
                             tau=point["kendall_tau"], tau_lo=bs.kendall_tau.quantile(.025),
                             tau_hi=bs.kendall_tau.quantile(.975),
                             agree=point["wilcoxon_agree"], agree_lo=bs.wilcoxon_agree.quantile(.025),
                             agree_hi=bs.wilcoxon_agree.quantile(.975),
                             rev=point["n_reversed"], p_rev0=(bs.n_reversed == 0).mean()))
        for x in PRED:
            if x == REF:
                continue
            for m in ["kendall_tau", "wilcoxon_agree", "n_reversed"]:
                dd = res[REF][m] - res[x][m]
                diffs.append(dict(dim=d, fraction=f, vs=x, metric=m, mean=dd.mean(),
                                  lo=dd.quantile(.025), hi=dd.quantile(.975)))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "cluster_bootstrap.csv"), index=False)
    pd.DataFrame(diffs).to_csv(os.path.join(a.out, "paired_diff.csv"), index=False)
    print("written to", a.out)


if __name__ == "__main__":
    main()
