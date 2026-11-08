"""MAE of each extrapolator by BBOB function group and by algorithm (Section 5.1, 5.6).

Usage:
    python code/error_by_group.py results/analysis_fold6_v3 --fraction 0.05
"""
import argparse
import os

import pandas as pd

PRED = ["LastValue", "LC-PFN", "ELA-Traj", "Curve+ELA", "PA-LCE", "PA-LCE+ELA"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis_dir")
    ap.add_argument("--fraction", type=float, default=0.05)
    a = ap.parse_args()
    p = pd.read_csv(os.path.join(a.analysis_dir, "predictions.csv.gz"))
    p = p[(p.kind == "within") & (p.fraction == a.fraction)]
    for x in PRED:
        p[x] = (p["pred_" + x] - p.y_true).abs()
    pd.set_option("display.width", 200)
    print("MAE by BBOB group (all dimensions)")
    print(p.groupby("group")[PRED].mean().round(2))
    print("\nMAE of PA-LCE+ELA by dimension and algorithm")
    print(p.groupby(["dim", "alg"])["PA-LCE+ELA"].mean().unstack().round(2))
    print("\nMean bias (prediction - truth) of PA-LCE+ELA by dimension and algorithm")
    print((p["pred_PA-LCE+ELA"] - p.y_true).groupby([p.dim, p.alg]).mean().unstack().round(2))


if __name__ == "__main__":
    main()
