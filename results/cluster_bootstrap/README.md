Output of `python code/cluster_bootstrap.py results/analysis_fold6_v3 --out results/cluster_bootstrap`
(500 function-level resamples, seed 2026, within scenario, 8 algorithms).

- `cluster_bootstrap.csv`: point value and 95% interval of Kendall tau and agreement,
  number of reversed conclusions, and P(0 rev) per dimension, budget fraction and extrapolator.
- `paired_diff.csv`: PA-LCE+ELA minus each other extrapolator on the same resamples
  (mean and 95% interval of the difference).
