# Early-stopped benchmarking of metaheuristics with trajectory-informed learning-curve extrapolation (PA-LCE)

Code and results supporting the manuscript

> T. H. Andika, *Can metaheuristic benchmarks be stopped early? A decision-level comparison of convergence-curve extrapolation with running fewer runs*, submitted to *Swarm and Evolutionary Computation* (2026).

The study asks whether a benchmark of metaheuristics can be stopped after 5–20% of its evaluation budget and its final results extrapolated **without changing the statistical conclusions of the study** (Friedman ranks and pairwise Wilcoxon–Holm tests). Eight population-based metaheuristics (plus CMA-ES and SHADE in an extension) were run on the 24 noiseless BBOB functions in 5, 10 and 20 dimensions (43,200 runs; 54,000 with the extension). Ten extrapolators are compared, from the last observed value and the pretrained LC-PFN to quantile gradient-boosting models over curve, population-state and trajectory-ELA features, calibrated by grouped cross-conformal prediction.

## Repository layout

```
code/
  pilot_lce_final.py      complete pipeline: generate -> lcpfn -> analyze -> extras
  cluster_bootstrap.py    function-level bootstrap, P(0 reversals), paired differences (Table 8, Sec. 5.4)
  error_by_group.py       MAE by BBOB group / algorithm and per-algorithm bias (Sec. 5.1, 5.6)
  extra_analyses.py       equal-cost and low-budget comparisons (Table 9), selective confirmation,
                          function-level sensitivity, effect sizes of disagreeing pairs, Fig. 5
kaggle/
  RUN_ON_KAGGLE.md        the exact notebook cells used to produce the results
results/
  analysis_fold6_v3/      main experiment, 8 algorithms, six-fold CV over functions
  analysis_train1_v3/     cheaper protocol: each fold predicted from one other fold (4 functions)
  analysis_loao_fold6_v3/ leave-one-algorithm-out x function folds (unseen algorithms)
  analysis_fold6_ext10/   extension with CMA-ES and SHADE (10 algorithms, 45 pairs)
  cluster_bootstrap/      output of code/cluster_bootstrap.py on analysis_fold6_v3
  extra_analyses/         output of code/extra_analyses.py on analysis_fold6_v3
requirements.txt
CITATION.cff
LICENSE
```

Each `analysis_*` folder contains `config.json` (all arguments of the run), `accuracy.csv` (MAE, Spearman, interval coverage and width), `benchmark_preservation.csv` (Kendall τ, agreement, reversed / spurious / lost conclusions, same best algorithm, with bootstrap intervals over problems), `stat_tests.csv` (run-level Wilcoxon tests and rank-biserial effect sizes), `feature_group_importance.csv`, `predictions.csv.gz` (on Zenodo, see below; one row per run, budget fraction and scenario, with the prediction of every extrapolator and the 90% intervals of PA-LCE+ELA and LC-PFN) and the figures. `analysis_fold6_v3` also holds the outputs of the `extras` stage: `stepness_vs_error.csv`, `coverage_by_group.csv`, `paired_decision_diff.csv`, `net_saving.csv` and `fig_example_curves.png`.

## Where each result of the manuscript comes from

| Manuscript | Source |
|---|---|
| Table 4, Fig. 2 (accuracy, RQ1) | `results/analysis_fold6_v3/accuracy.csv`, `stat_tests.csv`, `fig3_mae.png` |
| Sec. 5.1, error by BBOB group | `python code/error_by_group.py results/analysis_fold6_v3` |
| Table 5, Fig. 3 (calibration, RQ2a) | `accuracy.csv` (`coverage_raw`, `coverage_cv`, widths), `fig_calibration.png` |
| Table 6 (ablations) | `stat_tests.csv` |
| Fig. 4 (feature-group importance) | `feature_group_importance.csv`, `fig5_feature_importance.png` |
| Table 7 (feature cost) | printed by the `analyze` stage (log) |
| Table 8 (decision level, RQ3) | point values: `benchmark_preservation.csv`; intervals and P(0 rev): `results/cluster_bootstrap/cluster_bootstrap.csv` |
| Sec. 5.4, paired differences | `results/cluster_bootstrap/paired_diff.csv` |
| Fig. 5 | `results/extra_analyses/fig5_preservation_cluster_bootstrap.png` (bands from `cluster_bootstrap.csv`) |
| Fig. 6 | `analysis_fold6_v3/fig4_conclusion_matrix.png` |
| Sec. 5.4, effect sizes of spurious / lost / reversed pairs | `results/extra_analyses/disagreement_effect_sizes.csv` |
| Sec. 5.5 (pooled scenario) | rows with `kind == "pooled"` in the files above |
| Sec. 5.6, step-shaped curves | `analysis_fold6_v3/stepness_vs_error.csv` |
| Sec. 5.6, conditional coverage | `analysis_fold6_v3/coverage_by_group.csv` |
| Sec. 5.6, cheaper training set | `results/analysis_train1_v3/` |
| Sec. 5.6, unseen algorithms | `results/analysis_loao_fold6_v3/`; per-algorithm bias via `code/error_by_group.py` |
| Sec. 5.6, CMA-ES and SHADE | `results/analysis_fold6_ext10/` |
| Sec. 5.7 and Table 9, equal-cost and low-budget alternatives | `results/extra_analyses/equal_cost_cluster_bootstrap.csv`, `low_budget.csv` (and `equal_cost_subsampling.csv`) |
| Sec. 5.7, selective confirmation | `results/extra_analyses/protocol_step4.csv` |
| Sec. 5.6, unit of analysis | `results/extra_analyses/function_level_sensitivity.csv` |
| Sec. 6, cost of training data | `analysis_fold6_v3/net_saving.csv` |

## Reproducing the experiment

All experiments ran on free CPU-only Kaggle notebooks (4 cores, Python 3.13). The exact cells are in `kaggle/RUN_ON_KAGGLE.md`. Locally:

```bash
pip install -r requirements.txt
git clone https://github.com/automl/lcpfn.git          # LC-PFN code and pretrained weights

ALGS="PSO-w PSO-cf DE-rand1 DE-best1 DE-ctb1 GWO WOA HHO"
python code/pilot_lce_final.py generate --algs $ALGS     # 43,200 runs (resumable, per dimension)
python code/pilot_lce_final.py lcpfn    --algs $ALGS --lcpfn_repo lcpfn
python code/pilot_lce_final.py analyze  --tag v3                          # main experiment
python code/pilot_lce_final.py extras   --tag v3                          # stepness, coverage, savings
python code/pilot_lce_final.py analyze  --cv train1 --tag v3              # cheaper training set
python code/pilot_lce_final.py analyze  --cv loao_fold6 --scenarios within --tag v3   # unseen algorithms
python code/cluster_bootstrap.py results_final/analysis_fold6_v3 --out results_final/cluster_bootstrap
python code/extra_analyses.py results_final/analysis_fold6_v3 --boot results_final/cluster_bootstrap/cluster_bootstrap.csv \
    --out results_final/extra_analyses --n_boot 500 --n_draw 500   # values of Table 9

# extension with CMA-ES and SHADE (re-uses the runs above, generates only the new ones)
A10="$ALGS CMA-ES SHADE"
python code/pilot_lce_final.py generate --algs $A10
python code/pilot_lce_final.py lcpfn    --algs $A10 --lcpfn_repo lcpfn
python code/pilot_lce_final.py analyze  --algs $A10 --tag ext10
```

`python code/pilot_lce_final.py all --quick` runs a small version of every stage in a few minutes and is the recommended smoke test.

**Determinism.** Every run uses a seed derived from (dimension, algorithm, function, instance, run) and the global seed 2026, and the BBOB problems come from the official COCO implementation (`coco-experiment` 2.8.3), so `generate` reproduces the same curves. Gradient-boosting hyperparameters are fixed in the code and were not tuned.

**Runtime (4 CPU cores).** `generate` for eight algorithms takes roughly 12–16 hours in total and can be split by dimension (`--dims 20`) over several sessions; it skips combinations that already exist. Each `analyze` with six-fold CV takes about 65 minutes, `loao_fold6` about 6–8 hours.

**LC-PFN.** The official code and pretrained model `EMSIZE512_NLAYERS12_NBUCKETS1000` are used zero-shot. Two shims make the saved model load under current PyTorch (an ignored `is_causal` argument and defaults for attributes added to `GELU` and `TransformerEncoder` later); `stage_lcpfn` prints a sanity check on a synthetic saturating curve (true 0.750, predicted median 0.753, 90% interval 0.727–0.787).

**ELA features.** The `distr`, `meta`, `disp` and `nbc` feature groups are reimplemented following the flacco definitions, because `pflacco` is incompatible with current SciPy.

## Data on Zenodo

Two kinds of files are too large for this repository and are archived on Zenodo: **[Zenodo DOI to be added]**.

- `predictions.csv.gz` of each `analysis_*` folder (15–39 MB each; one row per run, budget fraction and scenario). Download `pa-lce-predictions.zip` and unzip it in the repository root; the files land in their `results/analysis_*/` folders. They are needed by `code/cluster_bootstrap.py`, `code/error_by_group.py` and `code/extra_analyses.py`.
- The raw convergence curves, population-state series and ELA features (`results_final/parts/`, one `.npz` file per dimension × algorithm × function), if deposited. They can also be regenerated exactly with the `generate` stage.

## Citation

See `CITATION.cff`. Please cite the manuscript once published.

## License

Code: MIT (see `LICENSE`). Results and figures: CC BY 4.0.
