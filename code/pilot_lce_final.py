"""
PA-LCE: early-stopped benchmarking of metaheuristics
====================================================

Pipeline supporting the manuscript "Can metaheuristic benchmarks be stopped early?"
(see README.md). Stages can be run separately, per dimension, and resumed:

  generate  Run the metaheuristics on the 24 official BBOB functions
            (coco-experiment), recording best-so-far curves, population state
            (diversity, fitness spread) and trajectory ELA features at 5/10/20%
            of the evaluation budget.
  lcpfn     LC-PFN baseline (Adriaensen et al., 2023), zero-shot, official
            pretrained weights (needs torch and github.com/automl/lcpfn).
            Incremental: only curves not yet predicted are processed.
  analyze   All extrapolators, within/pooled scenarios, grouped cross-conformal
            calibration, statistical tests, decision-level preservation (RQ3),
            figures. --cv: fold6 (main), train1, loao, loao_fold6, logo, lofo.
  extras    Step-shaped curves vs error, conditional coverage, paired
            decision-level differences, net savings, example curves.

Extrapolators: LastValue, PowerLaw, Exponential, LC-PFN, ELA-Traj, Meta-Curve,
Curve+ELA, PA-LCE (curve + population), PA-LCE+ELA (all groups), PA-LCE-rel.
Algorithms: PSO-w, PSO-cf, DE-rand1, DE-best1, DE-ctb1, GWO, WOA, HHO
(default) and optionally CMA-ES, SHADE.

Examples:
  python pilot_lce_final.py all --quick            # smoke test of every stage
  python pilot_lce_final.py generate --dims 20     # split work by dimension
  python pilot_lce_final.py lcpfn --lcpfn_repo lcpfn
  python pilot_lce_final.py analyze --tag v3

Inline comments and console messages are in Indonesian.
"""

import argparse
import itertools
import json
import os
import sys
import time
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist, pdist
from scipy.special import gamma as gamma_fn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

warnings.filterwarnings("ignore")

LB, UB = -5.0, 5.0
FLOOR = -8.0          # log10 error minimum (presisi target 1e-8, seperti BBOB/COCO)
ALPHA = 0.10          # interval 90%
FRACTIONS = (0.05, 0.10, 0.20)
PROPOSED = "PA-LCE+ELA"   # metode usulan: fitur kurva + populasi + ELA trajektori
BBOB_GROUPS = {       # kelompok fungsi BBOB (Hansen dkk., 2009)
    1: "separable", 2: "separable", 3: "separable", 4: "separable", 5: "separable",
    6: "low-cond", 7: "low-cond", 8: "low-cond", 9: "low-cond",
    10: "high-cond", 11: "high-cond", 12: "high-cond", 13: "high-cond", 14: "high-cond",
    15: "multi-global", 16: "multi-global", 17: "multi-global", 18: "multi-global",
    19: "multi-global",
    20: "multi-weak", 21: "multi-weak", 22: "multi-weak", 23: "multi-weak", 24: "multi-weak",
}


def stable_seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) % (2 ** 31)


# ---------------------------------------------------------------------------
# 1. Fungsi uji BBOB resmi (coco-experiment)
# ---------------------------------------------------------------------------
def make_problem(fid, dim, instance):
    import cocoex
    fn = cocoex.BareProblem("bbob", int(fid), int(dim), int(instance))
    fopt = float(fn.best_value())

    def f(X):
        # evaluasi sekaligus per populasi; hasilnya identik dengan evaluasi per titik
        return np.maximum(np.asarray(fn(np.atleast_2d(X)), dtype=float).reshape(-1) - fopt, 0.0)

    return f


# ---------------------------------------------------------------------------
# 2. Penghitung anggaran FE + perekaman keadaan populasi + sampel untuk ELA
# ---------------------------------------------------------------------------
class Budget:
    """Membungkus fungsi objektif: menghitung FE, merekam best-so-far, diversitas
    populasi, sebaran fitness pada grid checkpoint FE, dan menyimpan sampel
    (x, f) selama fraksi anggaran terbesar yang diamati (untuk fitur ELA)."""

    def __init__(self, f, dim, max_fe, grid, rec_frac=max(FRACTIONS)):
        self.f = f
        self.max_fe = max_fe
        self.cp = np.round(np.linspace(max_fe / grid, max_fe, grid)).astype(int)
        self.best_tr = np.full(grid, np.nan)
        self.div_tr = np.full(grid, np.nan)
        self.fstd_tr = np.full(grid, np.nan)
        self.norm = np.sqrt(dim) * (UB - LB)
        self.fe, self.best, self.ptr = 0, np.inf, 0
        self.div, self.fstd = np.nan, np.nan
        self.rec_limit = int(np.ceil(rec_frac * max_fe))
        self.rec_X, self.rec_y = [], []

    @property
    def done(self):
        return self.ptr >= len(self.cp)

    @property
    def progress(self):
        return min(self.fe / self.max_fe, 1.0)

    def pop(self, X, fit):
        self.div = np.mean(np.linalg.norm(X - X.mean(0), axis=1)) / self.norm
        self.fstd = np.std(np.log10(fit + 1e-12))

    def samples(self, n_fe):
        X = np.vstack(self.rec_X)[:n_fe]
        y = np.concatenate(self.rec_y)[:n_fe]
        return X, y

    def __call__(self, X):
        X = np.atleast_2d(X)
        vals = self.f(X)
        m = len(vals)
        if self.fe < self.rec_limit:
            self.rec_X.append(X.copy())
            self.rec_y.append(vals.copy())
        if np.isnan(self.div) and m > 1:
            self.pop(X, vals)
        cm = np.minimum.accumulate(np.concatenate([[self.best], vals]))[1:]
        while self.ptr < len(self.cp) and self.cp[self.ptr] <= self.fe + m:
            j = self.cp[self.ptr] - self.fe - 1
            self.best_tr[self.ptr] = cm[j]
            self.div_tr[self.ptr] = self.div
            self.fstd_tr[self.ptr] = self.fstd
            self.ptr += 1
        self.fe += m
        self.best = cm[-1]
        return vals


# ---------------------------------------------------------------------------
# 3. Metaheuristik (parameter waktu memakai progres FE, bukan iterasi)
# ---------------------------------------------------------------------------
def run_pso(bud, dim, n, rng, mode="inertia"):
    X = rng.uniform(LB, UB, (n, dim))
    V = np.zeros_like(X)
    vmax = 0.2 * (UB - LB)
    fit = bud(X)
    bud.pop(X, fit)
    P, pfit = X.copy(), fit.copy()
    g = P[np.argmin(pfit)].copy()
    while not bud.done:
        r1, r2 = rng.random((2, n, dim))
        if mode == "inertia":
            w = 0.9 - 0.5 * bud.progress
            V = w * V + 2.0 * r1 * (P - X) + 2.0 * r2 * (g - X)
        else:  # constriction (Clerc & Kennedy)
            V = 0.7298 * (V + 2.05 * r1 * (P - X) + 2.05 * r2 * (g - X))
        V = np.clip(V, -vmax, vmax)
        X = np.clip(X + V, LB, UB)
        fit = bud(X)
        imp = fit < pfit
        P[imp], pfit[imp] = X[imp], fit[imp]
        g = P[np.argmin(pfit)].copy()
        bud.pop(X, fit)


def run_de(bud, dim, n, rng, strategy="rand1", F=0.5, CR=0.9):
    X = rng.uniform(LB, UB, (n, dim))
    fit = bud(X)
    bud.pop(X, fit)
    ar = np.arange(n)
    while not bud.done:
        R = rng.random((n, n))
        R[ar, ar] = np.inf
        i0, i1, i2 = np.argsort(R, axis=1)[:, :3].T
        b = X[np.argmin(fit)]
        if strategy == "rand1":
            M = X[i0] + F * (X[i1] - X[i2])
        elif strategy == "best1":
            M = b + F * (X[i0] - X[i1])
        else:  # current-to-best/1
            M = X + F * (b - X) + F * (X[i0] - X[i1])
        cross = rng.random((n, dim)) < CR
        cross[ar, rng.integers(0, dim, n)] = True
        U = np.clip(np.where(cross, M, X), LB, UB)
        ufit = bud(U)
        imp = ufit <= fit
        X[imp], fit[imp] = U[imp], ufit[imp]
        bud.pop(X, fit)


def run_gwo(bud, dim, n, rng):
    X = rng.uniform(LB, UB, (n, dim))
    fit = bud(X)
    bud.pop(X, fit)
    while not bud.done:
        leaders = X[np.argsort(fit)[:3]]
        a = 2 - 2 * bud.progress
        Xn = np.zeros_like(X)
        for L in leaders:
            A = 2 * a * rng.random((n, dim)) - a
            C = 2 * rng.random((n, dim))
            Xn += L - A * np.abs(C * L - X)
        X = np.clip(Xn / 3.0, LB, UB)
        fit = bud(X)
        bud.pop(X, fit)


def run_woa(bud, dim, n, rng):
    X = rng.uniform(LB, UB, (n, dim))
    fit = bud(X)
    bud.pop(X, fit)
    best, bfit = X[np.argmin(fit)].copy(), fit.min()
    while not bud.done:
        a = 2 - 2 * bud.progress
        A = 2 * a * rng.random((n, 1)) - a
        C = 2 * rng.random((n, 1))
        p = rng.random((n, 1))
        l = rng.uniform(-1, 1, (n, 1))
        xr = X[rng.integers(n, size=n)]
        encircle = best - A * np.abs(C * best - X)
        search = xr - A * np.abs(C * xr - X)
        spiral = np.abs(best - X) * np.exp(l) * np.cos(2 * np.pi * l) + best
        X = np.clip(np.where(p < 0.5, np.where(np.abs(A) < 1, encircle, search), spiral), LB, UB)
        fit = bud(X)
        if fit.min() < bfit:
            best, bfit = X[np.argmin(fit)].copy(), fit.min()
        bud.pop(X, fit)


def _levy_step(dim, rng, beta=1.5):
    sigma = (gamma_fn(1 + beta) * np.sin(np.pi * beta / 2)
             / (gamma_fn((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    return rng.standard_normal(dim) * sigma / np.abs(rng.standard_normal(dim)) ** (1 / beta)


def run_hho(bud, dim, n, rng):
    X = rng.uniform(LB, UB, (n, dim))
    fit = bud(X)
    bud.pop(X, fit)
    rab, rfit = X[np.argmin(fit)].copy(), fit.min()
    while not bud.done:
        E1 = 2 * (1 - bud.progress)
        Xm = X.mean(0)
        for i in range(n):
            if bud.done:
                break
            E = E1 * (2 * rng.random() - 1)
            if abs(E) >= 1:
                if rng.random() >= 0.5:
                    xr = X[rng.integers(n)]
                    X[i] = xr - rng.random() * np.abs(xr - 2 * rng.random() * X[i])
                else:
                    X[i] = (rab - Xm) - rng.random() * (LB + rng.random() * (UB - LB))
            else:
                r = rng.random()
                J = 2 * (1 - rng.random())
                if r >= 0.5 and abs(E) >= 0.5:
                    X[i] = (rab - X[i]) - E * np.abs(J * rab - X[i])
                elif r >= 0.5:
                    X[i] = rab - E * np.abs(rab - X[i])
                else:
                    base = X[i] if abs(E) >= 0.5 else Xm
                    Y = np.clip(rab - E * np.abs(J * rab - base), LB, UB)
                    if bud(Y)[0] < fit[i]:
                        X[i] = Y
                    elif not bud.done:
                        Z = np.clip(Y + rng.random(dim) * _levy_step(dim, rng), LB, UB)
                        if bud(Z)[0] < fit[i]:
                            X[i] = Z
        if bud.done:
            break
        X = np.clip(X, LB, UB)
        fit = bud(X)
        if fit.min() < rfit:
            rab, rfit = X[np.argmin(fit)].copy(), fit.min()
        bud.pop(X, fit)


def run_cmaes(bud, dim, n, rng):
    """CMA-ES (Hansen & Ostermeier 2001) via pycma, populasi n, sigma0 = 0.2(UB-LB),
    restart dari titik acak bila kriteria berhenti internal terpenuhi."""
    import cma
    while not bud.done:
        es = cma.CMAEvolutionStrategy(rng.uniform(LB, UB, dim), 0.2 * (UB - LB),
                                      {"popsize": n, "bounds": [LB, UB], "verbose": -9,
                                       "seed": int(rng.integers(1, 2 ** 31 - 1))})
        while not bud.done and not es.stop():
            X = np.clip(np.array(es.ask()), LB, UB)
            fit = bud(X)
            es.tell(list(X), list(fit))
            bud.pop(X, fit)


def run_shade(bud, dim, n, rng, H=None):
    """SHADE (Tanabe & Fukunaga 2013): current-to-pbest/1/bin, arsip eksternal,
    memori historis keberhasilan untuk F dan CR."""
    H = H or n
    MF, MCR, kmem = np.full(H, 0.5), np.full(H, 0.5), 0
    X = rng.uniform(LB, UB, (n, dim))
    fit = bud(X)
    bud.pop(X, fit)
    A = np.empty((0, dim))
    ar = np.arange(n)
    while not bud.done:
        r = rng.integers(0, H, n)
        CR = np.clip(rng.normal(MCR[r], 0.1), 0, 1)
        F = np.empty(n)
        for i in range(n):
            f = 0.0
            while f <= 0:
                f = MF[r[i]] + 0.1 * np.tan(np.pi * (rng.random() - 0.5))
            F[i] = min(f, 1.0)
        p = rng.uniform(2 / n, 0.2, n)
        order = np.argsort(fit)
        pbest = X[order[(rng.random(n) * np.maximum(1, np.round(p * n))).astype(int)]]
        r1 = np.array([rng.choice(np.delete(ar, i)) for i in range(n)])
        pool = np.vstack([X, A]) if len(A) else X
        r2 = rng.integers(0, len(pool), n)
        V = X + F[:, None] * (pbest - X) + F[:, None] * (X[r1] - pool[r2])
        V = np.where(V < LB, (LB + X) / 2, np.where(V > UB, (UB + X) / 2, V))
        cross = rng.random((n, dim)) < CR[:, None]
        cross[ar, rng.integers(0, dim, n)] = True
        U = np.where(cross, V, X)
        uf = bud(U)
        better = uf < fit
        if better.any():
            w = fit[better] - uf[better]
            w = w / w.sum() if w.sum() > 0 else np.full(better.sum(), 1 / better.sum())
            sF, sCR = F[better], CR[better]
            MF[kmem] = np.sum(w * sF ** 2) / max(np.sum(w * sF), 1e-12)
            MCR[kmem] = np.sum(w * sCR)
            kmem = (kmem + 1) % H
            A = np.vstack([A, X[better]])
            if len(A) > n:
                A = A[rng.choice(len(A), n, replace=False)]
        imp = uf <= fit
        X[imp], fit[imp] = U[imp], uf[imp]
        bud.pop(X, fit)


ALGORITHMS = {
    "PSO-w": lambda b, d, n, r: run_pso(b, d, n, r, "inertia"),
    "PSO-cf": lambda b, d, n, r: run_pso(b, d, n, r, "constriction"),
    "DE-rand1": lambda b, d, n, r: run_de(b, d, n, r, "rand1"),
    "DE-best1": lambda b, d, n, r: run_de(b, d, n, r, "best1"),
    "DE-ctb1": lambda b, d, n, r: run_de(b, d, n, r, "ctb1"),
    "GWO": run_gwo,
    "WOA": run_woa,
    "HHO": run_hho,
    "CMA-ES": run_cmaes,   # opsional (perlu: pip install cma)
    "SHADE": run_shade,    # opsional
}
DEFAULT_ALGS = ["PSO-w", "PSO-cf", "DE-rand1", "DE-best1", "DE-ctb1", "GWO", "WOA", "HHO"]


# ---------------------------------------------------------------------------
# 4. Fitur ELA dari sampel trajektori (definisi mengikuti flacco/pflacco)
# ---------------------------------------------------------------------------
# Diimplementasikan langsung karena pflacco 1.2.x tidak kompatibel dengan
# scipy >= 1.11 / Python 3.13. Kelompok yang dipakai: ela_distr, ela_meta,
# disp, nbc (Mersmann dkk., 2011; Kerschke & Trautmann, 2019). Kelompok
# ela_level dan ic tidak dipakai karena mahal untuk dihitung per run.
ELA_NAMES = (
    ["distr_skewness", "distr_kurtosis", "distr_n_peaks"]
    + ["meta_lin_adj_r2", "meta_lin_intercept", "meta_lin_coef_min", "meta_lin_coef_max",
       "meta_lin_coef_ratio", "meta_lin_inter_adj_r2", "meta_quad_adj_r2", "meta_quad_cond",
       "meta_quad_inter_adj_r2"]
    + [f"disp_{s}_{q:02d}" for q in (2, 5, 10, 25)
       for s in ("ratio_mean", "ratio_median", "diff_mean", "diff_median")]
    + ["nbc_sd_ratio", "nbc_mean_ratio", "nbc_cor", "nbc_dist_ratio_cv", "nbc_fitness_cor"]
)


def _adj_r2(A, y):
    n, p = A.shape
    if n <= p + 1:
        return np.nan, None
    A1 = np.column_stack([np.ones(n), A])
    coef, *_ = np.linalg.lstsq(A1, y, rcond=None)
    ss_res = np.sum((y - A1 @ coef) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot <= 0:
        return np.nan, coef
    r2 = 1 - ss_res / ss_tot
    return 1 - (1 - r2) * (n - 1) / (n - p - 1), coef


def _safe_cor(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def ela_features(X, y):
    out = []
    n, d = X.shape
    # --- ela_distr ---
    if np.std(y) > 0:
        out += [stats.skew(y), stats.kurtosis(y)]
        z = (y - y.mean()) / y.std()
        try:
            kde = stats.gaussian_kde(z)
            g = np.linspace(z.min() - 0.1, z.max() + 0.1, 512)
            dens = kde(g)
            mins = np.r_[0, np.where((dens[1:-1] < dens[:-2]) & (dens[1:-1] < dens[2:]))[0] + 1, 511]
            mass = [np.trapezoid(dens[a:b + 1], g[a:b + 1]) for a, b in zip(mins[:-1], mins[1:])]
            out.append(float(np.sum(np.array(mass) > 0.01)))
        except Exception:
            out.append(np.nan)
    else:
        out += [np.nan, np.nan, np.nan]
    # --- ela_meta ---
    r_lin, c_lin = _adj_r2(X, y)
    if c_lin is not None:
        ac = np.abs(c_lin[1:])
        out += [r_lin, c_lin[0], ac.min(), ac.max(), ac.max() / max(ac.min(), 1e-300)]
    else:
        out += [np.nan] * 5
    iu = np.triu_indices(d, 1)
    inter = (X[:, :, None] * X[:, None, :])[:, iu[0], iu[1]]
    out.append(_adj_r2(np.hstack([X, inter]), y)[0])
    r_q, c_q = _adj_r2(np.hstack([X, X ** 2]), y)
    if c_q is not None:
        aq = np.abs(c_q[1 + d:])
        out += [r_q, aq.max() / max(aq.min(), 1e-300)]
    else:
        out += [np.nan, np.nan]
    out.append(_adj_r2(np.hstack([X, X ** 2, inter]), y)[0])
    # --- disp ---
    d_all = pdist(X)
    mean_all, med_all = d_all.mean(), np.median(d_all)
    for q in (0.02, 0.05, 0.10, 0.25):
        idx = np.argsort(y)[:max(2, int(np.ceil(q * n)))]
        db = pdist(X[idx])
        out += [db.mean() / mean_all, np.median(db) / med_all,
                db.mean() - mean_all, np.median(db) - med_all]
    # --- nbc ---
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    nn = D.min(1)
    better = y[None, :] < y[:, None]
    Db = np.where(better, D, np.inf)
    nb = Db.min(1)
    has = np.isfinite(nb)
    if has.sum() >= 3:
        nb_idx = np.argmin(Db, axis=1)
        indeg = np.bincount(nb_idx[has], minlength=n).astype(float)
        ratio = nn[has] / np.maximum(nb[has], 1e-300)
        out += [np.std(nn[has]) / max(np.std(nb[has]), 1e-300),
                np.mean(nn[has]) / max(np.mean(nb[has]), 1e-300),
                _safe_cor(nn[has], nb[has]),
                np.std(ratio) / max(np.mean(ratio), 1e-300),
                _safe_cor(indeg, y)]
    else:
        out += [np.nan] * 5
    return np.array(out, dtype=float)


# ---------------------------------------------------------------------------
# 5. Tahap 1: pembangkitan kurva + fitur ELA (paralel, bisa dilanjutkan)
# ---------------------------------------------------------------------------
def part_path(out, dim, alg, fid):
    return os.path.join(out, "parts", f"d{dim}", f"{alg}__f{fid:02d}.npz")


def run_combo(dim, alg, fid, c):
    # 1 thread BLAS/OpenMP per proses: tanpa ini tiap proses bisa membuka puluhan
    # thread (sesuai jumlah core mesin host) dan saling berebut CPU
    with threadpool_limits(1):
        return _run_combo(dim, alg, fid, c)


def _run_combo(dim, alg, fid, c):
    B, D, S, E, II, RR = [], [], [], [], [], []
    for inst in range(1, c["instances"] + 1):
        f = make_problem(fid, dim, inst)
        for r in range(c["runs"]):
            rng = np.random.default_rng(stable_seed(c["seed"], alg, fid, dim, inst, r))
            bud = Budget(f, dim, c["max_fe"], c["grid"])
            ALGORITHMS[alg](bud, dim, c["pop"], rng)
            B.append(np.maximum(np.log10(bud.best_tr + 1e-12), FLOOR))
            D.append(bud.div_tr)
            S.append(bud.fstd_tr)
            ela = []
            for frac in FRACTIONS:
                Xs, ys = bud.samples(int(round(frac * c["max_fe"])))
                cap = c["ela_mult"] * dim
                if len(Xs) > cap:
                    sel = np.random.default_rng(stable_seed("ela", alg, fid, dim, inst, r, frac)) \
                        .choice(len(Xs), cap, replace=False)
                    Xs, ys = Xs[sel], ys[sel]
                try:
                    ela.append(ela_features(Xs, ys))
                except Exception:
                    ela.append(np.full(len(ELA_NAMES), np.nan))
            E.append(np.array(ela))
            II.append(inst)
            RR.append(r)
    path = part_path(c["out"], dim, alg, fid)
    np.savez_compressed(path + ".tmp.npz", best=np.array(B), div=np.array(D), fstd=np.array(S),
                        ela=np.array(E), inst=np.array(II), run=np.array(RR))
    os.replace(path + ".tmp.npz", path)
    return dim, alg, fid


def signature(cfg, dim):
    return dict(suite="bbob", dim=dim, pop=cfg.pop, max_fe=cfg.fe_mult * dim, grid=cfg.grid,
                runs=cfg.runs, instances=cfg.instances, seed=cfg.seed, ela_mult=cfg.ela_mult,
                fractions=list(FRACTIONS))


def check_signature(cfg, dim, write):
    d = os.path.join(cfg.out, "parts", f"d{dim}")
    os.makedirs(d, exist_ok=True)
    sig, p = signature(cfg, dim), os.path.join(d, "signature.json")
    if os.path.exists(p):
        with open(p) as fh:
            if json.load(fh) != sig:
                raise SystemExit(f"[data] {d} berisi data dengan konfigurasi lain. Pakai --out "
                                 "berbeda, samakan argumen, atau hapus folder tersebut.")
    elif write:
        with open(p, "w") as fh:
            json.dump(sig, fh)


def stage_generate(cfg):
    import cocoex  # noqa: F401  (gagal cepat bila paket belum terpasang)
    todo, total = [], 0
    for dim in cfg.dims:
        check_signature(cfg, dim, write=True)
        for alg, fid in itertools.product(cfg.algs, cfg.funcs):
            total += 1
            if not os.path.exists(part_path(cfg.out, dim, alg, fid)):
                todo.append((dim, alg, fid))
    todo.sort(key=lambda t: (-t[0], t[1] != "HHO"))  # yang paling berat lebih dulu
    print(f"[generate] {total} kombinasi, {total - len(todo)} sudah ada, {len(todo)} dijalankan "
          f"dengan {cfg.jobs} proses", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=cfg.jobs) as ex:
        futs = {}
        for dim, alg, fid in todo:
            c = dict(signature(cfg, dim), out=cfg.out)
            futs[ex.submit(run_combo, dim, alg, fid, c)] = (dim, alg, fid)
        for k, fut in enumerate(as_completed(futs), 1):
            dim, alg, fid = fut.result()
            el = time.time() - t0
            print(f"[generate] {k}/{len(todo)} d{dim} {alg:9s} f{fid:02d}  ({el / 60:.1f} mnt, "
                  f"sisa ~{el / k * (len(todo) - k) / 60:.1f} mnt)", flush=True)


def load_dim(cfg, dim, required=True):
    meta, B, D, S, E = [], [], [], [], []
    missing = []
    for alg, fid in itertools.product(cfg.algs, cfg.funcs):
        p = part_path(cfg.out, dim, alg, fid)
        if not os.path.exists(p):
            missing.append(f"{alg}/f{fid}")
            continue
        z = np.load(p)
        for i, r in zip(z["inst"], z["run"]):
            meta.append(dict(dim=dim, alg=alg, func=int(fid), group=BBOB_GROUPS[int(fid)],
                             fold6=(int(fid) - 1) % 6, inst=int(i), run=int(r)))
        B.append(z["best"])
        D.append(z["div"])
        S.append(z["fstd"])
        E.append(z["ela"])
    if missing and required:
        raise SystemExit(f"[data] d{dim}: {len(missing)} kombinasi belum dibangkitkan "
                         f"(mis. {missing[:3]}). Jalankan tahap generate dulu.")
    return pd.DataFrame(meta), np.vstack(B), np.vstack(D), np.vstack(S), np.concatenate(E)


# ---------------------------------------------------------------------------
# 6. Tahap 2: pembanding LC-PFN (zero-shot, bobot pretrained resmi)
# ---------------------------------------------------------------------------
LCPFN_POINTS = 100  # LC-PFN dilatih pada kurva sepanjang 100 titik, nilai naik dalam [0, 1]


def to_progress(B):
    """Kurva log-error turun -> progres naik dalam [0, 1] terhadap error awal dan batas 1e-8."""
    y0 = B[:, :1]
    room = np.maximum(y0 - FLOOR, 1e-9)
    return np.clip(np.maximum.accumulate((y0 - B) / room, axis=1), 0.0, 1.0)


def from_progress(B, p):
    y0 = B[:, 0]
    return y0 - p * np.maximum(y0 - FLOOR, 0.0)


def load_lcpfn_model(cfg):
    import torch
    sys.path.insert(0, os.path.abspath(cfg.lcpfn_repo))
    import lcpfn  # noqa: E402  (repo https://github.com/automl/lcpfn)
    path = getattr(lcpfn, cfg.lcpfn_model)  # mengunduh bobot bila belum ada
    model = torch.load(path, map_location="cpu", weights_only=False)
    model.eval()
    # _pa_lce_gelu_fix: model disimpan dengan torch 1.11; modul bawaan torch versi baru
    # mengharapkan atribut yang belum ada saat itu. Isi dengan nilai bawaan lamanya
    # (perilaku perhitungannya sama dengan saat model dilatih).
    nn = torch.nn
    for m in model.modules():
        if isinstance(m, nn.GELU) and not hasattr(m, "approximate"):
            m.approximate = "none"
        if isinstance(m, nn.TransformerEncoder):
            for a, v in (("enable_nested_tensor", False), ("use_nested_tensor", False),
                         ("mask_check", True)):
                if not hasattr(m, a):
                    setattr(m, a, v)
        if isinstance(m, nn.MultiheadAttention) and not hasattr(m, "batch_first"):
            m.batch_first = False
        if isinstance(m, nn.LayerNorm) and not hasattr(m, "bias"):
            m.bias = None
    # Kompatibilitas torch >= 2: nn.TransformerEncoder kini mengirim argumen is_causal
    # ke setiap lapisan, sedangkan lapisan LC-PFN (ditulis untuk torch 1.11) tidak
    # menerimanya. Lapisan dibungkus agar argumen itu diabaikan; perhitungannya sama.
    for m in model.modules():
        cls = type(m)
        if (cls.__name__ == "TransformerEncoderLayer" and not cls.__module__.startswith("torch.")
                and not getattr(cls.forward, "_pa_lce_patched", False)):
            orig = cls.forward

            def fwd(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False,
                    _orig=orig, **kw):
                return _orig(self, src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

            fwd._pa_lce_patched = True
            cls.forward = fwd
    torch.set_num_threads(max(1, cfg.jobs))
    return model


def lcpfn_quantiles(model, P, k, qs=(ALPHA / 2, 0.5, 1 - ALPHA / 2), chunk=1024):
    """P: (n, 100) progres. Mengamati k titik pertama, memprediksi titik ke-100."""
    import torch
    n = len(P)
    out = np.empty((n, len(qs)))
    xs = torch.arange(1, LCPFN_POINTS + 1, dtype=torch.float32)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        m = b - a
        x = torch.cat([xs[:k], xs[-1:]]).view(-1, 1, 1).expand(k + 1, m, 1).contiguous()
        y = torch.from_numpy(P[a:b, :k].T.astype(np.float32)).unsqueeze(-1)
        with torch.no_grad():
            logits = model((x, y), single_eval_pos=k)  # (1, m, buckets)
            for j, q in enumerate(qs):
                out[a:b, j] = model.criterion.icdf(logits, q).reshape(-1).numpy()
    return np.clip(out, 0.0, 1.0)


def stage_lcpfn(cfg):
    model = load_lcpfn_model(cfg)
    # uji kewarasan pada kurva sintetis yang jenuh
    t = np.arange(1, LCPFN_POINTS + 1)
    demo = 0.8 * (1 - t ** -0.6)
    q = lcpfn_quantiles(model, demo[None, :], 20)[0]
    print(f"[lcpfn] uji kewarasan: nilai akhir sebenarnya {demo[-1]:.3f}, prediksi median "
          f"{q[1]:.3f} (interval 90% {q[0]:.3f}-{q[2]:.3f})", flush=True)
    for dim in cfg.dims:
        path = os.path.join(cfg.out, "parts", f"lcpfn_d{dim}.npz")
        check_signature(cfg, dim, write=False)
        meta, B, *_ = load_dim(cfg, dim)
        old = None
        if os.path.exists(path) and not cfg.force:
            z = np.load(path, allow_pickle=True)
            old = dict(preds=z["preds"], alg=z["alg"], func=z["func"], inst=z["inst"], run=z["run"])
            done = set(zip(old["alg"], old["func"], old["inst"], old["run"]))
            need = np.array([(a, f, i, r) not in done for a, f, i, r in
                             zip(meta.alg, meta.func, meta.inst, meta.run)])
            if not need.any():
                print(f"[lcpfn] d{dim} sudah lengkap, dilewati (pakai --force untuk mengulang)")
                continue
            print(f"[lcpfn] d{dim}: {need.sum()} kurva baru (algoritma tambahan) diprediksi", flush=True)
            meta, B = meta[need].reset_index(drop=True), B[need]
        G = B.shape[1]
        idx = np.round(np.linspace(G / LCPFN_POINTS, G, LCPFN_POINTS)).astype(int) - 1
        P = to_progress(B)[:, idx]
        preds = np.empty((len(B), len(FRACTIONS), 3))
        t0 = time.time()
        for j, frac in enumerate(FRACTIONS):
            k = max(int(round(frac * LCPFN_POINTS)), 3)
            qp = lcpfn_quantiles(model, P, k)
            y_k = B[:, max(int(round(frac * G)), 8) - 1]
            ys = np.column_stack([from_progress(B, qp[:, c]) for c in (2, 1, 0)])  # lo, md, hi
            preds[:, j, :] = np.clip(ys, FLOOR, y_k[:, None])
            print(f"[lcpfn] d{dim} {frac:.0%} selesai ({(time.time() - t0) / 60:.1f} mnt)", flush=True)
        new = dict(preds=preds, alg=meta.alg.values, func=meta.func.values,
                   inst=meta.inst.values, run=meta.run.values)
        if old is not None:
            new = {k: np.concatenate([old[k], new[k]]) for k in new}
        np.savez_compressed(path, **new)


# ---------------------------------------------------------------------------
# 7. Fitur dan model prediktor
# ---------------------------------------------------------------------------
def _pow(t, a, b, c):
    return a * np.power(t, -b) + c


def _exp(t, a, b, c):
    return a * np.exp(-b * t) + c


def parametric_fit(y, T, model):
    k = len(y)
    t = np.arange(1, k + 1, dtype=float)
    yl = y[-1]
    try:
        fn, p0 = ((_pow, [max(y[0] - yl, 1e-3), 0.5, yl]) if model == "power"
                  else (_exp, [max(y[0] - yl, 1e-3), 3.0 / k, yl]))
        popt, _ = curve_fit(fn, t, y, p0=p0, bounds=([0, 0, FLOOR - 5], [1e4, 5, 20]),
                            maxfev=5000)
        pred = fn(T, *popt)
    except Exception:
        pred = yl
    return float(np.clip(pred, FLOOR, yl))


def _resample(x, m=10):
    return x[np.linspace(0, len(x) - 1, m).round().astype(int)]


def _tail_slope(x):
    k = len(x)
    tail = x[int(0.8 * k):]
    return np.polyfit(np.arange(len(tail)), tail, 1)[0] * k if len(tail) > 1 else 0.0


def _stagnation(y):
    k = len(y)
    return (k - 1 - int(np.argmax(y <= y[-1] + 1e-12))) / k


CURVE_NAMES = ([f"y_r{i}" for i in range(10)]
               + ["y_last", "y_drop", "y_tail_slope", "stagnation_frac", "at_floor"])
CURVE_REL_NAMES = ([f"prog_r{i}" for i in range(10)]
                   + ["prog_last", "prog_tail_slope", "room_left", "stagnation_frac", "at_floor"])
POP_NAMES = ([f"logdiv_r{i}" for i in range(10)]
             + ["logdiv_last", "logdiv_drop", "logdiv_tail_slope", "fstd_last", "fstd_tail_mean"])


def curve_features(y):
    return np.concatenate([_resample(y), [y[-1], y[0] - y[-1], _tail_slope(y), _stagnation(y),
                                         float(y[-1] <= FLOOR + 1e-9)]])


def curve_features_rel(y):
    room0 = max(y[0] - FLOOR, 1e-6)
    p = (y[0] - y) / room0
    return np.concatenate([_resample(p), [p[-1], _tail_slope(p), y[-1] - FLOOR,
                                         _stagnation(y), float(y[-1] <= FLOOR + 1e-9)]])


def population_features(div, fstd):
    k = len(div)
    ld = np.log10(np.nan_to_num(div, nan=1e-12) + 1e-12)
    fs = np.nan_to_num(fstd)
    return np.concatenate([_resample(ld), [ld[-1], ld[0] - ld[-1], _tail_slope(ld),
                                         fs[-1], np.mean(fs[int(0.8 * k):])]])


BLOCK_NAMES = {"Fc": CURVE_NAMES, "Fr": CURVE_REL_NAMES, "Fp": POP_NAMES, "Fe": ELA_NAMES,
               "Fy": ["y_now"], "Fd": ["log2_dim"]}
FEATURE_GROUPS = {
    "Kurva: bentuk (trajektori)": [f"y_r{i}" for i in range(10)],
    "Kurva: level & laju": ["y_last", "y_drop", "y_tail_slope", "at_floor"],
    "Kurva: stagnasi": ["stagnation_frac"],
    "Populasi: trajektori diversitas": [f"logdiv_r{i}" for i in range(10)],
    "Populasi: level & tren diversitas": ["logdiv_last", "logdiv_drop", "logdiv_tail_slope"],
    "Populasi: sebaran fitness": ["fstd_last", "fstd_tail_mean"],
    "ELA: distribusi": [n for n in ELA_NAMES if n.startswith("distr_")],
    "ELA: meta-model": [n for n in ELA_NAMES if n.startswith("meta_")],
    "ELA: dispersi": [n for n in ELA_NAMES if n.startswith("disp_")],
    "ELA: nearest-better": [n for n in ELA_NAMES if n.startswith("nbc_")],
}

# nama -> (blok fitur, mode target)
MODEL_SPECS = {
    "ELA-Traj": (["Fe", "Fy"], "abs"),            # pembanding: regresi ELA berbasis trajektori
    "Meta-Curve": (["Fc"], "abs"),                # ablasi: kurva saja
    "Curve+ELA": (["Fc", "Fe"], "abs"),           # ablasi: kurva + ELA, TANPA fitur populasi
    "PA-LCE": (["Fc", "Fp"], "abs"),              # ablasi: kurva + populasi, tanpa ELA
    PROPOSED: (["Fc", "Fp", "Fe"], "abs"),        # metode usulan: kurva + populasi + ELA
    "PA-LCE-rel": (["Fr", "Fp", "Fd"], "rel"),    # ablasi: representasi relatif
}
INTERVAL_MODELS = ("ELA-Traj", "Meta-Curve", "Curve+ELA", "PA-LCE", PROPOSED)


def _fit_chunk(args):
    Y, G = args
    with threadpool_limits(1):
        return np.array([[parametric_fit(y, G, "power"), parametric_fit(y, G, "exp")] for y in Y])


def parametric_all(B, k, jobs):
    G = B.shape[1]
    chunks = np.array_split(np.arange(len(B)), max(1, jobs * 8))
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        res = list(ex.map(_fit_chunk, [(B[c, :k], G) for c in chunks]))
    return np.vstack(res)


def compute_block(B, D, S, E, k, j, dim, jobs=1):
    n = len(B)
    par = parametric_all(B, k, jobs)
    return dict(
        y_k=B[:, k - 1], y_T=B[:, -1],
        pl=par[:, 0], ex=par[:, 1],
        Fc=np.array([curve_features(B[i, :k]) for i in range(n)]),
        Fr=np.array([curve_features_rel(B[i, :k]) for i in range(n)]),
        Fp=np.array([population_features(D[i, :k], S[i, :k]) for i in range(n)]),
        Fe=E[:, j, :], Fy=B[:, k - 1:k], Fd=np.full((n, 1), np.log2(dim)),
    )


def concat_blocks(blocks):
    return {key: np.concatenate([b[key] for b in blocks]) for key in blocks[0]}


def assemble(block, keys):
    return np.hstack([block[k] for k in keys]), sum((BLOCK_NAMES[k] for k in keys), [])


def to_target(block, mode):
    d = block["y_T"] - block["y_k"]
    if mode == "abs":
        return d
    return np.clip(d / np.maximum(block["y_k"] - FLOOR, 1e-6), -1.0, 0.0)


def from_target(block, mask, pred, mode):
    y_k = block["y_k"][mask]
    if mode == "abs":
        return np.clip(y_k + np.minimum(pred, 0.0), FLOOR, y_k)
    return np.clip(y_k + np.clip(pred, -1.0, 0.0) * np.maximum(y_k - FLOOR, 0.0), FLOOR, y_k)


def _hgb(q, seed):
    """Model median 120 iterasi; model batas interval lebih ringan (60 iterasi)
    karena lebarnya dikoreksi kalibrasi cross-conformal."""
    it, lr = (120, 0.1) if q == 0.5 else (60, 0.15)
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=it,
                                         learning_rate=lr, max_leaf_nodes=15,
                                         min_samples_leaf=20, random_state=seed)


def cv_predict(src, tgt, spec, gs, gt, cfg, importance=False, intervals=True, train_mask=None):
    """Validasi silang berkelompok dari data sumber ke data target: untuk setiap
    kelompok g (lipatan fungsi, satu fungsi, atau satu kelas BBOB; lihat --cv),
    model dilatih tanpa fungsi-fungsi g di dimensi mana pun dan diuji pada g."""
    keys, mode = spec
    Xs, names = assemble(src, keys)
    Xt, _ = assemble(tgt, keys)
    ts = to_target(src, mode)
    y_T = tgt["y_T"]
    n = len(Xt)
    md, lo, hi = np.empty(n), np.empty(n), np.empty(n)
    imp_rows = []
    rng = np.random.default_rng(cfg.seed)
    for g in np.unique(gt):
        tr = train_mask(g) if train_mask is not None else gs != g
        te = gt == g
        m_md = _hgb(0.5, cfg.seed).fit(Xs[tr], ts[tr])
        md[te] = from_target(tgt, te, m_md.predict(Xt[te]), mode)
        if intervals:
            a = from_target(tgt, te, _hgb(ALPHA / 2, cfg.seed).fit(Xs[tr], ts[tr]).predict(Xt[te]), mode)
            b = from_target(tgt, te, _hgb(1 - ALPHA / 2, cfg.seed).fit(Xs[tr], ts[tr]).predict(Xt[te]), mode)
            lo[te], hi[te] = np.minimum(a, b), np.maximum(a, b)
        if importance:
            base_mae = np.mean(np.abs(md[te] - y_T[te]))
            for grp, cols in FEATURE_GROUPS.items():
                idx = [names.index(c) for c in cols if c in names]
                if not idx:
                    continue
                incs = []
                for _ in range(cfg.perm_repeats):
                    Xp = Xt[te].copy()
                    Xp[:, idx] = Xp[rng.permutation(len(Xp))][:, idx]
                    pm = from_target(tgt, te, m_md.predict(Xp), mode)
                    incs.append(np.mean(np.abs(pm - y_T[te])) - base_mae)
                imp_rows.append(dict(cv_group=str(g), group=grp, n=int(te.sum()),
                                     delta_MAE=float(np.mean(incs))))
    out = dict(md=md)
    if intervals:
        out.update(lo=lo, hi=hi)
    return out, imp_rows


def cross_conformal(lo, hi, y_T, y_k, gt):
    """Cross-conformal per kelompok CV: kuantil skor dari kelompok LAIN."""
    E = np.maximum(lo - y_T, y_T - hi)
    lo_c, hi_c = np.empty(len(lo)), np.empty(len(lo))
    for g in np.unique(gt):
        te = gt == g
        Eo = E[~te]
        nc = len(Eo)
        q = np.quantile(Eo, min(1.0, np.ceil((nc + 1) * (1 - ALPHA)) / nc), method="higher")
        lo_c[te] = np.clip(lo[te] - q, FLOOR, y_k[te])
        hi_c[te] = np.clip(hi[te] + q, FLOOR, y_k[te])
    return lo_c, hi_c


# ---------------------------------------------------------------------------
# 8. Statistik
# ---------------------------------------------------------------------------
def wilcoxon_p(a, b):
    """Wilcoxon signed-rank dua sisi (aproksimasi normal + koreksi ties) -> (p, rank-biserial)."""
    d = np.asarray(a) - np.asarray(b)
    d = d[np.abs(d) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0, 0.0
    r = stats.rankdata(np.abs(d))
    w_plus = r[d > 0].sum()
    _, cnt = np.unique(r, return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24 - np.sum(cnt ** 3 - cnt) / 48
    if var <= 0:
        return 1.0, 0.0
    z = (w_plus - n * (n + 1) / 4) / np.sqrt(var)
    return 2 * stats.norm.sf(abs(z)), (2 * w_plus - n * (n + 1) / 2) / (n * (n + 1) / 2)


def holm(p):
    p = np.asarray(p, dtype=float)
    adj, run = np.empty_like(p), 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def conclusions(tab):
    """tab: (problem x algoritma) rata-rata log10 error akhir ->
    peringkat rata-rata, kesimpulan Wilcoxon-Holm (+1 baris lebih buruk / -1 lebih baik / 0), pemenang."""
    ranks = stats.rankdata(tab, axis=1).mean(0)
    pairs = list(itertools.combinations(range(tab.shape[1]), 2))
    pv, sg = [], []
    for i, j in pairs:
        pv.append(wilcoxon_p(tab[:, i], tab[:, j])[0])
        sg.append(np.sign(np.median(tab[:, i] - tab[:, j])))
    return ranks, np.where(holm(pv) < 0.05, np.array(sg), 0), np.argmin(tab, axis=1)


def compare(true_c, pred_c):
    rt, ct, wt = true_c
    rp, cp, wp = pred_c
    return dict(kendall_tau=stats.kendalltau(rt, rp).correlation,
                same_best=float(np.argmin(rt) == np.argmin(rp)),
                winner_agree=float(np.mean(wt == wp)),
                wilcoxon_agree=float(np.mean(ct == cp)),
                n_reversed=int(np.sum((ct != 0) & (cp != 0) & (ct != cp))),
                n_false_sig=int(np.sum((ct == 0) & (cp != 0))),
                n_lost_sig=int(np.sum((ct != 0) & (cp == 0))))


# ---------------------------------------------------------------------------
# 9. Tahap 3: analisis
# ---------------------------------------------------------------------------
def build_scenarios(dims, kinds):
    sc = []
    if "within" in kinds:
        sc += [("within", [d], [d]) for d in dims]
    if len(dims) > 1 and "pooled" in kinds:
        sc += [("pooled", list(dims), list(dims))]  # satu model gabungan untuk semua dimensi
    if len(dims) > 1 and "lodo" in kinds:
        sc += [("lodo", [x for x in dims if x != d], [d]) for d in dims]
    return sc


def load_lcpfn_preds(cfg, dim, meta):
    p = os.path.join(cfg.out, "parts", f"lcpfn_d{dim}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    key = pd.DataFrame(dict(alg=z["alg"], func=z["func"], inst=z["inst"], run=z["run"]))
    key["row"] = np.arange(len(key))
    m = meta[["alg", "func", "inst", "run"]].merge(key, how="left", on=["alg", "func", "inst", "run"])
    if m.row.isna().any():
        print(f"[analyze] peringatan: prediksi LC-PFN d{dim} tidak lengkap, LC-PFN dilewati")
        return None
    return z["preds"][m.row.values.astype(int)]


def stage_analyze(cfg):
    with threadpool_limits(cfg.jobs):
        return _stage_analyze(cfg)


def _stage_analyze(cfg):
    t0 = time.time()
    out = os.path.join(cfg.out, f"analysis_{cfg.cv}" + (f"_{cfg.tag}" if cfg.tag else ""))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(vars(cfg), fh, indent=2)
    data, lcp = {}, {}
    for d in cfg.dims:
        check_signature(cfg, d, write=False)
        data[d] = load_dim(cfg, d)
        lcp[d] = load_lcpfn_preds(cfg, d, data[d][0])
    has_lcpfn = all(v is not None for v in lcp.values())
    print(f"[analyze] {sum(len(v[0]) for v in data.values())} kurva; LC-PFN "
          f"{'tersedia' if has_lcpfn else 'TIDAK tersedia (lewati)'}; CV = {cfg.cv}", flush=True)
    for d in cfg.dims:   # label gabungan untuk skema berbasis algoritma
        m = data[d][0]
        m["alg_fold"] = m["alg"] + "|" + m["fold6"].astype(str)
    gcol = {"fold6": "fold6", "lofo": "func", "logo": "group", "loao": "alg",
            "loao_fold6": "alg_fold", "train1": "fold6"}[cfg.cv]
    scenarios = build_scenarios(cfg.dims, cfg.scenarios)

    acc_rows, test_rows, imp_rows, preds_all = [], [], [], []
    G = data[cfg.dims[0]][1].shape[1]
    for j, frac in enumerate(FRACTIONS):
        k = max(int(round(frac * G)), 8)
        blocks = {d: compute_block(*data[d][1:], k, j, d, cfg.jobs) for d in cfg.dims}
        print(f"[analyze] fitur {frac:.0%} siap ({(time.time() - t0) / 60:.1f} mnt)", flush=True)
        units = []
        for kind, src_dims, tgt_dims in scenarios:
            src = concat_blocks([blocks[d] for d in src_dims])
            gs = np.concatenate([data[d][0][gcol].values for d in src_dims])
            s_alg = np.concatenate([data[d][0]["alg"].values for d in src_dims])
            s_fold = np.concatenate([data[d][0]["fold6"].values for d in src_dims])
            if cfg.cv == "loao_fold6":
                # algoritma BARU pada fungsi BARU: latih tanpa algoritma uji dan tanpa lipatan fungsi uji
                def tmask(g, s_alg=s_alg, s_fold=s_fold):
                    a, f = g.split("|")
                    return (s_alg != a) & (s_fold != int(f))
            elif cfg.cv == "train1":
                # anggaran latih kecil: lipatan h diprediksi model yang dilatih HANYA pada lipatan (h+1) mod 6
                def tmask(g, s_fold=s_fold):
                    return s_fold == (int(g) + 1) % 6
            else:
                tmask = None
            bt_all = concat_blocks([blocks[d] for d in tgt_dims])
            gt_all = np.concatenate([data[d][0][gcol].values for d in tgt_dims])
            dim_all = np.concatenate([np.full(len(data[d][0]), d) for d in tgt_dims])
            outs = {}
            for name, spec in MODEL_SPECS.items():
                o, rows = cv_predict(src, bt_all, spec, gs, gt_all, cfg,
                                     importance=(name == PROPOSED and kind == "within"),
                                     intervals=name in INTERVAL_MODELS, train_mask=tmask)
                outs[name] = o
                for r in rows:
                    imp_rows.append(dict(r, fraction=frac, kind=kind, dim=tgt_dims[0]))
            for d in tgt_dims:
                m = dim_all == d
                bt = blocks[d]
                gd = data[d][0][gcol].values
                P = {"LastValue": bt["y_k"], "PowerLaw": bt["pl"], "Exponential": bt["ex"]}
                I = {}
                if has_lcpfn:
                    lp = lcp[d][:, j, :]
                    P["LC-PFN"] = lp[:, 1]
                    I["LC-PFN"] = (lp[:, 0], lp[:, 2])
                for name, o in outs.items():
                    P[name] = o["md"][m]
                    if "lo" in o:
                        I[name] = (o["lo"][m], o["hi"][m])
                I = {n: (lo, hi) + cross_conformal(lo, hi, bt["y_T"], bt["y_k"], gd)
                     for n, (lo, hi) in I.items()}
                units.append((kind, d, P, I))
            print(f"[analyze] {frac:.0%} {kind} -> {tgt_dims} selesai "
                  f"({(time.time() - t0) / 60:.1f} mnt)", flush=True)

        for kind, d, P, I in units:
            meta_t = data[d][0]
            y_T = blocks[d]["y_T"]
            prob = (meta_t["func"].astype(str) + "#" + meta_t["inst"].astype(str)).values
            err = {n: np.abs(p - y_T) for n, p in P.items()}
            for name, p in P.items():
                sp = [stats.spearmanr(p[prob == q], y_T[prob == q]).correlation
                      for q in np.unique(prob)
                      if np.std(y_T[prob == q]) > 0 and np.std(p[prob == q]) > 0]
                row = dict(kind=kind, dim=d, fraction=frac, predictor=name,
                           MAE=err[name].mean(), MedAE=np.median(err[name]),
                           Spearman_within_problem=np.nanmean(sp))
                if name in I:
                    lo, hi, lo_c, hi_c = I[name]
                    row.update(coverage_raw=np.mean((y_T >= lo - 1e-9) & (y_T <= hi + 1e-9)),
                               width_raw=np.mean(hi - lo),
                               coverage_cv=np.mean((y_T >= lo_c - 1e-9) & (y_T <= hi_c + 1e-9)),
                               width_cv=np.mean(hi_c - lo_c))
                acc_rows.append(row)
            names = list(P)
            mae_prob = np.array([[err[n][prob == q].mean() for n in names] for q in np.unique(prob)])
            _, fr_p = stats.friedmanchisquare(*mae_prob.T)
            for n, r in zip(names, stats.rankdata(mae_prob, axis=1).mean(0)):
                test_rows.append(dict(kind=kind, dim=d, fraction=frac, test="Friedman avg rank",
                                      predictor=n, value=r, p_value=fr_p))
            others = [n for n in names if n != PROPOSED]
            res = [wilcoxon_p(err[PROPOSED], err[n]) for n in others]
            for n, (p, rbc), pa in zip(others, res, holm([r[0] for r in res])):
                test_rows.append(dict(kind=kind, dim=d, fraction=frac,
                                      test=f"Wilcoxon |err| {PROPOSED} vs", predictor=n,
                                      value=rbc, p_value=pa))
            tmp = meta_t.copy()
            tmp["kind"], tmp["fraction"], tmp["y_true"] = kind, frac, y_T
            for n, p in P.items():
                tmp[f"pred_{n}"] = p
            for n in (PROPOSED, "LC-PFN"):
                if n in I:
                    tmp[f"lo_{n}"], tmp[f"hi_{n}"] = I[n][2], I[n][3]
            preds_all.append(tmp)

    acc = pd.DataFrame(acc_rows)
    tests = pd.DataFrame(test_rows)
    preds = pd.concat(preds_all, ignore_index=True)
    acc.to_csv(os.path.join(out, "accuracy.csv"), index=False)
    tests.to_csv(os.path.join(out, "stat_tests.csv"), index=False)
    preds.to_csv(os.path.join(out, "predictions.csv.gz"), index=False)
    imp = pd.DataFrame(imp_rows)
    if len(imp):
        imp = (imp.assign(w=imp.delta_MAE * imp.n)
                  .groupby(["kind", "dim", "fraction", "group"])
                  .apply(lambda g: g.w.sum() / g.n.sum()).rename("delta_MAE").reset_index())
        imp.to_csv(os.path.join(out, "feature_group_importance.csv"), index=False)

    # RQ3: pelestarian kesimpulan benchmarking + CI bootstrap atas problem
    print("[analyze] RQ3 bootstrap...", flush=True)
    algs = cfg.algs
    bench_rows, true_rank, concl_store = [], {}, {}
    rng = np.random.default_rng(cfg.seed)
    for (kind, d, frac), sub in preds.groupby(["kind", "dim", "fraction"], sort=False):
        tabs = {c: sub.groupby(["func", "inst", "alg"])[c].mean().unstack("alg")[algs].values
                for c in ["y_true"] + [c for c in sub.columns if c.startswith("pred_")]}
        tt = tabs.pop("y_true")
        tc = conclusions(tt)
        true_rank[d] = " < ".join(np.array(algs)[np.argsort(tc[0])])
        concl_store[(kind, d, frac, "actual")] = tc[1]
        # bootstrap berklaster: fungsi diresampel beserta seluruh instance-nya
        fidx = sub.groupby(["func", "inst", "alg"])["y_true"].mean().unstack("alg").index.get_level_values(0).values
        ufun = np.unique(fidx)
        by_f = {u: np.where(fidx == u)[0] for u in ufun}
        boots = [np.concatenate([by_f[u] for u in rng.choice(ufun, len(ufun))]) for _ in range(cfg.n_boot)]
        tcb = [conclusions(tt[ix]) for ix in boots]
        for c, tp in tabs.items():
            pc = conclusions(tp)
            concl_store[(kind, d, frac, c.replace("pred_", ""))] = pc[1]
            point = compare(tc, pc)
            bs = pd.DataFrame([compare(t, conclusions(tp[ix])) for t, ix in zip(tcb, boots)])
            row = dict(kind=kind, dim=d, fraction=frac, predictor=c.replace("pred_", ""))
            for key, v in point.items():
                row[key] = v
                row[key + "_ci_lo"] = bs[key].quantile(0.025)
                row[key + "_ci_hi"] = bs[key].quantile(0.975)
            row["pred_ranking"] = " < ".join(np.array(algs)[np.argsort(pc[0])])
            bench_rows.append(row)
    bench = pd.DataFrame(bench_rows)
    bench.to_csv(os.path.join(out, "benchmark_preservation.csv"), index=False)

    make_figures(cfg, out, acc, imp, bench, concl_store)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for d, r in sorted(true_rank.items()):
        print(f"Peringkat aktual d{d}: {r}")
    print("\n=== RQ1: MAE log10 error akhir (rata-rata dimensi) ===")
    print(acc.pivot_table(index="predictor", columns=["kind", "fraction"], values="MAE").round(3).to_string())
    print("\n=== RQ1: Spearman dalam problem ===")
    print(acc.pivot_table(index="predictor", columns=["kind", "fraction"],
                          values="Spearman_within_problem").round(3).to_string())
    print("\n=== RQ2: coverage interval 90% (mentah | cross-conformal) dan lebar ===")
    cv = acc.dropna(subset=["coverage_cv"])
    print(cv.pivot_table(index="predictor", columns=["kind", "fraction"],
                         values=["coverage_raw", "coverage_cv", "width_cv"]).round(3).to_string())
    if len(imp):
        print(f"\n=== RQ2: kepentingan kelompok fitur {PROPOSED} (kenaikan MAE jika diacak) ===")
        print(imp.pivot_table(index="group", columns=["dim", "fraction"], values="delta_MAE")
                 .round(3).to_string())
    print(f"\n=== Wilcoxon |err| {PROPOSED} vs lainnya (rank-biserial, negatif = {PROPOSED} lebih baik) ===")
    w = tests[tests.test.str.startswith("Wilcoxon")]
    print(w.pivot_table(index=["predictor", "kind", "dim"], columns="fraction",
                        values=["value", "p_value"]).round(4).to_string())
    print("\n=== Ablasi kunci: apakah fitur populasi menambah nilai di atas ELA? ===")
    w2 = w[w.predictor == "Curve+ELA"]
    print(f"{PROPOSED} vs Curve+ELA (rank-biserial negatif = fitur populasi membantu)")
    print(w2.pivot_table(index=["kind", "dim"], columns="fraction", values=["value", "p_value"])
            .round(4).to_string())
    print("\n=== RQ3: pelestarian kesimpulan benchmarking (rata-rata dimensi) ===")
    for m in ["kendall_tau", "wilcoxon_agree", "n_reversed", "n_false_sig", "n_lost_sig", "same_best"]:
        print(f"-- {m}")
        print(bench.pivot_table(index="predictor", columns=["kind", "fraction"], values=m)
                   .round(3).to_string())
    print(f"\nTotal waktu analisis: {(time.time() - t0) / 60:.1f} menit. Output: {out}/")


# ---------------------------------------------------------------------------
# 10. Gambar
# ---------------------------------------------------------------------------
PRED_ORDER = ["LastValue", "PowerLaw", "Exponential", "LC-PFN", "ELA-Traj", "Meta-Curve",
              "Curve+ELA", "PA-LCE", PROPOSED, "PA-LCE-rel"]


def make_figures(cfg, out, acc, imp, bench, concl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preds = [p for p in PRED_ORDER if p in set(acc.predictor)]
    colors = {p: plt.cm.tab10(i) for i, p in enumerate(preds)}

    # Akurasi per jenis skenario
    kinds = list(dict.fromkeys(acc.kind))
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.6 * len(kinds), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, kd in zip(axes, kinds):
        sub = acc[acc.kind == kd].groupby(["predictor", "fraction"]).MAE.mean().reset_index()
        for p in preds:
            g = sub[sub.predictor == p]
            ax.plot(g.fraction * 100, g.MAE, marker="o", color=colors[p],
                    lw=2.6 if p == PROPOSED else 1.1, label=p)
        ax.set_title(f"{kd} (rata-rata dimensi)")
        ax.set_xlabel("Anggaran FE diamati (%)")
    axes[0].set_ylabel("MAE log10 error akhir")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_mae.png"), dpi=200)
    plt.close(fig)

    # Kalibrasi
    cv = acc.dropna(subset=["coverage_cv"])
    cv = cv[cv.kind == "within"].groupby(["predictor", "fraction"])[["coverage_raw", "coverage_cv"]].mean()
    if len(cv):
        cv = cv.reset_index()
        ip = [p for p in preds if p in set(cv.predictor)]
        fig, axes = plt.subplots(1, len(FRACTIONS), figsize=(4.2 * len(FRACTIONS), 3.6), sharey=True)
        for ax, f in zip(np.atleast_1d(axes), FRACTIONS):
            s = cv[cv.fraction == f].set_index("predictor").reindex(ip)
            x = np.arange(len(ip))
            ax.bar(x - 0.2, s.coverage_raw, 0.4, label="mentah")
            ax.bar(x + 0.2, s.coverage_cv, 0.4, label="cross-conformal")
            ax.axhline(1 - ALPHA, color="k", ls="--", lw=1)
            ax.set_xticks(x, ip, rotation=45, ha="right", fontsize=7)
            ax.set_title(f"{f:.0%} FE")
            ax.set_ylim(0, 1.05)
        np.atleast_1d(axes)[0].set_ylabel("Coverage interval 90% (within)")
        np.atleast_1d(axes)[-1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_calibration.png"), dpi=200)
        plt.close(fig)

    # Kepentingan fitur
    if len(imp):
        pv = imp.pivot_table(index="group", columns="fraction", values="delta_MAE", aggfunc="mean")
        pv = pv.loc[pv.mean(1).sort_values().index]
        fig, ax = plt.subplots(figsize=(7, 3.8))
        h = 0.8 / len(pv.columns)
        for jj, f in enumerate(pv.columns):
            ax.barh(np.arange(len(pv)) + jj * h, pv[f], h, label=f"{f:.0%} FE")
        ax.set_yticks(np.arange(len(pv)) + 0.4 - h / 2, pv.index, fontsize=8)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("Kenaikan MAE saat kelompok fitur diacak")
        ax.set_title(f"Kepentingan kelompok fitur {PROPOSED} (within, rata-rata dimensi)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig5_feature_importance.png"), dpi=200)
        plt.close(fig)

    # Gambar 6: trade-off anggaran vs pelestarian kesimpulan (within)
    show = [p for p in ["LastValue", "PowerLaw", "LC-PFN", "ELA-Traj", "Curve+ELA", PROPOSED]
            if p in set(bench.predictor)]
    b = bench[bench.kind == "within"]
    dims = sorted(b.dim.unique())
    fig, axes = plt.subplots(2, len(dims), figsize=(4.2 * len(dims), 6.4), sharex=True, sharey="row",
                             squeeze=False)
    for c, d in enumerate(dims):
        for r, (m, lab) in enumerate([("kendall_tau", "Kendall τ peringkat"),
                                      ("wilcoxon_agree", "Kesimpulan Wilcoxon sama")]):
            ax = axes[r, c]
            for p in show:
                g = b[(b.dim == d) & (b.predictor == p)].sort_values("fraction")
                ax.plot(g.fraction * 100, g[m], marker="o", color=colors[p],
                        lw=2.6 if p == PROPOSED else 1.1, label=p)
                if p == PROPOSED:
                    ax.fill_between(g.fraction * 100, g[m + "_ci_lo"], g[m + "_ci_hi"],
                                    color=colors[p], alpha=0.15)
            if r == 0:
                ax.set_title(f"d{d}")
            if c == 0:
                ax.set_ylabel(lab)
            if r == 1:
                ax.set_xlabel("Anggaran FE diamati (%)")
    axes[0, -1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig6_budget_tradeoff.png"), dpi=200)
    plt.close(fig)

    # Gambar 4: matriks kesimpulan berpasangan, aktual vs prediksi (within, 5% FE)
    comp = [p for p in ["LastValue", "LC-PFN", "ELA-Traj", PROPOSED]
            if ("within", dims[0], FRACTIONS[0], p) in concl]
    algs = cfg.algs
    na = len(algs)
    pairs = list(itertools.combinations(range(na), 2))
    fig, axes = plt.subplots(len(dims), len(comp), figsize=(3.6 * len(comp), 3.4 * len(dims)),
                             squeeze=False)
    cmap = matplotlib.colors.ListedColormap(["#4c9be8", "#f2f2f2", "#e8774c"])
    for r, d in enumerate(dims):
        act = concl.get(("within", d, FRACTIONS[0], "actual"))
        for c, p in enumerate(comp):
            ax = axes[r, c]
            pr = concl.get(("within", d, FRACTIONS[0], p))
            M = np.full((na, na), np.nan)
            for (i, jj), a in zip(pairs, act):
                M[i, jj], M[jj, i] = a, -a
            ax.imshow(M, cmap=cmap, vmin=-1, vmax=1)
            nrev = nfs = nls = 0
            for (i, jj), a, q in zip(pairs, act, pr):
                mark = None
                if a != 0 and q != 0 and a != q:
                    mark, nrev = "X", nrev + 1
                elif a == 0 and q != 0:
                    mark, nfs = "+", nfs + 1
                elif a != 0 and q == 0:
                    mark, nls = "o", nls + 1
                if mark:
                    for (u, v) in [(i, jj), (jj, i)]:
                        ax.text(v, u, mark, ha="center", va="center", fontsize=9, fontweight="bold")
            ax.set_xticks(range(na), algs, rotation=90, fontsize=6)
            ax.set_yticks(range(na), algs, fontsize=6)
            ax.set_title(f"d{d} — {p}\nX berbalik {nrev}, + palsu {nfs}, o hilang {nls}", fontsize=8)
    fig.suptitle("Kesimpulan Wilcoxon-Holm aktual (warna: biru = baris lebih baik) vs prediksi pada 5% FE",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_conclusion_matrix.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 10b. Analisis tambahan untuk revisi (butuh kurva + prediksi hasil analyze)
# ---------------------------------------------------------------------------
def stepness(B, k, jump=1.0):
    """Ukuran 'bentuk tangga' sisa kurva setelah titik henti k:
    n_jumps  = jumlah penurunan >= `jump` dekade antar checkpoint berturutan,
    max_share = porsi perbaikan sisa yang terjadi pada satu lompatan terbesar."""
    tail = B[:, k - 1:]
    dd = -np.diff(tail, axis=1)
    total = B[:, k - 1] - B[:, -1]
    return (dd >= jump).sum(1), np.where(total > 1e-9, dd.max(1) / np.maximum(total, 1e-9), 0.0)


def stage_extras(cfg):
    out = os.path.join(cfg.out, f"analysis_{cfg.cv}" + (f"_{cfg.tag}" if cfg.tag else ""))
    pr = pd.read_csv(os.path.join(out, "predictions.csv.gz"))
    pw = pr[pr.kind == "within"]
    res = []
    # (a) bentuk kurva vs galat ekstrapolasi
    rows = []
    for d in cfg.dims:
        meta, B, D, S, E = load_dim(cfg, d)
        G = B.shape[1]
        for frac in FRACTIONS:
            k = max(int(round(frac * G)), 8)
            nj, ms = stepness(B, k)
            m = meta[["alg", "func", "inst", "run"]].assign(n_jumps=nj, max_share=ms)
            sub = pw[(pw.dim == d) & (pw.fraction == frac)].merge(m, on=["alg", "func", "inst", "run"])
            for c in [c for c in sub.columns if c.startswith("pred_")]:
                err = (sub[c] - sub.y_true).abs()
                rows.append(dict(dim=d, fraction=frac, predictor=c[5:],
                                 rho_err_vs_jumps=stats.spearmanr(err, sub.n_jumps).correlation,
                                 rho_err_vs_maxshare=stats.spearmanr(err, sub.max_share).correlation,
                                 mae_no_jump=err[sub.n_jumps == 0].mean(),
                                 mae_with_jump=err[sub.n_jumps > 0].mean(),
                                 share_runs_with_jump=float((sub.n_jumps > 0).mean())))
        if d == cfg.dims[len(cfg.dims) // 2]:
            fig_curves(cfg, out, meta, B, D, d)
    st = pd.DataFrame(rows)
    st.to_csv(os.path.join(out, "stepness_vs_error.csv"), index=False)
    # (b) coverage per kelompok BBOB dan per algoritma
    cov = []
    for n in (PROPOSED, "LC-PFN"):
        if f"lo_{n}" not in pw:
            continue
        inside = (pw.y_true >= pw[f"lo_{n}"] - 1e-9) & (pw.y_true <= pw[f"hi_{n}"] + 1e-9)
        for key in ("group", "alg"):
            g = pw.assign(inside=inside).groupby(["fraction", key]).inside.mean().reset_index()
            g["predictor"], g["by"] = n, key
            cov.append(g.rename(columns={key: "level"}))
    if cov:
        pd.concat(cov).to_csv(os.path.join(out, "coverage_by_group.csv"), index=False)
    # (c) selisih tingkat keputusan berpasangan (bootstrap berklaster per fungsi)
    algs = cfg.algs
    rng = np.random.default_rng(cfg.seed)
    others = [p for p in ["ELA-Traj", "Curve+ELA", "PA-LCE", "LC-PFN", "LastValue"] if f"pred_{p}" in pw]
    for (d, frac), sub in pw.groupby(["dim", "fraction"]):
        g = sub.groupby(["func", "inst", "alg"])
        tabs = {c: g[c].mean().unstack("alg")[algs].values
                for c in ["y_true", f"pred_{PROPOSED}"] + [f"pred_{p}" for p in others]}
        fidx = g["y_true"].mean().unstack("alg").index.get_level_values(0).values
        ufun = np.unique(fidx)
        by_f = {u: np.where(fidx == u)[0] for u in ufun}
        boots = [np.concatenate([by_f[u] for u in rng.choice(ufun, len(ufun))]) for _ in range(cfg.n_boot)]
        tt = tabs["y_true"]
        bs = {c: [] for c in tabs if c != "y_true"}
        for ix in boots:
            tc = conclusions(tt[ix])
            for c in bs:
                bs[c].append(compare(tc, conclusions(tabs[c][ix])))
        bp = pd.DataFrame(bs[f"pred_{PROPOSED}"])
        for p in others:
            bo = pd.DataFrame(bs[f"pred_{p}"])
            for mtr in ("kendall_tau", "wilcoxon_agree", "n_reversed"):
                diff = bp[mtr] - bo[mtr]
                res.append(dict(dim=d, fraction=frac, vs=p, metric=mtr, mean_diff=diff.mean(),
                                ci_lo=diff.quantile(0.025), ci_hi=diff.quantile(0.975)))
    pd.DataFrame(res).to_csv(os.path.join(out, "paired_decision_diff.csv"), index=False)
    # (d) penghematan FE bersih per skenario pemakaian
    n_f = len(cfg.funcs)
    sav = []
    for frac in FRACTIONS:
        sav += [dict(fraction=frac, scenario="predictor trained on another benchmark", net_saving=1 - frac),
                dict(fraction=frac, scenario="train on 4 functions run in full (train1)",
                     net_saving=(n_f - 4) / n_f * (1 - frac)),
                dict(fraction=frac, scenario="train on 20 functions run in full (fold6 as protocol)",
                     net_saving=4 / n_f * (1 - frac))]
    pd.DataFrame(sav).to_csv(os.path.join(out, "net_saving.csv"), index=False)
    pd.set_option("display.width", 250)
    print(st[st.predictor.isin(["LC-PFN", "PowerLaw", "LastValue", PROPOSED])].round(3).to_string(index=False))
    if cov:
        print(pd.concat(cov).pivot_table(index=["predictor", "by", "level"], columns="fraction",
                                         values="inside").round(3).to_string())
    print(pd.DataFrame(res).round(3).to_string(index=False))
    print(pd.DataFrame(sav).round(3).to_string(index=False))


def fig_curves(cfg, out, meta, B, D, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show_f = [f for f in (1, 15, 21) if f in set(meta.func)]
    algs = [a for a in ("DE-rand1", "PSO-cf", "GWO", "HHO") if a in set(meta.alg)]
    G = B.shape[1]
    x = np.linspace(100 / G, 100, G)
    fig, axs = plt.subplots(2, len(show_f), figsize=(2.6 * len(show_f), 4.4), sharex=True, squeeze=False)
    for c, f in enumerate(show_f):
        for a in algs:
            ix = meta[(meta.func == f) & (meta.alg == a) & (meta.inst == 1)].index[:3]
            for j, i in enumerate(ix):
                kw = dict(lw=0.9, alpha=0.85, label=a if j == 0 else None)
                l0 = axs[0, c].plot(x, B[i], **kw)[0]
                axs[1, c].plot(x, np.log10(np.nan_to_num(D[i], nan=1e-12) + 1e-12), color=l0.get_color(), **kw)
        for r in range(2):
            axs[r, c].axvline(5, color="k", ls="--", lw=0.7)
        axs[0, c].set_title(f"f{f}, d = {d}")
        axs[1, c].set_xlabel("Budget used (%)")
    axs[0, 0].set_ylabel("log$_{10}$ best-so-far error")
    axs[1, 0].set_ylabel("log$_{10}$ diversity")
    axs[0, -1].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_example_curves.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 11. CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["generate", "lcpfn", "analyze", "extras", "all"])
    ap.add_argument("--quick", action="store_true", help="konfigurasi kecil untuk uji cepat")
    ap.add_argument("--dims", nargs="+", type=int, default=[5, 10, 20])
    ap.add_argument("--algs", nargs="+", default=list(DEFAULT_ALGS),
                    help="tambahkan CMA-ES SHADE untuk portofolio diperluas")
    ap.add_argument("--funcs", nargs="+", type=int, default=list(range(1, 25)))
    ap.add_argument("--instances", type=int, default=5)
    ap.add_argument("--runs", type=int, default=15)
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--fe_mult", type=int, default=2000, help="anggaran = fe_mult * dimensi")
    ap.add_argument("--grid", type=int, default=500)
    ap.add_argument("--ela_mult", type=int, default=50, help="maks. sampel ELA = ela_mult * dimensi")
    ap.add_argument("--scenarios", nargs="+", default=["within", "pooled"],
                    choices=["within", "pooled", "lodo"])
    ap.add_argument("--cv", default="fold6",
                    choices=["fold6", "lofo", "logo", "loao", "loao_fold6", "train1"],
                    help="fold6 = 6 lipatan berkelompok per fungsi (tiap lipatan 4 fungsi dari "
                         "kelas BBOB berbeda); lofo = tinggalkan satu fungsi; "
                         "logo = tinggalkan satu kelas fungsi BBOB; loao = tinggalkan satu "
                         "algoritma; loao_fold6 = algoritma DAN lipatan fungsi baru; "
                         "train1 = latih hanya pada 4 fungsi (1 lipatan)")
    ap.add_argument("--perm_repeats", type=int, default=3)
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--lcpfn_repo", default="lcpfn", help="folder hasil git clone automl/lcpfn")
    ap.add_argument("--lcpfn_model", default="EMSIZE512_NLAYERS12_NBUCKETS1000",
                    choices=["EMSIZE512_NLAYERS12_NBUCKETS1000", "EMSIZE512_NLAYERS6_NBUCKETS1000"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tag", default="v2", help="akhiran nama folder analisis (hasil lama tidak tertimpa)")
    ap.add_argument("--jobs", type=int, default=len(os.sched_getaffinity(0)) or 1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="results_final")
    cfg = ap.parse_args()
    if cfg.quick:
        cfg.instances, cfg.runs, cfg.fe_mult, cfg.grid, cfg.n_boot = 2, 3, 400, 200, 100
        cfg.perm_repeats = 1
        if cfg.dims == [5, 10, 20]:
            cfg.dims = [5, 10]
        if cfg.out == "results_final":
            cfg.out = "results_final_quick"
    os.makedirs(cfg.out, exist_ok=True)
    if cfg.stage in ("generate", "all"):
        stage_generate(cfg)
    if cfg.stage in ("lcpfn", "all"):
        try:
            stage_lcpfn(cfg)
        except Exception as e:
            if cfg.stage == "lcpfn":
                raise
            print(f"[lcpfn] dilewati: {type(e).__name__}: {e}")
    if cfg.stage in ("analyze", "all"):
        stage_analyze(cfg)
    if cfg.stage in ("extras", "all"):
        stage_extras(cfg)


if __name__ == "__main__":
    main()
