"""
tune_hyperparameters.py
=======================
Performs a grid search over LR and RF hyperparameters using an inner
3-fold stratified CV on a fixed 80% tuning subset, with minority-class
oversampling inside each training fold. The remaining 20% is reserved
as a held-out test split that is also re-used downstream by
run_analysis.py — tuning never touches it.

Selection criterion: mean macro F1 across the 3 inner folds.

Outputs:
  results/hyperparameter_search.csv  — full grid + scores
  results/best_hyperparameters.json  — selected configs per model
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent

from ml_lib import (
    GradientBoostingClassifier, LogisticRegression, RandomForestClassifier,
    StandardScaler, StratifiedKFold, accuracy, balanced_accuracy, macro_f1,
    roc_auc_ovr, train_test_split,
)
from run_analysis import build_cohort, build_features, impute, \
    derive_risk_profiles, make_feature_matrix, oversample, RNG


def inner_cv_score(model_fn, X, y, k=3, seed=0):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    scores = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr_s, Xte_s = sc.transform(X[tr]), sc.transform(X[te])
        Xtr_b, ytr_b = oversample(Xtr_s, y[tr], random_state=seed)
        m = model_fn()
        m.fit(Xtr_b, ytr_b)
        scores.append(macro_f1(y[te], m.predict(Xte_s)))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    print("==> Loading and processing data ...")
    df = build_cohort()
    df = build_features(df)
    df = df.dropna(subset=["PHQ9_total"]).copy()
    df = impute(df)
    df, _ = derive_risk_profiles(df, k_range=(2, 3, 4), seed=RNG)

    X, _ = make_feature_matrix(df, include_depression=True)
    y = df["risk_profile"].to_numpy()

    # 80% tuning, 20% holdout (never touched here)
    X_tune, _X_hold, y_tune, _y_hold = train_test_split(
        X, y, test_size=0.20, random_state=RNG, stratify=y
    )
    print(f"    tuning n={len(X_tune)}; holdout n={len(_X_hold)}")

    rows = []

    # ---- LR grid ---------------------------------------------------------
    lr_grid = [(C, lr_rate)
               for C in [0.01, 0.1, 1.0, 10.0]
               for lr_rate in [0.1, 0.3, 0.5, 1.0]]
    print(f"==> LR grid: {len(lr_grid)} configs")
    for C, lr_rate in lr_grid:
        def fac(C=C, lr_rate=lr_rate):
            return LogisticRegression(C=C, lr=lr_rate, max_iter=500,
                                      random_state=0)
        m, s = inner_cv_score(fac, X_tune, y_tune, k=3, seed=0)
        rows.append(dict(model="LR", C=C, lr=lr_rate, n_estimators=None,
                         max_depth=None, min_samples_leaf=None,
                         mean_f1=m, sd_f1=s))

    # ---- RF grid ---------------------------------------------------------
    rf_grid = [(n, d, leaf)
               for n in [40, 60, 100]
               for d in [4, 6, 8]
               for leaf in [20, 30, 50]]
    print(f"==> RF grid: {len(rf_grid)} configs")
    for n, d, leaf in rf_grid:
        def fac(n=n, d=d, leaf=leaf):
            return RandomForestClassifier(
                n_estimators=n, max_depth=d,
                min_samples_leaf=leaf, min_samples_split=2 * leaf,
                max_features="sqrt", random_state=0)
        m, s = inner_cv_score(fac, X_tune, y_tune, k=3, seed=0)
        rows.append(dict(model="RF", C=None, lr=None,
                         n_estimators=n, max_depth=d,
                         min_samples_leaf=leaf, learning_rate=None,
                         mean_f1=m, sd_f1=s))

    # ---- GB grid ---------------------------------------------------------
    gb_grid = [(n, lr_rate, d)
               for n in [40, 60, 100]
               for lr_rate in [0.05, 0.1]
               for d in [2, 3]]
    print(f"==> GB grid: {len(gb_grid)} configs")
    for n, lr_rate, d in gb_grid:
        def fac(n=n, lr_rate=lr_rate, d=d):
            return GradientBoostingClassifier(
                n_estimators=n, learning_rate=lr_rate, max_depth=d,
                min_samples_leaf=30, min_samples_split=60,
                subsample=0.8, random_state=0)
        m, s = inner_cv_score(fac, X_tune, y_tune, k=3, seed=0)
        rows.append(dict(model="GB", C=None, lr=None,
                         n_estimators=n, max_depth=d,
                         min_samples_leaf=30, learning_rate=lr_rate,
                         mean_f1=m, sd_f1=s))

    grid = pd.DataFrame(rows)
    grid.to_csv(REPO / "results" / "hyperparameter_search.csv", index=False)
    print("    saved results/hyperparameter_search.csv")

    # ---- Best per model --------------------------------------------------
    best = {}
    for model in ["LR", "RF", "GB"]:
        sub = grid[grid["model"] == model].copy()
        sub = sub.sort_values("mean_f1", ascending=False)
        top = sub.iloc[0].to_dict()
        best[model] = {k: v for k, v in top.items()
                       if pd.notna(v) and k != "model"}
        print(f"  {model} best: {best[model]}")

    with open(REPO / "results" / "best_hyperparameters.json", "w") as f:
        json.dump(best, f, indent=2)

    for model in ["LR", "RF", "GB"]:
        sub = grid[grid["model"] == model].sort_values(
            "mean_f1", ascending=False).head(3)
        print(f"\n{model} top 3:")
        print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
