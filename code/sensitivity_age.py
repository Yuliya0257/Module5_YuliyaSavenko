"""
sensitivity_age.py
==================
Sensitivity analysis: re-run the modelling on broader age windows
(18-39 primary, 18-64, 18+) to test whether the depression signal
generalises beyond young adulthood.

For tractability on the larger cohorts we use a smaller RF (40 trees,
larger leaves) and 2 random seeds; the conclusion is robustness across
age, not maximising point performance.

Output: results/sensitivity_age.csv
"""
from __future__ import annotations

import sys, time
from pathlib import Path
from math import sqrt, erf

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent

from ml_lib import LogisticRegression, RandomForestClassifier
from run_analysis import (
    build_features, impute, derive_risk_profiles,
    make_feature_matrix, cross_validate, RNG, load_module,
)


def build_cohort_age(min_age, max_age):
    demo = load_module("P_DEMO")
    dpq = load_module("P_DPQ")
    bpq = load_module("P_BPQ")
    mcq = load_module("P_MCQ")
    slq = load_module("P_SLQ")
    smq = load_module("P_SMQFAM")
    cdemo = ["SEQN","RIDAGEYR","RIAGENDR","RIDRETH3","DMDEDUC2",
             "DMDMARTZ","INDFMPIR","RIDEXPRG"]
    cbpq  = ["SEQN","BPQ020","BPQ050A","BPQ080","BPQ100D"]
    cmcq  = ["SEQN","MCQ080","MCQ366A","MCQ366B","MCQ366C","MCQ366D",
             "MCQ300A","MCQ300C"]
    cslq  = ["SEQN","SLD012"]; csmq=["SEQN","SMD460"]
    cdpq  = ["SEQN"] + [f"DPQ0{i}0" for i in range(1,10)]
    df = (demo[cdemo].merge(dpq[cdpq], on="SEQN")
                     .merge(bpq[cbpq], on="SEQN", how="left")
                     .merge(mcq[cmcq], on="SEQN", how="left")
                     .merge(slq[cslq], on="SEQN", how="left")
                     .merge(smq[csmq], on="SEQN", how="left"))
    df = df[(df["RIDAGEYR"]>=min_age) & (df["RIDAGEYR"]<=max_age)].copy()
    df = df[df["RIDEXPRG"].fillna(0) != 1].copy()
    df.drop(columns=["RIDEXPRG"], inplace=True)
    return df.reset_index(drop=True)


def _paired_t(a, b):
    d = np.asarray(a) - np.asarray(b)
    t = d.mean() / (d.std(ddof=1) / sqrt(len(d)))
    p = 2 * (1 - 0.5*(1 + erf(abs(t)/sqrt(2))))
    return t, p


def run_window(min_age, max_age, label, seeds=(1, 2)):
    t0 = time.time()
    print(f"\n==> Cohort {label}: ages {min_age}-{max_age}")
    feats = build_cohort_age(min_age, max_age)
    feats = build_features(feats).dropna(subset=["PHQ9_total"]).copy()
    feats = impute(feats)
    feats, info = derive_risk_profiles(feats, k_range=(2, 3, 4), seed=RNG)
    elev_pct = round(100 * (feats["risk_profile"] == feats["risk_profile"].max()).mean(), 1)
    phq_pos = round(100 * feats["PHQ9_pos"].mean(), 1)
    print(f"    n={len(feats)}; K*={info['k_star']}; elev={elev_pct}%; PHQ-9+={phq_pos}%")

    X_full, _ = make_feature_matrix(feats, include_depression=True)
    X_demo, _ = make_feature_matrix(feats, include_depression=False)
    y = feats["risk_profile"].to_numpy()

    def lr_fac():
        return LogisticRegression(C=1.0, lr=1.0, max_iter=400, random_state=0)
    def rf_fac():
        return RandomForestClassifier(n_estimators=40, max_depth=6,
                                      min_samples_leaf=40,
                                      min_samples_split=80,
                                      max_features="sqrt", random_state=0)

    lr_full = cross_validate(lr_fac, X_full, y, seeds=seeds)
    lr_demo = cross_validate(lr_fac, X_demo, y, seeds=seeds)
    rf_full = cross_validate(rf_fac, X_full, y, seeds=seeds)
    rf_demo = cross_validate(rf_fac, X_demo, y, seeds=seeds)
    print(f"    done ({time.time()-t0:.1f}s)")

    lr_t, lr_p = _paired_t(lr_full["macro_f1"], lr_demo["macro_f1"])
    rf_t, rf_p = _paired_t(rf_full["macro_f1"], rf_demo["macro_f1"])

    return dict(
        cohort=label, age_window=f"{min_age}-{max_age}", n=len(feats),
        elev_profile_pct=elev_pct, phq9_pos_pct=phq_pos, K=int(info["k_star"]),
        LR_demo_AUC=round(lr_demo["roc_auc"].mean(), 3),
        LR_full_AUC=round(lr_full["roc_auc"].mean(), 3),
        LR_dAUC=round(lr_full["roc_auc"].mean()-lr_demo["roc_auc"].mean(), 3),
        RF_demo_AUC=round(rf_demo["roc_auc"].mean(), 3),
        RF_full_AUC=round(rf_full["roc_auc"].mean(), 3),
        RF_dAUC=round(rf_full["roc_auc"].mean()-rf_demo["roc_auc"].mean(), 3),
        LR_f1_t=round(lr_t, 2), LR_f1_p=f"{lr_p:.1e}",
        RF_f1_t=round(rf_t, 2), RF_f1_p=f"{rf_p:.1e}",
    )


def main():
    rows = []
    rows.append(run_window(18, 39, "Primary: 18-39"))
    rows.append(run_window(18, 64, "Working-age: 18-64"))
    rows.append(run_window(18, 99, "All adults: 18+"))
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results" / "sensitivity_age.csv", index=False)
    print("\n", df.to_string(index=False))


if __name__ == "__main__":
    main()
