"""
sensitivity_clustering.py
=========================
Clustering-method sensitivity analysis.

The primary analysis derives cardiovascular risk profiles with K-means on
standardised indicators. Because the clustering inputs are dominated by
binary indicators, K-means' spherical-Gaussian assumption is not strictly
appropriate. This script re-derives the profiles with K-modes — the
clustering method designed for purely categorical (binary) data, which
uses matching dissimilarity and modal centroids — applied to the 13
binary risk indicators, and checks that:

  1. K-modes also favours a two-cluster solution;
  2. its cluster assignments agree with the K-means profiles
     (raw agreement and adjusted Rand index);
  3. the depression -> risk-profile association is preserved
     (elevated-risk prevalence in PHQ-9 >= 10 vs < 10).

K-modes is run via the KPrototypes class with no numeric columns, which
reduces it exactly to K-modes.

Output: results/sensitivity_clustering.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent

from ml_lib import KPrototypes
from run_analysis import (
    build_cohort, build_features, impute, derive_risk_profiles,
    RISK_BINARY, RNG,
)


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index between two labellings."""
    a, b = np.asarray(a), np.asarray(b)
    ua, ub = np.unique(a), np.unique(b)
    cont = np.zeros((len(ua), len(ub)), dtype=float)
    ia = {v: i for i, v in enumerate(ua)}
    ib = {v: i for i, v in enumerate(ub)}
    for x, y in zip(a, b):
        cont[ia[x], ib[y]] += 1

    def c2(x):
        return x * (x - 1) / 2

    sum_ij = c2(cont).sum()
    sum_a = c2(cont.sum(axis=1)).sum()
    sum_b = c2(cont.sum(axis=0)).sum()
    total = c2(len(a))
    expected = sum_a * sum_b / total
    maxv = 0.5 * (sum_a + sum_b)
    return 1.0 if maxv == expected else float((sum_ij - expected) / (maxv - expected))


def best_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Raw agreement under the better of the two binary label alignments."""
    a, b = np.asarray(a), np.asarray(b)
    return float(max((a == b).mean(), (a == (1 - b)).mean()))


def main():
    print("==> Building cohort and primary (K-means) profiles ...")
    feats = build_features(build_cohort()).dropna(subset=["PHQ9_total"]).copy()
    feats = impute(feats)
    feats, info = derive_risk_profiles(feats, k_range=(2, 3, 4), seed=RNG)
    km_labels = feats["risk_profile"].to_numpy()

    # K-modes operates on the 13 binary risk indicators (no continuous term)
    Xb = feats[RISK_BINARY].to_numpy(dtype=float)
    cat_idx = list(range(Xb.shape[1]))

    print("==> Re-deriving profiles with K-modes ...")
    costs = {}
    for k in (2, 3, 4):
        km_mode = KPrototypes(n_clusters=k, cat_idx=cat_idx, num_idx=[],
                              n_init=10, random_state=RNG).fit(Xb)
        costs[k] = km_mode.cost_
        print(f"    K={k}: K-modes cost = {km_mode.cost_:.1f}")

    kmode = KPrototypes(n_clusters=2, cat_idx=cat_idx, num_idx=[],
                        n_init=20, random_state=RNG).fit(Xb)
    kmode_labels = kmode.labels_
    # Relabel so 0 = lower mean indicator prevalence, matching K-means
    if Xb[kmode_labels == 0].mean() > Xb[kmode_labels == 1].mean():
        kmode_labels = 1 - kmode_labels

    agreement = best_agreement(km_labels, kmode_labels)
    ari = adjusted_rand_index(km_labels, kmode_labels)
    print(f"\n    K-means vs K-modes raw agreement: {agreement:.3f}")
    print(f"    Adjusted Rand index:              {ari:.3f}")

    phq_pos = feats["PHQ9_pos"].to_numpy().astype(bool)

    def elev_split(labels):
        hi = labels == 1
        return (round(100 * hi[phq_pos].mean(), 1),
                round(100 * hi[~phq_pos].mean(), 1))

    km_hi_dep, km_hi_nodep = elev_split(km_labels)
    kmode_hi_dep, kmode_hi_nodep = elev_split(kmode_labels)
    print(f"\n    K-means  elevated-risk: PHQ-9>=10 {km_hi_dep}% vs <10 {km_hi_nodep}%")
    print(f"    K-modes  elevated-risk: PHQ-9>=10 {kmode_hi_dep}% vs <10 {kmode_hi_nodep}%")

    out = pd.DataFrame([
        dict(metric="K-modes cost K=2", value=round(costs[2], 1)),
        dict(metric="K-modes cost K=3", value=round(costs[3], 1)),
        dict(metric="K-modes cost K=4", value=round(costs[4], 1)),
        dict(metric="K-means elevated-profile size %",
             value=round(100 * (km_labels == 1).mean(), 1)),
        dict(metric="K-modes elevated-profile size %",
             value=round(100 * (kmode_labels == 1).mean(), 1)),
        dict(metric="Raw agreement (K-means vs K-modes)", value=round(agreement, 3)),
        dict(metric="Adjusted Rand index", value=round(ari, 3)),
        dict(metric="K-means elevated-risk %, PHQ-9>=10", value=km_hi_dep),
        dict(metric="K-means elevated-risk %, PHQ-9<10", value=km_hi_nodep),
        dict(metric="K-modes elevated-risk %, PHQ-9>=10", value=kmode_hi_dep),
        dict(metric="K-modes elevated-risk %, PHQ-9<10", value=kmode_hi_nodep),
    ])
    out.to_csv(REPO / "results" / "sensitivity_clustering.csv", index=False)
    print(f"\nWrote {REPO / 'results' / 'sensitivity_clustering.csv'}")


if __name__ == "__main__":
    main()
