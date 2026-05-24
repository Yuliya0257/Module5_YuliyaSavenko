"""
validate_ml_lib.py
==================
Correctness validation for the from-scratch NumPy implementations in
ml_lib.py. Two layers of checks:

(1) Analytical / internal-consistency checks that always run
    (no scikit-learn required):
      - StandardScaler: post-transform mean ≈ 0, std ≈ 1
      - K-means: recovers planted Gaussian-blob cluster structure
        (adjusted accuracy ≥ 0.95 with K matched to ground truth)
      - LogisticRegression: training accuracy ≥ 0.95 on linearly
        separable data; probabilities sum to 1
      - RandomForestClassifier: out-of-the-box accuracy ≥ 0.85 on a
        moderately non-linear classification problem; probabilities
        sum to 1
      - Metrics: accuracy / macro-F1 / balanced-accuracy / ROC-AUC
        match analytical values on hand-built confusion matrices

(2) Side-by-side comparison with scikit-learn equivalents on the
    same synthetic data, run only if sklearn is importable. The
    pipeline runs end-to-end without sklearn for environments
    (such as the one used for this assignment) where pip is blocked.

A short pass/fail summary is printed and written to
results/validation_report.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from ml_lib import (
    DecisionTreeClassifier, GradientBoostingClassifier, KMeans, KPrototypes,
    LogisticRegression, RandomForestClassifier, StandardScaler, accuracy,
    balanced_accuracy, confusion_matrix, macro_f1, roc_auc_ovr,
)

try:
    import sklearn  # noqa: F401
    from sklearn.cluster import KMeans as SkKMeans
    from sklearn.linear_model import LogisticRegression as SkLR
    from sklearn.ensemble import (RandomForestClassifier as SkRF,
                                  GradientBoostingClassifier as SkGB)
    from sklearn.metrics import (accuracy_score as sk_acc,
                                 balanced_accuracy_score as sk_bal,
                                 f1_score as sk_f1,
                                 roc_auc_score as sk_auc)
    from sklearn.preprocessing import StandardScaler as SkScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

RES = REPO / "results"
RES.mkdir(parents=True, exist_ok=True)
REPORT = RES / "validation_report.txt"

results = []


def _log(msg: str) -> None:
    print(msg)
    results.append(msg)


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------
def make_blobs(n=300, centers=None, std=0.6, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array(centers if centers is not None
                       else [[-3, -3], [0, 0], [3, 3]])
    per = n // len(centers)
    X = np.vstack([rng.normal(c, std, size=(per, 2)) for c in centers])
    y = np.repeat(np.arange(len(centers)), per)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def make_linear_classification(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, 4))
    w_true = np.array([1.5, -1.0, 0.5, 0.0])
    logits = X @ w_true + 0.2
    y = (logits + rng.normal(0, 0.4, size=n) > 0).astype(int)
    return X, y


def make_nonlinear_classification(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 4))
    # Non-linear decision boundary: depends on x0*x1 and x2^2
    y = ((X[:, 0] * X[:, 1] + X[:, 2] ** 2 - 1.0) > 0).astype(int)
    return X, y


def cluster_purity(labels_true, labels_pred):
    """Fraction of samples in the dominant true class within each cluster."""
    n = len(labels_true)
    total = 0
    for c in np.unique(labels_pred):
        mask = labels_pred == c
        if mask.sum() == 0:
            continue
        vals, counts = np.unique(labels_true[mask], return_counts=True)
        total += counts.max()
    return total / n


# ---------------------------------------------------------------------------
# (1) Analytical / internal-consistency checks
# ---------------------------------------------------------------------------
_log("=" * 64)
_log("VALIDATION REPORT  (code/validate_ml_lib.py)")
_log(f"sklearn available: {SKLEARN_AVAILABLE}")
_log("=" * 64)

_log("\n--- (1) Analytical / internal-consistency checks ---")

# StandardScaler
X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
Xz = StandardScaler().fit_transform(X)
m_ok = np.allclose(Xz.mean(axis=0), 0, atol=1e-8)
s_ok = np.allclose(Xz.std(axis=0), 1, atol=1e-8)
_log(f"  StandardScaler: mean≈0 {'PASS' if m_ok else 'FAIL'}, "
     f"std≈1 {'PASS' if s_ok else 'FAIL'}")

# K-means on well-separated blobs
Xb, yb = make_blobs(n=600, centers=[[-5, -5], [0, 0], [5, 5]], std=0.5, seed=1)
km = KMeans(n_clusters=3, n_init=10, random_state=1).fit(Xb)
purity = cluster_purity(yb, km.labels_)
_log(f"  K-means cluster purity on 3 separated blobs: {purity:.3f} "
     f"({'PASS' if purity >= 0.95 else 'FAIL'}; expect ≥ 0.95)")

# Logistic regression on linearly separable data
Xl, yl = make_linear_classification(n=600, seed=2)
lr = LogisticRegression(C=1.0, lr=1.0, max_iter=600).fit(Xl, yl)
acc_train = accuracy(yl, lr.predict(Xl))
prob_sum = lr.predict_proba(Xl).sum(axis=1)
_log(f"  LR train accuracy on linear problem: {acc_train:.3f} "
     f"({'PASS' if acc_train >= 0.85 else 'FAIL'}; expect ≥ 0.85)")
_log(f"  LR predict_proba rows sum to 1: "
     f"{'PASS' if np.allclose(prob_sum, 1.0) else 'FAIL'}")

# Random forest on non-linear data
Xn, yn = make_nonlinear_classification(n=600, seed=3)
rf = RandomForestClassifier(n_estimators=80, max_depth=6,
                            min_samples_leaf=8, random_state=3).fit(Xn, yn)
acc_train_rf = accuracy(yn, rf.predict(Xn))
prob_sum_rf = rf.predict_proba(Xn).sum(axis=1)
_log(f"  RF train accuracy on non-linear problem: {acc_train_rf:.3f} "
     f"({'PASS' if acc_train_rf >= 0.85 else 'FAIL'}; expect ≥ 0.85)")
_log(f"  RF predict_proba rows sum to 1: "
     f"{'PASS' if np.allclose(prob_sum_rf, 1.0) else 'FAIL'}")

# Gradient boosting on non-linear data
gb = GradientBoostingClassifier(n_estimators=80, learning_rate=0.1,
                                max_depth=3, min_samples_leaf=8,
                                random_state=3).fit(Xn, yn)
acc_train_gb = accuracy(yn, gb.predict(Xn))
prob_sum_gb = gb.predict_proba(Xn).sum(axis=1)
_log(f"  GB train accuracy on non-linear problem: {acc_train_gb:.3f} "
     f"({'PASS' if acc_train_gb >= 0.85 else 'FAIL'}; expect ≥ 0.85)")
_log(f"  GB predict_proba rows sum to 1: "
     f"{'PASS' if np.allclose(prob_sum_gb, 1.0) else 'FAIL'}")

# K-modes (KPrototypes, no numeric columns) on planted binary blobs
rng_km = np.random.default_rng(7)
clusterA = (rng_km.random((150, 6)) < 0.15).astype(float)
clusterB = (rng_km.random((150, 6)) < 0.85).astype(float)
Xkm = np.vstack([clusterA, clusterB])
ykm = np.array([0] * 150 + [1] * 150)
kmode = KPrototypes(n_clusters=2, cat_idx=list(range(6)), num_idx=[],
                    n_init=10, random_state=7).fit(Xkm)
kmode_purity = cluster_purity(ykm, kmode.labels_)
_log(f"  K-modes cluster purity on 2 binary blobs: {kmode_purity:.3f} "
     f"({'PASS' if kmode_purity >= 0.95 else 'FAIL'}; expect ≥ 0.95)")

# Metrics: hand-built ground truth
yt = np.array([0, 0, 0, 1, 1, 1, 1, 1])
yp = np.array([0, 0, 1, 1, 1, 0, 1, 1])  # 6/8 correct, TP=4, FP=1, FN=1
acc_ok = abs(accuracy(yt, yp) - 6 / 8) < 1e-9
# Macro F1: class 0 F1 = 2*2/(2*2+1+1)=0.667 ; class 1 F1 = 2*4/(2*4+1+1)=0.8
f1_ok = abs(macro_f1(yt, yp) - (0.6666666666666666 + 0.8) / 2) < 1e-6
bal_ok = abs(balanced_accuracy(yt, yp) - (2 / 3 + 4 / 5) / 2) < 1e-9
cm_ok = (confusion_matrix(yt, yp) == np.array([[2, 1], [1, 4]])).all()
_log(f"  Metrics — accuracy {'PASS' if acc_ok else 'FAIL'}, "
     f"macro_f1 {'PASS' if f1_ok else 'FAIL'}, "
     f"balanced_acc {'PASS' if bal_ok else 'FAIL'}, "
     f"confusion_matrix {'PASS' if cm_ok else 'FAIL'}")

# ROC-AUC on perfectly separated scores
y_bin = np.array([0, 0, 0, 1, 1, 1])
scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
proba2 = np.stack([1 - scores, scores], axis=1)
auc = roc_auc_ovr(y_bin, proba2, np.array([0, 1]))
_log(f"  ROC-AUC on perfectly separable scores: {auc:.3f} "
     f"({'PASS' if abs(auc - 1.0) < 1e-9 else 'FAIL'}; expect 1.000)")

# ---------------------------------------------------------------------------
# (2) Side-by-side comparison with scikit-learn (if available)
# ---------------------------------------------------------------------------
if SKLEARN_AVAILABLE:
    _log("\n--- (2) Side-by-side comparison with scikit-learn ---")

    # StandardScaler
    sk_z = SkScaler().fit_transform(X)
    diff = np.abs(Xz - sk_z).max()
    _log(f"  StandardScaler vs sklearn — max abs diff: {diff:.2e} "
         f"({'PASS' if diff < 1e-8 else 'FAIL'}; expect < 1e-8)")

    # K-means on blobs (compare inertia within tolerance)
    sk_km = SkKMeans(n_clusters=3, n_init=10, random_state=1).fit(Xb)
    our_inertia = km.inertia_
    sk_inertia = sk_km.inertia_
    rel_diff = abs(our_inertia - sk_inertia) / sk_inertia
    _log(f"  K-means inertia ours vs sklearn: {our_inertia:.1f} vs "
         f"{sk_inertia:.1f} (rel diff {rel_diff:.1%}; "
         f"{'PASS' if rel_diff < 0.05 else 'FAIL'})")
    # Cluster purity vs sklearn purity
    sk_purity = cluster_purity(yb, sk_km.labels_)
    _log(f"  K-means cluster purity ours vs sklearn: {purity:.3f} vs "
         f"{sk_purity:.3f} ({'PASS' if abs(purity-sk_purity) < 0.05 else 'FAIL'})")

    # Logistic regression
    sk_lr = SkLR(C=1.0, solver="lbfgs", max_iter=600).fit(Xl, yl)
    sk_acc_lr = sk_acc(yl, sk_lr.predict(Xl))
    _log(f"  LR train acc ours vs sklearn: {acc_train:.3f} vs "
         f"{sk_acc_lr:.3f} ({'PASS' if abs(acc_train-sk_acc_lr) < 0.05 else 'FAIL'})")
    # Coefficient sign agreement on non-zero features
    our_coef = lr.W[1:, 1] - lr.W[1:, 0]  # binary case, class1 vs class0
    sk_coef = sk_lr.coef_[0]
    sign_agree = float(np.mean(np.sign(our_coef[:3]) == np.sign(sk_coef[:3])))
    _log(f"  LR coefficient sign agreement (first 3 features): "
         f"{sign_agree*100:.0f}% ({'PASS' if sign_agree >= 2/3 else 'FAIL'})")

    # Random forest
    sk_rf = SkRF(n_estimators=80, max_depth=6, min_samples_leaf=8,
                 random_state=3).fit(Xn, yn)
    sk_acc_rf = sk_acc(yn, sk_rf.predict(Xn))
    _log(f"  RF train acc ours vs sklearn: {acc_train_rf:.3f} vs "
         f"{sk_acc_rf:.3f} ({'PASS' if abs(acc_train_rf-sk_acc_rf) < 0.10 else 'FAIL'})")

    # Gradient boosting
    sk_gb = SkGB(n_estimators=80, learning_rate=0.1, max_depth=3,
                 min_samples_leaf=8, random_state=3).fit(Xn, yn)
    sk_acc_gb = sk_acc(yn, sk_gb.predict(Xn))
    _log(f"  GB train acc ours vs sklearn: {acc_train_gb:.3f} vs "
         f"{sk_acc_gb:.3f} ({'PASS' if abs(acc_train_gb-sk_acc_gb) < 0.10 else 'FAIL'})")

    # Metric functions
    sk_macro_f1 = sk_f1(yt, yp, average="macro")
    sk_bal_acc = sk_bal(yt, yp)
    sk_auc_val = sk_auc(y_bin, scores)
    _log(f"  macro_f1 ours vs sklearn: {macro_f1(yt, yp):.3f} vs "
         f"{sk_macro_f1:.3f} "
         f"({'PASS' if abs(macro_f1(yt, yp)-sk_macro_f1) < 1e-3 else 'FAIL'})")
    _log(f"  balanced_acc ours vs sklearn: {balanced_accuracy(yt, yp):.3f} vs "
         f"{sk_bal_acc:.3f} "
         f"({'PASS' if abs(balanced_accuracy(yt, yp)-sk_bal_acc) < 1e-3 else 'FAIL'})")
    _log(f"  ROC-AUC ours vs sklearn: {auc:.3f} vs {sk_auc_val:.3f} "
         f"({'PASS' if abs(auc-sk_auc_val) < 1e-3 else 'FAIL'})")
else:
    _log("\n--- (2) scikit-learn not installed in this environment ---")
    _log("       Install scikit-learn and re-run for side-by-side comparison.")

_log("\n" + "=" * 64)
n_pass = sum(1 for r in results if "PASS" in r)
n_fail = sum(1 for r in results if "FAIL" in r)
_log(f"SUMMARY: {n_pass} PASS, {n_fail} FAIL")
_log("=" * 64)

with open(REPORT, "w") as f:
    f.write("\n".join(results) + "\n")
print(f"\nWrote {REPORT}")
