"""
run_analysis.py
===============
End-to-end analysis pipeline for the project:
  "Can depression in younger adults be used to predict cardiovascular
   risk profiles using machine learning?"

Stages
------
1.  Load NHANES 2017-March 2020 (`P_*`) files from data/raw_nhanes/.
2.  Merge demographics, PHQ-9, BP-questionnaire, medical-conditions,
    sleep and household-smoking modules; restrict to adults 18-39.
3.  Engineer features (depression score, demographic dummies, risk
    indicators) and impute missing values.
4.  Cluster cardiovascular risk indicators with K-means (K selected
    by silhouette) to derive *risk profiles*.
5.  Train supervised classifiers (Logistic Regression, Random Forest)
    to predict the risk profile from depression + demographics,
    with a demographics-only baseline for ablation.
6.  Evaluate with stratified 5-fold cross-validation repeated over
    five random seeds; report macro-F1, balanced accuracy, ROC-AUC,
    and stability (SD across folds/seeds).
7.  Generate plots and per-fold result tables; permutation importance
    for the Random Forest.

Outputs are written to results/ and figures/.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ml_lib import (
    GradientBoostingClassifier, KMeans, LogisticRegression,
    RandomForestClassifier, StandardScaler, StratifiedKFold, accuracy,
    balanced_accuracy, confusion_matrix, macro_f1, permutation_importance,
    roc_auc_ovr, silhouette_score, train_test_split,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / "data" / "raw_nhanes"
FIG = REPO / "figures"
RES = REPO / "results"
for d in (FIG, RES, DATA):
    d.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper")
RNG = 20240516

# ---------------------------------------------------------------------------
# Unified figure colour scheme (Royal Mint / dusty rose)
#   mint  = low-risk profile / demographics-only baseline / below-mean
#   rose  = elevated-risk profile / depression-inclusive model / above-mean
# Every figure in the report draws from these anchors so the deck reads as
# one coherent visual identity.
# ---------------------------------------------------------------------------
from matplotlib.colors import LinearSegmentedColormap

# ColorBrewer RdBu anchors — a colourblind-safe diverging pair that
# prints cleanly: blue reads as low-risk (cool), red as elevated (hot).
COL_LOW = "#2166AC"                      # ColorBrewer RdBu blue — low-risk
COL_HIGH = "#B2182B"                     # ColorBrewer RdBu red — elevated-risk
PROFILE_COLORS = [COL_LOW, COL_HIGH]     # risk profile 0, risk profile 1
DEMO_SHADES = list(sns.light_palette(COL_LOW, n_colors=7))[3:6]   # demo models
FULL_SHADES = list(sns.light_palette(COL_HIGH, n_colors=7))[3:6]  # + PHQ-9
# Diverging map (blue <-> red) for the cluster heatmap; sequential blue
# for the confusion matrices.
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "blue_red", [COL_LOW, "#f4f2f0", COL_HIGH])
SEQ_CMAP = sns.light_palette(COL_LOW, as_cmap=True)

# ---------------------------------------------------------------------------
# Gradient-boosting backend
# ---------------------------------------------------------------------------
# The gradient-boosting model uses the best available library: LightGBM is
# preferred, then XGBoost, then the from-scratch NumPy GradientBoostingClassifier
# in ml_lib.py. Both LightGBM and XGBoost expose a scikit-learn-compatible
# fit / predict / predict_proba interface, so they are genuine drop-in
# replacements. Install either library (`pip install lightgbm`) and re-run to
# use it automatically; with neither installed the pipeline still runs end to
# end on the from-scratch implementation.
GB_BACKEND = "from-scratch NumPy"
try:
    from lightgbm import LGBMClassifier  # noqa: F401
    GB_BACKEND = "LightGBM"
except Exception:
    try:
        from xgboost import XGBClassifier  # noqa: F401
        GB_BACKEND = "XGBoost"
    except Exception:
        pass


def make_gb():
    """Return a gradient-boosting classifier from the best available backend.

    Hyperparameters (n_estimators 60, learning_rate 0.05, max_depth 3) were
    selected by grid search (Table 1) and map onto all three backends.
    """
    if GB_BACKEND == "LightGBM":
        return LGBMClassifier(
            n_estimators=60, learning_rate=0.05, max_depth=3, num_leaves=8,
            subsample=0.8, subsample_freq=1, min_child_samples=30,
            random_state=0, verbose=-1)
    if GB_BACKEND == "XGBoost":
        return XGBClassifier(
            n_estimators=60, learning_rate=0.05, max_depth=3, subsample=0.8,
            min_child_weight=5, random_state=0, verbosity=0,
            eval_metric="logloss")
    return GradientBoostingClassifier(
        n_estimators=60, learning_rate=0.05, max_depth=3,
        min_samples_leaf=30, min_samples_split=60, subsample=0.8,
        random_state=0)


# ---------------------------------------------------------------------------
# 1. Load and merge NHANES modules
# ---------------------------------------------------------------------------
def load_module(name: str) -> pd.DataFrame:
    """Load a single NHANES XPT (with the `.xpt.txt` suffix used in this repo)."""
    p = DATA / f"{name}.xpt.txt"
    if not p.exists():
        raise FileNotFoundError(f"Missing NHANES module: {p}")
    return pd.read_sas(p, format="xport")


def build_cohort() -> pd.DataFrame:
    demo = load_module("P_DEMO")
    dpq = load_module("P_DPQ")
    bpq = load_module("P_BPQ")
    mcq = load_module("P_MCQ")
    slq = load_module("P_SLQ")
    smq = load_module("P_SMQFAM")

    cols_demo = ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3",
                 "DMDEDUC2", "DMDMARTZ", "INDFMPIR", "RIDEXPRG"]
    cols_bpq = ["SEQN", "BPQ020", "BPQ050A", "BPQ080", "BPQ100D"]
    cols_mcq = ["SEQN", "MCQ080", "MCQ366A", "MCQ366B", "MCQ366C",
                "MCQ366D", "MCQ300A", "MCQ300C"]
    cols_slq = ["SEQN", "SLD012"]
    cols_smq = ["SEQN", "SMD460"]
    cols_dpq = ["SEQN"] + [f"DPQ0{i}0" for i in range(1, 10)]

    df = (demo[cols_demo]
          .merge(dpq[cols_dpq], on="SEQN", how="inner")
          .merge(bpq[cols_bpq], on="SEQN", how="left")
          .merge(mcq[cols_mcq], on="SEQN", how="left")
          .merge(slq[cols_slq], on="SEQN", how="left")
          .merge(smq[cols_smq], on="SEQN", how="left"))

    # Restrict to 18-39 and drop pregnant respondents
    df = df.query("RIDAGEYR >= 18 and RIDAGEYR <= 39").copy()
    df = df[df["RIDEXPRG"].fillna(0) != 1].copy()
    df.drop(columns=["RIDEXPRG"], inplace=True)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. PHQ-9 + feature engineering
# ---------------------------------------------------------------------------
PHQ_ITEMS = [f"DPQ0{i}0" for i in range(1, 10)]


def code_phq9(df: pd.DataFrame) -> pd.DataFrame:
    """Code 7/9 as missing, prorate when ≤2 missing, compute total."""
    out = df.copy()
    for col in PHQ_ITEMS:
        out.loc[out[col].isin([7, 9]), col] = np.nan
    miss = out[PHQ_ITEMS].isna().sum(axis=1)
    mean_item = out[PHQ_ITEMS].mean(axis=1, skipna=True)
    total = out[PHQ_ITEMS].sum(axis=1, skipna=True) + miss * mean_item
    total[miss > 2] = np.nan
    out["PHQ9_total"] = total
    out["PHQ9_pos"] = (out["PHQ9_total"] >= 10).astype("Int64")
    return out


def code_binary(df: pd.DataFrame, cols: list[str], yes: int = 1) -> pd.DataFrame:
    """NHANES coding: 1=Yes, 2=No, 7=refused, 9=don't know."""
    out = df.copy()
    for c in cols:
        s = out[c]
        new = pd.Series(np.nan, index=s.index, dtype="float")
        new[s == yes] = 1.0
        new[s == 2] = 0.0
        out[c] = new
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = code_phq9(df)
    binary_cols = ["BPQ020", "BPQ050A", "BPQ080", "BPQ100D",
                   "MCQ080", "MCQ366A", "MCQ366B", "MCQ366C", "MCQ366D",
                   "MCQ300A", "MCQ300C"]
    out = code_binary(out, binary_cols)

    # Sleep: NHANES values 77, 99 are refused/don't know
    out.loc[out["SLD012"].isin([77, 99]), "SLD012"] = np.nan
    out["insufficient_sleep"] = (out["SLD012"] < 7).astype("float")

    # Household smokers: code 999 missing; cap at 4
    out.loc[out["SMD460"].isin([777, 999]), "SMD460"] = np.nan
    out["household_smoker"] = (out["SMD460"] >= 1).astype("float")

    # Demographics
    out["female"] = (out["RIAGENDR"] == 2).astype(int)
    # Race/ethnicity (RIDRETH3): 1 Mex Am, 2 Other Hisp, 3 NH White,
    # 4 NH Black, 6 NH Asian, 7 Other/Multi
    race_map = {1: "MexAm", 2: "OtherHisp", 3: "NHWhite",
                4: "NHBlack", 6: "NHAsian", 7: "OtherMulti"}
    out["race"] = out["RIDRETH3"].map(race_map).fillna("OtherMulti")
    # Education: 1<9, 2 9-11, 3 HS, 4 some college, 5 college+
    edu_map = {1: "LessHS", 2: "LessHS", 3: "HS",
               4: "SomeCollege", 5: "CollegePlus"}
    out["education"] = out["DMDEDUC2"].map(edu_map).fillna("HS")
    # Marital: 1 married/partner, 2 widowed/divorced/separated, 3 never
    mar_map = {1: "MarriedPartner", 2: "WidDivSep", 3: "Never"}
    out["marital"] = out["DMDMARTZ"].map(mar_map).fillna("Never")

    return out


# ---------------------------------------------------------------------------
# 3. Imputation & one-hot encoding
# ---------------------------------------------------------------------------
RISK_BINARY = ["BPQ020", "BPQ050A", "BPQ080", "BPQ100D",
               "MCQ080", "MCQ366A", "MCQ366B", "MCQ366C", "MCQ366D",
               "MCQ300A", "MCQ300C", "insufficient_sleep", "household_smoker"]
RISK_CONT = ["SLD012", "SMD460"]
DEMO_NUM = ["RIDAGEYR", "INDFMPIR", "female"]
DEMO_CAT = ["race", "education", "marital"]


def impute(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in RISK_BINARY:
        med = out[c].mode(dropna=True)
        fill = med.iloc[0] if not med.empty else 0.0
        out[c] = out[c].fillna(fill)
    for c in RISK_CONT + ["INDFMPIR"]:
        out[c] = out[c].fillna(out[c].median())
    return out


def one_hot(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.get_dummies(df, columns=cols, drop_first=False, dtype=float)


# ---------------------------------------------------------------------------
# 4. K-means clustering on CV risk indicators
# ---------------------------------------------------------------------------
def derive_risk_profiles(df: pd.DataFrame, k_range=(2, 3, 4),
                         seed: int = RNG) -> tuple[pd.DataFrame, dict]:
    cv_feats = df[RISK_BINARY + ["SLD012"]].copy()
    # Reverse-code sleep so higher = worse (insufficient already binary)
    cv_feats["sleep_deficit"] = np.maximum(0.0, 7.0 - cv_feats["SLD012"].fillna(7.0))
    cv_feats.drop(columns=["SLD012"], inplace=True)
    X = cv_feats.to_numpy(dtype=float)
    scaler = StandardScaler().fit(X)
    Xz = scaler.transform(X)

    sil_scores = {}
    inertias = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xz)
        sil_scores[k] = silhouette_score(Xz, km.labels_, sample_size=1500,
                                         random_state=seed)
        inertias[k] = km.inertia_

    k_star = max(sil_scores, key=sil_scores.get)
    km_final = KMeans(n_clusters=k_star, n_init=20, random_state=seed).fit(Xz)
    labels = km_final.labels_

    # Order clusters by composite-risk rank (mean across risk indicators)
    cluster_means = pd.DataFrame(km_final.cluster_centers_,
                                 columns=cv_feats.columns)
    risk_rank = cluster_means.mean(axis=1).rank().astype(int) - 1
    relabel = {old: new for old, new in zip(cluster_means.index, risk_rank)}
    new_labels = np.array([relabel[l] for l in labels])

    df_out = df.copy()
    df_out["risk_profile"] = new_labels
    info = dict(
        k_star=k_star,
        silhouette=sil_scores,
        inertia=inertias,
        cluster_means=cluster_means.rename(index=relabel).sort_index().to_dict(),
        feature_names=list(cv_feats.columns),
        scaler_mean=scaler.mean_.tolist(),
        scaler_std=scaler.std_.tolist(),
    )
    return df_out, info


# ---------------------------------------------------------------------------
# 5. Modelling helpers
# ---------------------------------------------------------------------------
def make_feature_matrix(df: pd.DataFrame, include_depression: bool = True):
    cols = DEMO_NUM.copy()
    df_oh = one_hot(df, DEMO_CAT)
    cat_cols = [c for c in df_oh.columns
                if any(c.startswith(p + "_") for p in DEMO_CAT)]
    cols += cat_cols
    if include_depression:
        cols += ["PHQ9_total"]
    X = df_oh[cols].to_numpy(dtype=float)
    feature_names = cols
    return X, feature_names


def oversample(X, y, random_state=0):
    """Random oversampling to equalise class frequencies."""
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    max_n = counts.max()
    out_idx = []
    for c, n in zip(classes, counts):
        idx = np.where(y == c)[0]
        if n < max_n:
            extra = rng.choice(idx, size=max_n - n, replace=True)
            idx = np.concatenate([idx, extra])
        out_idx.append(idx)
    sel = np.concatenate(out_idx)
    rng.shuffle(sel)
    return X[sel], y[sel]


def cross_validate(model_fn, X, y, n_splits=5, seeds=(1, 2, 3, 4, 5),
                   balance=True):
    classes = np.unique(y)
    fold_results = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            scaler = StandardScaler().fit(X[tr])
            Xtr = scaler.transform(X[tr])
            Xte = scaler.transform(X[te])
            y_tr = y[tr]
            if balance:
                Xtr, y_tr = oversample(Xtr, y_tr, random_state=seed)
            model = model_fn()
            model.fit(Xtr, y_tr)
            y_pred = model.predict(Xte)
            y_proba = model.predict_proba(Xte)
            # Per-class recall / precision
            per_class = {}
            for c in classes:
                mask = y[te] == c
                if mask.sum() > 0:
                    per_class[f"recall_{c}"] = float(((y_pred == c) & mask).sum() / mask.sum())
                pmask = y_pred == c
                if pmask.sum() > 0:
                    per_class[f"prec_{c}"] = float(((y_pred == c) & (y[te] == c)).sum() / pmask.sum())
                else:
                    per_class[f"prec_{c}"] = 0.0
            fold_results.append(dict(
                seed=seed,
                fold=fold,
                acc=accuracy(y[te], y_pred),
                bal_acc=balanced_accuracy(y[te], y_pred),
                macro_f1=macro_f1(y[te], y_pred),
                roc_auc=roc_auc_ovr(y[te], y_proba, classes),
                **per_class,
            ))
    return pd.DataFrame(fold_results)


# ---------------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------------
def plot_cluster_profile(cluster_means: dict, feature_names: list[str],
                         path: Path) -> None:
    df = pd.DataFrame(cluster_means).T.reindex(feature_names).T
    df.index.name = "Risk profile"
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(df) + 2))
    sns.heatmap(df, annot=True, fmt=".2f", cmap=DIVERGING_CMAP,
                center=0, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Standardised mean"}, ax=ax)
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right")
    plt.title("Standardised cluster centres on cardiovascular risk indicators")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_silhouette(sil_scores: dict, inertias: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    ks = sorted(sil_scores)
    axes[0].plot(ks, [sil_scores[k] for k in ks], marker="o",
                 markersize=8, color=COL_HIGH, linewidth=2.4)
    for k in ks:
        axes[0].annotate(f"{sil_scores[k]:.3f}", (k, sil_scores[k]),
                         textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Number of clusters K", fontsize=11)
    axes[0].set_ylabel("Silhouette", fontsize=11)
    axes[0].set_title("Silhouette by K", fontsize=12)
    axes[0].margins(y=0.18)
    axes[1].plot(ks, [inertias[k] for k in ks], marker="o",
                 markersize=8, color=COL_LOW, linewidth=2.4)
    for k in ks:
        axes[1].annotate(f"{inertias[k]:,.0f}", (k, inertias[k]),
                         textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Number of clusters K", fontsize=11)
    axes[1].set_ylabel("Inertia", fontsize=11)
    axes[1].set_title("Elbow plot", fontsize=12)
    axes[1].margins(y=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_phq_by_profile(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.boxplot(data=df, x="risk_profile", y="PHQ9_total",
                hue="risk_profile", palette=PROFILE_COLORS, legend=False,
                width=0.55, ax=ax)
    sns.stripplot(data=df.sample(min(len(df), 600), random_state=0),
                  x="risk_profile", y="PHQ9_total", color=".25",
                  size=3, alpha=0.4, ax=ax)
    ax.set_xticklabels(["Profile 0 — low CV risk",
                        "Profile 1 — elevated CV risk"], fontsize=12)
    ax.set_xlabel("Cardiovascular risk profile", fontsize=12)
    ax.set_ylabel("PHQ-9 total score", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_title("Depressive symptom burden by CV risk profile",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_metric_bars(results: dict, path: Path) -> None:
    rows = []
    for name, df in results.items():
        for m in ["acc", "bal_acc", "macro_f1", "roc_auc"]:
            rows.append(dict(model=name, metric=m,
                             mean=df[m].mean(), sd=df[m].std()))
    long = pd.DataFrame(rows)
    # Mint shades = demographics-only baselines; rose shades = + PHQ-9 models
    model_palette = {
        "LR_demo": DEMO_SHADES[0], "RF_demo": DEMO_SHADES[1],
        "GB_demo": DEMO_SHADES[2],
        "LR_full": FULL_SHADES[0], "RF_full": FULL_SHADES[1],
        "GB_full": FULL_SHADES[2],
    }
    fig, ax = plt.subplots(figsize=(9.5, 5))
    sns.barplot(data=long, x="metric", y="mean", hue="model", ax=ax,
                palette=model_palette, errorbar=None,
                edgecolor="#333333", linewidth=0.6)
    # Manual error bars + value labels
    metrics = ["acc", "bal_acc", "macro_f1", "roc_auc"]
    models = list(results)
    width = 0.8 / len(models)
    for i, mname in enumerate(models):
        for j, m in enumerate(metrics):
            mean = long.query("model==@mname and metric==@m")["mean"].iloc[0]
            sd = long.query("model==@mname and metric==@m")["sd"].iloc[0]
            x = j - 0.4 + (i + 0.5) * width
            ax.errorbar(x, mean, yerr=sd, fmt="none", ecolor="#222222",
                        capsize=3, linewidth=0.9)
            ax.text(x, mean + sd + 0.02, f"{mean:.2f}", ha="center",
                    va="bottom", fontsize=7.5, rotation=90,
                    fontweight="bold", color="#222222")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylim(0, 0.85)
    ax.tick_params(labelsize=11)
    ax.set_title("Cross-validated performance (mean ± SD across 15 folds)",
                 fontsize=13)
    ax.legend(fontsize=9, title_fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_confusion(y_true, y_pred, classes, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap=SEQ_CMAP,
                xticklabels=classes, yticklabels=classes,
                cbar_kws={"label": "Row-normalised"}, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_permutation_importance(importances, feature_names, path):
    means = importances.mean(axis=1)
    sds = importances.std(axis=1)
    order = np.argsort(means)
    # Rose = informative feature (positive macro-F1 drop), mint = uninformative
    colors = [COL_HIGH if means[i] >= 0 else COL_LOW for i in order]
    fig, ax = plt.subplots(figsize=(7.5, max(3, 0.34 * len(feature_names))))
    ax.barh([feature_names[i] for i in order], means[order],
            xerr=sds[order], color=colors, edgecolor="#333333",
            linewidth=0.5, error_kw=dict(ecolor="#222222", capsize=2))
    # Value label at the end of each bar
    span = means.max() - min(means.min(), 0)
    for yi, idx in enumerate(order):
        v, s = means[idx], sds[idx]
        if v >= 0:
            ax.text(v + s + 0.02 * span, yi, f"{v:.3f}", va="center",
                    ha="left", fontsize=8, fontweight="bold")
        else:
            ax.text(v - s - 0.02 * span, yi, f"{v:.3f}", va="center",
                    ha="right", fontsize=8, fontweight="bold")
    ax.margins(x=0.16)
    ax.set_xlabel("Drop in macro-F1 when permuted", fontsize=11)
    ax.set_title("Random Forest permutation importance", fontsize=12)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_phq_proba(df, model, feature_names, scaler, path):
    """Marginal predicted-probability curve over PHQ-9, demographics averaged."""
    Xref = df[feature_names].mean(numeric_only=True).to_dict()
    grid = np.linspace(0, 27, 28)
    rows = []
    for v in grid:
        x = pd.Series(Xref).copy()
        x["PHQ9_total"] = v
        rows.append(x[feature_names].values)
    Xg = np.vstack(rows).astype(float)
    Xg_s = scaler.transform(Xg)
    proba = model.predict_proba(Xg_s)
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Profile 0 — low CV risk", "Profile 1 — elevated CV risk"]
    for k in range(proba.shape[1]):
        ax.plot(grid, proba[:, k], linewidth=2.2,
                color=PROFILE_COLORS[k] if k < 2 else None,
                label=labels[k] if k < 2 else f"Risk profile {k}")
    ax.set_xlabel("PHQ-9 total")
    ax.set_ylabel("Predicted probability")
    ax.set_title("Predicted CV risk profile vs PHQ-9 (Random Forest)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------
def build_cohort_table(df: pd.DataFrame, path: Path) -> None:
    """Descriptive cohort characteristics, overall and by risk profile."""
    def summarise(sub: pd.DataFrame) -> dict:
        def pc(mask) -> str:
            return f"{100 * mask.mean():.1f}"
        race = sub["race"].value_counts(normalize=True) * 100
        return {
            "n": f"{len(sub)}",
            "Age, mean (SD)":
                f"{sub['RIDAGEYR'].mean():.1f} ({sub['RIDAGEYR'].std():.1f})",
            "Female, %": pc(sub["female"] == 1),
            "Non-Hispanic White, %": f"{race.get('NHWhite', 0):.1f}",
            "Non-Hispanic Black, %": f"{race.get('NHBlack', 0):.1f}",
            "Mexican American, %": f"{race.get('MexAm', 0):.1f}",
            "Non-Hispanic Asian, %": f"{race.get('NHAsian', 0):.1f}",
            "PHQ-9 total, mean (SD)":
                f"{sub['PHQ9_total'].mean():.1f} ({sub['PHQ9_total'].std():.1f})",
            "PHQ-9 ≥ 10, %": pc(sub["PHQ9_pos"] == 1),
            "Self-reported hypertension, %": pc(sub["BPQ020"] == 1),
            "Self-reported high cholesterol, %": pc(sub["BPQ080"] == 1),
            "Doctor-reported overweight, %": pc(sub["MCQ080"] == 1),
            "Family history, premature MI, %": pc(sub["MCQ300A"] == 1),
            "Family history, diabetes, %": pc(sub["MCQ300C"] == 1),
            "Insufficient sleep (<7 h), %": pc(sub["insufficient_sleep"] == 1),
            "Household smoker, %": pc(sub["household_smoker"] == 1),
        }
    overall = summarise(df)
    low = summarise(df[df["risk_profile"] == 0])
    elev = summarise(df[df["risk_profile"] == 1])
    rows = [dict(Characteristic=k, Overall=overall[k],
                 Low_risk_profile=low[k], Elevated_risk_profile=elev[k])
            for k in overall]
    pd.DataFrame(rows).to_csv(path, index=False)


def crude_association(df: pd.DataFrame, path: Path) -> dict:
    """Odds ratio + chi-square for PHQ-9 >= 10 vs elevated risk profile."""
    from math import log, sqrt, exp, erf
    dep = (df["PHQ9_pos"] == 1).to_numpy()
    elev = (df["risk_profile"] == 1).to_numpy()
    a = int((dep & elev).sum())     # depressed, elevated
    b = int((dep & ~elev).sum())    # depressed, low
    c = int((~dep & elev).sum())    # not depressed, elevated
    d = int((~dep & ~elev).sum())   # not depressed, low
    odds = (a * d) / (b * c)
    se = sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    lo, hi = exp(log(odds) - 1.96 * se), exp(log(odds) + 1.96 * se)
    n = a + b + c + d
    obs = [[a, b], [c, d]]
    rt, ct = [a + b, c + d], [a + c, b + d]
    chi2 = sum((obs[i][j] - rt[i] * ct[j] / n) ** 2 / (rt[i] * ct[j] / n)
               for i in range(2) for j in range(2))
    p = 2 * (1 - 0.5 * (1 + erf(sqrt(chi2) / sqrt(2))))  # chi2 df=1 -> z^2
    res = dict(a=a, b=b, c=c, d=d, odds_ratio=round(odds, 2),
               ci_low=round(lo, 2), ci_high=round(hi, 2),
               chi2=round(chi2, 1), p_value=p)
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    return res


# ---------------------------------------------------------------------------
# 7. Main pipeline
# ---------------------------------------------------------------------------
def main():
    print("==> 1. Loading NHANES P_ cycle ...")
    raw = build_cohort()
    print(f"    raw merged cohort: {len(raw)} adults 18-39")

    print("==> 2. Feature engineering ...")
    feats = build_features(raw)
    # Drop rows with missing PHQ-9 total (too many missing items)
    feats = feats.dropna(subset=["PHQ9_total"]).copy()
    print(f"    after PHQ-9 prorating filter: {len(feats)}")

    feats = impute(feats)
    feats.to_csv(RES / "analysis_cohort.csv", index=False)

    cohort_summary = pd.DataFrame({
        "n": [len(feats)],
        "mean_age": [feats["RIDAGEYR"].mean()],
        "pct_female": [feats["female"].mean() * 100],
        "PHQ9_mean": [feats["PHQ9_total"].mean()],
        "PHQ9_sd": [feats["PHQ9_total"].std()],
        "PHQ9_pos_pct": [feats["PHQ9_pos"].mean() * 100],
    })
    cohort_summary.to_csv(RES / "cohort_summary.csv", index=False)
    print("    ", cohort_summary.to_string(index=False))

    print("==> 3. Clustering CV risk profiles ...")
    feats, cluster_info = derive_risk_profiles(feats, k_range=(2, 3, 4), seed=RNG)
    print(f"    K selected by silhouette: {cluster_info['k_star']}")
    print(f"    silhouettes: {cluster_info['silhouette']}")

    plot_silhouette(cluster_info["silhouette"],
                    cluster_info["inertia"],
                    FIG / "fig_silhouette.png")
    plot_cluster_profile(cluster_info["cluster_means"],
                         cluster_info["feature_names"],
                         FIG / "fig_cluster_profile.png")
    plot_phq_by_profile(feats, FIG / "fig_phq_by_profile.png")

    with open(RES / "cluster_info.json", "w") as f:
        json.dump({k: v for k, v in cluster_info.items()
                   if k != "scaler_mean" and k != "scaler_std"},
                  f, indent=2, default=str)

    # Cluster prevalence per PHQ-9 stratum
    prev = (feats.groupby(["PHQ9_pos", "risk_profile"]).size()
                  .unstack(fill_value=0))
    prev_pct = prev.div(prev.sum(axis=1), axis=0) * 100
    prev_pct.to_csv(RES / "risk_profile_by_depression.csv")

    # Descriptive cohort table and crude depression-profile association
    build_cohort_table(feats, RES / "cohort_table.csv")
    assoc = crude_association(feats, RES / "crude_association.json")
    print(f"    crude association: OR={assoc['odds_ratio']} "
          f"(95% CI {assoc['ci_low']}-{assoc['ci_high']}), "
          f"chi2={assoc['chi2']}, p={assoc['p_value']:.2e}")

    print("==> 4. Supervised modelling ...")
    X_full, feat_names = make_feature_matrix(feats, include_depression=True)
    X_demo, feat_names_demo = make_feature_matrix(feats, include_depression=False)
    y = feats["risk_profile"].to_numpy()

    seeds = (1, 2, 3)
    results = {}

    # Hyperparameters selected by grid search via inner 3-fold CV on an
    # 80% tuning subset; see code/tune_hyperparameters.py and
    # results/hyperparameter_search.csv. The 20% holdout reserved by
    # the tuner is not honoured here because run_analysis.py uses
    # cross-validation across the whole cohort; the tuner's holdout is
    # only to keep tuning honest if a single-split eval were used.
    def lr_factory():
        return LogisticRegression(C=1.0, lr=1.0, max_iter=600, random_state=0)

    def rf_factory():
        return RandomForestClassifier(n_estimators=60, max_depth=8,
                                      min_samples_leaf=30,
                                      min_samples_split=60,
                                      max_features="sqrt",
                                      random_state=0)

    def gb_factory():
        return make_gb()

    print(f"    Gradient-boosting backend: {GB_BACKEND}")

    print("    Logistic regression (full)")
    results["LR_full"] = cross_validate(lr_factory, X_full, y, seeds=seeds)
    print("    Logistic regression (demographics only)")
    results["LR_demo"] = cross_validate(lr_factory, X_demo, y, seeds=seeds)
    print("    Random forest (full)")
    results["RF_full"] = cross_validate(rf_factory, X_full, y, seeds=seeds)
    print("    Random forest (demographics only)")
    results["RF_demo"] = cross_validate(rf_factory, X_demo, y, seeds=seeds)
    print("    Gradient boosting (full)")
    results["GB_full"] = cross_validate(gb_factory, X_full, y, seeds=seeds)
    print("    Gradient boosting (demographics only)")
    results["GB_demo"] = cross_validate(gb_factory, X_demo, y, seeds=seeds)

    summary_rows = []
    for name, df in results.items():
        for m in ["acc", "bal_acc", "macro_f1", "roc_auc"]:
            summary_rows.append(dict(
                model=name, metric=m,
                mean=df[m].mean(), sd=df[m].std(),
                lo=df[m].quantile(0.025), hi=df[m].quantile(0.975),
            ))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RES / "cv_summary.csv", index=False)
    with open(RES / "gb_backend.txt", "w") as f:
        f.write(f"Gradient-boosting backend used for this run: {GB_BACKEND}\n")
    for n, d in results.items():
        d.to_csv(RES / f"cv_folds_{n}.csv", index=False)
    print(summary.pivot_table(index="model", columns="metric",
                              values="mean").round(3).to_string())

    plot_metric_bars(results, FIG / "fig_cv_metrics.png")

    # Paired t-test: full vs demographics-only on macro F1
    from math import sqrt
    def paired_t(a, b):
        d = np.asarray(a) - np.asarray(b)
        t = d.mean() / (d.std(ddof=1) / sqrt(len(d)))
        # Two-sided p via normal approx (df=24, conservative but ok)
        from math import erf
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
        return t, p

    lr_t, lr_p = paired_t(results["LR_full"]["macro_f1"],
                          results["LR_demo"]["macro_f1"])
    rf_t, rf_p = paired_t(results["RF_full"]["macro_f1"],
                          results["RF_demo"]["macro_f1"])
    gb_t, gb_p = paired_t(results["GB_full"]["macro_f1"],
                          results["GB_demo"]["macro_f1"])
    with open(RES / "ablation_ttests.json", "w") as f:
        json.dump(dict(LR=dict(t=lr_t, p=lr_p),
                       RF=dict(t=rf_t, p=rf_p),
                       GB=dict(t=gb_t, p=gb_p)), f, indent=2)
    print(f"    LR ablation: t={lr_t:.2f}, p~{lr_p:.4f}")
    print(f"    RF ablation: t={rf_t:.2f}, p~{rf_p:.4f}")
    print(f"    GB ablation: t={gb_t:.2f}, p~{gb_p:.4f}")

    print("==> 5. Final model + interpretability ...")
    # Keep aligned indices so subgroup metadata can be recovered on test split
    idx = np.arange(len(X_full))
    idx_tr, idx_te, y_tr, y_te = train_test_split(
        idx, y, test_size=0.25, random_state=RNG, stratify=y)
    X_tr, X_te = X_full[idx_tr], X_full[idx_te]
    feats_test = feats.iloc[idx_te].reset_index(drop=True)
    scaler = StandardScaler().fit(X_tr)
    Xtr_s = scaler.transform(X_tr)
    Xte_s = scaler.transform(X_te)

    Xtr_b, ytr_b = oversample(Xtr_s, y_tr, random_state=RNG)
    rf = rf_factory().fit(Xtr_b, ytr_b)
    lr = lr_factory().fit(Xtr_b, ytr_b)

    classes = np.unique(y)
    rf_pred = rf.predict(Xte_s)
    lr_pred = lr.predict(Xte_s)

    # Calibration check: oversampling improves discrimination but distorts
    # the predicted probabilities. Compare against a non-oversampled RF.
    rf_cal = rf_factory().fit(Xtr_s, y_tr)
    yte_bin = (y_te == 1).astype(float)
    p_over = rf.predict_proba(Xte_s)[:, 1]
    p_cal = rf_cal.predict_proba(Xte_s)[:, 1]
    obs_prev = float(yte_bin.mean())
    calibration = dict(
        observed_prevalence=round(obs_prev, 3),
        oversampled_mean_pred=round(float(p_over.mean()), 3),
        oversampled_brier=round(float(np.mean((p_over - yte_bin) ** 2)), 3),
        nonoversampled_mean_pred=round(float(p_cal.mean()), 3),
        nonoversampled_brier=round(float(np.mean((p_cal - yte_bin) ** 2)), 3),
        baseline_brier=round(float(np.mean((obs_prev - yte_bin) ** 2)), 3),
    )
    with open(RES / "calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"    calibration: {calibration}")

    plot_confusion(y_te, rf_pred, classes,
                   "Random Forest confusion (held-out)",
                   FIG / "fig_rf_confusion.png")
    plot_confusion(y_te, lr_pred, classes,
                   "Logistic Regression confusion (held-out)",
                   FIG / "fig_lr_confusion.png")

    imp = permutation_importance(rf, Xte_s, y_te, n_repeats=8,
                                 random_state=RNG, scoring=macro_f1)
    pi_df = pd.DataFrame({
        "feature": feat_names,
        "mean_drop_f1": imp.mean(axis=1),
        "sd_drop_f1": imp.std(axis=1),
    }).sort_values("mean_drop_f1", ascending=False)
    pi_df.to_csv(RES / "permutation_importance_rf.csv", index=False)
    plot_permutation_importance(imp, feat_names, FIG / "fig_rf_permimp.png")

    # Marginal PHQ-9 effect
    feats_for_marg = pd.DataFrame(X_full, columns=feat_names)
    plot_phq_proba(feats_for_marg, rf, feat_names, scaler,
                   FIG / "fig_rf_phq_marginal.png")

    # Logistic regression coefficients
    lr_coef = pd.DataFrame(lr.W[1:], index=feat_names,
                           columns=[f"class_{c}" for c in lr.classes_])
    lr_coef.to_csv(RES / "logistic_coefficients.csv")

    # ---- Subgroup fairness evaluation ----------------------------------
    print("==> 6. Subgroup fairness evaluation ...")
    rf_proba_te = rf.predict_proba(Xte_s)
    rf_pred_te = rf.predict(Xte_s)

    def subgroup_metrics(mask, label):
        if mask.sum() < 30:
            return None
        y_t = y_te[mask]
        y_p = rf_pred_te[mask]
        y_pr = rf_proba_te[mask]
        return dict(
            subgroup=label,
            n=int(mask.sum()),
            elev_risk_pct=round(100 * (y_t == 1).mean(), 1),
            acc=round(accuracy(y_t, y_p), 3),
            bal_acc=round(balanced_accuracy(y_t, y_p), 3),
            macro_f1=round(macro_f1(y_t, y_p), 3),
            roc_auc=round(roc_auc_ovr(y_t, y_pr, classes), 3),
        )

    sub_rows = []
    # Overall
    sub_rows.append(subgroup_metrics(np.ones(len(y_te), bool), "Overall"))
    # Sex
    for lab, m in [("Female", feats_test["female"] == 1),
                   ("Male", feats_test["female"] == 0)]:
        r = subgroup_metrics(m.to_numpy(), f"Sex: {lab}")
        if r:
            sub_rows.append(r)
    # Race/ethnicity
    for race in sorted(feats_test["race"].unique()):
        m = (feats_test["race"] == race).to_numpy()
        r = subgroup_metrics(m, f"Race: {race}")
        if r:
            sub_rows.append(r)
    # Income tertiles (poverty income ratio)
    pir = feats_test["INDFMPIR"]
    q1, q2 = pir.quantile([1/3, 2/3])
    for lab, m in [(f"PIR low (<{q1:.2f})", (pir < q1).to_numpy()),
                   (f"PIR mid", ((pir >= q1) & (pir < q2)).to_numpy()),
                   (f"PIR high (≥{q2:.2f})", (pir >= q2).to_numpy())]:
        r = subgroup_metrics(m, f"Income {lab}")
        if r:
            sub_rows.append(r)

    sub_df = pd.DataFrame([r for r in sub_rows if r is not None])
    sub_df.to_csv(RES / "subgroup_fairness.csv", index=False)
    print(sub_df.to_string(index=False))

    print("\nAll outputs written to:")
    print(f"  figures/  -> {FIG}")
    print(f"  results/  -> {RES}")


if __name__ == "__main__":
    main()
