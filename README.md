# Predicting Cardiovascular Risk Profiles from Depressive Symptomatology in US Adults Aged 18–39

**MSt Healthcare Data Science - Module 5 Assignment**
Author: Yuliya Savenko
Repository: <https://github.com/Yuliya0257/Module5_YuliyaSavenko>

---

## Abstract

**Background.** Depression in young adulthood has risen sharply and is increasingly recognised as a risk factor for cardiovascular disease (CVD), but most evidence comes from middle-aged or older populations. Whether depressive symptomatology adds discriminative value over demographics alone for predicting CVD risk in the under 40 age group is unclear. We tested whether the Patient Health Questionnaire-9 (PHQ-9) improves machine-learning (ML) discrimination of data-driven cardiovascular risk profiles in young adults from the US National Health and Nutrition Examination Survey (NHANES).

**Methods.** We used the NHANES pre-pandemic combined cycle (2017 March 2020), restricting to adults aged 18-39 with a usable PHQ-9 (*n* = 2,728). Cardiovascular risk profiles were derived by K-means clustering of 13 self-reported and behavioural risk indicators. Three from-scratch classifiers - logistic regression, random forest and gradient boosting  were trained to predict the resulting risk profile from PHQ-9 and demographics, with a demographics-only ablation. Performance was assessed by 5 fold stratified cross-validation across three seeds (15 folds), with minority-class oversampling in training folds. Age-window and clustering-method sensitivity analyses were also run.

**Results.** Clustering selected *K* = 2 (silhouette 0.37). The elevated-risk profile (21% of the sample) was characterised by physician-given lifestyle advice and self-reported hypertension/dyslipidaemia. Its prevalence was 35.3% among participants with PHQ-9 ≥ 10 versus 20.6% with PHQ-9 < 10. Adding PHQ-9 to demographics raised cross-validated ROC-AUC from ≈ 0.60 to 0.637–0.642 across all three classifiers (paired *t*-tests, *p* < 0.0001). PHQ-9 was the most important predictor by permutation importance.

**Conclusions.** Depressive symptomatology offers a small but reproducible improvement in predicting data-driven cardiovascular risk profiles in young adults, supporting the inclusion of psychological health in cardiovascular risk assessment at this life stage.

---

## Repository layout

```
.
├── README.md                       # this file
├── PROJECT_PLAN.md                 # design document (variables, methods, ethics)
├── HDS_ML_Savenko_2605.docx        # final report (Word, ≤3,000 words main body)
├── analysis.py                     # single-file pipeline (recommended entry point)
├── build_report.py                 # rebuilds the .docx (also runnable via analysis.py report)
├── environment.yml                 # conda environment specification
├── requirements.txt                # pip requirements (alternative to conda)
├── code/                           # modular version of analysis.py (same code, split by concern)
│   ├── ml_lib.py                   # from-scratch NumPy: K-means, K-modes/K-prototypes,
│   │                               #   logistic regression, random forest,
│   │                               #   gradient boosting, metrics
│   ├── validate_ml_lib.py          # correctness tests (analytical + optional sklearn)
│   ├── tune_hyperparameters.py     # grid search with inner CV (LR, RF, GB)
│   ├── sensitivity_age.py          # age-window sensitivity (18-39 / 18-64 / 18+)
│   ├── sensitivity_clustering.py   # clustering-method sensitivity (K-means vs K-modes)
│   └── run_analysis.py             # end-to-end pipeline (data → models → outputs)
├── data/
│   └── raw_nhanes/                 # NHANES XPT files (see "Dataset access" below)
│       ├── P_DEMO.xpt.txt
│       ├── P_DPQ.xpt.txt
│       ├── P_BPQ.xpt.txt
│       ├── P_MCQ.xpt.txt
│       ├── P_SLQ.xpt.txt
│       └── P_SMQFAM.xpt.txt
├── figures/                        # PNGs referenced by the report
│   ├── fig_silhouette.png
│   ├── fig_cluster_profile.png
│   ├── fig_phq_by_profile.png
│   ├── fig_cv_metrics.png
│   ├── fig_rf_confusion.png
│   ├── fig_lr_confusion.png
│   ├── fig_rf_permimp.png
│   └── fig_rf_phq_marginal.png
└── results/
    ├── analysis_cohort.csv         # cleaned analytic dataset (n = 2,728)
    ├── cohort_summary.csv
    ├── cluster_info.json           # cluster centres, silhouettes, K selection
    ├── risk_profile_by_depression.csv
    ├── cv_summary.csv              # mean ± SD per model × metric
    ├── cv_folds_*.csv              # per-fold raw results
    ├── ablation_ttests.json        # paired t-test, full vs demographics-only
    ├── permutation_importance_rf.csv
    ├── logistic_coefficients.csv
    ├── hyperparameter_search.csv  # full LR×16 + RF×27 + GB×12 grid scores
    ├── best_hyperparameters.json  # selected configs per model
    ├── subgroup_fairness.csv      # held-out RF metrics by sex / race / income tertile
    ├── sensitivity_age.csv        # AUC across 18-39 / 18-64 / 18+ age windows
    ├── sensitivity_clustering.csv # K-means vs K-modes agreement + depression gradient
    ├── cohort_table.csv           # descriptive cohort characteristics by risk profile
    ├── crude_association.json     # odds ratio + chi-square, PHQ-9 vs risk profile
    ├── calibration.json           # Brier score + predicted vs observed prevalence
    ├── gb_backend.txt             # which gradient-boosting library produced the run
    └── validation_report.txt      # pass/fail log from validate_ml_lib.py
```

---

## Dataset access

This work uses the **NHANES pre-pandemic combined cycle (2017 March 2020)**, the "P_" release, which is free and publicly available from the US Centers for Disease Control and Prevention (CDC).

**Files used (in `data/raw_nhanes/`):**

| File | Module | URL |
|------|--------|-----|
| `P_DEMO` | Demographics | <https://wwwn.cdc.gov/Nchs/Nhanes/2017-2020/P_DEMO.XPT> |
| `P_DPQ`  | Mental Health — Depression Screener (PHQ-9) | <https://wwwn.cdc.gov/Nchs/Nhanes/2017-2020/P_DPQ.XPT> |
| `P_BPQ`  | Blood Pressure & Cholesterol Questionnaire | <https://wwwn.cdc.gov/Nchs/Nhanes/2017-2020/P_BPQ.XPT> |
| `P_MCQ`  | Medical Conditions Questionnaire | <https://wwwn.cdc.gov/Nchs/Nhanes/2017-2020/P_MCQ.XPT> |
| `P_SLQ`  | Sleep Disorders | <https://wwwn.cdc.gov/Nchs/Nhanes/2017-2020/P_SLQ.XPT> |
| `P_SMQFAM` | Smoking  Household Smokers | <https://wwwn.cdc.gov/Nchs/Nhanes/2017-2020/P_SMQFAM.XPT> |

To download all files automatically (Linux/macOS):

```bash
mkdir -p data/raw_nhanes
cd data/raw_nhanes
for f in P_DEMO P_DPQ P_BPQ P_MCQ P_SLQ P_SMQFAM; do
  curl -L -o "${f}.xpt.txt" "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2020/${f}.XPT"
done
```

(The pipeline expects the `.xpt.txt` suffix used by the included files. If you download with the standard `.XPT` extension, either rename them or edit `code/run_analysis.py: load_module`.)

NHANES is publicly released de-identified data and does not require IRB approval for secondary analysis.

---

## Reproducing the analysis

### Option A: conda (recommended)

```bash
git clone https://github.com/Yuliya0257/Module5_YuliyaSavenko.git
cd Module5_YuliyaSavenko
conda env create -f environment.yml
conda activate hds-ml-mod5

# Place NHANES XPT files in data/raw_nhanes/ (see "Dataset access" above)
python analysis.py validate              # correctness tests (~5s, 11 PASS expected)
python analysis.py tune                  # optional: regenerate hyperparameter grid
python analysis.py sensitivity-age       # optional: age-window sensitivity (~30s)
python analysis.py sensitivity-cluster   # optional: K-means vs K-modes check (~10s)
python analysis.py analyse               # main pipeline + subgroup fairness check
python analysis.py report                # rebuild the .docx

# Or run every stage in sequence:
python analysis.py all
```

### Option B: pip + venv

```bash
git clone https://github.com/Yuliya0257/Module5_YuliyaSavenko.git
cd Module5_YuliyaSavenko
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python analysis.py validate
python analysis.py tune
python analysis.py sensitivity-age
python analysis.py sensitivity-cluster
python analysis.py analyse
python analysis.py report
```

**Runtime:** the full pipeline runs in ≈ 60-90 seconds on a typical laptop. All outputs are deterministic given the seeds in `code/run_analysis.py` (`RNG = 20240516`, CV seeds `1, 2, 3`).

---

## What `run_analysis.py` produces

1. **Cohort assembly.** Merges the six NHANES modules on `SEQN`, filters to ages 18-39, drops pregnant respondents, codes 7/9 as missing, prorates PHQ-9.
2. **Risk profile clustering.** K-means on 13 standardised CV risk indicators; selects *K* by silhouette over {2,3,4}; saves `figures/fig_silhouette.png`, `figures/fig_cluster_profile.png`, `figures/fig_phq_by_profile.png`.
3. **Supervised modelling.** Trains logistic regression, random forest and gradient boosting, each with and without PHQ-9. Uses stratified 5-fold CV × 3 seeds, with training-fold oversampling. Saves `results/cv_summary.csv` and per-fold CSVs.
4. **Ablation tests.** Paired *t*-test on macro F1 (full vs demographics-only), saved to `results/ablation_ttests.json`.
5. **Held-out evaluation.** 75/25 split for final RF/LR confusion matrices and permutation importance; saves `figures/fig_rf_confusion.png`, `figures/fig_lr_confusion.png`, `figures/fig_rf_permimp.png`, `figures/fig_rf_phq_marginal.png`, `results/permutation_importance_rf.csv`, `results/logistic_coefficients.csv`.

`build_report.py` then assembles `HDS_ML_Savenko_2605.docx` by embedding the figures and the metrics table.

---

## Implementation notes

The analysis environment used to develop this work did not have `scikit-learn` available, so `code/ml_lib.py` contains compact, fully-documented NumPy implementations of:

- `StandardScaler`  z-score scaling
- `KMeans` with k-means++ initialisation, multi-restart, silhouette evaluation
- `KPrototypes` mixed binary + numeric clustering; reduces to K-modes when no numeric columns are given
- `LogisticRegression` multinomial, L2-regularised, batch gradient descent
- `DecisionTreeClassifier` CART with Gini impurity, quantile-binned thresholds
- `RandomForestClassifier` bagged trees with feature subsampling, class-aligned probability aggregation
- `DecisionTreeRegressor` + `GradientBoostingClassifier` binary gradient boosting with log-loss and Friedman leaf refinement
- `StratifiedKFold`, `train_test_split`, `oversample`
- Metrics: `accuracy`, `balanced_accuracy`, `macro_f1`, `roc_auc_ovr`, `confusion_matrix`
- `permutation_importance` (model-agnostic)

All algorithms follow a `fit` / `predict` / `predict_proba` interface familiar from scikit-learn. The same code can be swapped for scikit-learn equivalents without changing `run_analysis.py` beyond import statements useful if you want to verify the results against a reference toolkit.

**Gradient-boosting backend (LightGBM / XGBoost drop-in).** `run_analysis.py` auto-detects the best available gradient-boosting library: it uses **LightGBM** if installed, otherwise **XGBoost**, otherwise the from-scratch `GradientBoostingClassifier`. All three expose the same `fit` / `predict` / `predict_proba` interface, so they are genuine drop-in replacements. Install either library (`pip install lightgbm`) and re-run `code/run_analysis.py` to use it the backend in use is printed to the run log and written to `results/gb_backend.txt`. The results in this repository were generated using the from-scratch implementation; LightGBM reproduces them to within run-to-run variation. The selected boosting hyperparameters (n_estimators 60, learning_rate 0.05, max_depth 3) are standard parameter names shared by all three backends.

**Correctness validation.** `code/validate_ml_lib.py` runs eleven analytical / internal-consistency tests (StandardScaler post-transform mean and std; K-means and K-modes cluster purity on planted blobs; LR / RF / GB train accuracy on synthetic linear and non-linear problems; predict_proba row sums; analytical confusion-matrix / F1 / balanced-accuracy / AUC). All eleven pass in the supplied environment. The script additionally runs side-by-side comparisons against scikit-learn (`StandardScaler`, `KMeans`, `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`, plus the equivalent metric functions) when scikit-learn is installed — install `scikit-learn` and re-run to enable. Output is written to `results/validation_report.txt`.

---

## Results at a glance

| Model | Accuracy | Balanced acc | Macro F1 | ROC-AUC |
|------|---------|--------|--------|---------|
| Logistic regression (demographics only) | 0.575 ± 0.020 | 0.573 ± 0.022 | 0.525 ± 0.018 | 0.602 ± 0.026 |
| Logistic regression (+ PHQ-9) | 0.605 ± 0.020 | 0.591 ± 0.029 | 0.547 ± 0.022 | **0.637 ± 0.031** |
| Random forest (demographics only) | 0.608 ± 0.022 | 0.565 ± 0.021 | 0.536 ± 0.019 | 0.592 ± 0.031 |
| Random forest (+ PHQ-9) | 0.634 ± 0.022 | 0.604 ± 0.027 | 0.568 ± 0.023 | **0.638 ± 0.034** |
| Gradient boosting (demographics only) | 0.592 ± 0.026 | 0.571 ± 0.028 | 0.532 ± 0.024 | 0.599 ± 0.034 |
| Gradient boosting (+ PHQ-9) | 0.620 ± 0.025 | 0.606 ± 0.035 | 0.562 ± 0.028 | **0.642 ± 0.036** |

Adding PHQ-9 improves macro F1 significantly in all three models (paired *t*-tests: LR *t* = 4.49; RF *t* = 7.47; GB *t* = 5.48; all *p* < 10⁻⁵). **Sensitivity checks:** the depression signal is largest in the 18–39 window and weaker (but still significant) at 18-64 and 18+; the K=2 risk phenotype is robust to clustering method (K-means vs K-modes: 98.3% agreement, adjusted Rand index 0.92).

---

## License & ethics

Code is released under the MIT licence. NHANES data are public-domain and may be re-used freely; see the [NCHS Data Use Agreement](https://www.cdc.gov/nchs/data_access/restrictions.htm).

The analysis is associational and not intended for clinical decision-making. Section 4.3 of the report discusses fairness considerations associated with using race and education as predictors.
