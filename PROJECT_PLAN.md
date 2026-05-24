# Project Plan
## Can Depression in Younger Adults Be Used to Predict Cardiovascular Risk Profiles Using Machine Learning?

### Research question
In US adults aged 18–39, can self-reported depressive symptomatology (PHQ-9) — together with demographic and socioeconomic context — discriminate between data-driven cardiovascular (CV) risk profiles derived from self-reported and behavioural risk indicators?

### Hypothesis
H1. Younger adults with higher depressive-symptom burden are over-represented in CV risk profiles characterised by self-reported diagnosis of hypertension and/or hypercholesterolaemia, doctor-flagged lifestyle issues, short sleep, and household tobacco exposure.
H2. Supervised machine-learning models trained on depression + demographic features can predict the derived risk-profile cluster with discrimination superior to a no-information baseline and to a model using demographics alone.

### Dataset
**NHANES pre-pandemic combined cycle (2017–March 2020, the "P_" release).** Free public-use dataset from the US CDC/NCHS. Probability sample of the US non-institutionalised population, with PHQ-9 screening for those aged 12+. We use the P_ release because it is the only cycle in the supplied data that contains both the full PHQ-9 (P_DPQ) and the cardiovascular-relevant questionnaire modules (P_BPQ, P_MCQ, P_CDQ, P_SLQ, P_SMQFAM, P_DEMO). The 2021–2023 ("_L") release was inspected but does not include the depression module in the supplied set.

### Sample
- Inclusion: age 18–39 (inclusive); non-missing PHQ-9 (≤2 items missing, prorated).
- Exclusion: pregnant participants (RIDEXPRG=1) and respondents who refused or did not know more than two PHQ-9 items.
- Target n: ≈ 3,000 after merge.

### Variables

**Depression (primary predictor)**
- PHQ-9 total score (DPQ010–DPQ090, recoded 7/9→missing, summed; valid range 0–27).
- Binary clinically-significant depression (PHQ-9 ≥ 10).
- Item-level scores retained for feature analysis.

**Demographic / SES covariates**
- Age (years), sex, race/ethnicity (RIDRETH3), education (DMDEDUC2 → categorical), marital status (DMDMARTZ), family poverty-to-income ratio (INDFMPIR).

**CV risk indicators (used to derive the unsupervised target)**
- BPQ020 — ever told had high blood pressure
- BPQ050A — currently taking medication for high blood pressure
- BPQ080 — ever told had high cholesterol
- BPQ100D — currently taking cholesterol-lowering medication
- MCQ080 — doctor told overweight
- MCQ366A/B/C/D — doctor told to control weight / increase activity / reduce salt / reduce fat
- MCQ300A — close relative had heart attack before age 50 (family history)
- MCQ300C — close relative had diabetes
- SLD012 — usual sleep duration on weekdays (hours)
- SMD460 — number of smokers in the household

### Pre-processing
1. Merge modules on SEQN, retain participants aged 18–39 only.
2. Code 7 (refused), 9 (don't know), and "." as missing for binary items.
3. Sum PHQ-9 with prorating when ≤2 items missing.
4. Median-impute remaining continuous predictors; mode-impute binary predictors. Sensitivity check with complete-case analysis.
5. One-hot encode multi-level categoricals (race, education, marital).
6. Standardise continuous variables for clustering and for the linear model.

### Unsupervised target derivation (CV risk profile)
- Apply K-means clustering on the standardised CV risk indicators (sleep reverse-scored so higher = more risk: insufficient_sleep = max(0, 7 − SLD012)).
- Select K by silhouette score over K∈{2,3,4} and elbow inspection.
- Label clusters by mean composite-risk-index rank (low → high). Profile names assigned post-hoc based on member-feature means.

### Supervised modelling
- Target: cluster label (k classes, ordinal interpretation).
- Features: PHQ-9 total, PHQ-9 binary, demographics & SES (age, sex, race, education, marital, poverty ratio).
- Algorithms (at least two; rubric satisfied):
  1. **Logistic Regression** (multinomial, L2-regularised) — interpretable linear baseline.
  2. **Random Forest** — handles non-linearities and feature interactions.
  3. **Gradient Boosting (HistGradientBoostingClassifier)** — state-of-the-art tabular performance with regularisation.
- A demographics-only baseline (no depression features) is fit for each algorithm to quantify depression's incremental contribution.

### Model selection & tuning
- Stratified 5-fold cross-validation with three repeats (15 evaluations).
- Hyperparameter search: small grids tuned by macro-F1 within an inner 3-fold CV.
- Random seeds: 5 seeds for stability assessment.

### Evaluation metrics
- Macro F1, balanced accuracy, ROC-AUC (one-vs-rest), confusion matrix.
- Comparison vs. demographics-only baseline using paired t-tests across folds.
- Stability: standard deviation across seeds/folds.

### Interpretability
- Permutation importance for tree-based models.
- SHAP values (Tree SHAP for HistGB and RF) on the test fold.
- Coefficient inspection for logistic regression.

### Ethics & reflection (to discuss in report)
- Self-reported diagnoses introduce ascertainment bias correlated with healthcare access.
- Race/sex feature use risks recapitulating historical inequities; we discuss feature-removal sensitivity.
- The model is *associational*: depression may be a marker, not a cause, of CV risk.
- We avoid generating individual-level clinical predictions; framing is population-level risk stratification.
- We acknowledge fairness considerations (disparate prevalence across subgroups) and run subgroup performance metrics.

### Limitations (acknowledged ex ante)
- Cross-sectional design — no causal claim.
- Self-reported CV risk indicators rather than measured (BP, lipids, HbA1c) because measurement modules were not available in the supplied subset of NHANES files. We will discuss how this skews estimates toward those with healthcare contact.
- PHQ-9 is a screener, not a diagnosis.
- Survey weights are not incorporated in the ML estimands (point estimates are unweighted); we discuss the implication.

### Deliverables
- `code/` — preprocessing, clustering, modelling, evaluation, plotting.
- `figures/` — PNG outputs referenced by the report.
- `results/` — metric tables (CSV).
- `report.docx` — 3,000-word report.
- `README.md` — title, abstract, dataset access, reproduction steps, environment.
