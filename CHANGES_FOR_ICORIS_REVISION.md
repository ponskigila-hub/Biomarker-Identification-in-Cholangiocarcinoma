# Changes made to address ICORIS 2026 reviewer comments (Paper 115)

This file maps every change in `app.py` to the specific reviewer comment
it addresses. Original code is preserved as `app_original_backup.py`.

## 🔴 Critical: data leakage (Reviewer 1, main concern)

Reviewer 1 asked whether DEA/mRMR/LASSO/scaling/threshold selection were
fit only on training data. Two real leakage bugs were found and fixed:

1. **Decision threshold was tuned on the external validation set.**
   Original: `roc_curve(y_test, y_prob)` → Youden's J computed directly
   from external-validation labels, then used to report external metrics
   on that same set. This inflated every reported external metric.
   Fix: the threshold is now chosen from out-of-fold probabilities on the
   *training* set only (`cross_val_predict`), before the external set is
   touched at all. See `train_models()`.

2. **Feature selection (DEA → mRMR → LASSO) was fit once on the full
   training set, then cross-validation ran only on the pre-selected
   genes.** This let every CV fold "see" information from its own
   held-out samples through the gene-selection step — the direct cause
   of the ~0.99 CV F1 vs 0.86 external F1 gap Reviewer 1 flagged.
   Fix: `nested_cv_pipeline_eval()` repeats DEA/mRMR/LASSO **inside**
   every fold, using only that fold's training split. The app now shows
   naive CV, honest nested CV, and external F1 side by side so the gap
   (and its resolution) is explicit rather than hidden.

   On synthetic test data this fix behaved exactly as expected: naive
   CV F1 ≈ 0.75–0.80 dropped to honest nested CV F1 ≈ 0.5–0.6, closing
   most of the gap toward the external-validation number.

   ComBat, KNN imputation, and StandardScaler were already fit on
   training data only and applied via `.transform()` to the external
   set — those steps were **not** leaky and are unchanged.

## 🟡 Dataset description (Reviewer 1 & 2)

- New "Dataset Summary" section reports, per GEO dataset: sample count,
  CCA vs normal count, and gene count after probe→gene mapping.
- Reports common-gene count across all three datasets, missing-value
  counts before imputation, and gene counts after each of
  DEA → mRMR → LASSO.

## 🟡 Ablation study (Reviewer 2, point 1)

`nested_cv_pipeline_eval()` is run for three stage combinations
(DEA only / DEA+mRMR / DEA+mRMR+LASSO), each using honest nested CV, so
the paper can report each stage's incremental contribution to F1.

## 🟡 Statistical significance test (Reviewer 2, point 3)

Paired t-test (`scipy.stats.ttest_rel`) between LogisticRegression's and
each other model's nested-CV fold scores (same 5 folds for every model).
Flagged in-app as low-power given only 5 folds — report accordingly.

## 🟡 Before/after ComBat visualization (Reviewer 2, point 5)

New PCA plot pair (before/after batch correction, colored by batch)
under "Additional Analytics Dashboard".

## 🟡 Hyperparameter tuning description (Reviewer 2, point 6)

Model configurations (SVM: linear kernel, C=0.1; RandomForest: 1000
trees, max_depth=10; LogisticRegression: balanced class weights) are
unchanged manual settings — no search was run in the original code.
**This is not yet fixed in code** — decide whether to (a) run and
document a small grid/random search, or (b) state in the paper that
hyperparameters were set manually based on preliminary experiments
rather than an automated search. Recommend (a) if time allows, since
Reviewer 2 asked for the tuning *process* specifically.

## 🟢 False negative analysis (Reviewer 1)

New section lists every missed CCA sample per model with its predicted
probability and the decision threshold used, so borderline vs.
confidently-missed cases can be distinguished in the Discussion.

## 🐛 Bonus fix: broken dependency

`requirements.txt` listed `pycombat`, which on PyPI is a *different,
incompatible* package from the one the code actually imports
(`combat.pycombat`). A fresh `pip install -r requirements.txt` would
have failed. Fixed to `combat`.

## ⚠️ Not addressed in code (needs manual writing/literature work)

- **"Early diagnosis" claim** (Reviewer 1): soften or remove — the
  datasets don't distinguish early- vs late-stage CCA.
- **Biomarker validation claim** (Reviewer 1): SHAP importance ≠
  validated biomarker. Add pathway-enrichment or literature cross-check
  discussion, or explicitly frame NOX4/NMI/CORO2B/TLL1/GFOD1 as
  "SHAP-prioritized candidates" pending independent validation.
- **Comparison table with related work** (Reviewer 2, point 7): build
  from references [4]–[8], [26]–[28], [35]–[37] already in the
  bibliography.
- Minor text fixes: "ensure that normal samples are e." (truncated
  sentence), "Youndex index" → "Youden index".
