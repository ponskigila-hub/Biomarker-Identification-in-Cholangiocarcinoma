"""
diagnose_threshold_transfer.py

Run this AFTER applying the log2 fix (i.e. app.py already has
auto_log2_transform wired in -- the file I sent as app_fixed.py).

Investigates why AUC improved a lot (0.78-0.90) but recall on
CCA dropped to 0/149 for all 3 models -- i.e. checks whether this
is a threshold-transfer / calibration problem (ranking is fine,
fixed cutoff isn't) rather than a ranking problem.

IMPORTANT: this script imports your app.py as a module, so save
it in the SAME folder as app.py (the one with auto_log2_transform
already added), and make sure the `data/` folder is a subfolder
right next to it, exactly like when you run `streamlit run app.py`.

Usage:
    python diagnose_threshold_transfer.py > threshold_diag.txt

Paste the full output back to Claude.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import app  # <- must be your fixed app.py, in the same folder

data_dir = "data"

# ---- load + log2 fix (same as app.py's main block) ----
expr1, y1 = app.parse_series_matrix(
    app.FileLike(f"{data_dir}/GSE76297_series_matrix.txt"))
expr1 = app.auto_log2_transform(expr1, "GSE76297")

expr2, y2 = app.parse_series_matrix(
    app.FileLike(f"{data_dir}/GSE132305_series_matrix.txt"))
expr2 = app.auto_log2_transform(expr2, "GSE132305")

expr3, y3 = app.parse_series_matrix(
    app.FileLike(f"{data_dir}/GSE32225_series_matrix.txt"))
expr3 = app.auto_log2_transform(expr3, "GSE32225")

map1 = app.load_annotation(f"{data_dir}/GPL17586.txt", "GPL17586")
map2 = app.load_annotation(f"{data_dir}/GPL13667.txt", "GPL13667")
map3 = app.load_annotation(f"{data_dir}/GPL8432.txt", "GPL8432")

expr1 = app.convert_probe_to_gene(expr1, map1)
expr2 = app.convert_probe_to_gene(expr2, map2)
expr3 = app.convert_probe_to_gene(expr3, map3)

common_genes = (
    expr1.columns.intersection(expr2.columns).intersection(expr3.columns)
)

X_train = pd.concat([expr1[common_genes], expr2[common_genes]])
y_train = pd.concat([y1, y2])
X_test = expr3[common_genes]
y_test = y3

batch_labels = ["batch1"] * len(expr1) + ["batch2"] * len(expr2)
X_train = app.pycombat(X_train.T, batch_labels).T

imputer = app.KNNImputer(n_neighbors=5)
X_train = pd.DataFrame(imputer.fit_transform(X_train),
                        columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(imputer.transform(X_test),
                       columns=X_test.columns, index=X_test.index)

scaler = app.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train),
                        columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test),
                       columns=X_test.columns, index=X_test.index)

dea_genes, _ = app.differential_expression(X_train, y_train, 1.0, 0.05)
mrmr_genes = app.mrmr_selection(X_train[dea_genes], y_train, 50)
final_features = app.lasso_selection(X_train[mrmr_genes], y_train)

X_train_final = X_train[final_features]
X_test_final = X_test[final_features]

print(f"Selected genes ({len(final_features)}): {final_features}\n")
print(f"Train class balance:    {dict(y_train.value_counts())}  "
      f"({(y_train == 1).mean() * 100:.1f}% CCA)")
print(f"External class balance: {dict(y_test.value_counts())}  "
      f"({(y_test == 1).mean() * 100:.1f}% CCA)\n")

cv = app.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in app.get_model_defs().items():
    print("=" * 70)
    print(name)
    print("=" * 70)

    oof_probs = app.cross_val_predict(
        app.clone(model), X_train_final, y_train,
        cv=cv, method="predict_proba"
    )[:, 1]

    fpr_tr, tpr_tr, thr_tr = app.roc_curve(y_train, oof_probs)
    youden_threshold = thr_tr[np.argmax(tpr_tr - fpr_tr)]

    model.fit(X_train_final, y_train)
    y_prob_test = model.predict_proba(X_test_final)[:, 1]

    print(f"Train OOF prob | CCA (y=1):    mean={oof_probs[y_train.values==1].mean():.4f}  "
          f"median={np.median(oof_probs[y_train.values==1]):.4f}")
    print(f"Train OOF prob | Normal (y=0): mean={oof_probs[y_train.values==0].mean():.4f}  "
          f"median={np.median(oof_probs[y_train.values==0]):.4f}")
    print(f"Youden threshold (from train OOF only): {youden_threshold:.4f}\n")

    print(f"External prob  | CCA (y=1):    mean={y_prob_test[y_test.values==1].mean():.4f}  "
          f"median={np.median(y_prob_test[y_test.values==1]):.4f}  "
          f"max={y_prob_test[y_test.values==1].max():.4f}")
    print(f"External prob  | Normal (y=0): mean={y_prob_test[y_test.values==0].mean():.4f}  "
          f"median={np.median(y_prob_test[y_test.values==0]):.4f}\n")

    n_above = int((y_prob_test[y_test.values == 1] >= youden_threshold).sum())
    print(f"External CCA samples above Youden threshold: "
          f"{n_above} / {(y_test.values == 1).sum()}")

    auc = app.roc_auc_score(y_test, y_prob_test)
    print(f"External AUC (threshold-independent, sanity check): {auc:.4f}\n")

    # --- Alternative threshold, still computed from y_train ONLY ---
    # Targets a chosen sensitivity level within the training OOF
    # probabilities instead of Youden's J (which is pulled around by
    # the ~68/32 train class balance). This is standard practice for
    # cancer-screening models where missing a positive is costlier
    # than a false alarm. Never touches y_test -> not leakage, just a
    # different pre-specified decision rule.
    for target_sensitivity in [0.90, 0.95]:
        cca_oof = np.sort(oof_probs[y_train.values == 1])
        idx = int(np.floor((1 - target_sensitivity) * len(cca_oof)))
        sens_threshold = cca_oof[idx]

        y_pred_sens = (y_prob_test >= sens_threshold).astype(int)
        tn, fp, fn, tp = app.confusion_matrix(y_test, y_pred_sens).ravel()

        print(f"Alt. threshold targeting {target_sensitivity:.0%} train sensitivity: "
              f"{sens_threshold:.4f}")
        print(f"  -> External: TP={tp} FN={fn} TN={tn} FP={fp}  "
              f"(missed {fn}/{tp + fn} CCA, "
              f"specificity={tn/(tn+fp) if (tn+fp) else float('nan'):.3f})")
    print()

print("DONE. Paste everything above back to Claude.")
