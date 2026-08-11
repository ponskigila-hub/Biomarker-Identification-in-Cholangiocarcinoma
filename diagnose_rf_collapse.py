"""
Diagnostic script #3: why does RandomForest collapse to constant
probability and miss ALL 149 CCA samples in external validation,
even after calibration?

Reuses app.py's own parsing/pipeline functions so results match your
real pipeline exactly. Uses the same hyperparameters you reported:
KNN k=5, logFC>=1.0, p<0.05, mRMR K=50.

Only summary statistics are printed -- no raw expression values.
Paste the full output back to Claude.

Usage:
    python diagnose_rf_collapse.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import (
        FileLike,
        parse_series_matrix,
        load_annotation,
        convert_probe_to_gene,
        differential_expression,
        mrmr_selection,
        lasso_selection,
    )
except ImportError as e:
    print(f"ERROR: could not import from app.py in this folder: {e}")
    sys.exit(1)

try:
    from combat.pycombat import pycombat
except ImportError:
    print("ERROR: combat package not importable.")
    sys.exit(1)

DATA_DIR = "data"

# Same hyperparameters you reported for resultv3.
IMPUTE_K = 5
LOGFC_THRESH = 1.0
PVAL_THRESH = 0.05
MRMR_K = 50


def main():
    print("Loading and parsing series matrices...")
    expr1, y1 = parse_series_matrix(
        FileLike(os.path.join(DATA_DIR, "GSE76297_series_matrix.txt"))
    )
    expr2, y2 = parse_series_matrix(
        FileLike(os.path.join(DATA_DIR, "GSE132305_series_matrix.txt"))
    )
    expr3, y3 = parse_series_matrix(
        FileLike(os.path.join(DATA_DIR, "GSE32225_series_matrix.txt"))
    )

    map1 = load_annotation(os.path.join(DATA_DIR, "GPL17586.txt"), "GPL17586")
    map2 = load_annotation(os.path.join(DATA_DIR, "GPL13667.txt"), "GPL13667")
    map3 = load_annotation(os.path.join(DATA_DIR, "GPL8432.txt"), "GPL8432")

    expr1 = convert_probe_to_gene(expr1, map1)
    expr2 = convert_probe_to_gene(expr2, map2)
    expr3 = convert_probe_to_gene(expr3, map3)

    common_genes = expr1.columns.intersection(expr2.columns).intersection(expr3.columns)

    X_train = pd.concat([expr1[common_genes], expr2[common_genes]])
    y_train = pd.concat([y1, y2])
    X_test = expr3[common_genes]
    y_test = y3

    batch_labels = ["batch1"] * len(expr1) + ["batch2"] * len(expr2)

    print("Running ComBat...")
    X_train = pycombat(X_train.T, batch_labels).T

    print("Imputing + scaling (train-fit, applied to test)...")
    imputer = KNNImputer(n_neighbors=IMPUTE_K)
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    print("Running DEA -> mRMR -> LASSO feature selection...")
    dea_genes, _ = differential_expression(X_train, y_train, LOGFC_THRESH, PVAL_THRESH)
    mrmr_genes = mrmr_selection(X_train[dea_genes], y_train, MRMR_K)
    final_features = lasso_selection(X_train[mrmr_genes], y_train)

    X_train_final = X_train[final_features]
    X_test_final = X_test[final_features]

    print(f"\nFinal selected genes ({len(final_features)}): {final_features}")

    # ---- Fit RAW (uncalibrated) RandomForest, exactly as app.py's
    #      base estimator, to isolate whether the constant-score
    #      issue exists BEFORE calibration is even applied. ----
    print("\nFitting raw RandomForestClassifier (uncalibrated)...")
    rf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    rf.fit(X_train_final, y_train)

    raw_probs = rf.predict_proba(X_test_final)[:, 1]
    print(f"\nRaw external predict_proba -- unique values: {len(np.unique(raw_probs))} "
          f"out of {len(raw_probs)} samples")
    print(f"Min: {raw_probs.min():.6f}  Max: {raw_probs.max():.6f}  "
          f"Std: {raw_probs.std():.6f}")

    # value counts
    vc = pd.Series(raw_probs).value_counts().sort_index()
    print(f"\nTop 10 most common raw probability values (value: count):")
    for val, count in vc.sort_values(ascending=False).head(10).items():
        print(f"  {val:.6f}: {count} samples")

    # ---- Leaf-assignment check: do the tied samples land in the
    #      EXACT SAME leaf across all/most trees? ----
    print("\n" + "=" * 70)
    print("LEAF ASSIGNMENT CHECK")
    print("=" * 70)

    leaves = rf.apply(X_test_final)  # shape (n_samples, n_trees)

    # group external test sample indices by their raw predicted prob
    prob_series = pd.Series(raw_probs, index=X_test_final.index)
    most_common_val = vc.sort_values(ascending=False).index[0]
    tied_mask = np.isclose(raw_probs, most_common_val)
    tied_idx = np.where(tied_mask)[0]

    print(f"\n{tied_mask.sum()} external samples share the most common "
          f"probability value ({most_common_val:.6f}).")
    print(f"Their true labels: {y_test.values[tied_idx]}")

    if len(tied_idx) >= 2:
        leaves_tied = leaves[tied_idx]
        identical_leaf_frac = (leaves_tied == leaves_tied[0]).mean()
        print(f"\nOf the {leaves.shape[1]} trees, fraction where these tied "
              f"samples land in the EXACT SAME leaf as each other: "
              f"{identical_leaf_frac:.3f}")
        print("(1.0 = perfectly identical leaf path across all trees for "
              "every tied sample -> the model literally cannot tell them "
              "apart. Lower values mean leaves differ but still average "
              "to the same vote fraction by coincidence.)")

    # ---- Feature-range overlap check: for each selected gene, is
    #      the external test value range entirely outside the
    #      training range (pure extrapolation)? ----
    print("\n" + "=" * 70)
    print("TRAIN vs EXTERNAL FEATURE RANGE OVERLAP (per selected gene)")
    print("=" * 70)
    print(f"{'Gene':15s} {'Train min':>10s} {'Train max':>10s} "
          f"{'Test min':>10s} {'Test max':>10s} {'Test outside train range?':>28s}")

    n_fully_outside = 0
    for gene in final_features:
        tr_min, tr_max = X_train_final[gene].min(), X_train_final[gene].max()
        te_min, te_max = X_test_final[gene].min(), X_test_final[gene].max()
        fully_outside = (te_max < tr_min) or (te_min > tr_max)
        if fully_outside:
            n_fully_outside += 1
        flag = "YES <-- pure extrapolation" if fully_outside else ""
        print(f"{gene:15s} {tr_min:10.3f} {tr_max:10.3f} "
              f"{te_min:10.3f} {te_max:10.3f} {flag:>28s}")

    print(f"\n{n_fully_outside} / {len(final_features)} genes have their "
          f"ENTIRE external test range outside the training range "
          f"(the model has never seen values in that range and every "
          f"split on that feature routes all test samples the same way).")

    # ---- For the specific tied samples, show which are CCA (missed)
    #      vs normal (correctly predicted normal by coincidence) ----
    print("\n" + "=" * 70)
    print("Composition of the tied-probability group")
    print("=" * 70)
    tied_labels = y_test.values[tied_idx]
    print(f"CCA (1) in tied group: {(tied_labels == 1).sum()}")
    print(f"Normal (0) in tied group: {(tied_labels == 0).sum()}")

    print("\n" + "=" * 70)
    print("DONE. Paste everything above back to Claude.")
    print("=" * 70)


if __name__ == "__main__":
    main()
