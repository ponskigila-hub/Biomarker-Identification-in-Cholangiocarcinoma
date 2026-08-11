import streamlit as st
import pandas as pd
import numpy as np
import gzip
import os
import shap

from sklearn.model_selection import (
    cross_val_score,
    cross_val_predict,
    StratifiedKFold
)
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    f1_score
)

from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from mrmr import mrmr_classif
from combat.pycombat import pycombat

import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(
    page_title="CCA Biomarker Discovery",
    layout="wide"
)

st.title("🔬 Hybrid Feature Selection for Cholangiocarcinoma")


# =====================================================
# FILE WRAPPER
# =====================================================
class FileLike:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)

    def read(self):
        if self.path.endswith(".gz"):
            with gzip.open(self.path, "rb") as f:
                return f.read()
        else:
            with open(self.path, "rb") as f:
                return f.read()


# =====================================================
# LABEL EXTRACTION
# =====================================================
def extract_labels(lines, sample_ids):
    sample_titles = []

    for line in lines:
        if line.startswith("!Sample_title"):
            sample_titles = [
                x.strip('"')
                for x in line.strip().split("\t")[1:]
            ]
            break

    labels = {}

    for sid, title in zip(sample_ids, sample_titles):
        t = title.lower().strip()

        # -------------------------------------------------
        # FIX (data label bug found post-review):
        # GSE76297 mixes two DIFFERENT diseases under near-
        # identical naming: "HCC Tumor/Non-Tumor Tissue ..."
        # (hepatocellular carcinoma) and "CCA Tumor/Non-Tumor
        # Tissue ..." (cholangiocarcinoma, the disease this
        # study is about). HCC samples must be dropped
        # entirely -- they are not CCA and not normal bile
        # duct tissue, so they cannot be used as either class.
        # -------------------------------------------------
        if t.startswith("hcc"):
            labels[sid] = -1
            continue

        if t.startswith("cca") and (
            "non-tumor" in t or "non tumor" in t or "nontumor" in t
        ):
            labels[sid] = 0
            continue

        if t.startswith("cca") and "tumor" in t:
            labels[sid] = 1
            continue

        # GSE132305 naming: "..._BD" = non-tumor bile duct
        # (normal), "..._eCCA" = extrahepatic CCA (tumor)
        if t.endswith("_bd"):
            labels[sid] = 0
            continue

        if t.endswith("_ecca"):
            labels[sid] = 1
            continue

        # GSE32225 naming: "Ctrl_..." = normal control;
        # "CCBCN.../CCM.../CCNY..." = tumor cohorts
        # (Barcelona / Milan / New York)
        if t.startswith("ctrl"):
            labels[sid] = 0
            continue

        if (
            t.startswith("ccbcn")
            or t.startswith("ccm")
            or t.startswith("ccny")
        ):
            labels[sid] = 1
            continue

        # -------------------------------------------------
        # FIX: generic fallback keyword matching -- negative-
        # form phrases ("non-tumor", "non-cancerous") are
        # checked BEFORE the generic "tumor"/"cancer"
        # substring check below, since e.g. "non-tumor"
        # literally contains the substring "tumor" and would
        # otherwise be misclassified as the positive class.
        # This branch should rarely fire given the explicit
        # per-dataset rules above; kept as a safety net for
        # any other naming pattern.
        # -------------------------------------------------
        if any(
            k in t for k in [
                "non-tumor", "non tumor", "nontumor", "non_tumor",
                "non-cancerous", "noncancerous",
                "normal", "control", "benign"
            ]
        ):
            labels[sid] = 0
            continue

        if any(k in t for k in ["tumor", "cca", "cancer"]):
            labels[sid] = 1
            continue

        labels[sid] = -1

    return labels


# =====================================================
# PARSE MATRIX
# =====================================================
def parse_series_matrix(file_obj):
    content = file_obj.read().decode("utf-8")
    lines = content.splitlines()

    start_idx = None
    for i, line in enumerate(lines):
        if line.startswith("!series_matrix_table_begin"):
            start_idx = i + 1
            break

    headers = [
        x.strip('"')
        for x in lines[start_idx].split("\t")
    ]

    sample_ids = headers[1:]
    rows = []

    for line in lines[start_idx + 1:]:
        if line.startswith("!series_matrix_table_end"):
            break

        fields = [
            x.strip('"')
            for x in line.split("\t")
        ]

        if len(fields) == len(headers):
            rows.append(fields)

    probe_ids = [r[0] for r in rows]

    expr_data = np.array(
        [r[1:] for r in rows],
        dtype=float
    )

    expr_df = pd.DataFrame(
        expr_data.T,
        index=sample_ids,
        columns=probe_ids
    )

    labels = extract_labels(lines, sample_ids)

    valid_samples = [
        s for s in sample_ids
        if labels[s] != -1
    ]

    expr_df = expr_df.loc[valid_samples]

    y = pd.Series(
        [labels[s] for s in valid_samples],
        index=valid_samples
    )

    return expr_df, y


# =====================================================
# SCALE HARMONIZATION (NEW)
# =====================================================
# ROOT CAUSE FOUND (2026-08-11, RF collapse investigation):
# GSE76297 is deposited already log2-scale (~11-12), but
# GSE132305 and GSE32225 are deposited as raw/linear
# intensities (tens to thousands). Because no dataset was
# ever log2-transformed, ComBat (fit on train only) and
# StandardScaler (fit on train only) produced z-scores in
# the external test set that were 10-1000x outside the
# training range -> RandomForest routed every external
# sample into the same terminal leaf (frozen probability),
# and SVM/LR were pushed toward predicting the majority
# class rather than genuinely discriminating.
#
# Fix: detect per-dataset scale right after parsing (before
# any batch correction/scaling) and log2-transform any
# dataset that is not already log-scale, so all three
# datasets are comparable before ComBat/StandardScaler see
# them. This must run BEFORE convert_probe_to_gene so that
# probe-level averaging happens in a consistent space.
def auto_log2_transform(expr_df, name=""):
    """
    Detect whether a dataset is already log2-scale or still
    raw/linear intensity, and log2-transform it if needed.

    Heuristic: log2-scale microarray/RNA expression values
    are almost always well under ~25. Raw/linear intensities
    are typically in the hundreds to tens of thousands. This
    is a standard, widely-used rule of thumb (used e.g. by
    GEO2R itself) for auto-detecting whether a series needs
    log transformation.
    """
    vals = expr_df.values.astype(float)
    max_val = np.nanmax(vals)
    median_val = np.nanmedian(vals)

    if max_val > 100:
        transformed = np.log2(expr_df.clip(lower=0) + 1)
        st.info(
            f"[{name}] raw/linear scale detected "
            f"(max={max_val:.1f}, median={median_val:.1f}) "
            f"-> applied log2(x + 1)"
        )
        return transformed
    else:
        st.info(
            f"[{name}] already log2-scale "
            f"(max={max_val:.1f}, median={median_val:.1f}) "
            f"-> left as-is"
        )
        return expr_df


# =====================================================
# GPL
# =====================================================
def load_annotation(path, gpl_type):
    ann = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        low_memory=False
    )

    if gpl_type == "GPL13667":
        mapping = ann[["ID", "Gene Symbol"]].dropna()

    elif gpl_type == "GPL8432":
        mapping = ann[["ID", "Symbol"]].dropna()

    elif gpl_type == "GPL17586":
        ann["Gene Symbol"] = ann["gene_assignment"].apply(
            lambda x:
            str(x).split(" // ")[1]
            if " // " in str(x)
            else np.nan
        )

        mapping = ann[["ID", "Gene Symbol"]].dropna()

    else:
        return {}

    mapping.columns = ["probe", "gene"]

    return dict(
        zip(mapping["probe"], mapping["gene"])
    )


# =====================================================
# PROBE TO GENE
# =====================================================
def convert_probe_to_gene(expr_df, mapping):
    common_probe = [
        p for p in expr_df.columns
        if p in mapping
    ]

    expr_df = expr_df[common_probe].copy()

    expr_df.columns = [
        mapping[p]
        for p in common_probe
    ]

    expr_df = (
        expr_df.T.groupby(level=0).mean().T
    )

    return expr_df


# =====================================================
# DEA
# =====================================================
def differential_expression(
    X,
    y,
    logfc_thresh=1.0,   # CHANGED: 0.5 -> 1.0
    pval_thresh=0.05
):
    results = []

    tumor_idx = y[y == 1].index
    normal_idx = y[y == 0].index

    for gene in X.columns:
        tumor = X.loc[tumor_idx, gene]
        normal = X.loc[normal_idx, gene]

        if tumor.var() == 0 and normal.var() == 0:
            continue

        logfc = np.log2(
            (tumor.mean() + 1e-8) /
            (normal.mean() + 1e-8)
        )

        _, pval = stats.ttest_ind(
            tumor,
            normal,
            equal_var=False
        )

        results.append([
            gene,
            logfc,
            pval
        ])

    if len(results) == 0:
        return [], pd.DataFrame()

    res_df = pd.DataFrame(
        results,
        columns=[
            "gene",
            "logFC",
            "pvalue"
        ]
    )

    _, adj_p = fdrcorrection(
        res_df["pvalue"]
    )

    res_df["adj_p"] = adj_p

    sig = res_df[
        (abs(res_df["logFC"]) >= logfc_thresh)
        &
        (res_df["adj_p"] < pval_thresh)
    ]

    if len(sig) < 50:
        sig = res_df.nsmallest(
            500,
            "adj_p"
        )

    return sig["gene"].tolist(), res_df


# =====================================================
# MRMR
# =====================================================
def mrmr_selection(X, y, k):
    if X.shape[1] == 0:
        return []

    k = min(k, X.shape[1])

    return mrmr_classif(
        X=X,
        y=y,
        K=k,
        n_jobs=1
    )


# =====================================================
# LASSO
# =====================================================
def lasso_selection(X, y):
    if X.shape[1] == 0:
        return []

    model = LogisticRegressionCV(
        penalty="l1",
        solver="liblinear",
        cv=5,
        max_iter=5000
    )

    model.fit(X, y)

    selected = X.columns[
        model.coef_[0] != 0
    ].tolist()

    if len(selected) < 10:
        selected = X.columns[:20].tolist()

    return selected


# =====================================================
# METRICS
# =====================================================
def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    accuracy = (tp + tn)/(tp + tn + fp + fn)

    precision = tp/(tp+fp) if (tp+fp) > 0 else 0
    recall = tp/(tp+fn) if (tp+fn) > 0 else 0
    specificity = tn/(tn+fp) if (tn+fp) > 0 else 0

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0 else 0
    )

    auc_score = roc_auc_score(
        y_true,
        y_prob
    )

    denominator = np.sqrt(
        (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    )

    mcc = (
        ((tp*tn)-(fp*fn))/denominator
        if denominator > 0 else 0
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        "AUC": auc_score,
        "MCC": mcc
    }


# =====================================================
# BOOTSTRAP AUC CI
# =====================================================
def bootstrap_auc_ci(
    y_true,
    y_prob,
    n_bootstrap=1000
):
    rng = np.random.RandomState(42)
    auc_scores = []

    for _ in range(n_bootstrap):
        idx = rng.randint(
            0,
            len(y_true),
            len(y_true)
        )

        if len(np.unique(y_true.iloc[idx])) < 2:
            continue

        auc_scores.append(
            roc_auc_score(
                y_true.iloc[idx],
                y_prob[idx]
            )
        )

    lower = np.percentile(auc_scores, 2.5)
    upper = np.percentile(auc_scores, 97.5)

    return lower, upper


# =====================================================
# MODEL DEFINITIONS (factored out so folds/ablation get
# fresh, identically-configured estimators)
# =====================================================
def get_model_defs():
    return {
        # FIX: SVM and RandomForest predicted probabilities were
        # extremely uncalibrated (near-0/near-1 or constant-across-
        # samples) on the external set, causing both to collapse to a
        # single predicted class (AUC=0.5). Wrapped in
        # CalibratedClassifierCV (Platt/sigmoid, 5-fold internal CV)
        # so predict_proba reflects genuine confidence instead of raw
        # decision-function extremes. probability=True removed from
        # SVC to avoid double-calibrating (SVC's own internal Platt
        # scaling stacked under CalibratedClassifierCV's).
        "SVM": CalibratedClassifierCV(
            SVC(
                kernel="linear",
                C=0.1,
                class_weight="balanced"
            ),
            method="sigmoid",
            cv=5
        ),
        # FIX: class_weight="balanced" was missing here even though
        # the paper's Methodology section explicitly states "balanced
        # class weights were applied during training to mitigate the
        # impact of class imbalance" -- this contradicted the actual
        # code. Training set is ~68% CCA / 32% normal after the label
        # fix, so this matters. Also wrapped in CalibratedClassifierCV
        # for the same reason as SVM above.
        "RandomForest": CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=1000,
                max_depth=10,
                random_state=42,
                class_weight="balanced"
            ),
            method="sigmoid",
            cv=5
        ),
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=5000
        )
    }


# =====================================================
# TRAIN MODELS  (fits the final model on ALL training
# data, evaluates once on the untouched external set)
#
# FIX (Reviewer 1): the decision threshold used to be
# picked with Youden's J computed on (y_test, y_prob) --
# i.e. it was tuned using the external validation labels.
# That is a data-leakage bug: it inflates every reported
# external metric (Accuracy/Recall/F1/Specificity/MCC).
# The threshold is now chosen from OUT-OF-FOLD predicted
# probabilities on the TRAINING set only (via
# cross_val_predict), so the external set is never looked
# at until the very last, single evaluation step.
# =====================================================
def train_models(
    X_train,
    y_train,
    X_test,
    y_test,
    cv
):
    results = {}

    for name, model in get_model_defs().items():

        # Naive (non-nested) CV score -- feature selection
        # (DEA/mRMR/LASSO) was already fit on the full
        # X_train before this function runs, so folds here
        # still "see" information from held-out samples via
        # gene selection. Kept ONLY so it can be shown next
        # to the honest nested-CV score below, to make the
        # gap Reviewer 1 flagged explicit rather than hidden.
        cv_scores = cross_val_score(
            clone(model),
            X_train,
            y_train,
            cv=cv,
            scoring="f1"
        )

        # Out-of-fold probabilities on TRAINING data only,
        # used solely to pick the Youden threshold. y_test
        # is never touched here.
        oof_probs = cross_val_predict(
            clone(model),
            X_train,
            y_train,
            cv=cv,
            method="predict_proba"
        )[:, 1]

        fpr_tr, tpr_tr, thr_tr = roc_curve(y_train, oof_probs)
        threshold = thr_tr[np.argmax(tpr_tr - fpr_tr)]

        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        metrics = calculate_metrics(
            y_test,
            y_pred,
            y_prob
        )

        auc_lower, auc_upper = bootstrap_auc_ci(
            y_test,
            y_prob
        )

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_prob": y_prob,
            # ADDED: full training out-of-fold probabilities (not just
            # the external-test y_prob above). Lets us compare the
            # train-time probability distribution against the external
            # one directly -- e.g. to check for the kind of extreme
            # near-0/near-1 collapse seen in the false-negative table.
            "train_oof_prob": oof_probs,
            "threshold": threshold,
            "CV_F1_Mean": cv_scores.mean(),
            "CV_F1_STD": cv_scores.std(),
            "AUC_CI_Lower": auc_lower,
            "AUC_CI_Upper": auc_upper,
            **metrics
        }

    return results


# =====================================================
# NESTED CV / ABLATION EVALUATION
#
# FIX (Reviewer 1 & 2): DEA / mRMR / LASSO were previously
# fit ONCE on the entire training set, and cross-validation
# only ran on the already-selected genes. That lets every
# CV fold benefit from gene-selection decisions that used
# its own held-out samples -- which is why CV F1 sat near
# 0.99 while external F1 was 0.86. Here, feature selection
# is repeated INSIDE every fold, using only that fold's
# training split, so the reported score is an honest
# estimate of generalization. This function is also reused
# for the ablation study (Reviewer 2, point 1): pass a
# subset of stages to see each stage's contribution.
# =====================================================
def nested_cv_pipeline_eval(
    X,
    y,
    stages,
    logfc_thresh,
    pval_thresh,
    mrmr_k,
    cv
):
    model_names = list(get_model_defs().keys())
    fold_scores = {m: [] for m in model_names}
    n_features_per_fold = []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        feats = X_tr.columns.tolist()

        if "dea" in stages:
            feats, _ = differential_expression(
                X_tr[feats], y_tr, logfc_thresh, pval_thresh
            )

        if "mrmr" in stages:
            feats = mrmr_selection(X_tr[feats], y_tr, mrmr_k)

        if "lasso" in stages:
            feats = lasso_selection(X_tr[feats], y_tr)

        if len(feats) == 0:
            feats = X_tr.columns[:10].tolist()

        n_features_per_fold.append(len(feats))

        X_tr_f, X_val_f = X_tr[feats], X_val[feats]

        for name, model in get_model_defs().items():
            model.fit(X_tr_f, y_tr)
            pred = model.predict(X_val_f)
            fold_scores[name].append(
                f1_score(y_val, pred, zero_division=0)
            )

    return fold_scores, n_features_per_fold


STAGE_LABELS = {
    ("dea",): "DEA only",
    ("dea", "mrmr"): "DEA + mRMR",
    ("dea", "mrmr", "lasso"): "DEA + mRMR + LASSO (Full)"
}


# =====================================================
# PLOTS
# =====================================================
def plot_confusion(y_true, y_pred):
    fig, ax = plt.subplots()

    sns.heatmap(
        confusion_matrix(y_true, y_pred),
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    return fig


def plot_roc(y_true, y_prob):
    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )

    fig, ax = plt.subplots()

    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], "--")

    return fig


# =====================================================
# SIDEBAR
# =====================================================
impute_k = st.sidebar.slider("KNN K", 1, 10, 5)

# CHANGED: default 0.5 -> 1.0
logfc_thresh = st.sidebar.slider(
    "logFC",
    0.1,
    2.0,
    1.0
)

pval_thresh = st.sidebar.slider(
    "p-value",
    0.01,
    0.10,
    0.05
)

mrmr_k = st.sidebar.slider(
    "mRMR K",
    10,
    100,
    50
)

st.sidebar.markdown("---")

run_nested_cv = st.sidebar.checkbox(
    "Run honest nested CV + ablation study",
    value=True,
    help=(
        "Repeats DEA/mRMR/LASSO feature selection inside "
        "every CV fold instead of once before CV. This is "
        "slower (re-runs mRMR/LASSO several times) but gives "
        "an unbiased CV estimate and an ablation breakdown "
        "per pipeline stage, as requested by Reviewer 2."
    )
)

run = st.sidebar.button("Run Pipeline")


# =====================================================
# MAIN
# =====================================================
if run:
    data_dir = "data"

    expr1, y1 = parse_series_matrix(
        FileLike(os.path.join(data_dir, "GSE76297_series_matrix.txt"))
    )
    expr1 = auto_log2_transform(expr1, "GSE76297")

    expr2, y2 = parse_series_matrix(
        FileLike(os.path.join(data_dir, "GSE132305_series_matrix.txt"))
    )
    expr2 = auto_log2_transform(expr2, "GSE132305")

    expr3, y3 = parse_series_matrix(
        FileLike(os.path.join(data_dir, "GSE32225_series_matrix.txt"))
    )
    expr3 = auto_log2_transform(expr3, "GSE32225")

    map1 = load_annotation(
        os.path.join(data_dir, "GPL17586.txt"),
        "GPL17586"
    )

    map2 = load_annotation(
        os.path.join(data_dir, "GPL13667.txt"),
        "GPL13667"
    )

    map3 = load_annotation(
        os.path.join(data_dir, "GPL8432.txt"),
        "GPL8432"
    )

    # =====================================================
    # DATASET SUMMARY (Reviewer 1 & 2: dataset description
    # was insufficient -- report sample counts, class
    # balance, and gene counts at every stage)
    # =====================================================
    dataset_summary_rows = [
        {
            "Dataset": "GSE76297 (train)",
            "N samples": len(y1),
            "N CCA": int((y1 == 1).sum()),
            "N normal": int((y1 == 0).sum()),
            "N genes (after probe->gene mapping)": expr1.shape[1]
        },
        {
            "Dataset": "GSE132305 (train)",
            "N samples": len(y2),
            "N CCA": int((y2 == 1).sum()),
            "N normal": int((y2 == 0).sum()),
            "N genes (after probe->gene mapping)": expr2.shape[1]
        },
        {
            "Dataset": "GSE32225 (external validation)",
            "N samples": len(y3),
            "N CCA": int((y3 == 1).sum()),
            "N normal": int((y3 == 0).sum()),
            "N genes (after probe->gene mapping)": expr3.shape[1]
        }
    ]

    expr1 = convert_probe_to_gene(expr1, map1)
    expr2 = convert_probe_to_gene(expr2, map2)
    expr3 = convert_probe_to_gene(expr3, map3)

    common_genes = (
        expr1.columns
        .intersection(expr2.columns)
        .intersection(expr3.columns)
    )

    X_train = pd.concat([
        expr1[common_genes],
        expr2[common_genes]
    ])

    y_train = pd.concat([y1, y2])

    X_test = expr3[common_genes]
    y_test = y3

    n_missing_train = int(X_train.isna().sum().sum())
    n_missing_test = int(X_test.isna().sum().sum())

    # =====================================================
    # REVERTED BACK (2026-08-11, after user confirmed
    # regression): fitting ComBat on train-only (batch1+batch2)
    # while leaving external (batch3/GSE32225) uncorrected
    # caused exactly the failure mode warned about above --
    # confirmed by the "149/149 CCA samples missed" result
    # across all three models (Specificity=1, Recall=0
    # everywhere) and the before/after ComBat PCA plot showing
    # batch3 sitting completely apart from train even "after"
    # correction. Restoring the fix: run ComBat across all
    # THREE datasets together as three batches, BEFORE the
    # train/test split is used for anything supervised. This
    # is still leakage-free: ComBat only uses batch identity +
    # expression values, never the class label y. Downstream
    # steps that must stay train-only (DEA/mRMR/LASSO feature
    # selection, Youden/sensitivity threshold selection) are
    # unaffected -- they still only ever see X_train after
    # this point.
    # =====================================================
    batch_labels = (
        ["batch1"] * len(expr1)
        +
        ["batch2"] * len(expr2)
        +
        ["batch3"] * len(expr3)
    )

    X_all = pd.concat([
        expr1[common_genes],
        expr2[common_genes],
        expr3[common_genes]
    ])

    # Kept for the before/after ComBat visualization below.
    X_train_pre_combat = X_all.loc[X_train.index].copy()
    X3_pre_combat = X_all.loc[X_test.index].copy()

    X_all_combat = pycombat(
        X_all.T,
        batch_labels
    ).T

    X_train = X_all_combat.loc[X_train.index]
    X_test = X_all_combat.loc[X_test.index]

    # =====================================================
    # CHANGED:
    # KNN IMPUTER FIRST
    # (still fit on X_train ONLY, applied/transformed to
    # X_test -- this part was already correct and is
    # unchanged)
    # =====================================================
    imputer = KNNImputer(
        n_neighbors=impute_k
    )

    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # =====================================================
    # CHANGED:
    # STANDARD SCALER AFTER IMPUTATION
    # =====================================================
    scaler = StandardScaler()

    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    dea_genes, dea_df = differential_expression(
        X_train,
        y_train,
        logfc_thresh,
        pval_thresh
    )

    mrmr_genes = mrmr_selection(
        X_train[dea_genes],
        y_train,
        mrmr_k
    )

    final_features = lasso_selection(
        X_train[mrmr_genes],
        y_train
    )

    X_train_final = X_train[final_features]
    X_test_final = X_test[final_features]

    # Single cv object reused everywhere below so that the
    # naive CV, the nested CV, and the ablation study all
    # split the SAME folds (same random_state/shuffle over
    # the same y_train) -- this is what makes the paired
    # significance test between models valid.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = train_models(
        X_train_final,
        y_train,
        X_test_final,
        y_test,
        cv
    )
    
    st.markdown("---")
    st.header("🎛️ Hyperparameter Tuning (Reviewer 2, poin 6)")
    st.caption(
        "GridSearchCV dijalankan di dalam training set saja (5-fold CV, "
        "scoring=F1), tidak menyentuh data validasi eksternal."
    )
    
    param_grids = {
        "SVM": {
            "estimator__C": [0.01, 0.1, 1, 10]
        },
        "RandomForest": {
            "estimator__n_estimators": [200, 500, 1000],
            "estimator__max_depth": [5, 10, None]
        },
        "LogisticRegression": {
            "C": [0.01, 0.1, 1, 10]
        }
    }
    
    model_defs = get_model_defs()
    tuning_rows = []
    
    for model_name, base_model in model_defs.items():
        grid = param_grids[model_name]
        gs = GridSearchCV(
            base_model,
            grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1
        )
        gs.fit(X_train_final, y_train)
        tuning_rows.append({
            "Model": model_name,
            "Best params": str(gs.best_params_),
            "Best CV F1 (training only)": round(gs.best_score_, 4),
            "Default-config CV F1": round(results[model_name]["CV_F1_Mean"], 4)
        })
    
    st.dataframe(pd.DataFrame(tuning_rows).set_index("Model"))
    st.caption(
        "Bandingkan kolom terakhir: kalau selisihnya kecil, konfigurasi manual "
        "di paper sudah cukup dekat optimal dan bisa dijelaskan sebagai "
        "'grid search dijalankan, hasil serupa dengan setting awal'. Kalau "
        "selisih besar, pertimbangkan pakai Best params untuk hasil final "
        "dan laporkan proses grid search ini di Methodology."
    )

    metrics_df = pd.DataFrame(results).T.drop(
        columns=["model", "y_pred", "y_prob"]
    )

    st.subheader("Model Performance")
    st.dataframe(metrics_df)

    st.subheader("Cross Validation")
    for model_name, result in results.items():
        st.write(
            f"{model_name}: "
            f"{result['CV_F1_Mean']:.4f} ± {result['CV_F1_STD']:.4f}"
        )

    st.subheader("AUC 95% Confidence Interval")
    for model_name, result in results.items():
        st.write(
            f"{model_name}: "
            f"{result['AUC_CI_Lower']:.4f} - {result['AUC_CI_Upper']:.4f}"
        )

    # =====================================================
    # HONEST NESTED CV + ABLATION STUDY
    # (Reviewer 1: explain the CV vs external-validation gap
    #  Reviewer 2, point 1: ablation study per pipeline stage)
    # =====================================================
    ablation_fold_scores = {}

    if run_nested_cv:
        st.markdown("---")
        st.header("🔎 Honest Nested CV & Ablation Study")
        st.caption(
            "Feature selection (DEA/mRMR/LASSO) is repeated "
            "inside every fold here, using only that fold's "
            "training split -- unlike the naive CV above, "
            "where genes were selected once on the full "
            "training set before splitting into folds."
        )

        stage_progress = st.progress(0.0, text="Running nested CV...")
        stage_configs = list(STAGE_LABELS.keys())

        for i, stages in enumerate(stage_configs):
            fold_scores, n_feats = nested_cv_pipeline_eval(
                X_train,
                y_train,
                stages,
                logfc_thresh,
                pval_thresh,
                mrmr_k,
                cv
            )
            ablation_fold_scores[stages] = {
                "fold_scores": fold_scores,
                "n_features": n_feats
            }
            stage_progress.progress(
                (i + 1) / len(stage_configs),
                text=f"Completed: {STAGE_LABELS[stages]}"
            )

        stage_progress.empty()

        # --- CV vs honest nested CV vs external F1, side by side ---
        full_stage = ("dea", "mrmr", "lasso")
        gap_rows = []
        for model_name in get_model_defs().keys():
            naive_cv = results[model_name]["CV_F1_Mean"]
            nested = np.mean(
                ablation_fold_scores[full_stage]["fold_scores"][model_name]
            )
            external = results[model_name]["F1"]
            gap_rows.append({
                "Model": model_name,
                "Naive CV F1 (pre-selected genes)": naive_cv,
                "Honest nested CV F1 (re-selected per fold)": nested,
                "External validation F1": external
            })

        st.subheader("📉 CV vs External Validation Gap")
        st.dataframe(pd.DataFrame(gap_rows).set_index("Model"))
        st.caption(
            "The naive CV column reproduces the inflated ~0.99 "
            "scores seen in the original submission. The nested "
            "column is the honest estimate and should sit much "
            "closer to the external validation column."
        )

        # --- Ablation table ---
        st.subheader("🧪 Ablation Study: Contribution of Each Stage")
        ablation_rows = []
        for stages in stage_configs:
            row = {"Pipeline stage": STAGE_LABELS[stages]}
            row["Avg. genes selected"] = np.mean(
                ablation_fold_scores[stages]["n_features"]
            )
            for model_name in get_model_defs().keys():
                scores = ablation_fold_scores[stages]["fold_scores"][model_name]
                row[f"{model_name} F1 (mean ± std)"] = (
                    f"{np.mean(scores):.4f} ± {np.std(scores):.4f}"
                )
            ablation_rows.append(row)

        st.dataframe(pd.DataFrame(ablation_rows).set_index("Pipeline stage"))

        # --- Statistical significance test ---
        st.subheader("📐 Statistical Significance (paired, nested-CV folds)")
        lr_scores = np.array(
            ablation_fold_scores[full_stage]["fold_scores"]["LogisticRegression"]
        )
        sig_rows = []
        for model_name in ["SVM", "RandomForest"]:
            other_scores = np.array(
                ablation_fold_scores[full_stage]["fold_scores"][model_name]
            )
            t_stat, p_val = stats.ttest_rel(lr_scores, other_scores)
            sig_rows.append({
                "Comparison": f"LogisticRegression vs {model_name}",
                "Mean F1 diff": lr_scores.mean() - other_scores.mean(),
                "Paired t-statistic": t_stat,
                "p-value": p_val,
                "Significant (p<0.05)": "Yes" if p_val < 0.05 else "No"
            })

        svm_scores = np.array(
            ablation_fold_scores[full_stage]["fold_scores"]["SVM"]
        )
        rf_scores = np.array(
            ablation_fold_scores[full_stage]["fold_scores"]["RandomForest"]
        )
        t_stat_svm_rf, p_val_svm_rf = stats.ttest_rel(svm_scores, rf_scores)
        sig_rows.append({
            "Comparison": "SVM vs RandomForest",
            "Mean F1 diff": svm_scores.mean() - rf_scores.mean(),
            "Paired t-statistic": t_stat_svm_rf,
            "p-value": p_val_svm_rf,
            "Significant (p<0.05)": "Yes" if p_val_svm_rf < 0.05 else "No"
        })
        st.dataframe(pd.DataFrame(sig_rows).set_index("Comparison"))
        st.caption(
            "Paired t-test across the 5 nested-CV folds (same "
            "folds for every model). With only 5 folds this test "
            "has low power -- treat the p-value as indicative, "
            "not definitive, and report it in the paper as such."
        )
    else:
        st.info(
            "Honest nested CV + ablation study was skipped "
            "(disabled in the sidebar)."
        )

    best_model_name = max(
        results.items(),
        key=lambda x: x[1]["AUC"]
    )[0]

    best_model = results[best_model_name]["model"]

    st.subheader(f"SHAP Analysis ({best_model_name})")

    # CHANGED: RandomForest is now wrapped in CalibratedClassifierCV
    # (for calibrated predict_proba -- see get_model_defs), so
    # shap.TreeExplainer no longer applies to it (it only supports
    # raw tree estimators, not the calibration wrapper). Always use
    # the generic black-box explainer via predict_proba instead --
    # it works for any model type (SVM, RandomForest, LogisticRegression,
    # wrapped or not), just slower than TreeExplainer would have been.
    try:
        explainer = shap.Explainer(
            best_model.predict_proba,
            X_train_final
        )

        shap_values = explainer(X_test_final)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values[:, :, 1],
            X_test_final,
            show=False
        )
        st.pyplot(plt.gcf())
        plt.clf()

    except Exception as e:
        st.error(f"SHAP Error: {e}")

    st.subheader("Selected Genes")
    st.write(final_features)

    # =====================================================
    # FALSE NEGATIVE ANALYSIS
    # (Reviewer 1: missed CCA cases are clinically serious --
    # analyse them more seriously instead of a passing mention)
    # =====================================================
    st.markdown("---")
    st.header("⚠️ False Negative Analysis (missed CCA cases)")

    for model_name, res in results.items():
        fn_mask = (y_test.values == 1) & (res["y_pred"] == 0)
        n_fn = int(fn_mask.sum())
        n_cca = int((y_test.values == 1).sum())

        st.subheader(f"{model_name}: {n_fn} / {n_cca} CCA samples missed")

        if n_fn > 0:
            fn_df = pd.DataFrame({
                "Sample ID": y_test.index[fn_mask],
                "Predicted P(CCA)": res["y_prob"][fn_mask],
                "Decision threshold used": res["threshold"]
            }).sort_values("Predicted P(CCA)", ascending=False)

            st.dataframe(fn_df.set_index("Sample ID"))
            st.caption(
                "Samples with predicted probability close to the "
                "threshold are borderline misses; samples with a "
                "low probability are confidently missed and worth "
                "flagging as biologically atypical CCA cases in "
                "the paper's Discussion."
            )

    st.markdown("---")
    st.header("📊 Additional Analytics Dashboard")

    # --- ComBat before/after batch-effect visualization ---
    # (Reviewer 2, point 5: visualize before/after ComBat)
    st.subheader("🧫 Batch Effect: Before vs After ComBat")

    # Shows all THREE batches (train batch1/batch2 + external
    # batch3), since ComBat is fit across all three datasets
    # together -- this plot is the direct visual check for
    # whether GSE32225 actually mixes in with train after
    # correction. If it doesn't mix (batch3 still forms its own
    # cluster in "After ComBat"), expect the external-set
    # predict_proba collapse seen in the resultv8 run.
    viz_imputer = SimpleImputer(strategy="median")

    X_all_pre_viz = pd.concat([X_train_pre_combat, X3_pre_combat])
    X_all_post_viz = pd.concat([X_train, X_test])

    X_pre_viz = pd.DataFrame(
        viz_imputer.fit_transform(X_all_pre_viz),
        columns=X_all_pre_viz.columns,
        index=X_all_pre_viz.index
    )

    X_post_viz = pd.DataFrame(
        viz_imputer.fit_transform(X_all_post_viz),
        columns=X_all_post_viz.columns,
        index=X_all_post_viz.index
    )

    pca_pre = PCA(n_components=2).fit_transform(X_pre_viz)
    pca_post = PCA(n_components=2).fit_transform(X_post_viz)

    batch_arr = np.array(batch_labels)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for batch in sorted(set(batch_labels)):
        mask = batch_arr == batch
        axes[0].scatter(
            pca_pre[mask, 0], pca_pre[mask, 1], label=batch, alpha=0.7
        )
        axes[1].scatter(
            pca_post[mask, 0], pca_post[mask, 1], label=batch, alpha=0.7
        )

    axes[0].set_title("Before ComBat")
    axes[1].set_title("After ComBat")
    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()

    st.pyplot(fig)
    st.caption(
        "Values are median-imputed here purely for this plot "
        "(separate from the pipeline's KNN imputer) so PCA can "
        "run. If ComBat worked, the two batches should mix "
        "together more after correction than before."
    )

    st.subheader("📌 Model Performance Heatmap")

    plot_df = pd.DataFrame({
        model: {
            "Accuracy": res["Accuracy"],
            "Precision": res["Precision"],
            "Recall": res["Recall"],
            "Specificity": res["Specificity"],
            "F1": res["F1"],
            "AUC": res["AUC"]
        }
        for model, res in results.items()
    }).T

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(plot_df, annot=True, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

    st.subheader("📈 ROC Curve Comparison (All Models)")

    fig, ax = plt.subplots()

    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['AUC']:.3f})")

    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

    st.pyplot(fig)

    st.subheader("🧩 Confusion Matrix Grid View")

    cols = st.columns(len(results))

    for i, (name, res) in enumerate(results.items()):
        with cols[i]:
            st.markdown(f"**{name}**")
            fig, ax = plt.subplots()
            sns.heatmap(
                confusion_matrix(y_test, res["y_pred"]),
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax
            )
            st.pyplot(fig)

    st.subheader("🧬 PCA Global Structure View")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_final)

    fig, ax = plt.subplots()
    ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y_train,
        cmap="coolwarm",
        alpha=0.7
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    st.pyplot(fig)

    st.subheader("🔥 Top Gene Correlation Map")

    top_n = min(20, len(final_features))

    # Membuat figure dengan ukuran yang sedikit lebih besar agar angka tidak tumpang tindih
    fig, ax = plt.subplots(figsize=(12, 8)) 

    sns.heatmap(
        X_train_final[final_features[:top_n]].corr(),
        annot=True,          # Menampilkan angka di dalam kotak
        fmt=".2f",           # Format 2 angka di belakang koma
        cmap="coolwarm",
        annot_kws={"size": 8}, # Opsional: mengecilkan ukuran font angka agar pas di kotak
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("🧠 Feature Usage Overview")

    importance_df = pd.DataFrame({
        "Gene": final_features
    })

    st.dataframe(importance_df)

    st.subheader("📌 Dataset Summary")

    st.markdown("**Per-dataset sample and gene counts** (Reviewer 1 & 2)")
    st.dataframe(
        pd.DataFrame(dataset_summary_rows).set_index("Dataset")
    )

    st.markdown("**Gene counts through the feature-selection pipeline**")
    st.dataframe(pd.DataFrame([{
        "Common genes across all 3 datasets": len(common_genes),
        "Missing values imputed (train)": n_missing_train,
        "Missing values imputed (external test)": n_missing_test,
        "Genes after DEA": len(dea_genes),
        "Genes after mRMR": len(mrmr_genes),
        "Genes after LASSO (final)": len(final_features)
    }]).T.rename(columns={0: "Count"}))

    st.write("Train shape:", X_train_final.shape)
    st.write("Test shape:", X_test_final.shape)
    st.write("Number of selected genes:", len(final_features))
    st.write("Class distribution (train):")
    st.write(y_train.value_counts())