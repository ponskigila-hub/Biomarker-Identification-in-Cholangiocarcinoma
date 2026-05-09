import streamlit as st
import pandas as pd
import numpy as np
import gzip
import os
import shap

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.base import clone
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score
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

        if t.endswith("_bd"):
            labels[sid] = 0
        elif t.endswith("_ecca"):
            labels[sid] = 1
        elif t.startswith("ctrl"):
            labels[sid] = 0
        elif (
            t.startswith("ccbcn")
            or t.startswith("ccm")
            or t.startswith("ccny")
        ):
            labels[sid] = 1
        elif any(k in t for k in ["normal", "control", "benign"]):
            labels[sid] = 0
        elif any(k in t for k in ["tumor", "cca", "cancer"]):
            labels[sid] = 1
        else:
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
# TRAIN MODELS
# =====================================================
def train_models(
    X_train,
    y_train,
    X_test,
    y_test
):
    models = {
        "SVM": SVC(
            kernel="linear",
            C=0.1,
            probability=True,
            class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=1000,
            max_depth=10,
            random_state=42
        ),
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=5000
        )
    }

    results = {}

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    for name, model in models.items():

        cv_scores = cross_val_score(
            clone(model),
            X_train,
            y_train,
            cv=cv,
            scoring="f1"
        )

        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]

        fpr, tpr, thresholds = roc_curve(
            y_test,
            y_prob
        )

        threshold = thresholds[
            np.argmax(tpr - fpr)
        ]

        y_pred = (
            y_prob >= threshold
        ).astype(int)

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
            "CV_F1_Mean": cv_scores.mean(),
            "CV_F1_STD": cv_scores.std(),
            "AUC_CI_Lower": auc_lower,
            "AUC_CI_Upper": auc_upper,
            **metrics
        }

    return results


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

run = st.sidebar.button("Run Pipeline")


# =====================================================
# MAIN
# =====================================================
if run:
    data_dir = "data"

    expr1, y1 = parse_series_matrix(
        FileLike(os.path.join(data_dir, "GSE76297_series_matrix.txt"))
    )

    expr2, y2 = parse_series_matrix(
        FileLike(os.path.join(data_dir, "GSE132305_series_matrix.txt"))
    )

    expr3, y3 = parse_series_matrix(
        FileLike(os.path.join(data_dir, "GSE32225_series_matrix.txt"))
    )

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

    batch_labels = (
        ["batch1"] * len(expr1)
        +
        ["batch2"] * len(expr2)
    )

    X_train = pycombat(
        X_train.T,
        batch_labels
    ).T

    # =====================================================
    # CHANGED:
    # KNN IMPUTER FIRST
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

    results = train_models(
        X_train_final,
        y_train,
        X_test_final,
        y_test
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

    best_model_name = max(
        results.items(),
        key=lambda x: x[1]["AUC"]
    )[0]

    best_model = results[best_model_name]["model"]

    st.subheader(f"SHAP Analysis ({best_model_name})")

    try:
        if best_model_name == "RandomForest":
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test_final)

            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values[1],
                X_test_final,
                show=False
            )
            st.pyplot(plt.gcf())
            plt.clf()

        else:
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

    st.markdown("---")
    st.header("📊 Additional Analytics Dashboard")

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

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        X_train_final[final_features[:top_n]].corr(),
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("🧠 Feature Usage Overview")

    importance_df = pd.DataFrame({
        "Gene": final_features
    })

    st.dataframe(importance_df)

    st.subheader("📌 Dataset Summary")

    st.write("Train shape:", X_train_final.shape)
    st.write("Test shape:", X_test_final.shape)
    st.write("Number of selected genes:", len(final_features))
    st.write("Class distribution:")
    st.write(y_train.value_counts())
