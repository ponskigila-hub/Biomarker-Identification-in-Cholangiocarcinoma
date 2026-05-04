import streamlit as st
import pandas as pd
import numpy as np
import gzip
import os

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve
)

from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CCA Biomarker Discovery",
    layout="wide"
)

st.title("🔬 Hybrid Feature Selection for Cholangiocarcinoma")

st.markdown("""
### Workflow

Automatically reading GEO Series Matrix files from `data/`

Training:
- GSE76297
- GSE132305

Testing:
- GSE32225

Pipeline:

Expression Parsing → Label Extraction → Probe Mapping → DEA → mRMR → LASSO → Classification
""")


# =========================================================
# FILE WRAPPER
# =========================================================
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


# =========================================================
# LABEL EXTRACTION
# =========================================================
def extract_labels(lines, sample_ids):

    sample_titles = []

    for line in lines:
        if line.startswith("!Sample_title"):
            parts = line.strip().split("\t")[1:]
            sample_titles = [x.strip('"') for x in parts]
            break

    if len(sample_titles) == 0:
        st.error("No !Sample_title row found.")
        return None

    labels = {}

    for sid, title in zip(sample_ids, sample_titles):

        title_lower = title.lower()

        # GSE32225 special
        if title_lower.startswith("ctrl"):
            labels[sid] = 0

        elif (
            title_lower.startswith("ccbcn")
            or title_lower.startswith("ccm")
            or title_lower.startswith("ccny")
        ):
            labels[sid] = 1

        # generic normal
        elif any(keyword in title_lower for keyword in [
            "normal",
            "control",
            "non-tumor",
            "adjacent normal"
        ]):
            labels[sid] = 0

        # generic tumor
        elif any(keyword in title_lower for keyword in [
            "tumor",
            "cca",
            "cholangiocarcinoma",
            "cancer"
        ]):
            labels[sid] = 1

        else:
            labels[sid] = -1

    return labels


# =========================================================
# PARSE SERIES MATRIX
# =========================================================
def parse_series_matrix(file_obj):

    if file_obj.name.endswith(".gz"):
        content = gzip.decompress(
            file_obj.read()
        ).decode("utf-8")
    else:
        content = file_obj.read().decode("utf-8")

    lines = content.splitlines()

    start_idx = None

    for i, line in enumerate(lines):
        if line.startswith("!series_matrix_table_begin"):
            start_idx = i + 1
            break

    if start_idx is None:
        st.error("Could not find expression table.")
        return None, None

    headers = [
        x.strip('"')
        for x in lines[start_idx].split("\t")
    ]

    sample_ids = headers[1:]

    st.info(f"Found {len(sample_ids)} samples")

    rows = []

    for line in lines[start_idx + 1:]:

        if line.startswith("!series_matrix_table_end"):
            break

        fields = [
            x.strip('"')
            for x in line.split("\t")
        ]

        if len(fields) != len(headers):
            continue

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

    labels = extract_labels(
        lines,
        sample_ids
    )

    valid_samples = [
        s for s in sample_ids
        if labels[s] != -1
    ]

    expr_df = expr_df.loc[valid_samples]

    y = pd.Series(
        [labels[s] for s in valid_samples],
        index=valid_samples
    )

    st.success(
        f"Parsed {expr_df.shape[0]} samples and {expr_df.shape[1]} probes."
    )

    preview_df = pd.DataFrame({
        "Sample_ID": valid_samples,
        "Label": y.values
    })

    st.dataframe(preview_df)

    return expr_df, y


# =========================================================
# LOAD GPL ANNOTATION
# =========================================================
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
            lambda x: str(x).split(" // ")[1]
            if " // " in str(x)
            else np.nan
        )

        mapping = ann[["ID", "Gene Symbol"]].dropna()

    else:
        return {}

    mapping.columns = ["probe", "gene"]

    mapping["gene"] = (
        mapping["gene"]
        .astype(str)
        .str.strip()
    )

    return dict(
        zip(
            mapping["probe"],
            mapping["gene"]
        )
    )


# =========================================================
# PROBE TO GENE CONVERSION
# =========================================================
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

    # merge duplicate genes
    expr_df = (
        expr_df.T
        .groupby(level=0)
        .mean()
        .T
    )

    return expr_df


# =========================================================
# DEA
# =========================================================
def differential_expression(
    X,
    y,
    logfc_thresh=1.0,
    pval_thresh=0.05
):

    results = []

    for gene in X.columns:

        tumor_idx = y[y == 1].index
        normal_idx = y[y == 0].index

        tumor = X.loc[tumor_idx, gene]
        normal = X.loc[normal_idx, gene]

        if len(tumor) < 2 or len(normal) < 2:
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

        results.append(
            [gene, logfc, pval]
        )

    if len(results) == 0:
        return []

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
        (abs(res_df["logFC"]) > logfc_thresh)
        &
        (res_df["adj_p"] < pval_thresh)
    ]

    return sig["gene"].tolist()


# =========================================================
# MRMR
# =========================================================
def mrmr_selection(X, y, k=20):

    mi = mutual_info_classif(
        X,
        y,
        random_state=42
    )

    mi_df = pd.DataFrame({
        "feature": X.columns,
        "mi": mi
    }).sort_values(
        "mi",
        ascending=False
    )

    return mi_df.head(k)["feature"].tolist()


# =========================================================
# LASSO
# =========================================================
def lasso_selection(X, y, C=0.1):

    model = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=C,
        random_state=42,
        max_iter=3000
    )

    model.fit(X, y)

    selected = X.columns[
        model.coef_[0] != 0
    ].tolist()

    return selected


# =========================================================
# TRAIN MODELS
# =========================================================
def train_models(
    X_train,
    y_train,
    X_test,
    y_test
):

    models = {
        "SVM": SVC(
            probability=True,
            class_weight="balanced",
            random_state=42
        ),

        "RandomForest": RandomForestClassifier(
            class_weight="balanced",
            random_state=42
        ),

        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            random_state=42,
            max_iter=3000
        )
    }

    results = {}

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(
            X_test
        )

        y_prob = model.predict_proba(
            X_test
        )[:, 1]

        results[name] = {
            "Accuracy":
                accuracy_score(
                    y_test,
                    y_pred
                ),

            "Precision":
                precision_score(
                    y_test,
                    y_pred
                ),

            "Recall":
                recall_score(
                    y_test,
                    y_pred
                ),

            "F1":
                f1_score(
                    y_test,
                    y_pred
                ),

            "AUC":
                roc_auc_score(
                    y_test,
                    y_prob
                ),

            "MCC":
                matthews_corrcoef(
                    y_test,
                    y_pred
                ),

            "y_pred": y_pred,
            "y_prob": y_prob
        }

    return results


# =========================================================
# PLOTS
# =========================================================
def plot_confusion(y_true, y_pred):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
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
    ax.plot(
        [0, 1],
        [0, 1],
        "--"
    )

    return fig


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Parameters")

impute_k = st.sidebar.slider(
    "KNN K",
    1,
    10,
    5
)

logfc_thresh = st.sidebar.slider(
    "logFC Threshold",
    1.0,
    3.0,
    1.5
)

pval_thresh = st.sidebar.slider(
    "Adjusted p-value",
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

lasso_c = st.sidebar.slider(
    "LASSO C",
    0.01,
    1.0,
    0.1
)

run = st.sidebar.button(
    "🚀 Run Pipeline"
)


# =========================================================
# MAIN PIPELINE
# =========================================================
if run:

    data_dir = "data"

    expr1, y1 = parse_series_matrix(
        FileLike(
            os.path.join(
                data_dir,
                "GSE76297_series_matrix.txt"
            )
        )
    )

    expr2, y2 = parse_series_matrix(
        FileLike(
            os.path.join(
                data_dir,
                "GSE132305_series_matrix.txt"
            )
        )
    )

    expr3, y3 = parse_series_matrix(
        FileLike(
            os.path.join(
                data_dir,
                "GSE32225_series_matrix.txt"
            )
        )
    )

    map1 = load_annotation(
        os.path.join(
            data_dir,
            "GPL17586.txt"
        ),
        "GPL17586"
    )

    map2 = load_annotation(
        os.path.join(
            data_dir,
            "GPL13667.txt"
        ),
        "GPL13667"
    )

    map3 = load_annotation(
        os.path.join(
            data_dir,
            "GPL8432.txt"
        ),
        "GPL8432"
    )

    expr1 = convert_probe_to_gene(
        expr1,
        map1
    )

    expr2 = convert_probe_to_gene(
        expr2,
        map2
    )

    expr3 = convert_probe_to_gene(
        expr3,
        map3
    )

    common_genes = (
        expr1.columns
        .intersection(expr2.columns)
        .intersection(expr3.columns)
    )

    st.success(
        f"Common genes: {len(common_genes)}"
    )

    if len(common_genes) == 0:
        st.error(
            "No common genes found."
        )
        st.stop()

    X_train = pd.concat([
        expr1[common_genes],
        expr2[common_genes]
    ])

    y_train = pd.concat([
        y1,
        y2
    ])

    X_test = expr3[
        common_genes
    ]

    y_test = y3

    # KNN IMPUTE (FIX INDEX)
    imputer = KNNImputer(
        n_neighbors=impute_k
    )

    X_train = pd.DataFrame(
        imputer.fit_transform(
            X_train
        ),
        index=X_train.index,
        columns=X_train.columns
    )

    X_test = pd.DataFrame(
        imputer.transform(
            X_test
        ),
        index=X_test.index,
        columns=X_test.columns
    )

    # SCALE (FIX INDEX)
    scaler = MinMaxScaler()

    X_train = pd.DataFrame(
        scaler.fit_transform(
            X_train
        ),
        index=X_train.index,
        columns=X_train.columns
    )

    X_test = pd.DataFrame(
        scaler.transform(
            X_test
        ),
        index=X_test.index,
        columns=X_test.columns
    )

    # DEA
    st.subheader(
        "Differential Expression Analysis"
    )

    dea_genes = differential_expression(
        X_train,
        y_train,
        logfc_thresh,
        pval_thresh
    )

    st.write(
        f"DEA selected: {len(dea_genes)}"
    )

    X_train_dea = X_train[
        dea_genes
    ]

    X_test_dea = X_test[
        dea_genes
    ]

    # MRMR
    st.subheader(
        "mRMR Selection"
    )

    mrmr_genes = mrmr_selection(
        X_train_dea,
        y_train,
        min(
            mrmr_k,
            len(dea_genes)
        )
    )

    st.write(
        f"mRMR selected: {len(mrmr_genes)}"
    )

    # LASSO
    st.subheader(
        "LASSO Selection"
    )

    lasso_genes = lasso_selection(
        X_train[mrmr_genes],
        y_train,
        lasso_c
    )

    final_features = (
        lasso_genes
        if len(lasso_genes) > 0
        else mrmr_genes
    )

    st.success(
        f"Final selected genes: {len(final_features)}"
    )

    # TRAIN
    st.subheader(
        "Model Performance"
    )

    results = train_models(
        X_train[final_features],
        y_train,
        X_test[final_features],
        y_test
    )

    metrics_df = pd.DataFrame(
        results
    ).T.drop(
        columns=[
            "y_pred",
            "y_prob"
        ]
    )

    st.dataframe(
        metrics_df
    )

    for model_name, result in results.items():

        st.subheader(
            model_name
        )

        col1, col2 = st.columns(2)

        with col1:
            st.pyplot(
                plot_confusion(
                    y_test,
                    result["y_pred"]
                )
            )

        with col2:
            st.pyplot(
                plot_roc(
                    y_test,
                    result["y_prob"]
                )
            )

    st.download_button(
        "📥 Download Selected Genes",
        "\n".join(
            final_features
        ),
        "selected_genes.txt"
    )
