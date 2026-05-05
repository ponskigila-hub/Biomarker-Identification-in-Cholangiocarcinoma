import streamlit as st
import pandas as pd
import numpy as np
import gzip
import os

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample          # <-- NEW import for downsampling
from imblearn.over_sampling import SMOTE

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

Expression Parsing → Label Extraction → Probe Mapping → DEA → mRMR → LASSO → (Downsampling + SMOTE) → Classification
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

        if title_lower.startswith("ctrl"):
            labels[sid] = 0

        elif (
            title_lower.startswith("ccbcn")
            or title_lower.startswith("ccm")
            or title_lower.startswith("ccny")
        ):
            labels[sid] = 1

        elif any(keyword in title_lower for keyword in [
            "normal",
            "control",
            "non-tumor",
            "adjacent normal"
        ]):
            labels[sid] = 0

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
# PROBE TO GENE
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

    tumor_idx = y[y == 1].index.intersection(X.index)
    normal_idx = y[y == 0].index.intersection(X.index)

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

        results.append(
            [gene, logfc, pval]
        )

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

    if len(sig) < 20:
        sig = res_df.nsmallest(
            50,
            "adj_p"
        )

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
def lasso_selection(X, y):

    model = LogisticRegressionCV(
        penalty="l1",
        solver="saga",
        cv=5,
        random_state=42,
        max_iter=5000
    )

    model.fit(X, y)

    selected = X.columns[
        model.coef_[0] != 0
    ].tolist()

    if len(selected) < 10:
        selected = X.columns[:20].tolist()

    return selected


# =========================================================
# TRAIN MODELS (UPDATED)
# =========================================================
# =========================================================
# TRAIN MODELS (FIXED BALANCE)
# =========================================================
def train_models(
    X_train,
    y_train,
    X_test,
    y_test
):

    models = {
        "SVM": SVC(
            kernel="rbf",
            C=0.5,
            gamma="scale",
            probability=True,
            class_weight=None,   # FIX
            random_state=42
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            bootstrap=True,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=None,   # FIX
            random_state=42
        ),

        "LogisticRegression": LogisticRegression(
            C=1.0,
            class_weight=None,   # FIX
            random_state=42,
            max_iter=5000
        )
    }

    results = {}

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        # probability
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(
                X_test
            )[:, 1]
        else:
            decision_scores = model.decision_function(
                X_test
            )

            y_prob = (
                decision_scores - decision_scores.min()
            ) / (
                decision_scores.max()
                - decision_scores.min()
                + 1e-8
            )

        # dynamic threshold
        threshold = 0.5

        y_pred = (
            y_prob >= threshold
        ).astype(int)

        results[name] = {
            "Accuracy":
                accuracy_score(
                    y_test,
                    y_pred
                ),

            "Precision":
                precision_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),

            "Recall":
                recall_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),

            "F1":
                f1_score(
                    y_test,
                    y_pred,
                    zero_division=0
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

# lasso_c is not used now because we use LogisticRegressionCV, but keep it for interface
lasso_c = st.sidebar.slider(
    "LASSO C (for CV)",
    0.01,
    1.0,
    0.1
)

use_smote = st.sidebar.checkbox(
    "Use SMOTE for imbalance handling",
    value=True
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

    st.success(
        f"Common genes: {len(common_genes)}"
    )

    X_train = pd.concat([
        expr1[common_genes],
        expr2[common_genes]
    ])

    y_train = pd.concat([
        y1,
        y2
    ])

    X_test = expr3[common_genes]
    y_test = y3

    imputer = KNNImputer(
        n_neighbors=impute_k
    )

    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns
    )

    X_test = pd.DataFrame(
        imputer.transform(X_test),
        index=X_test.index,
        columns=X_test.columns
    )

    scaler = MinMaxScaler(feature_range=(-1, 1))

    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns
    )

    X_test = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns
    )

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

    X_train_dea = X_train[dea_genes]
    X_test_dea = X_test[dea_genes]

    st.subheader(
        "mRMR Selection"
    )

    mrmr_genes = mrmr_selection(
        X_train_dea,
        y_train,
        min(mrmr_k, len(dea_genes))
    )

    st.write(
        f"mRMR selected: {len(mrmr_genes)}"
    )

    st.subheader(
        "LASSO Selection"
    )

    final_features = lasso_selection(
        X_train[mrmr_genes],
        y_train
    )

    st.success(
        f"Final selected genes: {len(final_features)}"
    )

    X_train_final = X_train[final_features]
    X_test_final = X_test[final_features]

    # =========================================================
    # NEW: BALANCING via DOWNSAMPLING of MAJORITY CLASS
    # =========================================================
    st.subheader("Class Distribution Before Balancing")
    st.write(pd.Series(y_train).value_counts())

    # Combine features and labels to a single DataFrame for easier resampling
    train_df = X_train_final.copy()
    train_df['class'] = y_train

    # Separate majority (tumor=1) and minority (normal=0)
    majority = train_df[train_df['class'] == 1]
    minority = train_df[train_df['class'] == 0]

    # Downsample majority to match size of minority
    if len(majority) > len(minority):
        majority_downsampled = resample(majority,
                                        replace=False,
                                        n_samples=len(minority),
                                        random_state=42)
        balanced_df = pd.concat([majority_downsampled, minority])
    else:
        # If majority is already smaller (unlikely), keep as is
        balanced_df = train_df

    X_train_final = balanced_df.drop('class', axis=1)
    y_train = balanced_df['class']

    st.subheader("Class Distribution After Downsampling (1:1 ratio)")
    st.write(pd.Series(y_train).value_counts())

    # Optional SMOTE (if user selected) – this will generate synthetic minority samples
    if use_smote:

        class_counts = pd.Series(y_train).value_counts()

        majority_count = class_counts.max()
        minority_count = class_counts.min()

        # hanya jalankan SMOTE jika memang masih imbalance
        if majority_count > minority_count:

            ratio = majority_count / minority_count

            k_neighbors = min(3, minority_count - 1)

            if k_neighbors < 1:
                k_neighbors = 1

            smote = SMOTE(
                sampling_strategy=1.0,
                random_state=42,
                k_neighbors=k_neighbors
            )

            X_train_final, y_train = smote.fit_resample(
                X_train_final,
                y_train
            )

            st.subheader("Class Distribution After SMOTE")
            st.write(pd.Series(y_train).value_counts())

        else:
            st.info("SMOTE skipped: dataset already balanced.")

    st.subheader(
        "Model Performance"
    )

    results = train_models(
        X_train_final,
        y_train,
        X_test_final,
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

    st.dataframe(metrics_df)

    for model_name, result in results.items():

        st.subheader(model_name)

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

    for model_name, result in results.items():
        st.write(model_name)
        st.write(pd.Series(result["y_pred"]).value_counts())
        
        st.write("Prediction probability summary")
        st.write(pd.Series(result["y_prob"]).describe())

    st.write("Final training distribution (after all balancing)")
    st.write(pd.Series(y_train).value_counts())

    st.write("Test distribution (original, imbalanced)")
    st.write(pd.Series(y_test).value_counts())

    st.write("Selected genes")
    st.write(final_features)
    
    # Show mean probability for the best model (last model in loop)
    st.write("Mean tumor probability (last model):", np.mean(list(results.values())[-1]["y_prob"]))
    st.write("Min tumor probability:", np.min(list(results.values())[-1]["y_prob"]))
    st.write("Max tumor probability:", np.max(list(results.values())[-1]["y_prob"]))

    st.download_button(
        "📥 Download Selected Genes",
        "\n".join(final_features),
        "selected_genes.txt"
    )
