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
import shap


st.set_page_config(
    page_title="CCA Biomarker Discovery",
    layout="wide"
)

st.title("🔬 Hybrid Feature Selection for Cholangiocarcinoma")

st.markdown("""
### Workflow
Automatically reading GEO Series Matrix files from `data/` folder:

- GSE76297 (training)
- GSE132305 (training)
- GSE32225 (testing)

Pipeline:

Expression Parsing → Label Extraction → DEA → mRMR → LASSO → Classification → SHAP
""")


# =========================================================
# Helper to create a file-like object from a local path
# =========================================================
class FileLike:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)   # for extension detection
    def read(self):
        if self.path.endswith(".gz"):
            with gzip.open(self.path, "rb") as f:
                return f.read()
        else:
            with open(self.path, "rb") as f:
                return f.read()

# =========================================================
# LABEL EXTRACTION (unchanged)
# =========================================================
def extract_labels(lines, sample_ids):
    """
    Extract labels from GEO !Sample_title row
    Supports:
    - GSE76297
    - GSE132305
    - GSE32225 (Ctrl = normal)
    """
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
        # GSE32225 special rule
        if title_lower.startswith("ctrl"):
            labels[sid] = 0
        elif (title_lower.startswith("ccbcn") or
              title_lower.startswith("ccm") or
              title_lower.startswith("ccny")):
            labels[sid] = 1
        elif any(keyword in title_lower for keyword in [
            "non-tumor",
            "non tumor",
            "normal",
            "adjacent normal",
            "control",
            "bile duct",
            "bd"
        ]):
            labels[sid] = 0
        elif any(keyword in title_lower for keyword in [
            "tumor",
            "cca",
            "cholangiocarcinoma",
            "ecca",
            "icc",
            "hcc",
            "cancer"
        ]):
            labels[sid] = 1
        else:
            labels[sid] = -1
    return labels


# =========================================================
# GEO PARSER (unchanged)
# =========================================================
def parse_series_matrix(file_obj):

    if file_obj.name.endswith(".gz"):
        content = gzip.decompress(file_obj.read()).decode("utf-8")
    else:
        content = file_obj.read().decode("utf-8")

    lines = content.splitlines()

    # find data table start
    start_idx = None

    for i, line in enumerate(lines):
        if line.startswith("!series_matrix_table_begin"):
            start_idx = i + 1
            break

    if start_idx is None:
        for i, line in enumerate(lines):
            if line.startswith('"ID_REF"') or line.startswith("ID_REF"):
                start_idx = i
                break

    if start_idx is None:
        st.error("Could not find expression table.")
        return None, None

    # header
    header_line = lines[start_idx].strip()
    headers = [x.strip('"') for x in header_line.split("\t")]
    sample_ids = headers[1:]

    st.info(f"Found {len(sample_ids)} samples")

    # parse expression rows
    data_rows = []

    for i in range(start_idx + 1, len(lines)):

        line = lines[i].strip()

        if line.startswith("!series_matrix_table_end"):
            break

        if not line:
            continue

        if line.startswith("!"):
            continue

        fields = [x.strip('"') for x in line.split("\t")]

        if len(fields) != len(headers):
            continue

        data_rows.append(fields)

    if len(data_rows) == 0:
        st.error("No expression rows found.")
        return None, None

    probe_ids = [row[0] for row in data_rows]

    expr_data = np.array(
        [row[1:] for row in data_rows],
        dtype=float
    )

    expr_df = pd.DataFrame(
        expr_data.T,
        index=sample_ids,
        columns=probe_ids
    )

    st.success(
        f"Parsed {expr_df.shape[0]} samples and {expr_df.shape[1]} probes."
    )

    # label extraction
    sample_labels = extract_labels(lines, sample_ids)

    if sample_labels is None:
        return None, None

    valid = [
        s for s in sample_ids
        if sample_labels.get(s, -1) != -1
    ]

    if len(valid) == 0:
        st.error("No valid labels found.")
        return None, None

    expr_df = expr_df.loc[valid]

    y = pd.Series(
        [sample_labels[s] for s in valid],
        index=valid
    )

    st.write(
        f"Class distribution: Tumor={sum(y==1)}, Normal={sum(y==0)}"
    )

    preview_df = pd.DataFrame({
        "Sample_ID": valid[:10],
        "Label": [sample_labels[s] for s in valid[:10]]
    })

    st.dataframe(preview_df)

    return expr_df, y


# =========================================================
# DEA, mRMR, LASSO, train, plots (unchanged)
# =========================================================
def differential_expression(X, y, logfc_thresh=1.0, pval_thresh=0.05):

    results = []

    for gene in X.columns:
        group1 = X[gene][y == 1]
        group0 = X[gene][y == 0]

        if len(group1) < 2 or len(group0) < 2:
            continue

        log2fc = np.log2(
            (group1.mean() + 1e-8) /
            (group0.mean() + 1e-8)
        )

        _, p_val = stats.ttest_ind(
            group1,
            group0,
            equal_var=False
        )

        results.append([gene, log2fc, p_val])

    if len(results) == 0:
        return []

    res_df = pd.DataFrame(
        results,
        columns=["gene", "logFC", "pvalue"]
    )

    _, adj_p = fdrcorrection(
        res_df["pvalue"].values
    )

    res_df["adj_p"] = adj_p

    sig = res_df[
        (abs(res_df["logFC"]) > logfc_thresh) &
        (res_df["adj_p"] < pval_thresh)
    ]

    return sig["gene"].tolist()


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

    selected = []
    remaining = list(X.columns)

    first = mi_df.iloc[0]["feature"]
    selected.append(first)
    remaining.remove(first)

    for _ in range(min(k - 1, len(remaining))):

        scores = []

        for feat in remaining:

            relevance = mi_df[
                mi_df["feature"] == feat
            ]["mi"].values[0]

            redundancy = np.mean([
                abs(np.corrcoef(
                    X[feat],
                    X[s]
                )[0, 1])
                for s in selected
            ])

            score = relevance - redundancy
            scores.append(score)

        best = remaining[np.argmax(scores)]

        selected.append(best)
        remaining.remove(best)

    return selected


def lasso_selection(X, y, C=0.1):

    model = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=C,
        random_state=42,
        max_iter=1000
    )

    model.fit(X, y)

    selected = X.columns[
        model.coef_[0] != 0
    ].tolist()

    return selected, model


def train_models(X_train, y_train, X_test, y_test):

    models = {
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            random_state=42,
            max_iter=1000
        )
    }

    results = {}
    trained_models = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_proba = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_proba),
            "MCC": matthews_corrcoef(y_test, y_pred),
            "y_pred": y_pred,
            "y_proba": y_proba
        }

        trained_models[name] = model

    return results, trained_models


def plot_confusion(y_true, y_pred, title):

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_title(title)

    return fig


def plot_roc(y_true, y_proba, model_name):

    fpr, tpr, _ = roc_curve(y_true, y_proba)

    fig, ax = plt.subplots()

    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], "--")

    ax.set_title(model_name)

    return fig


# =========================================================
# SIDEBAR – Parameters only (no uploaders)
# =========================================================
st.sidebar.header("Parameters")

impute_k = st.sidebar.slider(
    "KNN K",
    1,
    10,
    5
)

logfc_thresh = st.sidebar.slider(
    "logFC threshold",
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

run = st.sidebar.button("🚀 Run Pipeline")


# =========================================================
# Load data from data/ folder (cached in session state)
# =========================================================
data_dir = "data"
gse_ids = ["GSE76297", "GSE132305", "GSE32225"]
file_names = [f"{gse_id}_series_matrix.txt" for gse_id in gse_ids]

# Check if files exist (also allow .gz)
for gse_id, fname in zip(gse_ids, file_names):
    path = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        # try with .gz
        path_gz = path + ".gz"
        if os.path.exists(path_gz):
            file_names[gse_ids.index(gse_id)] = fname + ".gz"
        else:
            st.error(f"Missing file: {path} or {path_gz}")
            st.stop()

if "data_loaded" not in st.session_state:
    with st.spinner("Loading GSE76297..."):
        file1 = FileLike(os.path.join(data_dir, file_names[0]))
        expr1, y1 = parse_series_matrix(file1)
    with st.spinner("Loading GSE132305..."):
        file2 = FileLike(os.path.join(data_dir, file_names[1]))
        expr2, y2 = parse_series_matrix(file2)
    with st.spinner("Loading GSE32225..."):
        file3 = FileLike(os.path.join(data_dir, file_names[2]))
        expr3, y3 = parse_series_matrix(file3)

    if expr1 is None or expr2 is None or expr3 is None:
        st.error("Failed to parse datasets. Check file formats.")
        st.stop()

    st.session_state.expr1 = expr1
    st.session_state.y1 = y1
    st.session_state.expr2 = expr2
    st.session_state.y2 = y2
    st.session_state.expr3 = expr3
    st.session_state.y3 = y3
    st.session_state.data_loaded = True


# =========================================================
# MAIN PIPELINE (executes when run button is pressed)
# =========================================================
if run:

    expr1 = st.session_state.expr1
    y1 = st.session_state.y1
    expr2 = st.session_state.expr2
    y2 = st.session_state.y2
    expr3 = st.session_state.expr3
    y3 = st.session_state.y3

    common_genes = (
        expr1.columns
        .intersection(expr2.columns)
        .intersection(expr3.columns)
    )

    st.write("Train1 genes:", len(expr1.columns))
    st.write("Train2 genes:", len(expr2.columns))
    st.write("Test genes:", len(expr3.columns))
    st.write("Common genes:", len(common_genes))

    if len(common_genes) == 0:
        st.error("No common genes found across all datasets.")
        st.stop()

    X_train = pd.concat([
        expr1[common_genes],
        expr2[common_genes]
    ])

    y_train = pd.concat([y1, y2])

    X_test = expr3.reindex(columns=common_genes)
    y_test = y3

    # preprocessing
    imputer = KNNImputer(
        n_neighbors=impute_k
    )

    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns
    )

    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns
    )

    scaler = MinMaxScaler()

    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )

    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )

    # DEA
    st.subheader("Differential Expression Analysis")

    dea_genes = differential_expression(
        X_train,
        y_train,
        logfc_thresh,
        pval_thresh
    )

    st.write(f"DEA selected: {len(dea_genes)} genes")

    X_train_dea = X_train[dea_genes]
    X_test_dea = X_test[dea_genes]

    # mRMR
    st.subheader("mRMR Selection")

    mrmr_genes = mrmr_selection(
        X_train_dea,
        y_train,
        min(mrmr_k, len(dea_genes))
    )

    st.write(f"mRMR selected: {len(mrmr_genes)} genes")

    X_train_mrmr = X_train[mrmr_genes]
    X_test_mrmr = X_test[mrmr_genes]

    # LASSO
    st.subheader("LASSO Selection")

    lasso_genes, _ = lasso_selection(
        X_train_mrmr,
        y_train,
        lasso_c
    )

    if len(lasso_genes) == 0:
        final_features = mrmr_genes
    else:
        final_features = lasso_genes

    st.success(
        f"Final selected genes: {len(final_features)}"
    )

    X_train_final = X_train[final_features]
    X_test_final = X_test[final_features]

    # train
    st.subheader("Model Performance")

    results, models = train_models(
        X_train_final,
        y_train,
        X_test_final,
        y_test
    )

    metrics_df = pd.DataFrame(results).T.drop(
        columns=["y_pred", "y_proba"]
    )

    st.dataframe(metrics_df)

    for model_name in results:

        col1, col2 = st.columns(2)

        with col1:
            st.pyplot(
                plot_confusion(
                    y_test,
                    results[model_name]["y_pred"],
                    model_name
                )
            )

        with col2:
            st.pyplot(
                plot_roc(
                    y_test,
                    results[model_name]["y_proba"],
                    model_name
                )
            )

    # (Optional) SHAP – you can uncomment if you want
    # best_model_name = max(results, key=lambda x: results[x]["MCC"])
    # best_model = models[best_model_name]
    # st.subheader(f"SHAP Feature Importance ({best_model_name})")
    # explainer = shap.Explainer(best_model, X_train_final)
    # shap_values = explainer.shap_values(X_test_final)
    # fig, ax = plt.subplots()
    # shap.summary_plot(shap_values, X_test_final, show=False)
    # st.pyplot(fig)

    # download
    st.download_button(
        "📥 Download Selected Genes",
        "\n".join(final_features),
        "selected_genes.txt"
    )
