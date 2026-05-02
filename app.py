import os
import gzip
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
)
from scipy.stats import ttest_ind

import shap

# Optional ComBat (batch correction)
try:
    from pycombat import Combat
    HAS_COMBAT = True
except Exception:
    HAS_COMBAT = False

st.set_page_config(page_title="CCA Biomarker Pipeline", layout="wide")
st.title("Hybrid Feature Selection for Cholangiocarcinoma Biomarker Identification")

DATA_DIR = "data"
TRAIN_PATH_1 = os.path.join(DATA_DIR, "GSE76297_series_matrix.txt")
TRAIN_PATH_2 = os.path.join(DATA_DIR, "GSE132305_series_matrix.txt")
TEST_PATH = os.path.join(DATA_DIR, "GSE32225_series_matrix.txt")

# =========================
# GEO PARSER
# =========================
def parse_labels(dataset_name, sample_names):
    labels = []
    valid_samples = []

    for sample in sample_names:
        s = str(sample).replace('"', '').strip().upper()

        # GSE76297
        if dataset_name == "GSE76297":
            if "CCA" in s:
                if "NON-TUMOR" in s:
                    valid_samples.append(sample)
                    labels.append("Normal")
                elif "TUMOR" in s:
                    valid_samples.append(sample)
                    labels.append("CCA")

        # GSE132305
        elif dataset_name == "GSE132305":
            if "ECCA" in s:
                valid_samples.append(sample)
                labels.append("CCA")
            elif "BD" in s:
                valid_samples.append(sample)
                labels.append("Normal")

        # GSE32225
        elif dataset_name == "GSE32225":
            if "ICC" in s or "CCA" in s:
                valid_samples.append(sample)
                labels.append("CCA")
            elif "NORMAL" in s or "CONTROL" in s:
                valid_samples.append(sample)
                labels.append("Normal")

    return valid_samples, labels


def load_geo_dataset(path, dataset_name):
    sample_titles = None
    expression_lines = []
    data_started = False

    opener = gzip.open if str(path).endswith(".gz") else open

    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if line.startswith("!Sample_title"):
                sample_titles = line.split("\t")[1:]

            if (
                line.startswith("ID_REF")
                or line.startswith('"ID_REF"')
                or line.startswith("ID")
                or line.startswith("Probe")
            ):
                data_started = True

            if data_started:
                expression_lines.append(line)

    if len(expression_lines) == 0:
        st.error(f"No expression matrix found in {dataset_name}")
        st.stop()

    expression_text = "\n".join(expression_lines)
    df = pd.read_csv(StringIO(expression_text), sep="\t")

    first_col = df.columns[0]
    df = df.set_index(first_col).T
    df.columns.name = None

    if sample_titles is not None and len(sample_titles) == len(df):
        df.index = sample_titles

    valid_samples, labels = parse_labels(dataset_name, df.index)

    df = df.loc[valid_samples]

    if len(df) == 0:
        st.error(f"No valid samples found in {dataset_name}")
        st.stop()

    df["label"] = labels
    return df


# =========================
# PREPROCESSING
# =========================
def preprocess_data(df_train, df_test):
    X_train = df_train.drop(columns=["label"])
    y_train = df_train["label"]

    X_test = df_test.drop(columns=["label"])
    y_test = df_test["label"]

    # align genes across datasets
    common_genes = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_genes]
    X_test = X_test[common_genes]

    # KNN Imputation
    imputer = KNNImputer(n_neighbors=5)
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # MinMax Scaling
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Encode labels
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    return X_train, X_test, y_train, y_test


# =========================
# DEA
# =========================
def differential_expression_analysis(X, y, p_thresh=0.05):
    group1 = X[y == 1]
    group0 = X[y == 0]

    selected_genes = []

    for gene in X.columns:
        _, p = ttest_ind(group1[gene], group0[gene], equal_var=False)
        if p < p_thresh:
            selected_genes.append(gene)

    return selected_genes


# =========================
# mRMR (approx with MI ranking)
# =========================
def mrmr_selection(X, y, top_k=100):
    mi = mutual_info_classif(X, y)
    scores = pd.Series(mi, index=X.columns)
    selected = scores.sort_values(ascending=False).head(top_k).index.tolist()
    return selected


# =========================
# LASSO
# =========================
def lasso_selection(X, y):
    lasso = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        max_iter=2000,
        class_weight="balanced"
    )
    lasso.fit(X, y)

    coef = pd.Series(lasso.coef_[0], index=X.columns)
    selected = coef[coef != 0].index.tolist()
    return selected


# =========================
# MODELS
# =========================
def get_models():
    return {
        "SVM": SVC(probability=True, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ),
    }


# =========================
# EVALUATION
# =========================
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_test, y_prob),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm


# =========================
# MAIN PIPELINE
# =========================
if st.button("Run Pipeline"):
    with st.spinner("Loading GEO datasets..."):
        df1 = load_geo_dataset(TRAIN_PATH_1, "GSE76297")
        df2 = load_geo_dataset(TRAIN_PATH_2, "GSE132305")
        df_test = load_geo_dataset(TEST_PATH, "GSE32225")

        df_train = pd.concat([df1, df2], axis=0)

    st.subheader("Dataset Summary")
    st.write("Training shape:", df_train.shape)
    st.write("Testing shape:", df_test.shape)

    X_train, X_test, y_train, y_test = preprocess_data(df_train, df_test)

    # DEA
    st.subheader("Stage 1: Differential Expression Analysis")
    dea_genes = differential_expression_analysis(X_train, y_train)
    st.write("Selected genes after DEA:", len(dea_genes))

    X_train_dea = X_train[dea_genes]
    X_test_dea = X_test[dea_genes]

    # mRMR
    st.subheader("Stage 2: mRMR")
    mrmr_genes = mrmr_selection(X_train_dea, y_train, top_k=min(100, len(dea_genes)))
    st.write("Selected genes after mRMR:", len(mrmr_genes))

    X_train_mrmr = X_train_dea[mrmr_genes]
    X_test_mrmr = X_test_dea[mrmr_genes]

    # LASSO
    st.subheader("Stage 3: LASSO")
    lasso_genes = lasso_selection(X_train_mrmr, y_train)
    st.write("Final biomarkers:", len(lasso_genes))
    st.write(lasso_genes)

    X_train_final = X_train_mrmr[lasso_genes]
    X_test_final = X_test_mrmr[lasso_genes]

    models = get_models()

    for model_name, model in models.items():
        st.subheader(model_name)

        model.fit(X_train_final, y_train)
        metrics, cm = evaluate_model(model, X_test_final, y_test)

        metric_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
        st.dataframe(metric_df)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        ax.set_title(f"Confusion Matrix - {model_name}")
        st.pyplot(fig)

        # SHAP
        st.subheader(f"SHAP Interpretability - {model_name}")
        try:
            if model_name == "Random Forest":
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model, X_train_final)

            shap_values = explainer(X_test_final)
            fig = plt.figure(figsize=(8, 6))
            shap.plots.beeswarm(shap_values, show=False)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"SHAP failed: {e}")
