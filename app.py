import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import gzip
from io import StringIO

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import shap

# Optional ComBat for batch correction
try:
    from pycombat import Combat
    COMBAT_AVAILABLE = True
except Exception:
    COMBAT_AVAILABLE = False

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(page_title="CCA Biomarker Discovery", layout="wide")
st.title("🧬 Hybrid Feature Selection Framework for CCA")
st.markdown("Pipeline: GEO → Preprocessing → DEA → mRMR → LASSO → ML → SHAP")

# ======================================================
# FILE PATHS
# Folder structure:
# project/
# ├── app.py
# ├── requirements.txt
# └── data/
#     ├── GSE76297_series_matrix.txt
#     ├── GSE132305_series_matrix.txt
#     └── GSE32225_series_matrix.txt
# ======================================================
DATA_DIR = Path("data")
TRAIN_PATH_1 = DATA_DIR / "GSE76297_series_matrix.txt"
TRAIN_PATH_2 = DATA_DIR / "GSE132305_series_matrix.txt"
TEST_PATH = DATA_DIR / "GSE32225_series_matrix.txt"

# ======================================================
# LOAD GEO DATASET (.txt)
# GEO Series Matrix Format
# ======================================================
def load_geo_dataset(path):
    sample_titles = None
    expression_lines = []
    data_started = False

    # support .txt and .txt.gz
    opener = gzip.open if str(path).endswith(".gz") else open

    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            # sample metadata
            if line.startswith("!Sample_title"):
                sample_titles = line.split("\t")[1:]

            # detect beginning of expression matrix
            if (
                line.startswith("ID_REF")
                or line.startswith('"ID_REF"')
                or line.startswith("ID")
                or line.startswith("Probe")
            ):
                data_started = True

            if data_started:
                expression_lines.append(line)

    # validation
    if len(expression_lines) == 0:
        st.error(
            "Expression matrix not found in GEO file. "
            "Check whether you downloaded Series Matrix file."
        )
        st.stop()

    expression_text = "\n".join(expression_lines)

    df = pd.read_csv(
        StringIO(expression_text),
        sep="\t"
    )

    # normalize first column name
    first_col = df.columns[0]
    df = df.set_index(first_col).T
    df.columns.name = None

    # attach sample titles if found
    if sample_titles is not None and len(sample_titles) == len(df):
        df.index = sample_titles

    # filter only CCA samples
    filtered_samples = []
    labels = []

    for sample in df.index:
        sample_upper = str(sample).upper()

        if "CCA" in sample_upper:
            filtered_samples.append(sample)

            if "TUMOR" in sample_upper:
                labels.append("CCA")
            elif "NON-TUMOR" in sample_upper:
                labels.append("Normal")

    df = df.loc[filtered_samples]

    if len(labels) != len(df):
        st.error("Label extraction failed. Check sample naming format.")
        st.stop()

    df["label"] = labels

    return df

# ======================================================
# PREPROCESSING
# 1. Label Encoding
# 2. KNN Imputation
# 3. ComBat Batch Correction
# 4. MinMax Scaling
# ======================================================
def preprocess_data(df, label_col="label", batch_col="batch"):
    X = df.drop(columns=[label_col, batch_col])
    y = df[label_col].map({"Normal": 0, "CCA": 1})
    batch = df[batch_col]

    X = X.apply(pd.to_numeric, errors="coerce")

    # KNN Imputation
    imputer = KNNImputer(n_neighbors=5)
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # ComBat batch correction
    if COMBAT_AVAILABLE:
        combat = Combat()
        X = pd.DataFrame(
            combat.fit_transform(X.T, batch).T,
            columns=X.columns
        )

    # MinMax Scaling
    scaler = MinMaxScaler()
    X = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    return X, y

# ======================================================
# DEA (Differential Expression Analysis)
# |logFC| > 1 and adj p < 0.05
# ======================================================
def dea_filter(X, y):
    cancer = X[y == 1]
    normal = X[y == 0]

    pvals = []
    logfc = []

    for gene in X.columns:
        _, p = ttest_ind(cancer[gene], normal[gene], nan_policy="omit")
        fc = np.log2((cancer[gene].mean()+1e-8)/(normal[gene].mean()+1e-8))

        pvals.append(p)
        logfc.append(fc)

    adj_p = multipletests(pvals, method="fdr_bh")[1]

    dea_result = pd.DataFrame({
        "Gene": X.columns,
        "logFC": logfc,
        "adj_p": adj_p
    })

    selected_genes = dea_result[
        (dea_result["adj_p"] < 0.05) &
        (np.abs(dea_result["logFC"]) > 1)
    ]["Gene"].tolist()

    return X[selected_genes], dea_result

# ======================================================
# mRMR approximation
# Mutual Information + Redundancy Filter
# ======================================================
def mrmr_filter(X, y, top_k=50):
    mi = mutual_info_classif(X, y)

    mi_df = pd.DataFrame({
        "Gene": X.columns,
        "Score": mi
    }).sort_values("Score", ascending=False)

    selected = []

    for gene in mi_df["Gene"]:
        if len(selected) == 0:
            selected.append(gene)
        else:
            redundancy = np.mean([
                abs(X[gene].corr(X[g])) for g in selected
            ])

            if redundancy < 0.7:
                selected.append(gene)

        if len(selected) >= top_k:
            break

    return X[selected]

# ======================================================
# LASSO Selection
# ======================================================
def lasso_filter(X, y):
    model = LogisticRegressionCV(
        penalty="l1",
        solver="liblinear",
        cv=5,
        max_iter=3000
    )

    model.fit(X, y)

    selected_features = X.columns[
        model.coef_[0] != 0
    ]

    return X[selected_features]

# ======================================================
# MODEL EVALUATION
# ======================================================
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = model.decision_function(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "AUC": roc_auc_score(y_test, probs),
        "MCC": matthews_corrcoef(y_test, preds)
    }

    return metrics, model, preds

# ======================================================
# CONFUSION MATRIX PLOT
# ======================================================
def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig)

# ======================================================
# MAIN PIPELINE
# ======================================================
if st.button("Run Full Pipeline"):

    # Load datasets
    df1 = load_geo_dataset(TRAIN_PATH_1)
    df2 = load_geo_dataset(TRAIN_PATH_2)
    df_test = load_geo_dataset(TEST_PATH)

    # IMPORTANT:
    # Add labels manually after loading.
    # Edit according to your sample metadata.
    # Example only.

    if "label" not in df1.columns:
        st.error("Please add label column manually into training datasets.")
        st.stop()

    if "label" not in df2.columns:
        st.error("Please add label column manually into training datasets.")
        st.stop()

    if "label" not in df_test.columns:
        st.error("Please add label column manually into testing datasets.")
        st.stop()

    df1["batch"] = "GSE76297"
    df2["batch"] = "GSE132305"
    df_test["batch"] = "GSE32225"

    # Combine training datasets
    train_df = pd.concat([df1, df2], axis=0).reset_index(drop=True)

    st.subheader("Dataset Shape")
    st.write("Training:", train_df.shape)
    st.write("Testing:", df_test.shape)

    # Preprocessing
    X_train, y_train = preprocess_data(train_df)
    X_test, y_test = preprocess_data(df_test)

    st.success("Preprocessing finished")

    # DEA
    st.subheader("1. DEA")
    X_train_dea, dea_result = dea_filter(X_train, y_train)
    X_test_dea = X_test[X_train_dea.columns]

    st.write("Genes after DEA:", X_train_dea.shape[1])
    st.dataframe(dea_result.sort_values("adj_p").head(20))

    # mRMR
    st.subheader("2. mRMR")
    X_train_mrmr = mrmr_filter(X_train_dea, y_train, top_k=50)
    X_test_mrmr = X_test_dea[X_train_mrmr.columns]

    st.write("Genes after mRMR:", X_train_mrmr.shape[1])

    # LASSO
    st.subheader("3. LASSO")
    X_train_final = lasso_filter(X_train_mrmr, y_train)
    X_test_final = X_test_mrmr[X_train_final.columns]

    st.write("Final biomarkers:")
    st.write(list(X_train_final.columns))

    # Models
    models = {
        "SVM": SVC(probability=True, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(class_weight="balanced"),
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=3000)
    }

    results = []
    trained_models = {}

    st.subheader("4. Model Evaluation")

    for name, model in models.items():
        metrics, trained_model, preds = evaluate_model(
            model,
            X_train_final,
            y_train,
            X_test_final,
            y_test
        )

        metrics["Model"] = name
        results.append(metrics)
        trained_models[name] = trained_model

    result_df = pd.DataFrame(results)
    st.dataframe(result_df)

    # Best model by MCC
    best_model_name = result_df.sort_values(
        "MCC",
        ascending=False
    ).iloc[0]["Model"]

    best_model = trained_models[best_model_name]

    st.success(f"Best Model (MCC): {best_model_name}")

    # Confusion matrix
    best_preds = best_model.predict(X_test_final)
    plot_confusion(y_test, best_preds)

    # SHAP interpretability
    st.subheader("5. SHAP Analysis")

    explainer = shap.Explainer(best_model, X_test_final)
    shap_values = explainer(X_test_final)

    fig = plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(fig)
