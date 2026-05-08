# 🔬 CCA Biomarker Discovery Pipeline

A Streamlit-based machine learning pipeline for **hybrid feature selection** and **biomarker discovery** in **Cholangiocarcinoma (CCA)** using multi-cohort gene expression datasets.

## Overview

This project integrates:

- Differential Expression Analysis (DEA)
- mRMR Feature Selection
- LASSO Feature Selection
- Batch Effect Correction (ComBat)
- Machine Learning Classification
- SHAP Explainability

to identify potential biomarkers for CCA classification.

---

## Features

### Data Processing
- Parse GEO Series Matrix files
- Extract phenotype labels automatically
- Convert probe IDs into gene symbols
- Merge multiple cohorts by common genes
- Handle missing values using KNN imputation
- Standardize gene expression values

### Feature Selection Pipeline
Hybrid feature selection consists of:

#### 1. Differential Expression Analysis (DEA)
Select significant genes using:

- logFC threshold
- adjusted p-value (FDR)

#### 2. Minimum Redundancy Maximum Relevance (mRMR)
Reduce redundant features while maximizing relevance.

#### 3. LASSO (L1 Regularization)
Final sparse biomarker selection.

---

## Machine Learning Models

Three classification models are evaluated:

- Support Vector Machine (SVM)
- Random Forest
- Logistic Regression

Evaluation includes:

- Accuracy
- Precision
- Recall
- Specificity
- F1 Score
- AUC
- MCC

Cross-validation:
- Stratified 5-Fold CV

---

## Explainability

Uses SHAP for model interpretability:

- TreeExplainer (RandomForest)
- General Explainer (SVM/Logistic Regression)

Provides:

- SHAP summary plot
- feature importance interpretation

---

## Visualization Dashboard

Includes:

### Model Performance Heatmap
Compare metrics across models.

### ROC Curve Comparison
Visual comparison of model discrimination.

### Confusion Matrix Grid
Per-model confusion matrices.

### PCA Global Structure
Visualizes sample separation.

### Gene Correlation Heatmap
Correlation among selected biomarkers.

### Dataset Summary
Training/testing shape and class distribution.

---

## Project Structure

```text
project/
│── app.py
│── requirements.txt
│── README.md
│
├── data/
│   ├── GSE76297_series_matrix.txt
│   ├── GSE132305_series_matrix.txt
│   ├── GSE32225_series_matrix.txt
│   ├── GPL17586.txt
│   ├── GPL13667.txt
│   ├── GPL8432.txt
