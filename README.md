# 🔬 Hybrid Feature Selection for Cholangiocarcinoma

A Streamlit-based machine learning application for biomarker discovery in Cholangiocarcinoma (CCA) using integrated GEO gene expression datasets, hybrid feature selection, explainable AI, and classification models.

---

# 📌 Overview

This project implements a complete bioinformatics and machine learning pipeline for identifying potential biomarkers in Cholangiocarcinoma (CCA) from microarray gene expression datasets.

The system combines:

- Differential Expression Analysis (DEA)
- Minimum Redundancy Maximum Relevance (mRMR)
- LASSO Feature Selection
- Machine Learning Classification
- SHAP Explainability
- Batch Effect Correction (ComBat)

The application is built using Streamlit and provides an interactive analytics dashboard for model evaluation and biological interpretation.

---

# 🚀 Features

## 🧬 Bioinformatics Pipeline
- GEO Series Matrix parsing
- Automatic label extraction
- Probe-to-gene conversion
- Differential Expression Analysis
- Batch effect correction using ComBat
- Missing value imputation using KNN
- Feature scaling using StandardScaler

## 🧠 Hybrid Feature Selection
- Differential Expression Analysis (DEA)
- mRMR feature selection
- LASSO feature selection

## 🤖 Machine Learning Models
- Support Vector Machine (SVM)
- Random Forest
- Logistic Regression

## 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- Specificity
- F1 Score
- ROC-AUC
- MCC (Matthews Correlation Coefficient)
- Cross-validation performance
- Bootstrap AUC confidence interval

## 🔍 Explainable AI
- SHAP summary plots
- Feature importance interpretation

## 📈 Visualization Dashboard
- ROC Curve comparison
- Confusion Matrix heatmaps
- PCA visualization
- Gene correlation heatmap
- Model performance heatmap

---

# 📂 Project Structure

```bash
project/
│
├── app.py
├── requirements.txt
├── README.md
│
└── data/
    ├── GSE76297_series_matrix.txt
    ├── GSE132305_series_matrix.txt
    ├── GSE32225_series_matrix.txt
    ├── GPL17586.txt
    ├── GPL13667.txt
    └── GPL8432.txt
```

---

# 📚 Datasets

This project uses publicly available GEO datasets:

| Dataset | Description |
|---|---|
| GSE76297 | Cholangiocarcinoma dataset |
| GSE132305 | Cholangiocarcinoma dataset |
| GSE32225 | External validation dataset |

Platform annotation files:
- GPL17586
- GPL13667
- GPL8432

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/cca-biomarker-discovery.git
cd cca-biomarker-discovery
```

## 2. Create Virtual Environment

```bash
python -m venv venv
```

### Windows
```bash
venv\Scripts\activate
```

### Linux / Mac
```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Application

```bash
streamlit run app.py
```

---

# 🔄 Pipeline Workflow

```text
GEO Dataset Loading
        ↓
Probe-to-Gene Conversion
        ↓
Batch Effect Correction (ComBat)
        ↓
KNN Imputation
        ↓
Feature Scaling
        ↓
Differential Expression Analysis
        ↓
mRMR Feature Selection
        ↓
LASSO Feature Selection
        ↓
Machine Learning Training
        ↓
Model Evaluation
        ↓
SHAP Explainability
        ↓
Visualization Dashboard
```

---

# 🧠 Machine Learning Models

The application evaluates three classifiers:

| Model | Description |
|---|---|
| SVM | Linear classification |
| Random Forest | Ensemble tree-based classifier |
| Logistic Regression | Interpretable statistical classifier |

Cross-validation uses:

```python
StratifiedKFold(n_splits=5)
```

---

# 🔍 SHAP Explainability

The project integrates SHAP for model interpretation:

- TreeExplainer for Random Forest
- Generic SHAP Explainer for linear models

This helps identify:
- Important biomarker genes
- Feature contribution
- Biological relevance

---

# 📊 Dashboard Analytics

## 📌 Model Performance Heatmap
Compares evaluation metrics across all models.

## 📈 ROC Curve Comparison
Displays ROC curves and AUC scores for all classifiers.

## 🧩 Confusion Matrix Grid
Shows prediction distributions for each model.

## 🧬 PCA Global Structure View
Visualizes sample clustering and separation.

## 🔥 Gene Correlation Map
Displays correlations among selected genes.

## 🧠 SHAP Summary Plot
Highlights influential genes affecting predictions.

---

# 🛠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Streamlit | Interactive dashboard |
| Scikit-learn | Machine learning |
| SHAP | Explainable AI |
| Pandas / NumPy | Data processing |
| Matplotlib / Seaborn | Visualization |
| Statsmodels | Statistical analysis |

---

# 📦 Example Requirements

```txt
streamlit
pandas
numpy
scikit-learn
scipy
statsmodels
matplotlib
seaborn
shap
mrmr-selection
combat
```

---

# 🎯 Research Purpose

This project was developed for:
- Biomarker discovery
- Cancer bioinformatics research
- Explainable AI in healthcare
- Machine learning for genomics

Focused specifically on:
- Cholangiocarcinoma diagnosis support
- High-dimensional gene expression analysis

---

# 🔮 Future Improvements

Potential future enhancements:
- Deep learning integration
- Multi-omics analysis
- Survival analysis support
- Gene ontology enrichment
- Cloud deployment
- External API integration

---

# 👨‍💻 Author

Developed for research and educational purposes in:
- Bioinformatics
- Machine Learning
- Computational Biology

---

# 📜 License

This project is intended for academic and research purposes.

---

```bibtex
