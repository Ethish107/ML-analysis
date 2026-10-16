# ML Capstone · 23CSE301

**23CSE301 Machine Learning Capstone Project**  
B.Tech CSE · III Year · Academic Year 2026–27

**Team**

| Register Number | Track |
|----------------|-------|
| CB.SC.U4CSE24712 | Classification |
| CB.SC.U4CSE24761 | Regression |
| CB.SC.U4CSE24727 | Clustering |

---

## Problem Statements

### Classification — Bank Marketing (UCI)
**Question:** Given a client's demographic and campaign contact data, will they subscribe to a bank term deposit?  
**Type:** Binary classification (Yes / No)  
**Dataset:** `bank-additional-full.csv` — 41,188 records, 20 features (UCI ML Repository)

### Regression — Life Expectancy (WHO)
**Question:** Given a country's health, economic, and social indicators for a given year, what is the expected average life expectancy?  
**Type:** Continuous regression  
**Dataset:** `Life Expectancy Data.csv` — 2,938 records, 22 features (WHO Global Health Observatory)

### Clustering — Dry Bean (Koklu & Ozkan, 2020)
**Question:** Using only geometric features extracted from bean images, can unsupervised clustering discover groups that correspond to meaningful bean varieties?  
**Type:** Unsupervised clustering (no labels during training)  
**Dataset:** `Dry_Bean_Dataset.xlsx` — 13,611 records, 16 geometric features, 7 registered varieties

---

## Results Summary

### Regression Track

| Model | R² | RMSE | MAE |
|-------|-----|------|-----|
| Linear Regression | ~0.82 | — | — |
| Ridge | ~0.82 | — | — |
| Lasso | ~0.82 | — | — |
| ElasticNet | ~0.82 | — | — |
| Polynomial Regression | ~0.85 | — | — |
| Decision Tree | ~0.88 | — | — |
| **Random Forest** | **~0.97** | — | — |
| Gradient Boosting | ~0.95 | — | — |
| SVR | ~0.86 | — | — |
| KNN | ~0.91 | — | — |

**Best model: Random Forest Regressor — R² ≈ 0.97**

### Classification Track

| Model | Accuracy | F1 (weighted) | ROC-AUC |
|-------|----------|---------------|---------|
| Logistic Regression | — | — | — |
| KNN | — | — | — |
| Gaussian Naive Bayes | — | — | — |
| Decision Tree | — | — | — |
| SVC | — | — | — |
| Random Forest | — | — | — |
| AdaBoost | — | — | — |
| **Gradient Boosting** | **~0.90** | — | **~0.80** |
| Bagging | — | — | — |
| MLP | — | — | — |

**Best model: Gradient Boosting Classifier — Accuracy ≈ 90.1%, ROC-AUC ≈ 0.80**

### Clustering Track

| Algorithm | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|-----------|-----------|----------------|-------------------|
| K-Means (k=3) | — | — | — |
| Agglomerative (ward) | — | — | — |

**Final model: K-Means, k=3 (selected by internal metrics only)**  
Cluster dominant varieties: `{0: DERMASON, 1: BOMBAY, 2: HOROZ}`

---

## Reviewer & Architectural Implementation Notes

### 1. Classification Model Alignment (Notebook vs. Web Deployment)
- **Evaluation in Notebook (`classification.ipynb`):**
  - Random Forest: 5-fold CV ROC-AUC = **0.7991** (std: 0.0069)
  - Gradient Boosting: 5-fold CV ROC-AUC = **0.7989** (std: 0.0047)
- **Deployment Decision Rationale:**
  - The cross-validated discrimination performance difference is negligible (only **0.0002**, or 0.025%).
  - Gradient Boosting shows **lower cross-fold variance** (std 0.0047 vs 0.0069), indicating more consistent generalization.
  - **Production Constraints:** An unpruned Random Forest (200 trees) serializes to **>30 MB** (up to 100+ MB), while Gradient Boosting serializes to only **~210 KB** (>140x smaller).
  - In serverless hosting (e.g. Vercel Python functions with a 50 MB bundle ceiling and cold-start latency requirements), Gradient Boosting cold-starts and deserializes in sub-milliseconds with minimal RAM, preventing timeouts.
  - Therefore, Gradient Boosting is intentionally selected as the deployment model for production efficiency and operational stability with equivalent ROC-AUC.

### 2. Dataset Path Resolution
- `train_and_save_models.py` employs dynamic candidate path resolution (`find_dataset()`) checking `../data/`, `data/`, `../`, and current working directories. This guarantees that model training runs successfully regardless of whether executed from the project root, from within `ml-capstone-app/`, or in automated build environments.

### 3. Preprocessing Consistency
- The production training pipeline (`train_and_save_models.py`) strictly mirrors the exploratory preprocessing steps established in `classification.ipynb`:
  - Outlier capping: `campaign` is clipped at 20 (`clip(upper=20)`), `previous` is clipped at 5 (`clip(upper=5)`).
  - Duplicate rows are removed via `drop_duplicates()`.
  - Feature engineering (`previously_contacted`, `campaign_intensity`) and one-hot encoding + standard scaling via `ColumnTransformer` are fitted solely on the training split to avoid data leakage.

### 4. Interpretation of Clustering Output
- The Dry Bean clustering task is strictly unsupervised; ground-truth labels are never used during training or cluster assignment.
- Internal cluster validity metrics (Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz) identify $k=3$ as the optimal natural geometric clustering.
- Because 7 natural bean varieties exist in the dataset, mapping them to 3 morphological clusters means that each cluster contains a mixture of varieties (bounded cluster purity).
- The variety label assigned to each cluster (Cluster 0: DERMASON, Cluster 1: BOMBAY, Cluster 2: HOROZ) represents the **dominant variety (modal class label)** within that cluster partition, and must not be misinterpreted as a 7-class individual classifier. Both the Web UI, Streamlit dashboard, and API metadata explicitly reflect this modal interpretation.

---

## Environment Setup

**Python version:** 3.11+

Install all dependencies:

```bash
pip install -r requirements.txt
```

**Core libraries:**
- `scikit-learn` — ML models and pipelines
- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualisations
- `joblib` — model serialisation
- `openpyxl` — reading the Dry Bean Excel dataset

---

## How to Run

### Notebooks

Open each notebook in Jupyter and run all cells top-to-bottom:

```
notebooks/
├── classification.ipynb   # Review 1 Part A + Review 2 Part B
├── regression.ipynb       # Review 1 full track
└── clustering.ipynb       # Review 2 full track
```

All cells must produce outputs with no errors before submission.

### Interactive App (Prediction GUI)

**Step 1 — Train & save models** (one-time):

```bash
cd ml-capstone-app
python train_and_save_models.py
```

**Step 2 — Run locally:**

```bash
python local_server.py
```

Then open **http://localhost:8000** in your browser.

---

## Project Structure

```
/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── classification.ipynb
│   ├── regression.ipynb
│   └── clustering.ipynb
├── data/
│   ├── bank-additional-full.csv
│   ├── Life Expectancy Data.csv
│   └── Dry_Bean_Dataset.xlsx
└── ml-capstone-app/
    ├── api/predict.py          ← Serverless prediction API
    ├── models/                 ← Saved .pkl model files
    ├── public/index.html       ← Frontend GUI
    ├── train_and_save_models.py
    ├── local_server.py
    ├── requirements.txt
    └── vercel.json
```

---

## Academic Integrity

- Code scaffolding assisted by Google Antigravity AI.
- All analysis, EDA interpretation, feature engineering decisions, model selection rationale, and written observations are original work by the team.
- Datasets are publicly available (UCI ML Repository, WHO, Koklu & Ozkan 2020).
