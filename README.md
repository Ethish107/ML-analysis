# 23CSE301 — Machine Learning Capstone

Three machine learning tracks — **Classification**, **Regression**, and **Clustering** — each implemented as a self-contained Jupyter notebook with complete EDA, preprocessing, model evaluation, and an interactive full-stack prediction application.

| Track | Notebook | Dataset | Target / Task |
|---|---|---|---|
| Classification | `classification.ipynb` | `bank-additional-full.csv` (Bank Marketing) | `y` — will the client subscribe to a term deposit? (yes/no) |
| Regression | `regression.ipynb` | `Life Expectancy Data.csv` | `Life expectancy` (years) |
| Clustering | `clustering.ipynb` | `Dry_Bean_Dataset.xlsx` (Koklu & Ozkan) | Identify bean varieties from morphological geometry |

---

## Project Structure

```
.
├── data/
│   ├── Dry_Bean_Dataset.xlsx
│   ├── Life Expectancy Data.csv
│   └── bank-additional-full.csv
├── notebooks/
│   ├── classification.ipynb
│   ├── clustering.ipynb
│   └── regression.ipynb
├── ml-capstone-app/
│   ├── api/predict.py
│   ├── models/
│   ├── public/index.html
│   ├── app.py
│   ├── local_server.py
│   ├── requirements.txt
│   └── vercel.json
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Then open either notebook and run all cells top to bottom (each is fully reproducible with `RANDOM_STATE = 42`).

---

## 1. Classification — Bank Marketing (`classification.ipynb`)

**Goal:** predict whether a client subscribes to a term deposit (`y`) from demographic, financial, campaign, and macroeconomic attributes.

**Pipeline:**
- Full dataset audit (dtypes, missingness, duplicates, target balance) — 41,188 rows, 21 columns, no literal `NaN`s, but six categorical columns encode missingness as the string `"unknown"` (`default` worst at ~21%).
- EDA: target distribution, numeric feature distributions/boxplots, feature-vs-feature and feature-vs-target (jittered) scatter plots, categorical distribution and subscription-rate plots, correlation heatmap.
- Target is **imbalanced** (~89% "no" vs ~11% "yes", ≈8:1), which is called out explicitly as a reason plain accuracy is misleading.
- `duration` is dropped as a **leakage feature** (only known after a call takes place).
- Outlier audit via IQR on all numeric columns; `campaign` and `previous` are capped rather than dropped.
- Feature engineering: `previously_contacted` (from the `pdays == 999` sentinel) and `campaign_intensity`.
- 80/20 **stratified** train/test split; `OneHotEncoder` + `StandardScaler` via `ColumnTransformer`, fit on the training set only.

**Models trained (each in its own section, with metrics + confusion matrix):**
1. Logistic Regression (`class_weight="balanced"`)
2. K-Nearest Neighbors (k tuned by scanning odd values 3–99)
3. Gaussian Naive Bayes
4. Decision Tree Classifier (`max_depth` tuned, `class_weight="balanced"`)
5. Support Vector Classifier (linear/RBF, C tuned)

**Evaluation:** Accuracy, weighted Precision/Recall/F1, ROC-AUC, confusion matrices, ROC curves for all five models, 5-fold stratified cross-validation on the two strongest models by ROC-AUC.

**Key finding:** ranking by weighted F1 and ranking by ROC-AUC **disagree** — a model can look strong on weighted F1 purely by favoring the majority "no" class, while a threshold-independent metric (ROC-AUC) tells a different story. The notebook reports both a weighted-F1 ranking and a minority-class-only (subscribers-caught) ranking so the imbalance doesn't hide model quality, and explains the two rankings in a dedicated interpretation section.

## 2. Regression — Life Expectancy (`regression.ipynb`)

**Goal:** predict a country's life expectancy from demographic, health, immunization, economic, and education indicators.

**Pipeline:**
- Dataset audit: shape, dtypes, missing values (several health/economic columns have missing data), duplicates.
- EDA: target distribution (left-skewed, 36.3–89.0 years), numeric feature distributions, correlation heatmap (`Schooling` r≈0.75, `Income composition of resources` r≈0.72, `Adult Mortality` r≈−0.70 are the strongest relationships), feature-vs-target scatter plots, country `Status`/year coverage.
- Rows with a missing target are dropped; duplicates removed; predictor missingness is imputed **inside the pipeline** (median for numeric, most-frequent for categorical) so statistics come only from the training fold.
- Outlier detection via IQR is explicitly separated from outlier *treatment* — a domain-rule check (valid ranges for percentages/indices) confirms flagged values are extreme but not invalid, so none are removed.
- Feature engineering: `immunization_index` = mean(Hepatitis B, Polio, Diphtheria).
- `Country` is dropped as a high-cardinality identifier; `Status` is kept as a categorical predictor.
- 80/20 split, **stratified on quantile bins of the continuous target** to preserve its distribution.

**Models trained (10, all on the identical preprocessing + test split):**
Linear, Ridge, Lasso, ElasticNet, Polynomial (degree 1 vs 2, best degree carried forward), Decision Tree, Random Forest, Gradient Boosting, SVR, KNN Regressor.

**Evaluation:** R², RMSE, MAE for all ten; consolidated comparison table; `GridSearchCV` hyperparameter tuning on three models (Random Forest, Gradient Boosting, Decision Tree); 5-fold cross-validation on the two best models; an additional `GroupKFold`-by-country check to quantify how much of the random-split score is inflated by seeing the same country in both train and test; predicted-vs-actual and residual plots; Random Forest feature-importance plot; standardized-coefficient interpretation and Lasso sparsity analysis for the linear family.

**Reported results (held-out test set):**

| Model | R² | RMSE (years) | MAE (years) |
|---|---|---|---|
| Random Forest Regressor | **0.9729** | **1.58** | **1.03** |
| Support Vector Regressor | 0.9467 | — | — |
| Gradient Boosting Regressor | 0.9464 → **0.9591** (tuned) | 2.22 → 1.94 (tuned) | — |
| Linear / Ridge / Lasso / ElasticNet | 0.838 – 0.845 | — | — |

Random Forest is the best model overall; the four linear models cluster tightly, indicating regularization changes little on this data; GroupKFold-by-country validation shows some optimism in the random split because the dataset has ~16 repeated yearly rows per country.

---

## 3. Clustering — Dry Bean Dataset (`clustering.ipynb`)

**Goal:** Unsupervised grouping of 13,611 dry bean specimens across 16 morphological geometric features into coherent clusters without ground-truth labels during training.

**Pipeline:**
- Exploratory data analysis, box plots, IQR outlier checks, and correlation heatmaps.
- Preprocessing with StandardScaler and feature space variants (PCA dimensionality reduction).
- Algorithms evaluated: K-Means (k=3 to 10) and Agglomerative Hierarchical Clustering (Ward, Average, Complete linkage).
- Model selection using Elbow method, Silhouette scores, and post-hoc purity validation against registered varieties.
- Serialization of the optimal pipeline into `models/clustering_pipeline.pkl`.

---

## 4. Interactive Web Application (`ml-capstone-app/`)

An interactive application providing instant predictions across all three tracks:
- **Streamlit GUI (`app.py`):** Interactive parameter sliders, dropdowns, and live metric cards.
- **Web Interface (`public/index.html`):** Fast, clean, dependency-free responsive interface.
- **Serverless API (`api/predict.py`):** Python handler deployable to Vercel or run locally via `local_server.py`.

```bash
cd ml-capstone-app
pip install -r requirements.txt
python local_server.py    # runs web UI at http://localhost:8000
# or
streamlit run app.py      # runs Streamlit GUI
```

---

## Reproducibility

All notebooks set `RANDOM_STATE = 42` for all splits and stochastic models, and fit every preprocessing step (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) only on the training fold via `sklearn` `Pipeline`/`ColumnTransformer`, so results are deterministic and free of train/test leakage.

---

## 5. Architectural & Reviewer Alignment Notes

### Classification Model Alignment (Notebook vs. Web Deployment)
- **Evaluation in Notebook (`classification.ipynb`):**
  - Random Forest: 5-fold CV ROC-AUC = **0.7991** (std: 0.0069)
  - Gradient Boosting: 5-fold CV ROC-AUC = **0.7989** (std: 0.0047)
- **Deployment Decision Rationale:**
  - The cross-validated discrimination performance difference is negligible (only **0.0002**, or 0.025%).
  - Gradient Boosting shows **lower cross-fold variance** (std 0.0047 vs 0.0069), indicating greater stability across folds.
  - **Production Constraints:** An unpruned Random Forest (200 trees) serializes to **>30 MB** (up to 100+ MB), while Gradient Boosting serializes to only **~210 KB** (>140x smaller).
  - In serverless hosting (e.g. Vercel Python functions with a 50 MB bundle ceiling and cold-start latency requirements), Gradient Boosting cold-starts and deserializes in sub-milliseconds with minimal RAM, preventing timeouts.
  - Therefore, Gradient Boosting is intentionally selected as the deployment model for production efficiency and operational stability with equivalent ROC-AUC.

### Dataset Path Resolution
- `train_and_save_models.py` employs dynamic candidate path resolution (`find_dataset()`) checking `../data/`, `data/`, `../`, and current working directories. This guarantees that model training runs successfully regardless of whether executed from the project root, from within `ml-capstone-app/`, or in automated build environments.

### Preprocessing Consistency
- The production training pipeline (`train_and_save_models.py`) strictly mirrors the exploratory preprocessing steps established in `classification.ipynb`:
  - Outlier capping: `campaign` is clipped at 20 (`clip(upper=20)`), `previous` is clipped at 5 (`clip(upper=5)`).
  - Duplicate rows are removed via `drop_duplicates()`.
  - Feature engineering (`previously_contacted`, `campaign_intensity`) and one-hot encoding + standard scaling via `ColumnTransformer` are fitted solely on the training split to avoid data leakage.

### Interpretation of Clustering Output
- The Dry Bean clustering task is strictly unsupervised; ground-truth labels are never used during training or cluster assignment.
- Internal cluster validity metrics (Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz) identify $k=3$ as the optimal natural geometric clustering.
- Because 7 natural bean varieties exist in the dataset, mapping them to 3 morphological clusters means that each cluster contains a mixture of varieties (bounded cluster purity).
- The variety label assigned to each cluster (Cluster 0: DERMASON, Cluster 1: BOMBAY, Cluster 2: HOROZ) represents the **dominant variety (modal class label)** within that cluster partition, and must not be misinterpreted as a 7-class individual classifier. Both the Web UI, Streamlit dashboard, and API metadata explicitly reflect this modal interpretation.


