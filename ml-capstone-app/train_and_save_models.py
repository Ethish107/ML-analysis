"""
train_and_save_models.py
Trains and saves the three capstone models:
  1. Classification  -> models/classification_pipeline.pkl
  2. Regression      -> models/regression_pipeline.pkl
  3. Clustering      -> models/clustering_pipeline.pkl  +  models/cluster_names.json
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.cluster import KMeans

RANDOM_STATE = 42
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def find_dataset(filename: str) -> str:
    """Locate dataset file across common repo layouts (root or data/ folder)."""
    base_dirs = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
        os.path.join(os.path.dirname(os.path.abspath(__file__))),
        os.getcwd(),
        os.path.join(os.getcwd(), ".."),
    ]
    candidates = []
    for b in base_dirs:
        candidates.append(os.path.join(b, "data", filename))
        candidates.append(os.path.join(b, filename))
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError(f"Could not find '{filename}'. Searched: {candidates}")


# ===========================================================
# 1. CLASSIFICATION  (Bank Marketing)
# ===========================================================
print("=" * 60)
print("Training CLASSIFICATION model …")

cls_path = find_dataset("bank-additional-full.csv")
print(f"  Loaded dataset from: {cls_path}")
df_cls = pd.read_csv(cls_path, sep=";")

# Drop leaky column (mirrors notebook §11)
df_cls = df_cls.drop(columns=["duration"])

# Outlier treatment (mirrors notebook §10.2 / Cell 40)
# Capping campaign counts at 20 and previous contacts at 5
df_cls["campaign"] = df_cls["campaign"].clip(upper=20)
df_cls["previous"] = df_cls["previous"].clip(upper=5)

# Drop duplicate rows (mirrors notebook §10.3 / Cell 44)
df_cls = df_cls.drop_duplicates().reset_index(drop=True)

# Feature engineering (mirrors notebook §13)
df_cls["previously_contacted"] = (df_cls["pdays"] != 999).astype(int)
df_cls["campaign_intensity"]   = df_cls["campaign"] + df_cls["previous"]

X_cls = df_cls.drop(columns=["y"])
y_cls = (df_cls["y"] == "yes").astype(int)

num_cls = X_cls.select_dtypes(include=np.number).columns.tolist()
cat_cls = X_cls.select_dtypes(exclude=np.number).columns.tolist()

preprocessor_cls = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc",  StandardScaler())]), num_cls),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cls),
], remainder="drop")

# Note on Model Selection for Production Deployment:
# In the classification notebook (§23), Random Forest (CV ROC-AUC: 0.7991, std: 0.0069)
# and Gradient Boosting (CV ROC-AUC: 0.7989, std: 0.0047) were the top two candidates,
# virtually tied in cross-validation performance (difference of 0.0002).
# Gradient Boosting is selected for web deployment because:
#   1. Extremely lightweight serialization (~210 KB vs >30 MB for unconstrained Random Forest)
#   2. Fast cold starts and sub-millisecond inference in serverless runtime (Vercel)
#   3. Lower cross-fold variance (0.0047 vs 0.0069), providing consistent probability estimates
cls_pipeline = Pipeline([
    ("pre", preprocessor_cls),
    ("clf", GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                        random_state=RANDOM_STATE)),
])

X_tr, X_te, y_tr, y_te = train_test_split(
    X_cls, y_cls, test_size=0.20, stratify=y_cls, random_state=RANDOM_STATE
)
cls_pipeline.fit(X_tr, y_tr)
score = cls_pipeline.score(X_te, y_te)
print(f"  Accuracy on test set: {score:.4f}")

# Save metadata needed by app
cls_meta = {
    "num_features": num_cls,
    "cat_features": cat_cls,
    "feature_cols": X_cls.columns.tolist(),
}
joblib.dump(cls_pipeline, os.path.join(MODELS_DIR, "classification_pipeline.pkl"))
with open(os.path.join(MODELS_DIR, "classification_meta.json"), "w") as f:
    json.dump(cls_meta, f, indent=2)
print("  Saved classification_pipeline.pkl")


# ===========================================================
# 2. REGRESSION  (Life Expectancy)
# ===========================================================
print("=" * 60)
print("Training REGRESSION model …")

reg_path = find_dataset("Life Expectancy Data.csv")
print(f"  Loaded dataset from: {reg_path}")
df_reg = pd.read_csv(reg_path)
df_reg.columns = df_reg.columns.str.strip()

TARGET = "Life expectancy"
# Drop Country (not used), drop rows with missing target
df_reg = df_reg.drop(columns=["Country"]).dropna(subset=[TARGET])

# Feature engineering (mirrors notebook §12)
df_reg["immunization_index"] = df_reg[
    ["Hepatitis B", "Polio", "Diphtheria"]
].mean(axis=1)

X_reg = df_reg.drop(columns=[TARGET])
y_reg = df_reg[TARGET]

num_reg = X_reg.select_dtypes(include=np.number).columns.tolist()
cat_reg = X_reg.select_dtypes(exclude=np.number).columns.tolist()

preprocessor_reg = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc",  StandardScaler())]), num_reg),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_reg),
], remainder="drop")

reg_pipeline = Pipeline([
    ("pre", preprocessor_reg),
    ("reg", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)),
])

# Stratified-like split via quantile bins
strat_bins = pd.qcut(y_reg, q=5, labels=False, duplicates="drop")
X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
    X_reg, y_reg, test_size=0.20, stratify=strat_bins, random_state=RANDOM_STATE
)
reg_pipeline.fit(X_tr_r, y_tr_r)
r2 = reg_pipeline.score(X_te_r, y_te_r)
print(f"  R² on test set: {r2:.4f}")

reg_meta = {
    "num_features": num_reg,
    "cat_features": cat_reg,
    "feature_cols": X_reg.columns.tolist(),
    "target": TARGET,
}
joblib.dump(reg_pipeline, os.path.join(MODELS_DIR, "regression_pipeline.pkl"))
with open(os.path.join(MODELS_DIR, "regression_meta.json"), "w") as f:
    json.dump(reg_meta, f, indent=2)
print("  Saved regression_pipeline.pkl")


# ===========================================================
# 3. CLUSTERING  (Dry Bean)
# ===========================================================
print("=" * 60)
print("Training CLUSTERING model …")

clu_path = find_dataset("Dry_Bean_Dataset.xlsx")
print(f"  Loaded dataset from: {clu_path}")
df_clu = pd.read_excel(clu_path)

feature_cols_clu = [c for c in df_clu.columns if c != "Class"]
y_clu = df_clu["Class"]

# Recompute with full deduplicated data (mirrors clustering notebook)
df_clu_clean = df_clu.drop_duplicates().reset_index(drop=True)
X_clu = df_clu_clean[feature_cols_clu].copy()
y_clu = df_clu_clean["Class"].copy()

clu_pipeline = Pipeline([
    ("sc", StandardScaler()),
    ("km", KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=RANDOM_STATE)),
])
clu_pipeline.fit(X_clu)

labels = clu_pipeline.predict(X_clu)
ct = pd.crosstab(pd.Series(labels, name="cluster"), y_clu)

# Note on Clustering Output Interpretation:
# This is an unsupervised model (k=3) partitioning morphological feature space.
# It does NOT perform a 7-class supervised bean classification.
# The variety label assigned to each cluster is strictly a post-hoc modal/dominant
# variety within that geometric cluster.
cluster_names = {
    int(cl): str(ct.loc[cl].idxmax()) for cl in ct.index
}
print("  Cluster -> post-hoc dominant variety label: " + str(cluster_names))

clu_meta = {
    "feature_cols": feature_cols_clu,
    "cluster_names": cluster_names,
    "n_clusters": 3,
    "task": "unsupervised_clustering",
    "interpretation": "Post-hoc dominant variety label within morphological cluster (k=3). Not a 7-variety individual classifier.",
}
joblib.dump(clu_pipeline, os.path.join(MODELS_DIR, "clustering_pipeline.pkl"))
with open(os.path.join(MODELS_DIR, "clustering_meta.json"), "w") as f:
    json.dump(clu_meta, f, indent=2)
print("  Saved clustering_pipeline.pkl")

print("=" * 60)
print("All models saved to", MODELS_DIR)
