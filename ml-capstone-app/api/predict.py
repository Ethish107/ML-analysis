"""
api/predict.py  –  Single Vercel Python serverless function
Handles POST /api/predict?track=classification|regression|clustering
"""

import os, json, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# ── model cache (loaded once per warm invocation) ────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

_cache: dict = {}

def _load(name: str):
    if name not in _cache:
        pkl  = os.path.join(MODELS_DIR, f"{name}_pipeline.pkl")
        meta = os.path.join(MODELS_DIR, f"{name}_meta.json")
        _cache[name] = {
            "model": joblib.load(pkl),
            "meta":  json.load(open(meta)) if os.path.exists(meta) else {},
        }
    return _cache[name]


# ── prediction helpers ────────────────────────────────────────────
def predict_classification(body: dict) -> dict:
    obj = _load("classification")
    model, meta = obj["model"], obj["meta"]

    age      = int(body.get("age", 38))
    pdays    = int(body.get("pdays", 999))
    campaign = int(body.get("campaign", 2))
    previous = int(body.get("previous", 0))

    row = pd.DataFrame([{
        "age":               age,
        "job":               body.get("job", "admin."),
        "marital":           body.get("marital", "married"),
        "education":         body.get("education", "university.degree"),
        "default":           body.get("default", "no"),
        "housing":           body.get("housing", "yes"),
        "loan":              body.get("loan", "no"),
        "contact":           body.get("contact", "cellular"),
        "month":             body.get("month", "may"),
        "day_of_week":       body.get("day_of_week", "mon"),
        "campaign":          campaign,
        "pdays":             pdays,
        "previous":          previous,
        "poutcome":          body.get("poutcome", "nonexistent"),
        "emp.var.rate":      float(body.get("emp_var_rate", 1.1)),
        "cons.price.idx":    float(body.get("cons_price_idx", 93.9)),
        "cons.conf.idx":     float(body.get("cons_conf_idx", -36.4)),
        "euribor3m":         float(body.get("euribor3m", 4.9)),
        "nr.employed":       float(body.get("nr_employed", 5191.0)),
        "previously_contacted": int(pdays != 999),
        "campaign_intensity":   campaign + previous,
    }])

    pred  = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0].tolist()
    return {"prediction": pred, "label": "Yes" if pred == 1 else "No",
            "prob_yes": round(proba[1] * 100, 1), "prob_no": round(proba[0] * 100, 1)}


def predict_regression(body: dict) -> dict:
    obj = _load("regression")
    model = obj["model"]

    hep_b = float(body.get("hepatitis_b", 82))
    polio = float(body.get("polio", 83))
    dipht = float(body.get("diphtheria", 82))
    immunization_index = (hep_b + polio + dipht) / 3.0

    row = pd.DataFrame([{
        "Year":                          int(body.get("year", 2010)),
        "Status":                        body.get("status", "Developing"),
        "Adult Mortality":               float(body.get("adult_mortality", 150)),
        "infant deaths":                 float(body.get("infant_deaths", 20)),
        "Alcohol":                       float(body.get("alcohol", 4.5)),
        "percentage expenditure":        float(body.get("percentage_expenditure", 500)),
        "Hepatitis B":                   hep_b,
        "Measles":                       float(body.get("measles", 1000)),
        "BMI":                           float(body.get("bmi", 38)),
        "under-five deaths":             float(body.get("under_five_deaths", 30)),
        "Polio":                         polio,
        "Total expenditure":             float(body.get("total_expenditure", 5.9)),
        "Diphtheria":                    dipht,
        "HIV/AIDS":                      float(body.get("hiv_aids", 0.5)),
        "GDP":                           float(body.get("gdp", 7000)),
        "Population":                    float(body.get("population", 5_000_000)),
        "thinness  1-19 years":          float(body.get("thinness_1_19", 4.8)),
        "thinness 5-9 years":            float(body.get("thinness_5_9", 4.9)),
        "Income composition of resources": float(body.get("income_composition", 0.65)),
        "Schooling":                     float(body.get("schooling", 12.0)),
        "immunization_index":            immunization_index,
    }])

    pred = float(model.predict(row)[0])
    return {"prediction": round(pred, 1), "unit": "years",
            "immunization_index": round(immunization_index, 1)}


def predict_clustering(body: dict) -> dict:
    obj = _load("clustering")
    model, meta = obj["model"], obj["meta"]
    cluster_names = {int(k): v for k, v in meta.get("cluster_names", {}).items()}
    feat_cols     = meta.get("feature_cols", [])

    row_data = {
        "Area":           float(body.get("area", 80000)),
        "Perimeter":      float(body.get("perimeter", 1000)),
        "MajorAxisLength": float(body.get("major_axis_length", 350)),
        "MinorAxisLength": float(body.get("minor_axis_length", 220)),
        "AspectRation":   float(body.get("aspect_ratio", 1.7)),
        "Eccentricity":   float(body.get("eccentricity", 0.75)),
        "ConvexArea":     float(body.get("convex_area", 82000)),
        "EquivDiameter":  float(body.get("equiv_diameter", 300)),
        "Extent":         float(body.get("extent", 0.75)),
        "Solidity":       float(body.get("solidity", 0.98)),
        "roundness":      float(body.get("roundness", 0.85)),
        "Compactness":    float(body.get("compactness", 0.88)),
        "ShapeFactor1":   float(body.get("shape_factor1", 0.007)),
        "ShapeFactor2":   float(body.get("shape_factor2", 0.001)),
        "ShapeFactor3":   float(body.get("shape_factor3", 0.78)),
        "ShapeFactor4":   float(body.get("shape_factor4", 0.994)),
    }
    row = pd.DataFrame([row_data])
    if feat_cols:
        row = row.reindex(columns=feat_cols, fill_value=0.0)

    cluster = int(model.predict(row)[0])
    variety = cluster_names.get(cluster, f"Cluster {cluster}")
    return {"cluster": cluster, "variety": variety, "features": row_data}


# ── Vercel handler ────────────────────────────────────────────────
class handler(BaseHTTPRequestHandler):

    def _send_json(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json(200, {})

    def do_POST(self):
        qs    = parse_qs(urlparse(self.path).query)
        track = qs.get("track", ["classification"])[0]
        length = int(self.headers.get("Content-Length", 0))
        body  = json.loads(self.rfile.read(length) or b"{}")

        try:
            if track == "classification":
                result = predict_classification(body)
            elif track == "regression":
                result = predict_regression(body)
            elif track == "clustering":
                result = predict_clustering(body)
            else:
                self._send_json(400, {"error": f"Unknown track: {track}"})
                return
            self._send_json(200, result)
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def log_message(self, *args):
        pass   # suppress Vercel noise
