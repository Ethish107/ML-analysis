"""
app.py – ML Capstone Interactive GUI
23CSE301 Machine Learning Capstone | 2026-27

Three tracks on a single Streamlit app:
  • Classification  – Bank Marketing (will a client subscribe?)
  • Regression      – Life Expectancy prediction
  • Clustering      – Dry Bean variety identification
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="ML Capstone · 23CSE301",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Light CSS overrides  (simple / clean theme)
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── fonts ──────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

/* ── sidebar ─────────────────────────────── */
[data-testid="stSidebar"] {
    background: #f8f9fb;
    border-right: 1px solid #e4e7ec;
}

/* ── metric cards ────────────────────────── */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e4e7ec;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

/* ── header band ─────────────────────────── */
.hero {
    background: linear-gradient(120deg, #1a56db 0%, #0e9f6e 100%);
    border-radius: 12px;
    padding: 28px 32px;
    color: #ffffff;
    margin-bottom: 24px;
}
.hero h1 { margin: 0; font-size: 1.8rem; font-weight: 700; }
.hero p  { margin: 6px 0 0; opacity: .85; font-size: 0.97rem; }

/* ── result box ──────────────────────────── */
.result-box {
    border-radius: 10px;
    padding: 20px 24px;
    margin-top: 16px;
    font-size: 1.05rem;
    font-weight: 600;
}
.result-yes  { background:#d1fae5; border:1.5px solid #34d399; color:#065f46; }
.result-no   { background:#fee2e2; border:1.5px solid #f87171; color:#7f1d1d; }
.result-reg  { background:#e0f2fe; border:1.5px solid #38bdf8; color:#0c4a6e; }
.result-clu  { background:#f5f3ff; border:1.5px solid #a78bfa; color:#3b0764; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Model loading (cached)
# ──────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

@st.cache_resource(show_spinner="Loading models …")
def load_models():
    models = {}
    metas  = {}
    for name in ("classification", "regression", "clustering"):
        pkl  = os.path.join(MODELS_DIR, f"{name}_pipeline.pkl")
        meta = os.path.join(MODELS_DIR, f"{name}_meta.json")
        if os.path.exists(pkl):
            models[name] = joblib.load(pkl)
        if os.path.exists(meta):
            with open(meta) as f:
                metas[name] = json.load(f)
    return models, metas

models, metas = load_models()


# ──────────────────────────────────────────────
# Sidebar navigation
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ML Capstone")
    st.markdown("**23CSE301 | 2026–27**")
    st.markdown("---")
    nav_options = ["Home", "Classification", "Regression", "Clustering"]
    nav_index = nav_options.index(st.session_state["nav"]) if st.session_state.get("nav") in nav_options else 0
    track = st.radio(
        "Select Track",
        nav_options,
        index=nav_index,
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Datasets used:")
    st.caption("• Bank Marketing (UCI)")
    st.caption("• Life Expectancy (WHO)")
    st.caption("• Dry Bean (Koklu & Ozkan)")


# ──────────────────────────────────────────────
# HOME
# ──────────────────────────────────────────────
if track == "Home":
    st.markdown("""
    <div class="hero">
        <h1>ML Capstone · 23CSE301</h1>
        <p>Interactive prediction app — B.Tech CSE III Year · Academic Year 2026–27</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### Classification")
        st.markdown("""
        **Dataset:** Bank Marketing  
        **Task:** Predict if a client will subscribe to a term deposit  
        **Model:** Gradient Boosting Classifier  
        **Metric:** ROC-AUC
        """)
        if st.button("Try Classification →", use_container_width=True):
            st.session_state["nav"] = "Classification"
            st.rerun()

    with c2:
        st.markdown("### Regression")
        st.markdown("""
        **Dataset:** Life Expectancy (WHO)  
        **Task:** Predict life expectancy in years  
        **Model:** Random Forest Regressor  
        **Metric:** R²
        """)
        if st.button("Try Regression →", use_container_width=True):
            st.session_state["nav"] = "Regression"
            st.rerun()

    with c3:
        st.markdown("### Clustering")
        st.markdown("""
        **Dataset:** Dry Bean (Koklu & Ozkan)  
        **Task:** Identify bean variety cluster  
        **Model:** K-Means (k = 3)  
        **Metric:** Silhouette Score
        """)
        if st.button("Try Clustering →", use_container_width=True):
            st.session_state["nav"] = "Clustering"
            st.rerun()

    st.markdown("---")
    st.markdown("#### How to use")
    st.markdown("""
    1. Pick a **track** from the sidebar  
    2. Fill in the feature values using the sliders and dropdowns  
    3. Hit **Predict** to get an instant result  
    """)


# ──────────────────────────────────────────────
# CLASSIFICATION
# ──────────────────────────────────────────────
elif track == "Classification":
    st.markdown("""
    <div class="hero">
        <h1>Classification · Bank Marketing</h1>
        <p>Will this client subscribe to a term deposit? (Binary: Yes / No)</p>
    </div>
    """, unsafe_allow_html=True)

    if "classification" not in models:
        st.error("Classification model not found. Run `train_and_save_models.py` first.")
        st.stop()

    st.markdown("#### Enter Client Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        age       = st.slider("Age", 18, 95, 38, key="cls_age")
        job       = st.selectbox("Job", ['admin.','blue-collar','entrepreneur','housemaid',
                                          'management','retired','self-employed','services',
                                          'student','technician','unemployed','unknown'], key="cls_job")
        marital   = st.selectbox("Marital Status", ['married','single','divorced','unknown'], key="cls_marital")
        education = st.selectbox("Education", ['university.degree','high.school','professional.course',
                                                'basic.9y','basic.6y','basic.4y','illiterate','unknown'], key="cls_edu")
        default   = st.selectbox("Has Credit Default?", ['no','yes','unknown'], key="cls_default")
        housing   = st.selectbox("Has Housing Loan?",   ['yes','no','unknown'], key="cls_housing")
        loan      = st.selectbox("Has Personal Loan?",  ['no','yes','unknown'], key="cls_loan")

    with col2:
        st.markdown("**Campaign Contact**")
        contact     = st.selectbox("Contact Type", ['cellular','telephone'], key="cls_contact")
        month       = st.selectbox("Last Contact Month", ['may','jun','jul','aug','oct','nov',
                                                           'dec','jan','feb','mar','apr','sep'], key="cls_month")
        day_of_week = st.selectbox("Day of Week", ['mon','tue','wed','thu','fri'], key="cls_day")
        campaign    = st.slider("Contacts in Campaign", 1, 50, 2, key="cls_campaign")
        pdays       = st.slider("Days since Last Contact (999 = never)", 0, 999, 999, key="cls_pdays")
        previous    = st.slider("Previous Contacts", 0, 20, 0, key="cls_prev")
        poutcome    = st.selectbox("Previous Campaign Outcome", ['nonexistent','failure','success'], key="cls_pout")

    with col3:
        st.markdown("**Socio-economic Context**")
        emp_var_rate    = st.slider("Employment Variation Rate", -3.5, 1.5, 1.1, step=0.1, key="cls_evr")
        cons_price_idx  = st.slider("Consumer Price Index",      92.0, 95.0, 93.9, step=0.01, key="cls_cpi")
        cons_conf_idx   = st.slider("Consumer Confidence Index", -51.0, -26.0, -36.4, step=0.1, key="cls_cci")
        euribor3m       = st.slider("Euribor 3-Month Rate",      0.6, 5.1, 4.9, step=0.01, key="cls_eur")
        nr_employed     = st.slider("No. of Employees (×1000)",  4963.6, 5228.1, 5191.0, step=0.1, key="cls_nremp")

    # derived features
    previously_contacted = int(pdays != 999)
    campaign_intensity   = campaign + previous

    if st.button("Predict Subscription", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "age": age, "job": job, "marital": marital, "education": education,
            "default": default, "housing": housing, "loan": loan,
            "contact": contact, "month": month, "day_of_week": day_of_week,
            "campaign": campaign, "pdays": pdays, "previous": previous,
            "poutcome": poutcome, "emp.var.rate": emp_var_rate,
            "cons.price.idx": cons_price_idx, "cons.conf.idx": cons_conf_idx,
            "euribor3m": euribor3m, "nr.employed": nr_employed,
            "previously_contacted": previously_contacted,
            "campaign_intensity": campaign_intensity,
        }])

        pred   = models["classification"].predict(row)[0]
        prob   = models["classification"].predict_proba(row)[0]

        if pred == 1:
            st.markdown(f"""
            <div class="result-box result-yes">
                Prediction: <b>YES</b> — Client is likely to subscribe!<br>
                Probability: <b>{prob[1]*100:.1f}%</b>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box result-no">
                Prediction: <b>NO</b> — Client is unlikely to subscribe.<br>
                Probability of Yes: <b>{prob[1]*100:.1f}%</b>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        m1, m2 = st.columns(2)
        m1.metric("P(Subscribe = Yes)",  f"{prob[1]*100:.1f}%")
        m2.metric("P(Subscribe = No)",   f"{prob[0]*100:.1f}%")


# ──────────────────────────────────────────────
# REGRESSION
# ──────────────────────────────────────────────
elif track == "Regression":
    st.markdown("""
    <div class="hero">
        <h1>Regression · Life Expectancy</h1>
        <p>Predict the average life expectancy (in years) for a country-year observation</p>
    </div>
    """, unsafe_allow_html=True)

    if "regression" not in models:
        st.error("Regression model not found. Run `train_and_save_models.py` first.")
        st.stop()

    st.markdown("#### Enter Country / Year Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Basic Info**")
        status  = st.selectbox("Development Status", ["Developing", "Developed"], key="reg_status")
        year    = st.slider("Year", 2000, 2015, 2010, key="reg_year")
        schooling = st.slider("Schooling (years)", 0.0, 20.7, 12.0, step=0.1, key="reg_school")
        income_comp = st.slider("Income Composition of Resources (HDI proxy)", 0.0, 0.95, 0.65, step=0.01, key="reg_inc")

    with col2:
        st.markdown("**Health Indicators**")
        adult_mort    = st.slider("Adult Mortality (per 1000)", 1, 723, 150, key="reg_am")
        infant_deaths = st.slider("Infant Deaths (per 1000)", 0, 1800, 20, key="reg_id")
        under5        = st.slider("Under-5 Deaths (per 1000)", 0, 2500, 30, key="reg_u5")
        hiv_aids      = st.slider("HIV/AIDS deaths (per 1000, under-5)", 0.1, 50.6, 0.5, step=0.1, key="reg_hiv")
        bmi           = st.slider("Average BMI", 1.0, 87.3, 38.0, step=0.1, key="reg_bmi")
        alcohol       = st.slider("Alcohol Consumption (litres/capita)", 0.01, 17.87, 4.5, step=0.01, key="reg_alc")
        total_exp     = st.slider("Total Health Expenditure (% of GDP)", 0.37, 17.6, 5.9, step=0.01, key="reg_texp")

    with col3:
        st.markdown("**Immunisation & Economy**")
        hep_b    = st.slider("Hepatitis B immunisation (%)", 1, 99, 82, key="reg_hepb")
        polio    = st.slider("Polio immunisation (%)",       3, 99, 83, key="reg_polio")
        dipht    = st.slider("Diphtheria immunisation (%)",  2, 99, 82, key="reg_diph")
        measles  = st.slider("Measles reported cases",       0, 212183, 1000, key="reg_measles")
        pct_exp  = st.slider("Gov. Expenditure on Health (per capita, USD)", 0.0, 19479.9, 500.0, step=1.0, key="reg_pexp")
        gdp      = st.slider("GDP per capita (USD)", 1.68, 119172.74, 7000.0, step=1.0, key="reg_gdp")
        pop      = st.number_input("Population", min_value=100, max_value=1_400_000_000, value=5_000_000, step=100_000, key="reg_pop")
        thin1    = st.slider("Thinness 1–19 years (%)", 0.1, 27.7, 4.8, step=0.1, key="reg_thin1")
        thin2    = st.slider("Thinness 5–9 years (%)",  0.1, 28.6, 4.9, step=0.1, key="reg_thin2")

    immunization_index = np.mean([hep_b, polio, dipht])

    if st.button("Predict Life Expectancy", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "Year": year, "Status": status,
            "Adult Mortality": adult_mort, "infant deaths": infant_deaths,
            "Alcohol": alcohol, "percentage expenditure": pct_exp,
            "Hepatitis B": hep_b, "Measles": measles,
            "BMI": bmi, "under-five deaths": under5,
            "Polio": polio, "Total expenditure": total_exp,
            "Diphtheria": dipht, "HIV/AIDS": hiv_aids,
            "GDP": gdp, "Population": pop,
            "thinness  1-19 years": thin1, "thinness 5-9 years": thin2,
            "Income composition of resources": income_comp,
            "Schooling": schooling,
            "immunization_index": immunization_index,
        }])

        pred = models["regression"].predict(row)[0]

        st.markdown(f"""
        <div class="result-box result-reg">
            Predicted Life Expectancy: <b>{pred:.1f} years</b>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Life Expectancy", f"{pred:.1f} yrs")
        c2.metric("Immunization Index (auto)", f"{immunization_index:.1f}%")
        c3.metric("Status", status)

        # Contextual message
        if pred < 55:
            st.warning("Below global average — high-risk indicators detected.")
        elif pred < 70:
            st.info("Below global mean (~72 yrs) — room for improvement.")
        elif pred >= 80:
            st.success("Excellent life expectancy — comparable to developed nations.")
        else:
            st.success("Above global average life expectancy.")


# ──────────────────────────────────────────────
# CLUSTERING
# ──────────────────────────────────────────────
elif track == "Clustering":
    st.markdown("""
    <div class="hero">
        <h1>Clustering · Dry Bean Identification</h1>
        <p>Enter 16 geometric measurements to discover which bean cluster this sample belongs to</p>
    </div>
    """, unsafe_allow_html=True)

    if "clustering" not in models:
        st.error("Clustering model not found. Run `train_and_save_models.py` first.")
        st.stop()

    clu_meta = metas.get("clustering", {})
    cluster_names = {int(k): v for k, v in clu_meta.get("cluster_names", {}).items()}

    def_area = 80000
    def_peri = 1000.0
    def_maj  = 350.0
    def_min  = 220.0
    def_eqd  = 300.0
    def_carea= 82000
    def_ext  = 0.75
    def_sol  = 0.98

    def_ecc  = 0.75
    def_cperi= 990.0
    def_ar   = 1.7
    def_rnd  = 0.85
    def_comp = 0.88
    def_sf1  = 0.007
    def_sf2  = 0.001
    def_sf3  = 0.78
    def_sf4  = 0.994

    st.markdown("#### Enter Bean Measurements")
    st.caption("Enter 16 geometric measurements to find the cluster.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Size & Area**")
        area          = st.number_input("Area (px²)",         min_value=20000, max_value=254616, value=def_area, step=100, key="clu_area")
        perimeter     = st.number_input("Perimeter (px)",     min_value=500.0, max_value=1985.0, value=def_peri, step=1.0, key="clu_peri")
        major_axis    = st.number_input("Major Axis Length (px)", min_value=183.0, max_value=738.0, value=def_maj, step=0.5, key="clu_maj")
        minor_axis    = st.number_input("Minor Axis Length (px)", min_value=122.0, max_value=460.0, value=def_min, step=0.5, key="clu_min")
        axis_length   = st.number_input("Equiv. Diameter (px)", min_value=160.0, max_value=570.0, value=def_eqd, step=0.5, key="clu_eqdiam")
        conv_area     = st.number_input("Convex Area (px²)", min_value=20000, max_value=263261, value=def_carea, step=100, key="clu_carea")
        extent        = st.slider("Extent (0–1)", 0.55, 0.87, def_ext, step=0.001, key="clu_ext")
        solidity      = st.slider("Solidity (0–1)", 0.919, 0.994, def_sol, step=0.001, key="clu_sol")

    with col2:
        st.markdown("**Shape Descriptors**")
        eccentricity  = st.slider("Eccentricity (0–1)",   0.0, 1.0, def_ecc, step=0.001, key="clu_ecc")
        convex_peri   = st.number_input("Convex Perimeter (px)", min_value=500.0, max_value=1985.0, value=def_cperi, step=1.0, key="clu_cperi")
        aspectratio   = st.slider("Aspect Ratio",         1.0, 3.2, def_ar, step=0.01, key="clu_ar")
        roundness     = st.slider("Roundness (0–1)",      0.48, 0.99, def_rnd, step=0.001, key="clu_rnd")
        compactness   = st.slider("Compactness (0–1)",    0.64, 1.00, def_comp, step=0.001, key="clu_comp")
        shapefactor1  = st.slider("Shape Factor 1",       0.002, 0.012, def_sf1, step=0.0001, key="clu_sf1", format="%.4f")
        shapefactor2  = st.slider("Shape Factor 2",       0.0, 0.005, def_sf2, step=0.00001, key="clu_sf2", format="%.5f")
        shapefactor3  = st.slider("Shape Factor 3",       0.40, 1.00, def_sf3, step=0.001, key="clu_sf3")
        shapefactor4  = st.slider("Shape Factor 4",       0.95, 1.00, def_sf4, step=0.0001, key="clu_sf4", format="%.4f")

    if st.button("Identify Cluster", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "Area": area, "Perimeter": perimeter,
            "MajorAxisLength": major_axis, "MinorAxisLength": minor_axis,
            "AspectRation": aspectratio, "Eccentricity": eccentricity,
            "ConvexArea": conv_area, "EquivDiameter": axis_length,
            "Extent": extent, "Solidity": solidity,
            "roundness": roundness, "Compactness": compactness,
            "ShapeFactor1": shapefactor1, "ShapeFactor2": shapefactor2,
            "ShapeFactor3": shapefactor3, "ShapeFactor4": shapefactor4,
        }])

        # ensure column order matches training
        feat_cols = metas.get("clustering", {}).get("feature_cols", [])
        if feat_cols:
            row = row.reindex(columns=feat_cols, fill_value=0.0)

        cluster = int(models["clustering"].predict(row)[0])
        variety = cluster_names.get(cluster, f"Cluster {cluster}")

        st.markdown(f"""
        <div class="result-box result-clu">
            Cluster: <b>{cluster}</b> — Dominant Variety: <b>{variety}</b>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.metric("Assigned Cluster", str(cluster))
        c2.metric("Dominant Variety (post-hoc)", variety)

        st.info("""
        **Note:** Clusters are discovered using *only* geometric features — no labels are used during training.  
        The variety name shown is a post-hoc interpretation based on the dominant class within each cluster.
        """)
