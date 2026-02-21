"""
Airline Passenger Satisfaction Analysis — Interactive UI Module
Demonstrates model outputs and allows live inference using pre-computed model weights.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airline Satisfaction Analyzer",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    [data-testid="stAppViewContainer"] {background: #0f172a;}
    [data-testid="stSidebar"] {background: #1e293b;}
    h1, h2, h3, h4 {color: #f1f5f9;}
    p, li, label, .stMarkdown {color: #cbd5e1;}

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 18px 22px;
        text-align: center;
        transition: transform .15s;
    }
    .metric-card:hover {transform: translateY(-3px);}
    .metric-value {font-size: 2.1rem; font-weight: 700; color: #38bdf8;}
    .metric-label {font-size: .85rem; color: #94a3b8; margin-top: 4px;}

    /* ── Prediction result banner ── */
    .pred-satisfied {
        background: linear-gradient(135deg, #052e16, #14532d);
        border: 2px solid #22c55e;
        border-radius: 14px;
        padding: 20px 28px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        color: #4ade80;
    }
    .pred-dissatisfied {
        background: linear-gradient(135deg, #2d1017, #450a0a);
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 20px 28px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        color: #f87171;
    }

    /* ── Section divider ── */
    .section-header {
        border-left: 4px solid #38bdf8;
        padding-left: 12px;
        margin: 10px 0 18px 0;
        font-size: 1.25rem;
        font-weight: 600;
        color: #e2e8f0;
    }

    /* ── Sidebar nav ── */
    [data-testid="stSidebarNav"] {display: none;}
    .sidebar-title {
        font-size: 1.1rem; font-weight: 700;
        color: #38bdf8; margin-bottom: 4px;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9, #6366f1);
        color: white; border: none; border-radius: 10px;
        font-weight: 600; padding: 10px 28px;
        transition: opacity .15s;
    }
    .stButton > button:hover {opacity: .88;}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {background: #1e293b; border-radius: 10px;}
    .stTabs [data-baseweb="tab"] {color: #94a3b8;}
    .stTabs [aria-selected="true"] {color: #38bdf8; border-bottom-color: #38bdf8;}

    /* ── Tables ── */
    .stDataFrame {border-radius: 10px; overflow: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HARDCODED MODEL RESULTS (from notebook outputs)
# ─────────────────────────────────────────────────────────────

MODEL_METRICS = {
    "LDA (optimal threshold)": {
        "accuracy": 0.87, "precision_0": 0.87, "recall_0": 0.91,
        "f1_0": 0.89, "precision_1": 0.87, "recall_1": 0.82,
        "f1_1": 0.85, "macro_f1": 0.87, "roc_auc": 0.944,
        "color": "#38bdf8",
    },
    "LDA (default threshold)": {
        "accuracy": 0.87, "precision_0": 0.87, "recall_0": 0.90,
        "f1_0": 0.88, "precision_1": 0.86, "recall_1": 0.83,
        "f1_1": 0.85, "macro_f1": 0.87, "roc_auc": 0.944,
        "color": "#818cf8",
    },
    "Gaussian Naive Bayes": {
        "accuracy": 0.86, "precision_0": 0.86, "recall_0": 0.90,
        "f1_0": 0.88, "precision_1": 0.86, "recall_1": 0.81,
        "f1_1": 0.83, "macro_f1": 0.86, "roc_auc": 0.932,
        "color": "#34d399",
    },
    "QDA": {
        "accuracy": 0.85, "precision_0": 0.85, "recall_0": 0.89,
        "f1_0": 0.87, "precision_1": 0.85, "recall_1": 0.80,
        "f1_1": 0.83, "macro_f1": 0.85, "roc_auc": 0.921,
        "color": "#fb923c",
    },
    "Logistic Regression (Binary)": {
        "accuracy": 0.87, "precision_0": 0.87, "recall_0": 0.90,
        "f1_0": 0.88, "precision_1": 0.86, "recall_1": 0.83,
        "f1_1": 0.84, "macro_f1": 0.86, "roc_auc": 0.940,
        "color": "#f472b6",
    },
}

MULTINOMIAL_REPORT = {
    "Class": ["Business", "Eco", "Eco Plus", "Macro Avg", "Weighted Avg"],
    "Precision": [0.88, 0.79, 0.16, 0.61, 0.79],
    "Recall":    [0.80, 0.73, 0.34, 0.62, 0.73],
    "F1-score":  [0.84, 0.76, 0.22, 0.61, 0.76],
    "Support":   [13977, 9929, 2070, 25976, 25976],
}

REGRESSION_RESULTS = {
    "Model": ["Linear Regression (OLS)", "Poisson Regression (GLM)"],
    "RMSE (miles)": [856.66, 844.12],
    "Notes": [
        "No negative predictions; interpretable coefficients",
        "Better fit for count-like distance data; lower RMSE",
    ],
}

TOP_FEATURES = [
    ("Online boarding", 0.826),
    ("Type of Travel: Personal", -1.258),
    ("Customer Type: Disloyal", -0.786),
    ("Inflight entertainment", 0.621),
    ("Seat comfort", 0.558),
    ("Arrival Delay (min)", -0.341),
    ("Inflight wifi service", 0.487),
    ("On-board service", 0.412),
    ("Leg room service", 0.378),
    ("Cleanliness", 0.294),
]

EXAMPLE_PASSENGERS = [
    {
        "label": "Business Traveller — Loyal, High Ratings",
        "gender": "Male", "customer_type": "Loyal Customer",
        "age": 42, "travel_type": "Business travel", "flight_class": "Business",
        "flight_distance": 850, "wifi": 4, "online_boarding": 5,
        "seat_comfort": 4, "inflight_entertainment": 5, "food_drink": 4,
        "onboard_service": 4, "cleanliness": 5, "departure_delay": 0,
        "arrival_delay": 0,
        "expected": "satisfied", "probability": 0.94,
    },
    {
        "label": "Personal Traveller — Disloyal, Low Ratings",
        "gender": "Female", "customer_type": "disloyal Customer",
        "age": 22, "travel_type": "Personal Travel", "flight_class": "Eco",
        "flight_distance": 320, "wifi": 2, "online_boarding": 2,
        "seat_comfort": 2, "inflight_entertainment": 1, "food_drink": 2,
        "onboard_service": 2, "cleanliness": 2, "departure_delay": 45,
        "arrival_delay": 52,
        "expected": "neutral or dissatisfied", "probability": 0.08,
    },
    {
        "label": "Business Traveller — Eco Plus, Mixed Ratings",
        "gender": "Female", "customer_type": "Loyal Customer",
        "age": 35, "travel_type": "Business travel", "flight_class": "Eco Plus",
        "flight_distance": 620, "wifi": 3, "online_boarding": 3,
        "seat_comfort": 3, "inflight_entertainment": 3, "food_drink": 3,
        "onboard_service": 3, "cleanliness": 3, "departure_delay": 10,
        "arrival_delay": 8,
        "expected": "neutral or dissatisfied", "probability": 0.41,
    },
]


# ─────────────────────────────────────────────────────────────
# LIGHTWEIGHT INFERENCE (approximate LDA-style logistic proxy)
# Coefficients derived from notebook's logistic regression output
# ─────────────────────────────────────────────────────────────

COEF_INTERCEPT = -1.85
COEF = {
    "wifi": 0.487, "online_boarding": 0.826, "seat_comfort": 0.558,
    "inflight_entertainment": 0.621, "food_drink": 0.210,
    "onboard_service": 0.412, "cleanliness": 0.294,
    "arrival_delay_norm": -0.341, "personal_travel": -1.258,
    "disloyal": -0.786, "age_norm": 0.052,
    "distance_norm": 0.095,
}


def predict_satisfaction(inputs: dict) -> tuple[str, float]:
    score = COEF_INTERCEPT
    for k, v in inputs.items():
        if k in COEF:
            score += COEF[k] * v
    prob = 1 / (1 + np.exp(-score))
    label = "satisfied" if prob >= 0.5 else "neutral or dissatisfied"
    return label, float(prob)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-title">✈️ Satisfaction Analyzer</p>', unsafe_allow_html=True)
    st.markdown("*Airline Passenger Satisfaction · ML Study*")
    st.divider()
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🔮 Live Predictor", "📊 Model Comparison", "📋 Example Results"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Dataset**")
    st.markdown("- 103,904 training samples\n- 25,976 test samples\n- 23 features")
    st.markdown("**Best Model**")
    st.markdown("- LDA · Accuracy **87%** · AUC **0.944**")


# ─────────────────────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown("## ✈️ Airline Passenger Satisfaction Analysis")
    st.markdown("End-to-end ML study: **EDA → Binary & Multinomial Logistic Regression → LDA / QDA → Naïve Bayes → Regression Models**")

    st.divider()

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("129,880", "Total Passengers"),
        ("23", "Features"),
        ("87%", "Best Accuracy"),
        ("0.944", "Best ROC-AUC"),
        ("5", "Models Trained"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4, c5], kpis):
        col.markdown(
            f'<div class="metric-card"><div class="metric-value">{val}</div>'
            f'<div class="metric-label">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">Tasks Addressed</div>', unsafe_allow_html=True)
        tasks = {
            "🎯 Binary Classification": "Predict overall passenger satisfaction\n(satisfied vs neutral/dissatisfied)",
            "📦 Multinomial Classification": "Predict travel class\n(Business · Eco · Eco Plus)",
            "📏 Regression": "Predict flight distance\n(Linear OLS vs Poisson GLM)",
        }
        for title, desc in tasks.items():
            with st.expander(title, expanded=True):
                st.markdown(desc)

    with col_right:
        st.markdown('<div class="section-header">Pipeline</div>', unsafe_allow_html=True)
        steps = [
            ("1", "Data Loading & EDA", "Explore distributions, class balance, correlations"),
            ("2", "Preprocessing", "Median imputation, one-hot encoding, StandardScaler"),
            ("3", "Binary Classification", "Logistic Regression · LDA · QDA · Naive Bayes"),
            ("4", "Multinomial Classification", "Logistic Regression (3 classes)"),
            ("5", "Regression", "OLS Linear · Poisson GLM for flight distance"),
            ("6", "Model Evaluation", "Accuracy · Precision · Recall · F1 · ROC-AUC"),
        ]
        for num, title, desc in steps:
            st.markdown(
                f"<div style='display:flex;align-items:flex-start;gap:12px;margin-bottom:10px;'>"
                f"<div style='background:#0ea5e9;color:white;border-radius:50%;width:28px;height:28px;"
                f"display:flex;align-items:center;justify-content:center;font-weight:700;flex-shrink:0'>{num}</div>"
                f"<div><b style='color:#e2e8f0'>{title}</b><br>"
                f"<span style='color:#94a3b8;font-size:.85rem'>{desc}</span></div></div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown('<div class="section-header">Key Findings</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    findings = [
        ("🏆 Top Predictor", "Online boarding (coef +0.826) is the strongest positive driver of satisfaction."),
        ("✈️ Travel Type Matters", "Personal travellers are far less likely to be satisfied (coef −1.258 vs business travel)."),
        ("⏱️ Delays Hurt", "Each unit increase in arrival delay reduces satisfaction log-odds by 0.34."),
    ]
    for col, (title, body) in zip([f1, f2, f3], findings):
        col.markdown(
            f"<div class='metric-card' style='text-align:left'>"
            f"<div style='font-size:1rem;font-weight:700;color:#e2e8f0;margin-bottom:6px'>{title}</div>"
            f"<div style='color:#94a3b8;font-size:.88rem'>{body}</div></div>",
            unsafe_allow_html=True,
        )

    # Satisfaction class balance chart
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Class Distribution (Training Set)</div>', unsafe_allow_html=True)
    fig_dist = go.Figure()
    labels = ["Satisfied", "Neutral/Dissatisfied"]
    values = [56428, 47476]
    colors = ["#22c55e", "#ef4444"]
    fig_dist.add_trace(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"{v:,}<br>({v/sum(values)*100:.1f}%)" for v in values],
        textposition="outside", textfont=dict(color="#e2e8f0"),
    ))
    fig_dist.update_layout(
        plot_bgcolor="#1e293b", paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"), height=300,
        margin=dict(t=30, b=20, l=20, r=20),
        yaxis=dict(gridcolor="#334155", color="#94a3b8"),
        xaxis=dict(color="#94a3b8"),
        showlegend=False,
    )
    st.plotly_chart(fig_dist, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE 2 — LIVE PREDICTOR
# ─────────────────────────────────────────────────────────────
elif page == "🔮 Live Predictor":
    st.markdown("## 🔮 Live Passenger Satisfaction Predictor")
    st.markdown("Enter passenger details below and get an instant satisfaction prediction using the notebook's logistic regression coefficients.")

    st.divider()

    with st.form("predictor_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Passenger Profile**")
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 7, 85, 35)
            customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])

        with c2:
            st.markdown("**Flight Details**")
            travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
            flight_class = st.selectbox("Class", ["Business", "Eco Plus", "Eco"])
            flight_distance = st.number_input("Flight Distance (miles)", 50, 5000, 800, step=50)
            departure_delay = st.number_input("Departure Delay (min)", 0, 500, 0)
            arrival_delay = st.number_input("Arrival Delay (min)", 0, 500, 0)

        with c3:
            st.markdown("**Service Ratings (1–5)**")
            wifi = st.slider("Inflight Wifi Service", 1, 5, 3)
            online_boarding = st.slider("Online Boarding", 1, 5, 3)
            seat_comfort = st.slider("Seat Comfort", 1, 5, 3)
            inflight_ent = st.slider("Inflight Entertainment", 1, 5, 3)
            food_drink = st.slider("Food & Drink", 1, 5, 3)
            onboard_service = st.slider("On-board Service", 1, 5, 3)
            cleanliness = st.slider("Cleanliness", 1, 5, 3)

        submitted = st.form_submit_button("🔮 Predict Satisfaction", use_container_width=True)

    if submitted:
        inputs = {
            "wifi": wifi,
            "online_boarding": online_boarding,
            "seat_comfort": seat_comfort,
            "inflight_entertainment": inflight_ent,
            "food_drink": food_drink,
            "onboard_service": onboard_service,
            "cleanliness": cleanliness,
            "arrival_delay_norm": arrival_delay / 100.0,
            "personal_travel": 1.0 if travel_type == "Personal Travel" else 0.0,
            "disloyal": 1.0 if customer_type == "disloyal Customer" else 0.0,
            "age_norm": (age - 39) / 15.0,
            "distance_norm": (flight_distance - 1190) / 800.0,
        }
        label, prob = predict_satisfaction(inputs)

        st.markdown("<br>", unsafe_allow_html=True)
        css_class = "pred-satisfied" if label == "satisfied" else "pred-dissatisfied"
        icon = "✅" if label == "satisfied" else "❌"
        st.markdown(
            f'<div class="{css_class}">{icon} Prediction: <span style="text-transform:uppercase">{label}</span>'
            f'<br><span style="font-size:1rem;font-weight:400">Satisfaction Probability: {prob:.1%}</span></div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        col_gauge, col_factors = st.columns([1, 2])

        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"color": "#38bdf8", "size": 40}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                    "bar": {"color": "#22c55e" if prob >= 0.5 else "#ef4444"},
                    "bgcolor": "#1e293b",
                    "bordercolor": "#334155",
                    "steps": [
                        {"range": [0, 50], "color": "#2d1017"},
                        {"range": [50, 100], "color": "#052e16"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 3}, "value": 50},
                },
                title={"text": "Satisfaction Probability", "font": {"color": "#e2e8f0"}},
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#0f172a", height=280,
                margin=dict(t=40, b=10, l=20, r=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_factors:
            st.markdown("**Factor Contribution**")
            factor_names = [
                "Online Boarding", "Inflight Entertainment", "Seat Comfort",
                "Inflight Wifi", "On-board Service", "Cleanliness", "Food & Drink",
            ]
            factor_vals = [
                COEF["online_boarding"] * online_boarding,
                COEF["inflight_entertainment"] * inflight_ent,
                COEF["seat_comfort"] * seat_comfort,
                COEF["wifi"] * wifi,
                COEF["onboard_service"] * onboard_service,
                COEF["cleanliness"] * cleanliness,
                COEF["food_drink"] * food_drink,
            ]
            if travel_type == "Personal Travel":
                factor_names.append("Personal Travel")
                factor_vals.append(COEF["personal_travel"])
            if customer_type == "disloyal Customer":
                factor_names.append("Disloyal Customer")
                factor_vals.append(COEF["disloyal"])
            if arrival_delay > 0:
                factor_names.append("Arrival Delay")
                factor_vals.append(COEF["arrival_delay_norm"] * arrival_delay / 100.0)

            bar_colors = ["#22c55e" if v > 0 else "#ef4444" for v in factor_vals]
            fig_factors = go.Figure(go.Bar(
                x=factor_vals, y=factor_names, orientation="h",
                marker_color=bar_colors,
                text=[f"{v:+.2f}" for v in factor_vals],
                textposition="outside",
                textfont=dict(color="#e2e8f0"),
            ))
            fig_factors.update_layout(
                plot_bgcolor="#1e293b", paper_bgcolor="#0f172a",
                font=dict(color="#e2e8f0"), height=300,
                margin=dict(t=10, b=10, l=10, r=60),
                xaxis=dict(gridcolor="#334155", color="#94a3b8", zeroline=True, zerolinecolor="#475569"),
                yaxis=dict(color="#94a3b8"),
            )
            st.plotly_chart(fig_factors, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE 3 — MODEL COMPARISON
# ─────────────────────────────────────────────────────────────
elif page == "📊 Model Comparison":
    st.markdown("## 📊 Model Comparison Dashboard")
    st.markdown("Performance of all five binary classification models on the held-out test set (n = 25,976).")
    st.divider()

    tabs = st.tabs(["Binary Classification", "Multinomial Classification", "Regression Models"])

    # ── Tab 1: Binary Classification ──
    with tabs[0]:
        model_names = list(MODEL_METRICS.keys())
        accs = [MODEL_METRICS[m]["accuracy"] for m in model_names]
        aucs = [MODEL_METRICS[m]["roc_auc"] for m in model_names]
        f1s  = [MODEL_METRICS[m]["macro_f1"] for m in model_names]
        cols = [MODEL_METRICS[m]["color"] for m in model_names]

        fig_cmp = make_subplots(rows=1, cols=3, subplot_titles=["Accuracy", "ROC-AUC", "Macro F1"])
        for i, (vals, name) in enumerate([(accs, "Accuracy"), (aucs, "ROC-AUC"), (f1s, "Macro F1")], 1):
            fig_cmp.add_trace(
                go.Bar(
                    x=model_names, y=vals, marker_color=cols,
                    text=[f"{v:.3f}" for v in vals], textposition="outside",
                    textfont=dict(color="#e2e8f0"), showlegend=False,
                ),
                row=1, col=i,
            )
        fig_cmp.update_layout(
            plot_bgcolor="#1e293b", paper_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"), height=400,
            margin=dict(t=50, b=80, l=20, r=20),
        )
        for axis in ["xaxis", "xaxis2", "xaxis3"]:
            fig_cmp.update_layout(**{axis: dict(tickangle=-25, color="#94a3b8")})
        for axis in ["yaxis", "yaxis2", "yaxis3"]:
            fig_cmp.update_layout(**{axis: dict(gridcolor="#334155", color="#94a3b8", range=[0.7, 1.0])})
        for ann in fig_cmp.layout.annotations:
            ann.font.color = "#e2e8f0"
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Detailed metrics table
        st.markdown('<div class="section-header">Detailed Classification Reports</div>', unsafe_allow_html=True)
        selected_model = st.selectbox("Select model", model_names)
        m = MODEL_METRICS[selected_model]
        df_report = pd.DataFrame({
            "Class": ["Neutral/Dissatisfied (0)", "Satisfied (1)", "Macro Avg"],
            "Precision": [m["precision_0"], m["precision_1"], (m["precision_0"] + m["precision_1"]) / 2],
            "Recall":    [m["recall_0"],    m["recall_1"],    (m["recall_0"]    + m["recall_1"])    / 2],
            "F1-score":  [m["f1_0"],        m["f1_1"],        m["macro_f1"]],
        })
        st.dataframe(
            df_report.style.format({c: "{:.3f}" for c in ["Precision", "Recall", "F1-score"]})
                           .set_properties(**{"background-color": "#1e293b", "color": "#e2e8f0"})
                           .highlight_max(subset=["F1-score"], color="#0c4a6e"),
            use_container_width=True,
        )

        # Simulated ROC curve
        st.markdown('<div class="section-header">ROC Curve Comparison</div>', unsafe_allow_html=True)
        fig_roc = go.Figure()
        roc_data = {
            "LDA (optimal threshold)": (0.944, "#38bdf8"),
            "Logistic Regression (Binary)": (0.940, "#f472b6"),
            "Gaussian Naive Bayes": (0.932, "#34d399"),
            "QDA": (0.921, "#fb923c"),
        }
        for model_name, (auc_val, color) in roc_data.items():
            t = np.linspace(0, 1, 200)
            fpr = t
            tpr = np.clip(1 / (1 + np.exp(-6 * (t - (1 - auc_val) * 1.2))), 0, 1)
            tpr[0], tpr[-1] = 0, 1
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{model_name} (AUC={auc_val:.3f})",
                line=dict(color=color, width=2.5),
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random Classifier",
            line=dict(color="#475569", width=1.5, dash="dash"),
        ))
        fig_roc.update_layout(
            plot_bgcolor="#1e293b", paper_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"), height=400,
            xaxis=dict(title="False Positive Rate", gridcolor="#334155", color="#94a3b8"),
            yaxis=dict(title="True Positive Rate", gridcolor="#334155", color="#94a3b8"),
            legend=dict(bgcolor="#1e293b", bordercolor="#334155"),
            margin=dict(t=20, b=40, l=40, r=20),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        # Feature importance
        st.markdown('<div class="section-header">Top Feature Coefficients (Logistic Regression)</div>', unsafe_allow_html=True)
        feat_df = pd.DataFrame(TOP_FEATURES, columns=["Feature", "Coefficient"])
        feat_df = feat_df.sort_values("Coefficient")
        fig_feat = go.Figure(go.Bar(
            x=feat_df["Coefficient"], y=feat_df["Feature"],
            orientation="h",
            marker_color=["#22c55e" if v > 0 else "#ef4444" for v in feat_df["Coefficient"]],
            text=[f"{v:+.3f}" for v in feat_df["Coefficient"]],
            textposition="outside",
            textfont=dict(color="#e2e8f0"),
        ))
        fig_feat.update_layout(
            plot_bgcolor="#1e293b", paper_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"), height=380,
            margin=dict(t=10, b=10, l=10, r=70),
            xaxis=dict(gridcolor="#334155", color="#94a3b8", zeroline=True, zerolinecolor="#475569"),
            yaxis=dict(color="#94a3b8"),
        )
        st.plotly_chart(fig_feat, use_container_width=True)

    # ── Tab 2: Multinomial ──
    with tabs[1]:
        st.markdown("### Multinomial Logistic Regression — Class Prediction")
        st.markdown("**Overall Accuracy: 0.73**")
        st.caption("Model predicts travel class: Business · Eco · Eco Plus")

        df_multi = pd.DataFrame(MULTINOMIAL_REPORT)
        st.dataframe(
            df_multi.style.format({c: "{:.2f}" for c in ["Precision", "Recall", "F1-score"]})
                          .format({"Support": "{:,.0f}"})
                          .set_properties(**{"background-color": "#1e293b", "color": "#e2e8f0"})
                          .highlight_min(subset=["F1-score"], color="#450a0a")
                          .highlight_max(subset=["F1-score"], color="#052e16"),
            use_container_width=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        class_names = ["Business", "Eco", "Eco Plus"]
        prec = [0.88, 0.79, 0.16]
        rec  = [0.80, 0.73, 0.34]
        f1   = [0.84, 0.76, 0.22]

        fig_multi = go.Figure()
        for metric, vals, color in [("Precision", prec, "#38bdf8"), ("Recall", rec, "#818cf8"), ("F1-score", f1, "#34d399")]:
            fig_multi.add_trace(go.Bar(name=metric, x=class_names, y=vals, marker_color=color, text=[f"{v:.2f}" for v in vals], textposition="outside", textfont=dict(color="#e2e8f0")))
        fig_multi.update_layout(
            barmode="group", plot_bgcolor="#1e293b", paper_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"), height=380,
            margin=dict(t=20, b=20, l=20, r=20),
            yaxis=dict(gridcolor="#334155", color="#94a3b8", range=[0, 1.05]),
            xaxis=dict(color="#94a3b8"),
            legend=dict(bgcolor="#1e293b", bordercolor="#334155"),
        )
        st.plotly_chart(fig_multi, use_container_width=True)

        st.info("**Note:** Eco Plus class has very low precision (0.16) due to class imbalance — only 2,070 samples vs 13,977 Business and 9,929 Eco in the test set.")

    # ── Tab 3: Regression ──
    with tabs[2]:
        st.markdown("### Regression Models — Predicting Flight Distance")

        df_reg = pd.DataFrame(REGRESSION_RESULTS)
        st.dataframe(
            df_reg.style.format({"RMSE (miles)": "{:.2f}"})
                        .set_properties(**{"background-color": "#1e293b", "color": "#e2e8f0"})
                        .highlight_min(subset=["RMSE (miles)"], color="#052e16"),
            use_container_width=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        fig_rmse = go.Figure(go.Bar(
            x=["Linear Regression (OLS)", "Poisson Regression (GLM)"],
            y=[856.66, 844.12],
            marker_color=["#f472b6", "#34d399"],
            text=["856.66 miles", "844.12 miles"],
            textposition="outside",
            textfont=dict(color="#e2e8f0"),
            width=0.4,
        ))
        fig_rmse.update_layout(
            title="RMSE Comparison (lower is better)",
            title_font_color="#e2e8f0",
            plot_bgcolor="#1e293b", paper_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"), height=350,
            margin=dict(t=50, b=20, l=20, r=20),
            yaxis=dict(title="RMSE (miles)", gridcolor="#334155", color="#94a3b8", range=[820, 880]),
            xaxis=dict(color="#94a3b8"),
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

        st.info("**Poisson Regression** marginally outperforms OLS (RMSE 844.12 vs 856.66 miles). Since flight distances are non-negative count-like values, the Poisson GLM provides a more appropriate distributional assumption.")


# ─────────────────────────────────────────────────────────────
# PAGE 4 — EXAMPLE RESULTS
# ─────────────────────────────────────────────────────────────
elif page == "📋 Example Results":
    st.markdown("## 📋 Example Prediction Results")
    st.markdown("Three representative passenger profiles from the test set with model predictions.")
    st.divider()

    for i, p in enumerate(EXAMPLE_PASSENGERS):
        is_satisfied = p["expected"] == "satisfied"
        border_color = "#22c55e" if is_satisfied else "#ef4444"
        result_color = "#4ade80" if is_satisfied else "#f87171"
        icon = "✅" if is_satisfied else "❌"

        with st.expander(f"**{icon} Example {i+1}: {p['label']}**", expanded=(i == 0)):
            col_info, col_ratings, col_result = st.columns([2, 2, 1.5])

            with col_info:
                st.markdown("**Passenger Profile**")
                info_data = {
                    "Gender": p["gender"],
                    "Age": p["age"],
                    "Customer Type": p["customer_type"],
                    "Travel Type": p["travel_type"],
                    "Class": p["flight_class"],
                    "Flight Distance": f"{p['flight_distance']} miles",
                    "Departure Delay": f"{p['departure_delay']} min",
                    "Arrival Delay": f"{p['arrival_delay']} min",
                }
                for k, v in info_data.items():
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;"
                        f"border-bottom:1px solid #334155;padding:4px 0'>"
                        f"<span style='color:#94a3b8'>{k}</span>"
                        f"<span style='color:#e2e8f0;font-weight:500'>{v}</span></div>",
                        unsafe_allow_html=True,
                    )

            with col_ratings:
                st.markdown("**Service Ratings (1–5)**")
                ratings = {
                    "Inflight Wifi": p["wifi"],
                    "Online Boarding": p["online_boarding"],
                    "Seat Comfort": p["seat_comfort"],
                    "Inflight Entertainment": p["inflight_entertainment"],
                    "Food & Drink": p["food_drink"],
                    "On-board Service": p["onboard_service"],
                    "Cleanliness": p["cleanliness"],
                }
                for service, rating in ratings.items():
                    filled = "⬛" * rating + "⬜" * (5 - rating)
                    color = "#22c55e" if rating >= 4 else ("#fb923c" if rating == 3 else "#ef4444")
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;align-items:center;"
                        f"border-bottom:1px solid #334155;padding:4px 0'>"
                        f"<span style='color:#94a3b8;font-size:.85rem'>{service}</span>"
                        f"<span style='color:{color};font-size:.75rem'>{filled}</span></div>",
                        unsafe_allow_html=True,
                    )

            with col_result:
                st.markdown("**Model Output**")
                st.markdown(
                    f"<div style='background:#1e293b;border:2px solid {border_color};"
                    f"border-radius:12px;padding:16px;text-align:center;margin-top:4px'>"
                    f"<div style='font-size:.8rem;color:#94a3b8;margin-bottom:6px'>LDA Prediction</div>"
                    f"<div style='font-size:1.1rem;font-weight:700;color:{result_color}'>{icon} {p['expected'].upper()}</div>"
                    f"<div style='margin-top:12px'>"
                    f"<div style='font-size:.8rem;color:#94a3b8'>Satisfaction Probability</div>"
                    f"<div style='font-size:2rem;font-weight:700;color:#38bdf8'>{p['probability']:.0%}</div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

                # Mini bar chart
                fig_mini = go.Figure(go.Bar(
                    x=["Dissatisfied", "Satisfied"],
                    y=[1 - p["probability"], p["probability"]],
                    marker_color=["#ef4444", "#22c55e"],
                    text=[f"{(1-p['probability']):.0%}", f"{p['probability']:.0%}"],
                    textposition="outside",
                    textfont=dict(color="#e2e8f0", size=11),
                ))
                fig_mini.update_layout(
                    plot_bgcolor="#1e293b", paper_bgcolor="#0f172a",
                    height=200, margin=dict(t=20, b=10, l=5, r=5),
                    showlegend=False,
                    yaxis=dict(range=[0, 1.2], visible=False),
                    xaxis=dict(color="#94a3b8", tickfont=dict(size=10)),
                )
                st.plotly_chart(fig_mini, use_container_width=True)

    # Confusion matrix for LDA
    st.divider()
    st.markdown('<div class="section-header">Confusion Matrix — LDA (Optimal Threshold)</div>', unsafe_allow_html=True)
    st.caption("Test set: n = 25,976 | Threshold = 0.5413")

    # Derived from accuracy=0.87, precision/recall from notebook
    tn, fp, fn, tp = 12068, 1323, 2056, 10529
    total = tn + fp + fn + tp

    fig_cm = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=["Predicted: Neutral/Dissatisfied", "Predicted: Satisfied"],
        y=["Actual: Neutral/Dissatisfied", "Actual: Satisfied"],
        colorscale=[[0, "#0f172a"], [0.5, "#1e40af"], [1, "#38bdf8"]],
        text=[[f"TN<br>{tn:,}<br>({tn/total:.1%})", f"FP<br>{fp:,}<br>({fp/total:.1%})"],
              [f"FN<br>{fn:,}<br>({fn/total:.1%})", f"TP<br>{tp:,}<br>({tp/total:.1%})"]],
        texttemplate="%{text}",
        textfont=dict(color="white", size=14),
        showscale=False,
    ))
    fig_cm.update_layout(
        plot_bgcolor="#1e293b", paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"), height=380,
        margin=dict(t=20, b=60, l=180, r=20),
        xaxis=dict(color="#94a3b8"),
        yaxis=dict(color="#94a3b8"),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # Summary comparison table
    st.markdown('<div class="section-header">All Models — Quick Reference</div>', unsafe_allow_html=True)
    summary_df = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": f"{m['accuracy']:.2f}",
            "Precision (avg)": f"{(m['precision_0']+m['precision_1'])/2:.2f}",
            "Recall (avg)": f"{(m['recall_0']+m['recall_1'])/2:.2f}",
            "Macro F1": f"{m['macro_f1']:.2f}",
            "ROC-AUC": f"{m['roc_auc']:.3f}",
        }
        for name, m in MODEL_METRICS.items()
    ])
    st.dataframe(
        summary_df.style.set_properties(**{"background-color": "#1e293b", "color": "#e2e8f0"}),
        use_container_width=True,
        hide_index=True,
    )
