"""
================================================================================
 BANK CUSTOMER CHURN PREDICTION SYSTEM
 Algorithm : K-Nearest Neighbors (KNN)
 Framework : Streamlit
 Author    : Generated for Ranjith
 Description:
    A professional, deployment-ready Streamlit application that predicts
    whether a bank customer is likely to churn (leave the bank) using a
    K-Nearest Neighbors classifier. Includes synthetic data generation,
    CSV upload support, full preprocessing pipeline, model training with
    hyperparameter search, rich EDA & evaluation visualizations, a single
    customer prediction form, batch (CSV) prediction, and a prediction
    history log — all wrapped in a custom-styled, card-based UI.
================================================================================
"""

import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Bank Churn Predictor | KNN",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# 2. CUSTOM CSS — PROFESSIONAL THEME
# ==============================================================================
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    /* Hide default Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background: linear-gradient(180deg, #f5f7fb 0%, #eef1f8 100%);
    }

    /* ---------- Header Banner ---------- */
    .main-header {
        background: linear-gradient(120deg, #1e3c72 0%, #2a5298 50%, #6a3093 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px rgba(30, 60, 114, 0.35);
        position: relative;
        overflow: hidden;
    }
    .main-header::after {
        content: "";
        position: absolute;
        top: -50%; right: -10%;
        width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .main-header h1 {
        color: #ffffff;
        font-weight: 800;
        font-size: 2.2rem;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #dbe4f5;
        font-size: 1.02rem;
        margin-top: 0.4rem;
        font-weight: 400;
    }

    /* ---------- Section Titles ---------- */
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1e3c72;
        margin: 1.2rem 0 0.8rem 0;
        border-left: 5px solid #6a3093;
        padding-left: 12px;
    }

    /* ---------- Cards ---------- */
    .info-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.4rem 1.5rem;
        box-shadow: 0 4px 16px rgba(30, 60, 114, 0.08);
        border: 1px solid rgba(30, 60, 114, 0.06);
        height: 100%;
        transition: transform 0.15s ease;
    }
    .info-card:hover { transform: translateY(-3px); }
    .info-card h3 { color: #1e3c72; font-weight: 700; font-size: 1.05rem; margin-bottom: 0.5rem;}
    .info-card p { color: #566072; font-size: 0.92rem; line-height: 1.5; margin: 0;}

    /* ---------- Metric Cards ---------- */
    .metric-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.1rem 1rem;
        text-align: center;
        box-shadow: 0 4px 14px rgba(30, 60, 114, 0.08);
        border-top: 4px solid #2a5298;
    }
    .metric-card .value { font-size: 1.8rem; font-weight: 800; color: #1e3c72; }
    .metric-card .label { font-size: 0.85rem; color: #7a8398; font-weight: 500; margin-top: 2px;}

    /* ---------- Risk Badges ---------- */
    .risk-badge {
        display: inline-block;
        padding: 0.55rem 1.4rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
    }
    .risk-high { background: #ffe1e1; color: #c62828; border: 2px solid #ef9a9a; }
    .risk-medium { background: #fff4d6; color: #b8860b; border: 2px solid #ffe082; }
    .risk-low { background: #dff5e1; color: #1b7a3b; border: 2px solid #a5d6a7; }

    /* ---------- Buttons ---------- */
    div.stButton > button {
        background: linear-gradient(120deg, #1e3c72, #6a3093);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.6rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(30, 60, 114, 0.25);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(30, 60, 114, 0.35);
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #16294f 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #eef1f8 !important;
    }
    section[data-testid="stSidebar"] .stRadio > label {
        font-weight: 600;
    }

    /* ---------- Dataframe ---------- */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Badge pill for status */
    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .pill-green { background: #d4f5dd; color: #1b7a3b; }
    .pill-red { background: #fbdcdc; color: #c62828; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==============================================================================
# 3. CONSTANTS
# ==============================================================================
CATEGORICAL_COLS = ["Geography", "Gender"]
NUMERIC_COLS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]
TARGET_COL = "Exited"
FEATURE_ORDER = ["CreditScore", "Geography", "Gender", "Age", "Tenure",
                  "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
                  "EstimatedSalary"]

GEOGRAPHIES = ["France", "Germany", "Spain"]
GENDERS = ["Male", "Female"]

# ==============================================================================
# 4. SESSION STATE INITIALIZATION
# ==============================================================================
def init_session_state():
    defaults = {
        "data": None,
        "model": None,
        "scaler": None,
        "encoders": {},
        "trained": False,
        "best_k": 5,
        "metrics": {},
        "cv_results": None,
        "history": [],
        "X_test": None,
        "y_test": None,
        "y_pred": None,
        "y_prob": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()

# ==============================================================================
# 5. SYNTHETIC DATA GENERATION
# ==============================================================================
@st.cache_data(show_spinner=False)
def generate_synthetic_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic bank-churn dataset."""
    rng = np.random.default_rng(seed)

    credit_score = rng.normal(650, 96, n_samples).clip(350, 850).astype(int)
    geography = rng.choice(GEOGRAPHIES, size=n_samples, p=[0.5, 0.25, 0.25])
    gender = rng.choice(GENDERS, size=n_samples, p=[0.55, 0.45])
    age = rng.normal(38, 10, n_samples).clip(18, 92).astype(int)
    tenure = rng.integers(0, 11, n_samples)
    has_balance = rng.random(n_samples) > 0.35
    balance = np.where(
        has_balance,
        rng.normal(95000, 45000, n_samples).clip(0, 260000),
        0.0,
    ).round(2)
    num_products = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.5, 0.4, 0.08, 0.02])
    has_cr_card = rng.choice([0, 1], size=n_samples, p=[0.29, 0.71])
    is_active = rng.choice([0, 1], size=n_samples, p=[0.48, 0.52])
    estimated_salary = rng.uniform(11000, 200000, n_samples).round(2)

    # ---- Churn probability model (drives realistic patterns) ----
    prob = np.full(n_samples, 0.10)
    prob += (age > 50) * 0.22
    prob += (age > 60) * 0.15
    prob += (geography == "Germany") * 0.14
    prob += (num_products >= 3) * 0.30
    prob += (num_products == 1) * 0.05
    prob += (is_active == 0) * 0.18
    prob += (has_balance == 0) * 0.05
    prob += (credit_score < 500) * 0.10
    prob += (gender == "Female") * 0.04
    prob += (tenure <= 1) * 0.06
    prob -= (tenure >= 8) * 0.05
    prob = prob.clip(0.02, 0.92)

    exited = (rng.random(n_samples) < prob).astype(int)

    df = pd.DataFrame(
        {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember": is_active,
            "EstimatedSalary": estimated_salary,
            "Exited": exited,
        }
    )
    return df


# ==============================================================================
# 6. PREPROCESSING
# ==============================================================================
def fit_preprocessors(df: pd.DataFrame):
    """Fit LabelEncoders (categorical) + StandardScaler (numeric) on a dataframe."""
    encoders = {}
    df_enc = df.copy()

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

    X = df_enc[FEATURE_ORDER]
    y = df_enc[TARGET_COL] if TARGET_COL in df_enc.columns else None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, encoders


def transform_with_fitted(df: pd.DataFrame, scaler: StandardScaler, encoders: dict):
    """Transform new data using already-fitted encoders/scaler."""
    df_enc = df.copy()
    for col in CATEGORICAL_COLS:
        le = encoders[col]
        # handle unseen categories gracefully
        df_enc[col] = df_enc[col].astype(str).apply(
            lambda x: x if x in le.classes_ else le.classes_[0]
        )
        df_enc[col] = le.transform(df_enc[col])

    X = df_enc[FEATURE_ORDER]
    X_scaled = scaler.transform(X)
    return X_scaled


# ==============================================================================
# 7. MODEL TRAINING
# ==============================================================================
def train_knn_model(X_train, y_train, k_min=3, k_max=25):
    """Grid-search the best K for KNN using 5-fold CV, then fit final model."""
    param_grid = {"n_neighbors": list(range(k_min, k_max + 1, 2)), "weights": ["uniform", "distance"]}
    base_model = KNeighborsClassifier()
    grid = GridSearchCV(base_model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    return best_model, grid.best_params_["n_neighbors"], grid


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    return metrics, y_pred, y_prob


# ==============================================================================
# 8. CHART HELPERS
# ==============================================================================
def plot_churn_distribution(df):
    counts = df[TARGET_COL].value_counts().rename({0: "Retained", 1: "Churned"})
    fig = px.pie(
        names=counts.index,
        values=counts.values,
        color=counts.index,
        color_discrete_map={"Retained": "#2a9d68", "Churned": "#e0574c"},
        hole=0.5,
    )
    fig.update_traces(textinfo="percent+label", textfont_size=13)
    fig.update_layout(
        showlegend=True,
        margin=dict(t=20, b=20, l=10, r=10),
        height=340,
        legend=dict(orientation="h", y=-0.1),
    )
    return fig


def plot_age_distribution(df):
    fig = px.histogram(
        df,
        x="Age",
        color=df[TARGET_COL].map({0: "Retained", 1: "Churned"}),
        nbins=30,
        color_discrete_map={"Retained": "#2a5298", "Churned": "#e0574c"},
        barmode="overlay",
        opacity=0.65,
    )
    fig.update_layout(
        height=340, margin=dict(t=20, b=20, l=10, r=10), legend_title_text=""
    )
    return fig


def plot_balance_box(df):
    fig = px.box(
        df,
        x=df[TARGET_COL].map({0: "Retained", 1: "Churned"}),
        y="Balance",
        color=df[TARGET_COL].map({0: "Retained", 1: "Churned"}),
        color_discrete_map={"Retained": "#2a5298", "Churned": "#e0574c"},
    )
    fig.update_layout(height=340, margin=dict(t=20, b=20, l=10, r=10), showlegend=False,
                       xaxis_title="", legend_title_text="")
    return fig


def plot_geography_bar(df):
    grp = df.groupby(["Geography", TARGET_COL]).size().reset_index(name="Count")
    grp[TARGET_COL] = grp[TARGET_COL].map({0: "Retained", 1: "Churned"})
    fig = px.bar(
        grp,
        x="Geography",
        y="Count",
        color=TARGET_COL,
        barmode="group",
        color_discrete_map={"Retained": "#2a5298", "Churned": "#e0574c"},
    )
    fig.update_layout(height=340, margin=dict(t=20, b=20, l=10, r=10), legend_title_text="")
    return fig


def plot_correlation_heatmap(df):
    df_corr = df.copy()
    for col in CATEGORICAL_COLS:
        df_corr[col] = LabelEncoder().fit_transform(df_corr[col].astype(str))
    corr = df_corr[FEATURE_ORDER + [TARGET_COL]].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(height=480, margin=dict(t=20, b=20, l=10, r=10))
    return fig


def plot_confusion_matrix(cm):
    labels = ["Retained (0)", "Churned (1)"]
    fig = px.imshow(
        cm,
        text_auto=True,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    fig.update_layout(height=380, margin=dict(t=20, b=20, l=10, r=10))
    return fig


def plot_roc_curve(y_test, y_prob, auc_score):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                              name=f"KNN (AUC = {auc_score:.3f})",
                              line=dict(color="#6a3093", width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                              name="Random Guess",
                              line=dict(color="#bbb", width=2, dash="dash")))
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=380,
        margin=dict(t=20, b=20, l=10, r=10),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def plot_k_vs_accuracy(cv_results):
    results_df = pd.DataFrame(cv_results.cv_results_)
    results_df = results_df[results_df["param_weights"] == "distance"]
    fig = px.line(
        results_df.sort_values("param_n_neighbors"),
        x="param_n_neighbors",
        y="mean_test_score",
        markers=True,
    )
    fig.update_traces(line=dict(color="#1e3c72", width=3), marker=dict(size=8, color="#6a3093"))
    fig.update_layout(
        xaxis_title="K (Number of Neighbors)",
        yaxis_title="Mean CV Accuracy",
        height=340,
        margin=dict(t=20, b=20, l=10, r=10),
    )
    return fig


def plot_gauge(probability):
    color = "#e0574c" if probability >= 0.6 else ("#e8a33d" if probability >= 0.3 else "#2a9d68")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 40}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30], "color": "#dff5e1"},
                    {"range": [30, 60], "color": "#fff4d6"},
                    {"range": [60, 100], "color": "#ffe1e1"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": probability * 100,
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(t=30, b=10, l=30, r=30))
    return fig


# ==============================================================================
# 9. SIDEBAR
# ==============================================================================
def sidebar():
    st.sidebar.markdown(
        """
        <div style="text-align:center; padding: 0.5rem 0 1.2rem 0;">
            <div style="font-size:2.4rem;">🏦</div>
            <div style="font-size:1.25rem; font-weight:800;">Churn Predictor</div>
            <div style="font-size:0.78rem; opacity:0.75;">KNN-Powered Analytics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigate",
        [
            "🏠 Home",
            "📂 Dataset",
            "📊 Exploratory Analysis",
            "🧠 Train Model",
            "🔮 Predict Customer",
            "📑 Batch Prediction",
            "🕘 Prediction History",
            "ℹ️ About",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Status")

    if st.session_state.data is not None:
        st.sidebar.markdown(
            f'<span class="status-pill pill-green">Dataset Loaded ({len(st.session_state.data)} rows)</span>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<span class="status-pill pill-red">No Dataset</span>', unsafe_allow_html=True
        )

    if st.session_state.trained:
        acc = st.session_state.metrics.get("accuracy", 0)
        st.sidebar.markdown(
            f'<span class="status-pill pill-green">Model Trained (Acc: {acc*100:.1f}%)</span>',
            unsafe_allow_html=True,
        )
        st.sidebar.markdown(f"**Best K:** {st.session_state.best_k}")
    else:
        st.sidebar.markdown(
            '<span class="status-pill pill-red">Model Not Trained</span>', unsafe_allow_html=True
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<div style='font-size:0.75rem; opacity:0.7;'>Predictions logged this session: "
        f"<b>{len(st.session_state.history)}</b></div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<div style='font-size:0.72rem; opacity:0.55; margin-top:1.2rem;'>"
        "Built with Streamlit + scikit-learn</div>",
        unsafe_allow_html=True,
    )

    return page


# ==============================================================================
# 10. PAGE: HOME
# ==============================================================================
def page_home():
    st.markdown(
        """
        <div class="main-header">
            <h1>🏦 Bank Customer Churn Prediction System</h1>
            <p>Predict customer attrition risk using a K-Nearest Neighbors (KNN) classifier —
            complete with data exploration, model training, and real-time predictions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("📂", "1. Load Data", "Generate a synthetic dataset or upload your own bank customer CSV file."),
        ("📊", "2. Explore", "Visualize churn patterns across age, geography, balance, and activity."),
        ("🧠", "3. Train KNN", "Automatically tune K via cross-validation and evaluate performance."),
        ("🔮", "4. Predict", "Score individual customers or entire batches for churn risk."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], steps):
        col.markdown(
            f"""<div class="info-card"><h3>{icon} {title}</h3><p>{desc}</p></div>""",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">About K-Nearest Neighbors (KNN)</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-card">
        <p>
        KNN is a simple, instance-based supervised learning algorithm. To classify a new
        customer, it looks at the <b>K</b> most similar existing customers (based on scaled
        feature distance) and takes a majority vote of their churn outcomes. It works well
        for churn prediction because customers with similar profiles (age, balance, activity,
        product usage) tend to behave similarly. This app automatically searches for the best
        value of K using 5-fold cross-validation.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.data is not None:
        st.markdown('<div class="section-title">Current Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(st.session_state.data.head(10), use_container_width=True)
    else:
        st.info("👉 Head to **Dataset** in the sidebar to generate or upload data to get started.")


# ==============================================================================
# 11. PAGE: DATASET
# ==============================================================================
def page_dataset():
    st.markdown('<div class="section-title">📂 Load Dataset</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["⚙️ Generate Synthetic Data", "⬆️ Upload CSV"])

    with tab1:
        st.write("Generate a realistic synthetic bank churn dataset for training and testing.")
        n_samples = st.slider("Number of customers", 500, 20000, 5000, step=500)
        seed = st.number_input("Random seed", value=42, step=1)
        if st.button("🎲 Generate Dataset"):
            with st.spinner("Generating synthetic customer data..."):
                df = generate_synthetic_data(n_samples, seed)
                st.session_state.data = df
                st.session_state.trained = False
            st.success(f"Generated {len(df)} customer records!")

    with tab2:
        st.write(
            "Upload a CSV with columns: `CreditScore, Geography, Gender, Age, Tenure, "
            "Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited`"
        )
        uploaded = st.file_uploader("Choose CSV file", type=["csv"])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                missing = set(FEATURE_ORDER + [TARGET_COL]) - set(df.columns)
                if missing:
                    st.error(f"Missing required columns: {', '.join(missing)}")
                else:
                    st.session_state.data = df
                    st.session_state.trained = False
                    st.success(f"Loaded {len(df)} rows from uploaded file!")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    if st.session_state.data is not None:
        df = st.session_state.data
        st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        churn_rate = df[TARGET_COL].mean() * 100 if TARGET_COL in df.columns else 0
        for col, (val, label) in zip(
            [m1, m2, m3, m4],
            [
                (f"{len(df):,}", "Total Customers"),
                (f"{churn_rate:.1f}%", "Churn Rate"),
                (f"{df['Age'].mean():.1f}", "Avg Age"),
                (f"${df['Balance'].mean():,.0f}", "Avg Balance"),
            ],
        ):
            col.markdown(
                f'<div class="metric-card"><div class="value">{val}</div>'
                f'<div class="label">{label}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df.head(50), use_container_width=True, height=320)

        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download Full Dataset as CSV",
            data=csv_buf.getvalue(),
            file_name="bank_churn_dataset.csv",
            mime="text/csv",
        )
    else:
        st.warning("No dataset loaded yet. Generate or upload one above.")


# ==============================================================================
# 12. PAGE: EDA
# ==============================================================================
def page_eda():
    st.markdown('<div class="section-title">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    if st.session_state.data is None:
        st.warning("Please load a dataset first from the **Dataset** page.")
        return

    df = st.session_state.data

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Churn Distribution**")
        st.plotly_chart(plot_churn_distribution(df), use_container_width=True)
    with c2:
        st.markdown("**Age Distribution by Churn Status**")
        st.plotly_chart(plot_age_distribution(df), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Balance Spread by Churn Status**")
        st.plotly_chart(plot_balance_box(df), use_container_width=True)
    with c4:
        st.markdown("**Churn Count by Geography**")
        st.plotly_chart(plot_geography_bar(df), use_container_width=True)

    st.markdown("**Feature Correlation Heatmap**")
    st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)

    st.markdown('<div class="section-title">Summary Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df.describe().T, use_container_width=True)


# ==============================================================================
# 13. PAGE: TRAIN MODEL
# ==============================================================================
def page_train():
    st.markdown('<div class="section-title">🧠 Train KNN Model</div>', unsafe_allow_html=True)

    if st.session_state.data is None:
        st.warning("Please load a dataset first from the **Dataset** page.")
        return

    df = st.session_state.data

    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
    with col2:
        k_min = st.number_input("Min K to search", 1, 15, 3, step=2)
    with col3:
        k_max = st.number_input("Max K to search", 15, 51, 25, step=2)

    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Preprocessing data and searching for best K via 5-fold CV..."):
            X_scaled, y, scaler, encoders = fit_preprocessors(df)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42, stratify=y
            )
            model, best_k, grid = train_knn_model(X_train, y_train, k_min, k_max)
            metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)

        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.encoders = encoders
        st.session_state.best_k = best_k
        st.session_state.metrics = metrics
        st.session_state.cv_results = grid
        st.session_state.trained = True
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred
        st.session_state.y_prob = y_prob

        st.success(f"✅ Model trained successfully! Optimal K = {best_k}")

    if st.session_state.trained:
        st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
        m = st.session_state.metrics
        cols = st.columns(5)
        labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
        keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        for col, label, key in zip(cols, labels, keys):
            col.markdown(
                f'<div class="metric-card"><div class="value">{m[key]*100:.1f}%</div>'
                f'<div class="label">{label}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Confusion Matrix**")
            cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
            st.plotly_chart(plot_confusion_matrix(cm), use_container_width=True)
        with c2:
            st.markdown("**ROC Curve**")
            st.plotly_chart(
                plot_roc_curve(st.session_state.y_test, st.session_state.y_prob, m["roc_auc"]),
                use_container_width=True,
            )

        st.markdown("**K vs. Cross-Validation Accuracy**")
        st.plotly_chart(plot_k_vs_accuracy(st.session_state.cv_results), use_container_width=True)
    else:
        st.info("Configure the settings above and click **Train Model** to begin.")


# ==============================================================================
# 14. PAGE: SINGLE PREDICTION
# ==============================================================================
def page_predict():
    st.markdown('<div class="section-title">🔮 Predict Customer Churn Risk</div>', unsafe_allow_html=True)

    if not st.session_state.trained:
        st.warning("Please train the model first from the **Train Model** page.")
        return

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            credit_score = st.slider("Credit Score", 300, 850, 650)
            geography = st.selectbox("Geography", GEOGRAPHIES)
            gender = st.selectbox("Gender", GENDERS)
        with c2:
            age = st.slider("Age", 18, 92, 35)
            tenure = st.slider("Tenure (years with bank)", 0, 10, 5)
            balance = st.number_input("Account Balance ($)", 0.0, 300000.0, 50000.0, step=1000.0)
        with c3:
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
            has_cr_card = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
            is_active = st.radio("Active Member?", ["Yes", "No"], horizontal=True)
            estimated_salary = st.number_input("Estimated Salary ($)", 0.0, 250000.0, 80000.0, step=1000.0)

        submitted = st.form_submit_button("🔍 Predict Churn Risk", type="primary")

    if submitted:
        input_df = pd.DataFrame(
            [{
                "CreditScore": credit_score,
                "Geography": geography,
                "Gender": gender,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_products,
                "HasCrCard": 1 if has_cr_card == "Yes" else 0,
                "IsActiveMember": 1 if is_active == "Yes" else 0,
                "EstimatedSalary": estimated_salary,
            }]
        )

        X_input = transform_with_fitted(input_df, st.session_state.scaler, st.session_state.encoders)
        prediction = st.session_state.model.predict(X_input)[0]
        probability = st.session_state.model.predict_proba(X_input)[0][1]

        if probability >= 0.6:
            risk_label, risk_class = "HIGH RISK", "risk-high"
        elif probability >= 0.3:
            risk_label, risk_class = "MEDIUM RISK", "risk-medium"
        else:
            risk_label, risk_class = "LOW RISK", "risk-low"

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2 = st.columns([1, 1.2])
        with r1:
            st.markdown(
                f"""
                <div class="info-card" style="text-align:center;">
                    <h3>Prediction Result</h3>
                    <p style="font-size:1.4rem; font-weight:800; color:#1e3c72; margin:0.6rem 0;">
                        {"Customer Likely to Churn" if prediction == 1 else "Customer Likely to Stay"}
                    </p>
                    <span class="risk-badge {risk_class}">{risk_label}</span>
                    <p style="margin-top:1rem;">Churn Probability: <b>{probability*100:.1f}%</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r2:
            st.plotly_chart(plot_gauge(probability), use_container_width=True)

        # Save to history
        st.session_state.history.append(
            {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "CreditScore": credit_score,
                "Geography": geography,
                "Gender": gender,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_products,
                "HasCrCard": has_cr_card,
                "IsActiveMember": is_active,
                "EstimatedSalary": estimated_salary,
                "Prediction": "Churn" if prediction == 1 else "Retained",
                "Probability": f"{probability*100:.1f}%",
                "Risk Level": risk_label,
            }
        )
        st.success("Prediction saved to history log ✅")


# ==============================================================================
# 15. PAGE: BATCH PREDICTION
# ==============================================================================
def page_batch():
    st.markdown('<div class="section-title">📑 Batch Prediction</div>', unsafe_allow_html=True)

    if not st.session_state.trained:
        st.warning("Please train the model first from the **Train Model** page.")
        return

    st.write(
        "Upload a CSV of customers (same feature columns as training data, without `Exited`) "
        "to score churn risk in bulk."
    )
    uploaded = st.file_uploader("Choose CSV file for batch scoring", type=["csv"], key="batch_upload")

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            missing = set(FEATURE_ORDER) - set(batch_df.columns)
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
                return

            X_batch = transform_with_fitted(batch_df, st.session_state.scaler, st.session_state.encoders)
            preds = st.session_state.model.predict(X_batch)
            probs = st.session_state.model.predict_proba(X_batch)[:, 1]

            results = batch_df.copy()
            results["Churn_Prediction"] = np.where(preds == 1, "Churn", "Retained")
            results["Churn_Probability"] = (probs * 100).round(2)
            results["Risk_Level"] = np.where(
                probs >= 0.6, "HIGH", np.where(probs >= 0.3, "MEDIUM", "LOW")
            )

            st.success(f"Scored {len(results)} customers!")

            m1, m2, m3 = st.columns(3)
            m1.markdown(
                f'<div class="metric-card"><div class="value">{len(results)}</div>'
                f'<div class="label">Total Scored</div></div>',
                unsafe_allow_html=True,
            )
            m2.markdown(
                f'<div class="metric-card"><div class="value">{(preds==1).sum()}</div>'
                f'<div class="label">Predicted Churn</div></div>',
                unsafe_allow_html=True,
            )
            m3.markdown(
                f'<div class="metric-card"><div class="value">{(results["Risk_Level"]=="HIGH").sum()}</div>'
                f'<div class="label">High Risk Customers</div></div>',
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(results, use_container_width=True, height=380)

            csv_buf = io.StringIO()
            results.to_csv(csv_buf, index=False)
            st.download_button(
                "⬇️ Download Predictions as CSV",
                data=csv_buf.getvalue(),
                file_name="churn_batch_predictions.csv",
                mime="text/csv",
            )

            st.markdown("**Risk Level Breakdown**")
            risk_counts = results["Risk_Level"].value_counts().reset_index()
            risk_counts.columns = ["Risk_Level", "Count"]
            fig = px.bar(
                risk_counts,
                x="Risk_Level",
                y="Count",
                color="Risk_Level",
                color_discrete_map={"HIGH": "#e0574c", "MEDIUM": "#e8a33d", "LOW": "#2a9d68"},
            )
            fig.update_layout(height=340, margin=dict(t=20, b=20, l=10, r=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing file: {e}")


# ==============================================================================
# 16. PAGE: HISTORY
# ==============================================================================
def page_history():
    st.markdown('<div class="section-title">🕘 Prediction History</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No predictions logged yet. Try the **Predict Customer** page.")
        return

    hist_df = pd.DataFrame(st.session_state.history)

    m1, m2, m3 = st.columns(3)
    m1.markdown(
        f'<div class="metric-card"><div class="value">{len(hist_df)}</div>'
        f'<div class="label">Total Predictions</div></div>',
        unsafe_allow_html=True,
    )
    m2.markdown(
        f'<div class="metric-card"><div class="value">{(hist_df["Prediction"]=="Churn").sum()}</div>'
        f'<div class="label">Churn Predictions</div></div>',
        unsafe_allow_html=True,
    )
    m3.markdown(
        f'<div class="metric-card"><div class="value">{(hist_df["Risk Level"]=="HIGH RISK").sum()}</div>'
        f'<div class="label">High Risk Flags</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(hist_df.iloc[::-1], use_container_width=True, height=400)

    c1, c2 = st.columns(2)
    with c1:
        csv_buf = io.StringIO()
        hist_df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download History as CSV",
            data=csv_buf.getvalue(),
            file_name="prediction_history.csv",
            mime="text/csv",
        )
    with c2:
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()


# ==============================================================================
# 17. PAGE: ABOUT
# ==============================================================================
def page_about():
    st.markdown('<div class="section-title">ℹ️ About This Application</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-card">
        <h3>🏦 Bank Customer Churn Prediction System</h3>
        <p>
        This application demonstrates an end-to-end machine learning workflow for predicting
        bank customer churn using the <b>K-Nearest Neighbors (KNN)</b> algorithm. It covers
        data generation/ingestion, preprocessing, model training with hyperparameter tuning,
        evaluation, and real-time / batch prediction — all within an interactive Streamlit
        interface.
        </p>
        </div>
        <br>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="info-card">
            <h3>🛠️ Tech Stack</h3>
            <p>
            • Streamlit — UI framework<br>
            • scikit-learn — KNN, preprocessing, model evaluation<br>
            • Plotly — interactive charts<br>
            • Pandas / NumPy — data handling
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="info-card">
            <h3>📋 Features Used</h3>
            <p>
            CreditScore, Geography, Gender, Age, Tenure, Balance,
            NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <br>
        <div class="info-card">
        <h3>⚠️ Disclaimer</h3>
        <p>
        This app is built for educational and demonstration purposes. Predictions are based on
        a model trained on synthetic or user-supplied data and should not be used for real
        financial or business decisions without further validation.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 18. MAIN APP ROUTER
# ==============================================================================
def main():
    page = sidebar()

    if page == "🏠 Home":
        page_home()
    elif page == "📂 Dataset":
        page_dataset()
    elif page == "📊 Exploratory Analysis":
        page_eda()
    elif page == "🧠 Train Model":
        page_train()
    elif page == "🔮 Predict Customer":
        page_predict()
    elif page == "📑 Batch Prediction":
        page_batch()
    elif page == "🕘 Prediction History":
        page_history()
    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()
