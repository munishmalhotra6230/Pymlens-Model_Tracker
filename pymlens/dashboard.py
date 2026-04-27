import streamlit as st
import requests
import pandas as pd
import json
import plotly.express as px
from streamlit_agraph import agraph, Node, Edge, Config

import random

st.set_page_config(layout="wide")

# =========================
# DYNAMIC APP THEME ENGINE
# =========================
if "theme_seed" not in st.session_state:
    st.session_state.theme_seed = random.randint(0, 4)
if "chart_theme" not in st.session_state:
    st.session_state.chart_theme = random.choice(["plotly", "plotly_dark", "plotly_white", "ggplot2", "seaborn", "simple_white"])

app_themes = [
    {"primary": "#FF4B4B", "bg": "", "font": "'Inter', sans-serif"},
    {"primary": "#00ADB5", "bg": "linear-gradient(135deg, rgba(0,173,181,0.08) 0%, rgba(34,40,49,0.02) 100%)", "font": "'Roboto', sans-serif"},
    {"primary": "#FCA311", "bg": "linear-gradient(to right, rgba(252,163,17,0.05), rgba(20,33,61,0.02))", "font": "'Trebuchet MS', sans-serif"},
    {"primary": "#E94560", "bg": "radial-gradient(circle, rgba(233,69,96,0.05) 0%, rgba(26,26,46,0.02) 100%)", "font": "'Courier New', monospace"},
    {"primary": "#7161EF", "bg": "linear-gradient(to top, rgba(113,97,239,0.08), rgba(255,255,255,0.01))", "font": "'Georgia', serif"}
]

current_theme = app_themes[st.session_state.theme_seed]

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Roboto:wght@400;600&display=swap');
    
    .stApp {{
        background: {current_theme['bg']} !important;
    }}
    
    html, body, [class*="css"] {{
        font-family: {current_theme['font']} !important;
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        font-family: {current_theme['font']} !important;
    }}
    
    /* Dynamic Buttons */
    .stButton>button {{
        border: 2px solid {current_theme['primary']} !important;
        color: {current_theme['primary']} !important;
        border-radius: 8px;
        transition: 0.3s;
    }}
    
    .stButton>button:hover {{
        background-color: {current_theme['primary']} !important;
        color: white !important;
        transform: translateY(-2px);
    }}
</style>
""", unsafe_allow_html=True)

# =========================
# DATA FETCH
# =========================
@st.cache_data
def load_data():
    response = requests.get("http://127.0.0.1:5000/get_results")
    data = response.json()

    params_df = pd.DataFrame(data["params"])
    results_df = pd.DataFrame(data["results"])

    params_df["Params"] = params_df["Params"].apply(json.loads)

    df = pd.merge(results_df, params_df, on=["Experiment_name", "Model"], how="left")
    return df

df = load_data()

st.title("🚀 ML Decision Dashboard")

# =========================
# NAVIGATION
# =========================
page = st.sidebar.radio("Navigate", ["Model Comparison", "Infographics"])

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Appearance")

# Randomizer Button
if st.sidebar.button("🎲 Randomize Entire App Theme"):
    st.session_state.theme_seed = random.randint(0, 4)
    st.session_state.chart_theme = random.choice(["plotly", "plotly_dark", "plotly_white", "ggplot2", "seaborn", "simple_white"])
    st.rerun()

chart_themes = ["plotly", "plotly_dark", "plotly_white", "ggplot2", "seaborn", "simple_white", "none"]
selected_chart_theme = st.sidebar.selectbox("Chart Theme", chart_themes, index=chart_themes.index(st.session_state.chart_theme))

if selected_chart_theme != st.session_state.chart_theme:
    st.session_state.chart_theme = selected_chart_theme
    st.rerun()

chart_theme = st.session_state.chart_theme

experiments = df["Experiment_name"].unique()

# =========================================================
# 📊 PAGE 1: MODEL COMPARISON (UPGRADED)
# =========================================================
if page == "Model Comparison":

    st.header("📊 Model Comparison")

    selected_exp = st.sidebar.selectbox("Select Experiment", experiments)
    filtered_df = df[df["Experiment_name"] == selected_exp]

    # Leaderboard
    st.subheader("🏆 Leaderboard")
    st.dataframe(filtered_df.sort_values(by="F1_Score", ascending=False))

    # Melt for charts
    melted = filtered_df.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "F1_Score", "Precision", "Recall"],
        var_name="Metric",
        value_name="Score"
    )

    # Grouped Bar
    st.subheader("📈 Metric Comparison")
    fig = px.bar(melted, x="Model", y="Score", color="Metric", barmode="group", template=chart_theme)
    st.plotly_chart(fig, use_container_width=True)

    # Radar
    st.subheader("🕸️ Radar View")
    fig = px.line_polar(melted, r="Score", theta="Metric", color="Model", line_close=True, template=chart_theme)
    st.plotly_chart(fig, use_container_width=True)

    # Precision vs Recall
    st.subheader("⚖️ Precision vs Recall")
    fig = px.scatter(
        filtered_df,
        x="Precision",
        y="Recall",
        size="F1_Score",
        color="Model",
        text="Model",
        template=chart_theme
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cross-val stability
    st.subheader("📊 Stability (Cross Validation)")
    fig = px.bar(filtered_df, x="Model", y="cross_val_score", color="cross_val_score", template=chart_theme)
    st.plotly_chart(fig, use_container_width=True)

    # Model Parameters Copy Feature
    st.markdown("---")
    st.subheader("📋 Copy Model Parameters")
    st.write("Select a model below to view and copy its precise hyperparameters for reproduction.")
    
    models_in_exp = filtered_df["Model"].unique()
    selected_model_params = st.selectbox("Select Model to Copy Params", models_in_exp)
    
    if selected_model_params:
        model_params = filtered_df[filtered_df["Model"] == selected_model_params].iloc[0]["Params"]
        st.code(json.dumps(model_params, indent=4), language="json")


# =========================================================
# 🧠 PAGE 2: REAL NODE-BASED INFOGRAPHICS
# =========================================================
elif page == "Infographics":

    st.header("🧠 Interactive Model Explorer (Sunburst)")
    st.markdown("Click on an **Experiment** circle to see its Models. Click on a **Model** to see its Metrics. Click the center to zoom back out!")

    import plotly.graph_objects as go

    ids = ["Root"]
    labels = ["All Experiments"]
    parents = [""]
    hovertexts = ["Click to explore experiments"]

    for exp in experiments:
        exp_df = df[df["Experiment_name"] == exp]
        models = exp_df["Model"].unique()
        num_models = len(models)
        
        ids.append(exp)
        labels.append(exp)
        parents.append("Root")
        hovertexts.append(f"<b>{exp}</b><br>Total Models Run: {num_models}")
        
        for model in models:
            row = exp_df[exp_df["Model"] == model].iloc[0]
            model_id = f"{exp}____{model}"
            
            ids.append(model_id)
            labels.append(model)
            parents.append(exp)
            
            # Show all metrics in hover for the model itself!
            metrics_html = f"<b>{model}</b><br>"
            metrics_html += f"Accuracy: {row['Accuracy']:.4f}<br>"
            metrics_html += f"Precision: {row['Precision']:.4f}<br>"
            metrics_html += f"Recall: {row['Recall']:.4f}<br>"
            metrics_html += f"F1: {row['F1_Score']:.4f}"
            
            hovertexts.append(metrics_html)
            
            # Add metrics as leaf nodes
            for m_name in ["Accuracy", "Precision", "Recall", "F1_Score"]:
                m_val = row[m_name]
                m_id = f"{model_id}____{m_name}"
                
                ids.append(m_id)
                labels.append(f"{m_name}<br>{m_val:.4f}")
                parents.append(model_id)
                hovertexts.append(f"{m_name}: {m_val:.4f}")

    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        hoverinfo="text",
        hovertext=hovertexts,
        insidetextorientation='radial'
    ))
    
    fig.update_layout(
        margin=dict(t=40, l=0, r=0, b=0), 
        height=750,
        font=dict(size=14),
        template=chart_theme
    )

    st.plotly_chart(fig, use_container_width=True)
