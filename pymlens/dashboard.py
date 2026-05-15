import streamlit as st
import sqlite3
import os
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import random
from sklearn.metrics  import ConfusionMatrixDisplay
import numpy as np 

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
    .stApp {{ background: {current_theme['bg']} !important; }}
    html, body, [class*="css"] {{ font-family: {current_theme['font']} !important; }}
    h1, h2, h3, h4, h5, h6 {{ font-family: {current_theme['font']} !important; }}
    .stButton>button {{
        border: 2px solid {current_theme['primary']} !important;
        color: {current_theme['primary']} !important;
        border-radius: 8px; transition: 0.3s;
    }}
    .stButton>button:hover {{
        background-color: {current_theme['primary']} !important;
        color: white !important; transform: translateY(-2px);
    }}
</style>
""", unsafe_allow_html=True)

# =========================
# DATA FETCH
# =========================
DB_PATH = os.path.join(os.path.expanduser("~"), ".pymlens", "experiments.db")

@st.cache_data
def load_classification_data():
    conn = sqlite3.connect(DB_PATH)
    try:
        results = pd.read_sql_query("""
            SELECT e.Experiment_name, s.Model, s.Accuracy,
                   s.Precision, s.Recall, s.F1_Score,
                   s.cross_val_score, s.Params, s.confusion_matrix, s.Key_word
            FROM Experiments e
            JOIN Scores s ON e.ID = s.Experiment_ID
        """, conn)
    except:
        results = pd.DataFrame()
    conn.close()

    if not results.empty:
        if "Params" not in results.columns:
            results["Params"] = "{}"
        results["Params"] = results["Params"].apply(
            lambda x: json.loads(x) if x else {}
        )
        # Parse confusion matrix JSON (stored as text) into Python lists
        if "confusion_matrix" in results.columns:
            results["confusion_matrix"] = results["confusion_matrix"].apply(
                lambda x: json.loads(x) if x and isinstance(x, str) else x
            )
        # Ensure Key_word exists and provide a display name that prefers the keyword
        if "Key_word" not in results.columns:
            results["Key_word"] = ""
        results["Display_Name"] = results.apply(
            lambda row: row["Key_word"] if row["Key_word"] else row["Model"], axis=1
        )
        # Normalize possible training-score column names to `Train_Score`
        possible_train_cols = [
            "Training_Score", "Train_Score", "TrainingAccuracy",
            "Train_Accuracy", "train_score", "training_score", "TrainScore"
        ]
        for col in possible_train_cols:
            if col in results.columns:
                results["Train_Score"] = results[col]
                break
        if "Train_Score" not in results.columns:
            results["Train_Score"] = float("nan")
    return results

@st.cache_data
def load_regression_data():
    conn = sqlite3.connect(DB_PATH)
    try:
        results = pd.read_sql_query("""
            SELECT e.Experiment_name, s.Model, s.MSE,
                   s.MAE, s.R2, s.RMSE, s.cv_score, s.Params, s.Key_word
            FROM Experiments e
            JOIN Regression_Scores s ON e.ID = s.Experiment_ID
        """, conn)
    except:
        results = pd.DataFrame()
    conn.close()

    if not results.empty:
        if "Params" not in results.columns:
            results["Params"] = "{}"
        results["Params"] = results["Params"].apply(
            lambda x: json.loads(x) if x else {}
        )
        if "Key_word" not in results.columns:
            results["Key_word"] = ""
        results["Display_Name"] = results.apply(
            lambda row: row["Key_word"] if row["Key_word"] else row["Model"], axis=1
        )
        # Normalize possible training-score column names to `Train_Score` for regression
        possible_train_cols = ["Training_Score", "Train_Score", "train_score", "training_score", "TrainScore"]
        for col in possible_train_cols:
            if col in results.columns:
                results["Train_Score"] = results[col]
                break
        if "Train_Score" not in results.columns:
            results["Train_Score"] = float("nan")
    return results

df = load_classification_data()
df_reg = load_regression_data()

st.title("🚀 ML Decision Dashboard")


# =========================
# NAVIGATION
# =========================
page = st.sidebar.radio("Navigate", ["Model Comparison", "Infographics", "Critics"])
st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Appearance")

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

# =========================================================
# PAGE 1: MODEL COMPARISON
# =========================================================
if page == "Model Comparison":
    st.header("📊 Model Comparison")
    select_problem = st.selectbox("Select Problem Type", ["Classification", "Regression"])

    # ---- CLASSIFICATION ----
    if select_problem == "Classification":
        if df.empty:
            st.warning("No classification experiments found. Run some experiments first.")
        else:
            experiments = df["Experiment_name"].unique()
            selected_exp = st.sidebar.selectbox("Select Experiment", experiments)
            filtered_df = df[df["Experiment_name"] == selected_exp]

            st.subheader("🏆 Leaderboard")
            st.dataframe(filtered_df.sort_values(by="F1_Score", ascending=False))

            melted = filtered_df.melt(
                id_vars=["Display_Name"],
                value_vars=["Accuracy", "F1_Score", "Precision", "Recall"],
                var_name="Metric",
                value_name="Score"
            )

            st.subheader("📈 Metric Comparison")
            fig = px.bar(melted, x="Display_Name", y="Score", color="Metric", barmode="group", template=chart_theme)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🕸️ Radar View")
            fig = px.line_polar(melted, r="Score", theta="Metric", color="Display_Name", line_close=True, template=chart_theme)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("⚖️ Precision vs Recall")
            fig = px.scatter(
                filtered_df, x="Precision", y="Recall",
                size="F1_Score", color="Display_Name", text="Display_Name", template=chart_theme
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📊 Stability (Cross Validation)")
            fig = px.bar(filtered_df, x="Display_Name", y="cross_val_score", color="cross_val_score", template=chart_theme)
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Confusion Matrix")
            option=st.selectbox("Select Model to view confusion matrix",filtered_df["Display_Name"].unique())
            cmf=filtered_df[filtered_df["Display_Name"]==option]
            try:
                cm_val = cmf["confusion_matrix"].values[0]
                if cm_val is None:
                    st.error("The confusion matrix is not there")
                else:
                    mtx = np.array(cm_val)
                    fig = px.imshow(
                        mtx,
                        text_auto=True,
                        color_continuous_scale='Viridis',
                        labels=dict(x="Predicted", y="Actual")
                    )
                    fig.update_layout(template=chart_theme)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error("There is no Confusion matrix")
                st.write(e)
            

            st.markdown("---")
            st.subheader("📋 Copy Model Parameters")
            models_in_exp = filtered_df["Display_Name"].unique()
            selected_model_params = st.selectbox("Select Model to Copy Params", models_in_exp)
            if selected_model_params:
                row = filtered_df[filtered_df["Display_Name"] == selected_model_params].iloc[0]
                model_params = row["Params"] if "Params" in row.index else {}
                st.code(json.dumps(model_params, indent=4), language="json")
           

    # ---- REGRESSION ----
    else:
        if df_reg.empty:
            st.warning("No regression experiments found. Run some experiments first.")
        else:
            reg_experiments = df_reg["Experiment_name"].unique()
            selected_reg_exp = st.sidebar.selectbox("Select Experiment", reg_experiments)
            filtered_reg = df_reg[df_reg["Experiment_name"] == selected_reg_exp]

            st.subheader("🏆 Leaderboard")
            st.dataframe(filtered_reg.sort_values(by="R2", ascending=False))

            st.subheader("📈 Error Comparison")
            melted_reg = filtered_reg.melt(
                id_vars=["Display_Name"],
                value_vars=["MSE", "MAE", "RMSE"],
                var_name="Metric",
                value_name="Value"
            )
            fig = px.bar(melted_reg, x="Display_Name", y="Value", color="Metric", barmode="group", template=chart_theme)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📊 R2 Score Comparison")
            fig = px.bar(filtered_reg, x="Display_Name", y="R2", color="R2", template=chart_theme)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📉 MSE vs MAE")
            fig = px.scatter(
                filtered_reg, x="MSE", y="MAE",
                size="RMSE", color="Display_Name", text="Display_Name", template=chart_theme
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📊 Cross Validation Score")
            fig = px.bar(filtered_reg, x="Display_Name", y="cv_score", color="cv_score", template=chart_theme)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("📋 Copy Model Parameters")
            reg_models = filtered_reg["Display_Name"].unique()
            selected_reg_model = st.selectbox("Select Model to Copy Params", reg_models)
            if selected_reg_model:
                row = filtered_reg[filtered_reg["Display_Name"] == selected_reg_model].iloc[0]
                model_params = row["Params"] if "Params" in row.index else {}
                st.code(json.dumps(model_params, indent=4), language="json")

# =========================================================
# PAGE 2: INFOGRAPHICS
# =========================================================
elif page == "Infographics":
    st.header("🧠 Interactive Model Explorer (Sunburst)")
    st.markdown("Click on an **Experiment** to see Models. Click on a **Model** to see Metrics.")

    select_problem_inf = st.selectbox("Select Problem Type", ["Classification", "Regression"])

    if select_problem_inf == "Classification":
        data_inf = df
        metric_cols = ["Accuracy", "Precision", "Recall", "F1_Score"]
    else:
        data_inf = df_reg
        metric_cols = ["MSE", "MAE", "R2", "RMSE"]

    if data_inf.empty:
        st.warning("No data found for this problem type.")
    else:
        experiments_inf = data_inf["Experiment_name"].unique()

        ids = ["Root"]
        labels = ["All Experiments"]
        parents = [""]
        hovertexts = ["Click to explore"]

        for exp in experiments_inf:
            exp_df = data_inf[data_inf["Experiment_name"] == exp]
            models = exp_df["Display_Name"].unique()

            ids.append(exp)
            labels.append(exp)
            parents.append("Root")
            hovertexts.append(f"<b>{exp}</b><br>Models: {len(models)}")

            for model in models:
                row = exp_df[exp_df["Display_Name"] == model].iloc[0]
                model_id = f"{exp}____{model}"

                ids.append(model_id)
                labels.append(model)
                parents.append(exp)

                hover = f"<b>{model}</b><br>"
                for m in metric_cols:
                    if m in row:
                        hover += f"{m}: {row[m]:.4f}<br>"
                hovertexts.append(hover)

                for m_name in metric_cols:
                    if m_name in row:
                        m_val = row[m_name]
                        m_id = f"{model_id}____{m_name}"
                        ids.append(m_id)
                        labels.append(f"{m_name}\n{m_val:.4f}")
                        parents.append(model_id)
                        hovertexts.append(f"{m_name}: {m_val:.4f}")

        fig = go.Figure(go.Sunburst(
            ids=ids, labels=labels, parents=parents,
            hoverinfo="text", hovertext=hovertexts,
            insidetextorientation='radial'
        ))
        fig.update_layout(
            margin=dict(t=40, l=0, r=0, b=0),
            height=750, font=dict(size=14), template=chart_theme
        )
        st.plotly_chart(fig, use_container_width=True)
elif page=="Critics":
    from ai_approach import generate_dna_report
    api_key=st.secrets['MY_API_KEY']
    st.header("🧬 Model DNA Report Generator")
    select_problem_crit = st.selectbox("Select Problem Type", ["Classification", "Regression"])
    if select_problem_crit == "Classification":
        data_crit = df
        experiments_crit=st.selectbox("Select Experiment",data_crit["Experiment_name"].unique())
        data_crit=data_crit[data_crit["Experiment_name"]==experiments_crit]
        metrics=data_crit[["Model","Accuracy","F1_Score","Precision","Recall","cross_val_score","Train_Score"]]
        response=generate_dna_report(metrics,api_key,"Classification",experiments_crit)
        st.subheader("🧬 DNA Report")
        st.markdown(response)

    else:
        # Mirror the classification critics flow for regression
        data_crit = df_reg
        experiments_crit = st.selectbox("Select Experiment", data_crit["Experiment_name"].unique())
        data_crit = data_crit[data_crit["Experiment_name"] == experiments_crit]
        # Select regression-relevant metrics
        metrics = data_crit[["Model", "MSE", "MAE", "R2", "RMSE", "cv_score", "Train_Score"]]
        response = generate_dna_report(metrics, api_key, "Regression", experiments_crit)
        st.subheader("🧬 DNA Report")
        st.markdown(response)









