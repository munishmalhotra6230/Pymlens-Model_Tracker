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


def _classification_report(series):
    # series: a pandas Series with metric fields
    lines = []
    name = series.get("Display_Name", "Model")
    lines.append(f"### {name} — Model DNA Report")

    train = series.get("Train_Score", np.nan)
    test_acc = series.get("Accuracy", np.nan)
    prec = series.get("Precision", np.nan)
    rec = series.get("Recall", np.nan)
    f1 = series.get("F1_Score", np.nan)
    cv = series.get("cross_val_score", np.nan)

    # Summary
    summary = []
    if not pd.isna(f1):
        summary.append(f"F1: {f1:.3f}")
    if not pd.isna(prec) and not pd.isna(rec):
        summary.append(f"Precision/Recall: {prec:.3f}/{rec:.3f}")
    if not pd.isna(test_acc):
        summary.append(f"Test accuracy: {test_acc:.3f}")
    if not pd.isna(train):
        summary.append(f"Train score: {train:.3f}")
    if not pd.isna(cv):
        summary.append(f"Cross-val (avg): {cv:.3f}")

    lines.append("**Summary metrics:** " + ", ".join(summary))

    # Strengths / weaknesses
    strengths = []
    weaknesses = []

    if not pd.isna(f1) and f1 > 0.85:
        strengths.append("Strong overall balance between precision and recall (high F1)")
    else:
        weaknesses.append("F1 is moderate/low — model may struggle on overall class balance")

    if not pd.isna(prec) and not pd.isna(rec):
        if prec > 0.85 and rec > 0.85:
            strengths.append("High precision and high recall — suitable for general use")
        elif prec > 0.85 and rec < 0.6:
            weaknesses.append("High precision but low recall — good when false positives are costly, but misses positives")
        elif rec > 0.85 and prec < 0.6:
            weaknesses.append("High recall but low precision — good when missing positives is costly, but many false alarms")

    # Confusion matrix analysis
    try:
        cm = series.get("confusion_matrix")
        if cm is not None:
            mtx = np.array(cm)
            row_sums = mtx.sum(axis=1) + 1e-9
            diag = np.diag(mtx)
            recalls = diag / row_sums
            low = list(np.where(recalls < 0.5)[0])
            if len(low) > 0:
                weaknesses.append(f"Some classes have low recall (indices): {low}")
            if np.max(row_sums) / (np.min(row_sums) + 1e-9) > 3:
                weaknesses.append("Significant class imbalance detected")
    except Exception:
        pass

    # Overfitting/underfitting
    if not pd.isna(train) and not pd.isna(test_acc):
        diff = train - test_acc
        if diff > 0.10:
            weaknesses.append("Likely overfitting: train >> test")
        elif diff < -0.05:
            weaknesses.append("Possible underfitting or data mismatch: test > train")
        else:
            strengths.append("Training and test performance are comparable (no obvious overfitting)")

    # Stability
    if not pd.isna(cv):
        if cv < 0.6:
            weaknesses.append("Low cross-validation stability — results may be brittle")
        elif cv < 0.8:
            strengths.append("Moderate cross-validation stability")
        else:
            strengths.append("Stable across folds (good CV score)")

    lines.append("**Strengths:**")
    if strengths:
        for s in strengths:
            lines.append(f"- {s}")
    else:
        lines.append("- (none detected)")

    lines.append("**Weaknesses / Risks:**")
    if weaknesses:
        for w in weaknesses:
            lines.append(f"- {w}")
    else:
        lines.append("- (none detected)")

    # Recommendation
    recs = []
    if weaknesses and any("overfit" in w.lower() or "imbalance" in w.lower() for w in weaknesses):
        recs.append("Consider more data, regularization, class-weighting or resampling.")
    if weaknesses and any("low" in w.lower() and "cv" in w.lower() for w in weaknesses):
        recs.append("Run more robust cross-validation and investigate feature stability.")
    if not weaknesses and (f1 > 0.8 if not pd.isna(f1) else False):
        recs.append("Model appears production-ready for similar data and use-cases.")
    if not recs:
        recs.append("Collect more validation data and iterate on model tuning before production.")

    lines.append("**Recommendations:**")
    for r in recs:
        lines.append(f"- {r}")

    return "\n\n".join(lines)


def _regression_report(series):
    lines = []
    name = series.get("Display_Name", "Model")
    lines.append(f"### {name} — Model DNA Report")

    train = series.get("Train_Score", np.nan)
    r2 = series.get("R2", np.nan)
    mse = series.get("MSE", np.nan)
    mae = series.get("MAE", np.nan)
    rmse = series.get("RMSE", np.nan)
    cv = series.get("cv_score", np.nan)

    summary = []
    if not pd.isna(r2):
        summary.append(f"R2: {r2:.3f}")
    if not pd.isna(mse):
        summary.append(f"MSE: {mse:.3f}")
    if not pd.isna(mae):
        summary.append(f"MAE: {mae:.3f}")
    if not pd.isna(train):
        summary.append(f"Train score: {train:.3f}")
    if not pd.isna(cv):
        summary.append(f"CV (neg MSE avg): {cv:.3f}")

    lines.append("**Summary metrics:** " + ", ".join(summary))

    strengths = []
    weaknesses = []

    if not pd.isna(r2):
        if r2 > 0.75:
            strengths.append("High R2 — explains a large portion of variance")
        elif r2 < 0.3:
            weaknesses.append("Low R2 — model explains little of the variance")
        else:
            strengths.append("Moderate R2")

    if not pd.isna(mse) and not pd.isna(mae):
        strengths.append("Error metrics available for diagnostic comparison")

    if not pd.isna(train) and not pd.isna(r2):
        diff = train - r2
        if diff > 0.10:
            weaknesses.append("Likely overfitting: training >> test")
        elif r2 - train > 0.10:
            weaknesses.append("Possible data leakage or mismatch: test > train")
        else:
            strengths.append("Training and test performance are comparable")

    if not pd.isna(cv):
        if cv < -1.0:
            weaknesses.append("Poor cross-validation (very negative score) — unstable")
        else:
            strengths.append("Cross-validation results available")

    lines.append("**Strengths:**")
    if strengths:
        for s in strengths:
            lines.append(f"- {s}")
    else:
        lines.append("- (none detected)")

    lines.append("**Weaknesses / Risks:**")
    if weaknesses:
        for w in weaknesses:
            lines.append(f"- {w}")
    else:
        lines.append("- (none detected)")

    recs = []
    if weaknesses:
        recs.append("Consider feature engineering, regularization, or more training data.")
    else:
        recs.append("Model is promising; validate on holdout and run stress tests before production.")

    lines.append("**Recommendations:**")
    for r in recs:
        lines.append(f"- {r}")

    return "\n\n".join(lines)

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
    st.header("Models Criticism and Feedback")
    st.markdown("This features allows user to give a analysis about each experiment using metrics")
    problem=st.selectbox("Select problem type",["Classification","Regression"]) 
    # choose experiment list depending on problem
    experiments = df["Experiment_name"].unique() if problem=="Classification" else df_reg["Experiment_name"].unique()
    selected_exp = st.selectbox("Select Experiment to Critic", experiments)

    if problem=="Classification":
        df_exp = df[df["Experiment_name"]==selected_exp]
        st.subheader(f"Models Trained {len(df_exp['Model'])}")
        selected_model = st.selectbox("Select model to critic", df_exp["Display_Name"].unique())
        key_frame = df_exp[df_exp["Display_Name"]==selected_model]
        if key_frame.empty:
            st.error("No data available for selected model.")
        else:
            cv = key_frame["cross_val_score"].values[0]
            if cv == 0 or pd.isna(cv):
                st.warning("This model was not cross validated, consider running it with cross validation for better insights")
            else:
                if cv < 0.75:
                    st.error("This model has lower cross val score")
                elif cv < 0.85:
                    st.warning("This model has decent cross val score but can be improved")
                elif cv < 0.95:
                    st.success("This model has good cross val score")
                elif cv <= 1:
                    st.success("This model has excellent cross val score")
                else:
                    st.error("This model has invalid cross val score")

            f1 = key_frame["F1_Score"].values[0]
            if f1 > 0.85:
                st.success("This model has good F1 score")
            else:
                st.warning("There may be class imbalance or the model needs improvement")

            prec = key_frame["Precision"].values[0]
            rec = key_frame["Recall"].values[0]
            if prec > 0.85 and rec > 0.85:
                st.success("This model has good precision and recall")
            elif prec > 0.85 and rec < 0.52:
                st.markdown("⚠ High false negatives detected — model misses many actual positive cases")
            elif prec < 0.85 and rec > 0.85:
                st.markdown("⚠ High false positives detected — model incorrectly labeling many negative cases as positive")
            # Training vs test analysis
            try:
                train_score = key_frame["Train_Score"].values[0]
            except Exception:
                train_score = float("nan")

            test_acc = key_frame["Accuracy"].values[0] if "Accuracy" in key_frame else None
            overview = []
            overview.append(f"**Model:** {selected_model}")
            if not pd.isna(train_score):
                overview.append(f"Training score: {train_score:.4f}")
            if test_acc is not None:
                overview.append(f"Test accuracy: {test_acc:.4f}")

            # Overfitting / underfitting heuristics
            if not pd.isna(train_score) and test_acc is not None:
                diff = train_score - test_acc
                if diff > 0.10:
                    overview.append("⚠ Likely overfitting: training score significantly higher than test score")
                elif diff < -0.05:
                    overview.append("⚠ Possible underfitting or data mismatch: test score higher than training score")
                else:
                    overview.append("✅ Training and test scores are comparable")

            # Confusion matrix inspection
            try:
                cm_val = key_frame["confusion_matrix"].values[0]
                if cm_val is not None:
                    mtx = np.array(cm_val)
                    row_sums = mtx.sum(axis=1)
                    diag = np.diag(mtx)
                    recalls = diag / (row_sums + 1e-9)
                    low_recall_classes = np.where(recalls < 0.5)[0]
                    if len(low_recall_classes) > 0:
                        overview.append(f"⚠ Some classes have low recall: {list(map(int, low_recall_classes))}")
                    # class imbalance heuristic
                    if np.max(row_sums) / (np.min(row_sums) + 1e-9) > 3:
                        overview.append("⚠ Significant class imbalance detected — consider resampling or class-weighting")
            except Exception:
                pass

            report = _classification_report(key_frame.iloc[0])
            st.markdown(report, unsafe_allow_html=False)

    else:  # Regression critic
        df_exp = df_reg[df_reg["Experiment_name"]==selected_exp]
        st.subheader(f"Models Trained {len(df_exp['Model'])}")
        selected_model = st.selectbox("Select model to critic", df_exp["Display_Name"].unique())
        row = df_exp[df_exp["Display_Name"]==selected_model]
        if row.empty:
            st.error("No data available for selected model.")
        else:
            r2 = row["R2"].values[0]
            mse = row["MSE"].values[0]
            rmse = row["RMSE"].values[0] if "RMSE" in row else (mse ** 0.5 if mse is not None else None)
            cv = row.get("cv_score") if "cv_score" in row else None

            # R2 based guidance
            if pd.isna(r2):
                st.warning("R2 not available for this run.")
            elif r2 < 0:
                st.error("Model performs worse than a horizontal mean predictor (R2 < 0). Consider feature engineering or different algorithms.")
            elif r2 < 0.5:
                st.warning("Low R2 — the model explains little variance.")
            elif r2 < 0.75:
                st.info("Moderate R2 — acceptable but can be improved.")
            elif r2 <= 1:
                st.success("Good R2 — model explains a large portion of variance.")
            else:
                st.error("Invalid R2 value")

            # Basic error insights
            st.write(f"MSE: {mse:.4f}" if mse is not None else "MSE: N/A")
            if rmse is not None:
                st.write(f"RMSE: {rmse:.4f}")

            # CV score note
            try:
                cv_val = row["cv_score"].values[0]
                if cv_val == 0 or pd.isna(cv_val):
                    st.warning("This model was not cross validated, consider running cross validation for robust error estimates")
                else:
                    st.info(f"Cross-validation score (neg MSE): {cv_val:.4f}")
            except Exception:
                pass
            # Training vs test analysis for regression
            try:
                train_score = row["Train_Score"].values[0]
            except Exception:
                train_score = float("nan")

            overview = []
            overview.append(f"**Model:** {selected_model}")
            if not pd.isna(train_score):
                overview.append(f"Training score: {train_score:.4f}")
            if not pd.isna(r2):
                overview.append(f"Test R2: {r2:.4f}")

            if not pd.isna(train_score) and not pd.isna(r2):
                if train_score - r2 > 0.10:
                    overview.append("⚠ Likely overfitting: training R2 significantly higher than test R2")
                elif r2 - train_score > 0.10:
                    overview.append("⚠ Possible data leakage or mismatch: test R2 higher than training R2")
                else:
                    overview.append("✅ Training and test R2 are comparable")

            report = _regression_report(row.iloc[0])
            st.markdown(report, unsafe_allow_html=False)
                    
    






