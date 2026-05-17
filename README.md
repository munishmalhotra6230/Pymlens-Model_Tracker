# PyMLens 🧪

> A lightweight ML experiment tracking tool that helps data scientists log, compare, and visualize their model experiments — runs **fully locally** on your machine, no cloud required.

[![PyPI version](https://img.shields.io/pypi/v/pymlens?color=blue&style=flat-square)](https://pypi.org/project/pymlens/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.8-blue?style=flat-square)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

---

## Installation

```bash
pip install pymlens
```

---

## Why PyMLens?

Managing multiple experiments manually becomes chaotic and time-consuming. After running several models, it becomes difficult to track which model performed best and on which problem. PyMLens eliminates this by **automatically recording all your experiments** in a local SQLite database and giving you a rich Streamlit dashboard to explore results.

---
# documentation :https://pymlens.netlify.app/

## Features

- ✅ **Minimal code changes** — wrap your existing training code in a `with` block
- 🔒 **Fully local** — data stored in `~/.pymlens/experiments.db` (SQLite), nothing leaves your machine
- 📊 **Classification & Regression** support
- 🏷️ **Custom experiment keywords** — label individual runs with `exp_keyword`
- 📈 **Train accuracy tracking** — detects overfitting by comparing train vs. validation scores
- 🧮 **Confusion matrix** — stored and visualized per model for classification
- 🔁 **Cross-validation** — enabled by default for classification, optional for regression
- 🧬 **AI DNA Report** — powered by Groq (LLaMA 3.1), gives per-model analysis with a best-model verdict
- 🌈 **Dynamic themes** — randomize the full dashboard appearance on demand
- 📋 **Copy hyperparameters** — inspect and copy any model's params directly from the dashboard
- 🌀 **Interactive Sunburst explorer** — drill down from experiment → model → metric

---

## Supported Metrics

### Classification
| Metric | Description |
|---|---|
| Accuracy | Validation accuracy |
| Train Accuracy | Training accuracy (overfitting check) |
| Precision | Weighted precision |
| Recall | Weighted recall |
| F1 Score | Weighted F1 |
| Cross Validation Score | Mean CV score (3-fold, enabled by default) |
| Confusion Matrix | Stored as JSON, visualized as heatmap |

### Regression
| Metric | Description |
|---|---|
| MSE | Mean Squared Error |
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| R2 | R² Score |
| Cross Validation Score | Mean CV score using `neg_mean_squared_error` (opt-in) |

---

## Run Locally

Clone the project:

```bash
git clone https://github.com/munishmalhotra6230/model_tracker-MLENS-.git
cd model_tracker-MLENS-
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
python -m pymlens dashboard
# or
pymlens dashboard
```

---

## 🐳 Docker

### Build the image

```bash
docker build -t pymlens .
```

### Run the dashboard

```bash
docker run -p 8501:8501 \
  -v pymlens_data:/root/.pymlens \
  -e GROQ_API_KEY=your_groq_api_key_here \
  pymlens
```

Then open **http://localhost:8501** in your browser.

| Flag | Purpose |
|---|---|
| `-p 8501:8501` | Maps container port to your machine |
| `-v pymlens_data:/root/.pymlens` | Persists your experiment DB across restarts |
| `-e GROQ_API_KEY=...` | Passes your Groq key securely (for Critics page) |

> Leave out `-e GROQ_API_KEY` if you don't need the AI Critics page.

### Using docker-compose (recommended)

Create a `docker-compose.yml` in your project:

```yaml
services:
  pymlens:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - pymlens_data:/root/.pymlens
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}

volumes:
  pymlens_data:
```

Then run:

```bash
# Start
docker compose up

# Start in background
docker compose up -d

# Stop
docker compose down
```

Pass your key without hardcoding it:

```bash
# Windows PowerShell
$env:GROQ_API_KEY="your_key_here"; docker compose up

# Linux / Mac
GROQ_API_KEY="your_key_here" docker compose up
```

---

## Usage — Classification

```python
from pymlens import Classification_Experiment
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

x, y = load_iris(return_X_y=True)
xtrain, xval, ytrain, yval = train_test_split(x, y, test_size=0.2, random_state=42)

with Classification_Experiment("Iris_Classification", xtrain, ytrain, xval, yval) as exp:
    exp.Start_experiment(model=LogisticRegression(), exp_keyword="Logistic_reg", cross_val=True)
    exp.Start_experiment(model=RandomForestClassifier(), exp_keyword="RF_baseline", cross_val=True)
    exp.Start_experiment(model=SVC(), exp_keyword="SVM_rbf", cross_val=True)
```

> **Note:** `cross_val=True` is the default for classification. Set `cross_val=False` to skip cross-validation and speed up training.

---

## Usage — Regression

```python
from pymlens import Regression_Experiment
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

x, y = fetch_california_housing(return_X_y=True)
xtrain, xval, ytrain, yval = train_test_split(x, y, test_size=0.2, random_state=42)

with Regression_Experiment("House_Price", xtrain, ytrain, xval, yval) as exp:
    exp.Start_experiment(model=LinearRegression(), exp_keyword="Linear_baseline")
    exp.Start_experiment(model=RandomForestRegressor(), exp_keyword="RF_regressor", cross_val=True)
    exp.Start_experiment(model=GradientBoostingRegressor(), exp_keyword="GBR_v1", cross_val=True)
```

> **Note:** `cross_val=False` is the default for regression (uses `neg_mean_squared_error` scoring when enabled).

---

## `Start_experiment` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | sklearn estimator | required | Any scikit-learn compatible model |
| `exp_keyword` | `str` | `None` | Custom label for this run (falls back to class name) |
| `cross_val` | `bool` | `True` (Classification) / `False` (Regression) | Enable/disable 3-fold cross-validation |

---

## How Data Is Stored

All results are persisted in a local **SQLite database** at:

```
~/.pymlens/experiments.db
```

Three tables are used:

- **`Experiments`** — experiment name + timestamp
- **`Scores`** — classification results per model (accuracy, precision, recall, F1, CV score, confusion matrix, params)
- **`Regression_Scores`** — regression results per model (MSE, MAE, R2, RMSE, CV score, params)

Re-running an experiment with the same name **replaces** existing results for that model (upsert behavior).

---

## Dashboard Pages

Launch with:

```bash
pymlens dashboard
```

The dashboard has **3 pages**:

### 1. 📊 Model Comparison
- Leaderboard table sorted by F1 Score (classification) or R² (regression)
- Grouped bar chart comparing all metrics across models
- Radar (spider) chart for multi-metric comparison
- Precision vs. Recall scatter plot (classification) / MSE vs. MAE scatter (regression)
- Cross-validation stability bar chart
- Interactive confusion matrix heatmap (classification)
- Copy model hyperparameters as JSON

### 2. 🌀 Infographics — Sunburst Explorer
- Drill-down from **All Experiments → Experiment → Model → Metric**
- Hover to see all metric values per model
- Supports both classification and regression data

### 3. 🧬 Critics — AI DNA Report *(Groq-powered)*
- Sends model metrics to **LLaMA 3.1 (8B)** via Groq API
- Generates a structured per-model report:
  - Score interpretation, overfitting analysis, CV stability
  - One specific improvement recommendation per model
  - Final **VERDICT** — best model with justification
- Requires a Groq API key (see setup below)

---

## AI DNA Report Setup

The **Critics** page requires a free [Groq API key](https://console.groq.com/).

Save it using the built-in settings utility:

```python
from pymlens import Pymlens_settings

settings = Pymlens_settings()
settings.add_api_key()   # prompts you to paste your Groq key
```

This saves your key to `pymlens/.streamlit/secrets.toml` so the dashboard can load it securely.

---

## Managing Your Database

Use `Pymlens_settings` to manage stored data:

```python
from pymlens import Pymlens_settings

settings = Pymlens_settings()
settings.delete_db()   # interactive confirmation + 6-digit code required
```

> ⚠️ `delete_db()` clears **all experiments** from the database. A random 6-digit confirmation code is required to prevent accidental deletion.

---

## Screenshots

![Model Comparison Dashboard](<Screenshot 2026-05-12 175014.png>)
![Sunburst Infographics Explorer](<Screenshot 2026-05-12 175105.png>)
![Leaderboard View](<Screenshot 2026-05-12 174956.png>)
<img width="985" height="651" alt="Untitled Diagram drawio" src="https://github.com/user-attachments/assets/b32b18c3-3901-4257-ab42-b733306b1681" />


---


## Dependencies

| Package | Purpose |
|---|---|
| `scikit-learn` | Model training & metric computation |
| `numpy` | Numerical operations |
| `pandas` | Data manipulation |
| `streamlit` | Dashboard UI |
| `plotly` | Interactive charts |
| `groq` | AI DNA Report via LLaMA 3.1 |
| `sqlite3` | Local experiment storage (stdlib) |

---

## Feedback & Community

Have suggestions or found a bug? Join the Discord:

🔗 [discord.gg/svx4Sfckz](https://discord.gg/svx4Sfckz)

---

## Links

[![Portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://munish1234.pythonanywhere.com/)

---

## Author

- [@Munish Malhotra](https://github.com/munishmalhotra6230)
