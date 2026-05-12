# PyMLens 🧪

A lightweight ML experiment tracking tool that helps data scientists log, compare, and visualize their model experiments — runs fully locally on your machine.

## Installation

```bash
pip install pymlens
```

---

## Why PyMLens

Managing multiple experiments manually becomes chaotic and time-consuming. After running several models, it becomes difficult to track which model performed best and on which problem. PyMLens eliminates this problem by automatically recording all your experiments in one place.

---

## Features

- Easy to use — minimal code changes required
- Runs fully locally — no cloud, no data leaves your machine
- Supports both Classification and Regression problems
- Records each experiment and results automatically
- Compare model performance visually via Streamlit dashboard
- Confusion matrix tracking for classification
- MSE, MAE, RMSE, R2 tracking for regression
- Cross validation score tracking
- Copy model hyperparameters directly from dashboard
- Interactive Sunburst visualization for experiment exploration
- Dynamic themes — randomize dashboard appearance

---

## Supported Metrics

### Classification
- Accuracy
- Precision
- Recall
- F1 Score
- Cross Validation Score
- Confusion Matrix

### Regression
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R2 Score
- Cross Validation Score

---

## Run Locally

Clone the project

```bash
git clone https://github.com/munishmalhotra6230/model_tracker-MLENS-.git
cd model_tracker-MLENS-
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run dashboard

```bash
python -m pymlens dashboard
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
    exp.Start_experiment(model=LogisticRegression())
    exp.Start_experiment(model=RandomForestClassifier())
    exp.Start_experiment(model=SVC())
```

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
    exp.Start_experiment(model=LinearRegression())
    exp.Start_experiment(model=RandomForestRegressor())
    exp.Start_experiment(model=GradientBoostingRegressor())
```

---

## View Dashboard

```bash
python -m pymlens dashboard
```

Dashboard opens automatically in your browser with:
- Model leaderboard
- Metric comparison charts
- Radar view
- Precision vs Recall scatter (Classification)
- MSE vs MAE scatter (Regression)
- Cross validation stability
- Copy model hyperparameters
- Interactive Sunburst explorer

---

## Demo

---

## Screenshots
![alt text](<Screenshot 2026-05-12 175014.png>) ![alt text](<Screenshot 2026-05-12 175105.png>) ![alt text](<Screenshot 2026-05-12 174956.png>)

---

## Optimizations

Have suggestions? Join the Discord and share your ideas.

---

## Feedback

Join the Discord community:
https://discord.gg/svx4Sfckz

---

## Links

[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://munish1234.pythonanywhere.com/)

---

## Author

- [@Munish Malhotra](https://github.com/munishmalhotra6230)
