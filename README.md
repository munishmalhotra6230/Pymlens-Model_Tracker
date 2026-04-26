# MLens

MLens is a lightweight ML experiment tracking tool that helps data scientists 
log, compare, and visualize their model experiments — all running fully locally 
on your machine.

---

## Why MLens

Managing multiple experiments manually becomes chaotic and time-consuming. 
After running several models, it becomes difficult to track which model 
performed best and on which problem. MLens eliminates this problem by 
automatically recording all your experiments in one place.

## Installation

```bash
pip install pymlens
```

## Run Locally

Clone the project

```bash
git clone https://github.com/munishmalhotra6230/model_tracker-MLENS-.git
```

Go to the project directory

```bash
cd model_tracker-MLENS-
```

Install dependencies

```bash
pip install -r requirements.txt
```

Start the server

```bash
cd backend_monitoring
python backend.py
```

Run test scripts

```bash
python test_scripts.py
```

View results

```bash
streamlit run dashboard.py
```

## Features

- Easy to use — minimal code changes required
- Runs fully locally — no cloud, no data leaves your machine
- Records each experiment and their results automatically
- Compare model performance visually via dashboard
- Supports vanilla metrics out of the box
- Preferable for Supervised Learning (Scikit-learn)

## How to Use MLens

- Start the backend server before running experiments
- Declare an experiment with a meaningful name based on your problem
- Run multiple models inside the same experiment
- View and compare results on the Streamlit dashboard

## Demo

![Demo](<Screen Recording 2026-04-25 211626.gif>)

## Screenshots

![alt text](<Screenshot 2026-04-25 205636.png>)
![alt text](<Screenshot 2026-04-25 191931.png>)
![alt text](<Screenshot 2026-04-25 205210.png>)
![alt text](<Screenshot 2026-04-25 205516.png>)
![alt text](<Screenshot 2026-04-25 205552.png>)
![alt text](<Screenshot 2026-04-25 205624.png>)

## Usage/Examples

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from Experiment.Experiment import Experiment

x, y = load_iris(return_X_y=True)

with Experiment("Iris_Classification") as exp:
    try:
        xtrain, xval, ytrain, yval = train_test_split(
            x, y, test_size=0.2, random_state=42
        )
        exp.Start_experiment(
            xtrain, ytrain, Xtest=xval, ytest=yval,
            model=LogisticRegression()
        )
        exp.Start_experiment(
            xtrain, ytrain, Xtest=xval, ytest=yval,
            model=RandomForestClassifier()
        )
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Experiment Completed")
```

## Optimizations

Have suggestions? Join the Discord and share your ideas.

## Feedback

Join the Discord community:
https://discord.gg/svx4Sfckz

## Links

[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://munish1234.pythonanywhere.com/)

## Author

- [@Munish Malhotra](https://github.com/munishmalhotra6230)
