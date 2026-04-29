import sqlite3
import os
import datetime
import random
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, mean_squared_error, r2_score, 
    mean_absolute_error, confusion_matrix
)
from sklearn.model_selection import cross_val_score
import numpy as np
import json

DB_PATH = os.path.join(os.path.expanduser("~"), ".pymlens", "experiments.db")

def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    return conn
def _new_db():
    conn=get_db()
    conn.execute("DELETE FROM Experiments;")
    conn.execute("DELETE FROM Scores; ")
    conn.execute("DELETE FROM Regression_Scores;")
    conn.commit()
    conn.close()
def init_db():
    conn = get_db()
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS Experiments (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Experiment_name TEXT,
            timestamp TEXT
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS Scores (
            Score_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Experiment_ID INTEGER,
            Model TEXT,
            Accuracy REAL,
            Precision REAL,
            Recall REAL,
            F1_Score REAL,
            cross_val_score REAL,
            confusion_matrix TEXT,
            Params TEXT
        )
    """)
#     conn.execute("""
# ALTER TABLE Scores ADD confusion_matrix Text NULL;
# """)
def _inti_reg():  
    conn=get_db()  
    conn.execute("""
        CREATE TABLE IF NOT EXISTS Experiments (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Experiment_name TEXT,
            timestamp TEXT
        )
    """)
    conn.execute("""
            CREATE TABLE IF NOT EXISTS Regression_Scores (
                Score_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                Experiment_ID INTEGER,
                Model TEXT,
                MSE REAL,
                MAE REAL,
                R2 REAL,
                RMSE REAL,
                cv_score REAL,
                Params TEXT
            )
        """)
        
    conn.commit()
    conn.close()


class Classification_Experiment():
    def __init__(self, Experiment_name, xtrain, ytrain, Xtest, ytest):
        self.Experiment_name = Experiment_name
        self.all_runs = []
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = Xtest
        self.ytest = ytest
        
        init_db()

    def __enter__(self):
        print(f"Experiment running: {self.Experiment_name}")
        return self

    def __exit__(self, *args):
        conn = get_db()

        row = conn.execute(
            "SELECT ID FROM Experiments WHERE Experiment_name = ?",
            (self.Experiment_name,)
        ).fetchone()

        if row:
            exp_id = row[0]
        else:
            conn.execute(
                '''INSERT INTO Experiments (Experiment_name, timestamp) VALUES (?, ?)''',
                (self.Experiment_name, str(datetime.datetime.now())))
            conn.commit()
            exp_id = conn.execute(
                "SELECT ID FROM Experiments WHERE Experiment_name = ? ORDER BY timestamp DESC LIMIT 1",
                (self.Experiment_name,)
            ).fetchone()[0]

        for run in self.all_runs:
            scores = run['Scores']
            conn.execute(
                "DELETE FROM Scores WHERE Experiment_ID = ? AND Model = ?",
                (exp_id, run['Model'])
)
            conn.execute("""
                INSERT INTO Scores 
                (Experiment_ID, Model, Accuracy, Precision, Recall, F1_Score, cross_val_score, confusion_matrix, Params)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                exp_id,
                run['Model'],
                scores.get('accuracy'),
                scores.get('precision'),
                scores.get('recall'),
                scores.get('f1_score'),
                scores.get('cross_val_score'),
                scores.get('confusion_matrix'),
                json.dumps(scores.get('params', {}))
            ))
            print(f"Saved: {run['Model']}")

        conn.commit()
        conn.close()
        print(f"Experiment khatam: {self.Experiment_name}")

    def Start_experiment(self, model=None, cross_val=False):
        self.model = model
        print(f"Running: {model.__class__.__name__}")

        if cross_val:
            cv_score = np.mean(cross_val_score(self.model, self.xtrain, self.ytrain, cv=3))
        else:
            cv_score = 0

        self.model.fit(self.xtrain, self.ytrain)
        ypred = self.model.predict(self.xtest)
        cm = confusion_matrix(self.ytest, ypred).tolist()

        scores = {
            'accuracy': accuracy_score(self.ytest, ypred),
            'precision': precision_score(self.ytest, ypred, average="weighted"),
            'recall': recall_score(self.ytest, ypred, average="weighted"),
            'f1_score': f1_score(self.ytest, ypred, average="weighted"),
            'cross_val_score': cv_score,
            'confusion_matrix': json.dumps(cm),
            'params': self.model.get_params()
        }

        self.all_runs.append({
            'Model': self.model.__class__.__name__,
            'Scores': scores
        })
        print(f"{model.__class__.__name__}: Accuracy = {scores['accuracy']:.4f}")


class Regression_Experiment():
    def __init__(self, Experiment_name, xtrain, ytrain, Xtest, ytest):
        self.Experiment_name = Experiment_name
        self.all_runs = []
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.Xtest = Xtest
        self.ytest = ytest
        _inti_reg()

    def __enter__(self):
        print(f"Experiment running: {self.Experiment_name}")
        return self

    def __exit__(self, *args):
        conn = get_db()

        row = conn.execute(
            "SELECT ID FROM Experiments WHERE Experiment_name = ?",
            (self.Experiment_name,)
        ).fetchone()

        if row:
            exp_id = row[0]
        else:
            conn.execute(
                "INSERT INTO Experiments (Experiment_name, timestamp) VALUES (?, ?)",
                (self.Experiment_name, str(datetime.datetime.now()))
            )
            conn.commit()
            exp_id = conn.execute(
                "SELECT ID FROM Experiments WHERE Experiment_name = ? ORDER BY timestamp DESC LIMIT 1",
                (self.Experiment_name,)
            ).fetchone()[0]

        for run in self.all_runs:
            scores = run['Scores']
            conn.execute(
                "DELETE FROM Regression_Scores WHERE Experiment_ID = ? AND Model = ?",
                (exp_id, run['Model'])
            )
            conn.execute("""
                INSERT INTO Regression_Scores
                (Experiment_ID, Model, MSE, MAE, R2, RMSE, cv_score, Params)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                exp_id,
                run['Model'],
                scores.get('mse'),
                scores.get('mae'),
                scores.get('r2'),
                scores.get('rmse'),
                scores.get('cv_score'),
                json.dumps(scores.get('params', {}))
            ))
            print(f"Saved: {run['Model']}")

        conn.commit()
        conn.close()
        print(f"Experiment done: {self.Experiment_name}")

    def Start_experiment(self, model=None, cross_val=False):
        self.model = model
        print(f"Running: {model.__class__.__name__}")

        if cross_val:
            cv_score = np.mean(cross_val_score(
                self.model, self.xtrain, self.ytrain, 
                cv=3, scoring='neg_mean_squared_error'
            ))
        else:
            cv_score = 0

        self.model.fit(self.xtrain, self.ytrain)
        ypred = self.model.predict(self.Xtest)
        mse = mean_squared_error(self.ytest, ypred)

        scores = {
            'mse': mse,
            'mae': mean_absolute_error(self.ytest, ypred),
            'r2': r2_score(self.ytest, ypred),
            'rmse': np.sqrt(mse),
            'cv_score': cv_score,
            'params': self.model.get_params()
        }

        self.all_runs.append({
            'Model': self.model.__class__.__name__,
            'Scores': scores
        })
        print(f"{model.__class__.__name__}: R2 = {scores['r2']:.4f}")
    
    
    
class Pymlens_settings():
    def __init__(self):
            pass
    
    def delete_db(self):
        confirmation=input(" ARE YOU SURE TO DELETE THE DATA  Y/N :")
        if confirmation =='Y':
            numx=[]
            for  i in range(6):
                numx.append(random.choices([0,1,2,3,4,5,6,7,8,9])[0])
            codex=str(numx)
            print(codex)
            code=input("PASTE THE  ABOVE CONFIRMATION CODE  :")
            if code==codex:
                _new_db()
                print(" the existing db is deleted") 
            else: raise("MISMATCH ERROR")     
        else: return None             
     