from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from Experiment.Experiment import Experiment
# Load the Iris dataset
x,y=load_iris(return_X_y=True)
with Experiment("Experiment1") as exp:
    try:
        xtrain,xval,ytrain,yval=train_test_split(x,y,test_size=0.2,random_state=42)
        exp.Start_experiment(xtrain,ytrain,Xtest=xval,ytest=yval,model=LogisticRegression())
        exp.Start_experiment(xtrain,ytrain,Xtest=xval,ytest=yval,model=RandomForestClassifier())

    except Exception as e:
        print(f"An error occurred:")
    finally:
        print("Experiment Completed")       

with Experiment("Experiment2") as exp:
    try:
        xtrain,xval,ytrain,yval=train_test_split(x,y,test_size=0.2,random_state=42)
        exp.Start_experiment(xtrain,ytrain,Xtest=xval,ytest=yval,model=LogisticRegression())
        exp.Start_experiment(xtrain,ytrain,Xtest=xval,ytest=yval,model=RandomForestClassifier())

    except Exception as e:
        print(f"An error occurred:")
    finally:
        print("Experiment Completed")   