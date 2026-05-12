from sklearn.datasets import load_iris,load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from pymlens import Classification_Experiment,Regression_Experiment
# Load the Iris dataset
x,y=load_iris(return_X_y=True)
xtrain,xval,ytrain,yval=train_test_split(x,y,test_size=0.2,random_state=42)
with Classification_Experiment("Experiment1",xtrain=xtrain,ytrain=ytrain,Xtest=xval,ytest=yval) as exp1:
    try:
        exp1.Start_experiment(model=LogisticRegression())
        exp1.Start_experiment(model=RandomForestClassifier())

    except Exception as e:
        print(f"An error occurred:",e)
    finally:
        print("Experiment Completed")       

with Classification_Experiment(Experiment_name="Experiment2",xtrain=xtrain,ytrain=ytrain,Xtest=xval,ytest=yval) as exp2:
    try:
        xtrain,xval,ytrain,yval=train_test_split(x,y,test_size=0.2,random_state=42)
        exp2.Start_experiment(model=LogisticRegression())
        exp2.Start_experiment(model=RandomForestClassifier())

    except Exception as e:
        print(f"An error occurred:",e)
    finally:
        print("Experiment Completed")
diabx,diaby=load_diabetes(return_X_y=True)     
X_train,X_test,y_train,y_test = train_test_split(diabx,diaby,test_size=0.2)   
with Regression_Experiment("diabetest",xtrain=X_train,ytrain=y_train,Xtest=X_test,ytest=y_test) as regx:
    regx.Start_experiment(LinearRegression())
