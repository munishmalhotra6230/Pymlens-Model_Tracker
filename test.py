from pymlens.Experiment import Classification_Experiment, Regression_Experiment
from sklearn.datasets import load_iris,load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(load_iris().data, load_iris().target, test_size=0.2, random_state=42)
with Classification_Experiment("Iris_experiment", xtrain, ytrain, xtest, ytest) as exp:
    exp.Start_experiment(RandomForestClassifier(),exp_keyword="Rf1", cross_val=True)
    exp.Start_experiment(LogisticRegression(max_iter=200),exp_keyword="Logistic_reg", cross_val=True)
    exp.Start_experiment(SVC(),exp_keyword="Support_vec_mac", cross_val=True)
    exp.Start_experiment(DecisionTreeClassifier(),exp_keyword="Decision_tree", cross_val=True)
xtrain2, xtest2, ytrain2, ytest2 = train_test_split(load_diabetes().data, load_diabetes().target, test_size=0.2, random_state=42)
with Regression_Experiment("Diabetes_experiment", xtrain2, ytrain2, xtest2, ytest2) as exp2:
    exp2.Start_experiment(RandomForestRegressor(),exp_keyword="Rf_regressor", cross_val=True)
    exp2.Start_experiment(LinearRegression(),exp_keyword="Linear_reg", cross_val=True)
    exp2.Start_experiment(Ridge(),exp_keyword="Ridge_reg", cross_val=True)    

