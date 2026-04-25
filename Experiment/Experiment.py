import requests
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import cross_val_score
import numpy as np 

server="http://127.0.0.1:5000/"
class Experiment():
    def __init__(self,Experiment_name):
        self.Experiment_name=Experiment_name
        self.model=None
        self.all_runs=[]
    def __enter__(self):
        print(f"your experiment {self.Experiment_name} is Staring")
        return self
    def __exit__(self, *args):
        
        for score in self.all_runs:
            print(f"Saving the model {score['Model']} with scores {score['Scores']} to the server")
            respone=requests.post(server+"save",json={"Experiment_name":self.Experiment_name,"model_name":score['Model'],"Scores":score['Scores']})
            print(f"Your model {score['Model']} with scores {score['Scores']} is saved to the server  with {respone.status_code} status code")
        print(f"Experiments done successfully")
        return self
    def Start_experiment(self,xtrain,ytrain,Xtest,ytest,model=None,cross_val=False):
        self.model=model
        '''
        IF THE CROSS_VAL=TRUE
        THEN IT WILL FIND CROSS VALSCORE AS WELL AS TEST SCORES 
        AND IF THE CROSS_VAL=FALSE THEN IT WILL FIND ONLY TEST SCORES
        THERE CROSS_VAL SCORE WILL BE ZERO-->NO CROSS VAL 
        '''
        if  cross_val: 
            print("You Didn't provide any test data so we will provide the cross val score")
            cross_val=np.mean(cross_val_score(self.model,xtrain,ytrain,cv=3))
            self.model.fit(xtrain,ytrain)
            ypred=self.model.predict(Xtest)
            acc=accuracy_score(ytest,ypred)
            pre=precision_score(ytest,ypred,average="weighted")
            rec=recall_score(ytest,ypred,average="weighted")
            f1=f1_score(ytest,ypred,average="weighted")
            self.Scores=self.model_stats(accuracy=acc,precision=pre,recall=rec,f1_score=f1,params=self.model.get_params(),cross_val_score=cross_val)
            self.all_runs.append({
                'Model':self.model.__class__.__name__,
                'Scores':self.Scores
            })
            print(f"Scores: {self.Scores}")
        else:
            print("Running the experiment")
            self.model.fit(xtrain,ytrain)
            ypred=self.model.predict(Xtest)
            acc=accuracy_score(ytest,ypred)
            pre=precision_score(ytest,ypred,average="weighted")
            rec=recall_score(ytest,ypred,average="weighted")
            f1=f1_score(ytest,ypred,average="weighted")
            self.Scores = self.model_stats(accuracy=acc,precision=pre,recall=rec,f1_score=f1,params=self.model.get_params(),cross_val_score=0)
            self.all_runs.append({
                'Model': self.model.__class__.__name__,
                'Scores': self.Scores
            })
    def model_stats(self,**Scores):
        
        return Scores

  
    

        
