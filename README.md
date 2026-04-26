# MLens 
Mlens is a simple tool to manage and compare Models and via visualizing the models stats .This is completely free easy to use and run locally on  your computer.

---

## Why MLens
In the world of data and data science the main things are models. The engineers run various models and train various example after some time they didn't know which models is perfect and they even didn't on which Managing multiple experiments manually becomes 
chaotic and time-consuming for data scientists.
But now using MlensMLens eliminates this problem by automatically 
tracking and managing all your experiments. so much because Mlens takes all records of this and manage your models quicky.
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
run test scripts
```bash
python test_scripts.py
```
View_results
``` bash
streamlit run dashboard.py
```

## Features

- Easy to use 
- Runs fully locally
- Records each experiment and their results 
- No cloud integrations your data on your system 
- Provides visualization of each experiment you run on so you can comapare perfromance 
- uses vanilla metrics 
- preferable for Supervised learning (Scikit-learn)





## How to use Mlens 

- You need to add the db_path to store your Experiments

- After this run the server(Always make sure server should be ON)
- Then Train the model withing declaring a experiment(
NOTE:Choose Experiment name on the basis of problem you are working on  Like : Problem_Datasetname)
-and after that dasboard command to visualize the experiments 
# gif
![alt text](<Screen Recording 2026-04-25 211626.gif>)
## Screen shots:
![alt text](<Screenshot 2026-04-25 205636.png>) ![alt text](<Screenshot 2026-04-25 191931.png>) ![alt text](<Screenshot 2026-04-25 205210.png>) ![alt text](<Screenshot 2026-04-25 205516.png>) ![alt text](<Screenshot 2026-04-25 205552.png>) ![alt text](<Screenshot 2026-04-25 205624.png>)




## Usage/Examples

```python
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
```







## Optimizations
- Suggest Optimizations for mlens on Discord i will listen to it 


## Feedback

If you have any feedback, please reach out to us on discord here is the link to join Discord channel https://discord.gg/svx4Sfckz


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://munish1234.pythonanywhere.com/)
 
## Authors

- [@Munish Malhotra](https://github.com/munishmalhotra6230)
