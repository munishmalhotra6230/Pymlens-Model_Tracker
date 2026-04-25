from flask import Flask, request, jsonify
import duckdb
import json
from pathlib import Path

app = Flask(__name__)

# Ensure the repository-level `db_path` directory exists and use an
# absolute path so the DB file is created in the project root.
repo_root = Path(__file__).resolve().parent.parent
db_dir = repo_root / "db_path"
db_dir.mkdir(parents=True, exist_ok=True)
db_file = str(db_dir / "ML_experiments.db")

try:
    Experiment_db = duckdb.connect(db_file)
except duckdb.IOException as e:
    print("Could not open DB file (locked by another process):", e)
    print("Falling back to in-memory DuckDB. Persistent DB will not be available.")
    Experiment_db = duckdb.connect(":memory:")

Experiment_db.execute("""
CREATE SEQUENCE IF NOT EXISTS experiment_id_seq;
CREATE SEQUENCE IF NOT EXISTS score_id_seq;
CREATE SEQUENCE IF NOT EXISTS param_id_seq;

CREATE TABLE IF NOT EXISTS Experiments (
    ID INTEGER PRIMARY KEY DEFAULT nextval('experiment_id_seq'),
    Experiment_name TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS Scores (
    Score_ID INTEGER PRIMARY KEY DEFAULT nextval('score_id_seq'),
    Experiment_ID INTEGER REFERENCES Experiments(ID),

    Accuracy FLOAT, Precision FLOAT, Recall FLOAT, F1_Score FLOAT,cross_val_score FLOAT,Model TEXT
);

CREATE TABLE IF NOT EXISTS Params (
    Param_ID INTEGER PRIMARY KEY DEFAULT nextval('param_id_seq'),
    Experiment_ID INTEGER REFERENCES Experiments(ID),
    Params JSON
);
""")
try:
    Experiment_db.execute("ALTER TABLE Params ADD COLUMN Model TEXT;")
except:
    pass
Experiment_db.commit()

@app.route("/save",methods=['POST'])
def save():
    data=request.get_json()
    print("Received data:", data)
    if data is None:
        return {"message": "No data received"}, 400
    Experiment_name=data.get("Experiment_name")
    # 1. Check if the experiment exists
    result = Experiment_db.execute("SELECT ID FROM Experiments WHERE Experiment_name = ?", (Experiment_name,)).fetchone()

    if result:
        # 2. Get the ID from the result tuple
        existing_id = result[0]
        print(f"Session {Experiment_name} already exists with ID: {existing_id}")
        
        model_name = data.get('model_name')
        
        # Remove old data for the same model to prevent duplication
        Experiment_db.execute("DELETE FROM Scores WHERE Experiment_ID = ? AND Model = ?", (existing_id, model_name))


        # 3. Insert into Scores using that specific ID
        scores = data.get('Scores', {})
        Experiment_db.execute('''
            INSERT INTO Scores (Experiment_ID, Accuracy, Precision, Recall, F1_Score, cross_val_score, Model)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (existing_id, scores.get('accuracy'), scores.get('precision'), scores.get('recall'), scores.get('f1_score'), scores.get('cross_val_score'), model_name))

        params = scores.get('params')
        if params is not None:
            Experiment_db.execute('''
                INSERT INTO Params (Experiment_ID, Model, Params)
                VALUES (?, ?, ?)
            ''', (existing_id, model_name, json.dumps(params)))

        Experiment_db.commit()
        return jsonify({"message": "updated", "experiment_id": existing_id}), 200
    else:

        # 4. Insert a new experiment and get the new ID
        Experiment_db.execute("INSERT INTO Experiments (Experiment_name) VALUES (?)", (Experiment_name,))
        Experiment_db.commit()
        # Retrieve the newly created experiment ID
        new_id = Experiment_db.execute("SELECT ID FROM Experiments WHERE Experiment_name = ? ORDER BY timestamp DESC LIMIT 1", (Experiment_name,)).fetchone()[0]
        print(f"Created new session {Experiment_name} with ID: {new_id}")
        
        # 5. Insert into Scores using the new ID
        model_name = data.get('model_name')
        scores = data.get('Scores', {})
        Experiment_db.execute('''
            INSERT INTO Scores (Experiment_ID, Accuracy, Precision, Recall, F1_Score, cross_val_score, Model)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (new_id, scores.get('accuracy'), scores.get('precision'), scores.get('recall'), scores.get('f1_score'), scores.get('cross_val_score'), model_name))

        params = scores.get('params')
        if params is not None:
            Experiment_db.execute('''
                INSERT INTO Params (Experiment_ID, Model, Params)
                VALUES (?, ?, ?)
            ''', (new_id, model_name, json.dumps(params)))

        Experiment_db.commit()
        return jsonify({"message": "created", "experiment_id": new_id}), 201
@app.route('/get_results',methods=['GET'])
def get_results():
    results = Experiment_db.execute('''
        SELECT e.Experiment_name, s.Model, s.Accuracy, s.Precision, s.Recall, s.F1_Score,s.cross_val_score
        FROM Experiments e
        JOIN Scores s ON e.ID = s.Experiment_ID 
    ''').fetchall()
    params=Experiment_db.execute('''
        SELECT e.Experiment_name, p.Model, p.Params
        FROM Experiments e
        JOIN Params p ON e.ID = p.Experiment_ID 
    ''').fetchall()
    output= []

    for r in results:
        output.append({
            "Experiment_name": r[0],
            "Model": r[1],
            "Accuracy": r[2],
            "Precision": r[3],
            "Recall": r[4],
            "F1_Score": r[5],
            "cross_val_score": r[6]
        })
    output2=[]    
    for p in params:
        output2.append({
            "Experiment_name": p[0],
            "Model": p[1],
            "Params": p[2]
        })

    return jsonify({"results": output, "params": output2}), 200
   
Experiment_db.commit()
if __name__=="__main__":
    app.run(debug=True, use_reloader=False, port=5000)