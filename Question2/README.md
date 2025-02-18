python -m venv .venv
pip install -r requirements.txt
.\.venv\Scripts\activate
sqlite3 mlflow.db
mlflow server --backend-store-uri sqlite:///mlflow.db
dvc init
dvc add .\Question2\data\iris_dataset.csv
dvc remote add -d storage ./Simulate-Remote-Storage
dvc push
dvc pull .\Question2\data\iris_dataset.csv.dvc
