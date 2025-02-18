python -m venv .venv
pip install -r requirements.txt
.\.venv\Scripts\activate
sqlite3 mlflow.db
mlflow server --backend-store-uri sqlite:///backend.db
dvc init
dvc add .\data\iris_dataset.csv
dvc remote add -d storage ./Simulate-Remote-Storage
dvc pull .\data\iris_dataset.csv.dvc
