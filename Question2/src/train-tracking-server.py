import mlflow
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import pandas as pd

# Set tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("experiment-1")

with mlflow.start_run():
    # Load data
    df = pd.read_csv("data/iris_dataset.csv")
    # X, y = load_iris(return_X_y=True)
    X = df.iloc[:, :-1] 
    y = df.iloc[:, -1] 

    # Log parameters
    params = {"C": 0.1, "random_state": 42}
    for key, value in params.items():
        mlflow.log_param(key, value)

    # Train model
    lr = LogisticRegression(**params).fit(X, y)
    y_pred = lr.predict(X)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy_score(y, y_pred))

    # Log model
    mlflow.sklearn.log_model(lr, artifact_path="models")
    print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")

# Fetch the experiment ID and get run ID
client = MlflowClient("http://127.0.0.1:5000")
experiment_id = client.get_experiment_by_name("experiment-1").experiment_id
runs = client.search_runs(experiment_ids=[experiment_id], order_by=["start_time desc"], max_results=1)
run_id = runs[0].info.run_id

# Register model in Registry
mlflow.register_model(
    model_uri=f"runs:/{run_id}/models",
    name='iris-classifier-tracking-server'
)
