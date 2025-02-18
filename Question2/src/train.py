import mlflow
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import pandas as pd

# Set tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("my-experiment-1")

with mlflow.start_run():
    df = pd.read_csv("data/iris_dataset.csv")

    # Load data
    # X, y = load_iris(return_X_y=True)
    X = df.iloc[:, :-1]  # All columns except the last
    y = df.iloc[:, -1]   # The last column

    # Set parameters and log them
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

# Fetch the experiment ID
client = MlflowClient("http://127.0.0.1:5000")
experiment_id = client.get_experiment_by_name("my-experiment-1").experiment_id

# Get the most recent run's ID using search_runs
runs = client.search_runs(experiment_ids=[experiment_id], order_by=["start_time desc"], max_results=1)
print(runs)
run_id = runs[0].info.run_id

# Register model in the Model Registry
mlflow.register_model(
    model_uri=f"runs:/{run_id}/models",
    name='iris-classifier'
)
