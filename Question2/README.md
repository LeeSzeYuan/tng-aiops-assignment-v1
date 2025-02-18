# Local Setup Guide

Follow these steps to set up the project locally:

## 1. **Create a Virtual Environment**
Create a virtual environment to isolate project dependencies:

```bash
python -m venv .venv
```

## 2. **Chaging working directory**
Cd into Question 2 folder

```bash
cd Question2
```

## 3. **Install Required Dependencies**
Install the required dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## 4. **Activate the Virtual Environment**
Activate the virtual environment:

- On Windows:
  ```bash
  .\.venv\Scripts\activate
  ```

- On macOS/Linux:
  ```bash
  source .venv/bin/activate
  ```

## 5. **Set Up SQLite Database for MLflow**
Initialize the SQLite database for MLflow:

```bash
sqlite3 mlflow.db
```

Run the following to start the MLflow server:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db
```

## 6. **Initialize DVC (Data Version Control)**
Initialize DVC in your project:

```bash
dvc init
```

## 7. **Pull Dataset from DVC**
To pull the dataset from the DVC remote storage:

```bash
dvc pull .\data\iris_dataset.csv.dvc
```

## 8. **Add Dataset to DVC**
Add the dataset (`iris_dataset.csv`) to DVC for version control:

```bash
dvc add .\data\iris_dataset.csv
```

## 9. **Set Up DVC Remote Storage**
Configure a remote storage for DVC to store dataset versions:

```bash
dvc remote add -d storage ./Simulate-Remote-Storage
```

## 10. **Push Data to DVC Remote Storage**
Push the dataset to the configured DVC remote storage:

```bash
dvc push
```

## 11. **Run Python Scripts**
To run with tracking-server

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db
python .\src\train-tracking-server.py
```
To run **without** tracking-server

```bash
python .\src\train-local-filesystem.py
```