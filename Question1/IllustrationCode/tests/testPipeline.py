import pytest
from my_ml_pipeline import process_data, train_model, deploy_model

def test_process_data():
    data = load_sample_data()
    processed_data = process_data(data)
    assert processed_data is not None
    assert len(processed_data) > 0

def test_train_model():
    processed_data = load_processed_data()
    model = train_model(processed_data)
    assert model is not None

def test_deploy_model():
    model = load_trained_model()
    deploy_model(model)
    assert model.is_deployed()  
