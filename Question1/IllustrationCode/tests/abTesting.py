import time
import numpy as np

def monitor_model_performance(model, X_test, y_test):
    start_time = time.time()

    # Get model predictions, calc accuray and response timetaken
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test) 
    response_time = time.time() - start_time
    
    # Resource usage monitoring with DUMMY value
    cpu_usage = 50  
    memory_usage = 100  

    return accuracy, response_time, cpu_usage, memory_usage

# Monitor model performance
accuracy, response_time, cpu_usage, memory_usage = monitor_model_performance(model, X_test, y_test)
print(f"Accuracy: {accuracy}, Response Time: {response_time}s, CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%")
