import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# Example training data (reference data)
train_data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.normal(5, 2, 1000)
})

# New incoming data (for testing)
new_data = pd.DataFrame({
    'feature1': np.random.normal(0, 1.5, 500),  # Notice the variance change
    'feature2': np.random.normal(5, 2, 500)
})

# Compare mean and variance between train and new data for each feature
def compare_statistics(train_data, new_data):
    for column in train_data.columns:
        train_mean = train_data[column].mean()
        train_std = train_data[column].std()
        new_mean = new_data[column].mean()
        new_std = new_data[column].std()

        print(f"Comparing {column}:")
        print(f"Training Mean: {train_mean}, New Mean: {new_mean}")
        print(f"Training Std: {train_std}, New Std: {new_std}")
        
        if abs(train_mean - new_mean) > 0.1 or abs(train_std - new_std) > 0.2:
            print(f"Warning: Significant statistical change detected in {column}!")
        else:
            print(f"{column} appears similar.")
        print()

# Run comparison
compare_statistics(train_data, new_data)
