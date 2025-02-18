import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

train_data = pd.DataFrame()

# New incoming data
new_data = pd.DataFrame()

# Compare mean and variance between train and new data for each feature to check if there is any data drift
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

compare_statistics(train_data, new_data)
