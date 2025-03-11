import pandas as pd

# Load data
X_train = pd.read_csv("/Users/sujithsuresh/Downloads/Predictive_Maintenance/data/X_train.csv")
X_test = pd.read_csv("/Users/sujithsuresh/Downloads/Predictive_Maintenance/data/X_test.csv")
y_train = pd.read_csv("/Users/sujithsuresh/Downloads/Predictive_Maintenance/data/y_train.csv")
y_test = pd.read_csv("/Users/sujithsuresh/Downloads/Predictive_Maintenance/data/y_test.csv")

# Add labels to the datasets
X_train['failure'] = y_train
X_test['failure'] = y_test

# Combine train and test data
combined_data = pd.concat([X_train, X_test], ignore_index=True)

# Save the combined dataset
combined_data.to_csv("/Users/sujithsuresh/Downloads/Predictive_Maintenance/data/combined_data.csv", index=False)