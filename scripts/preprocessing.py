import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("/Users/sujithsuresh/Downloads/Predictive_Maintenance/data/raw_data.csv")
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Extract time-based features
data['hour'] = data['timestamp'].dt.hour
data['day'] = data['timestamp'].dt.day
data['day_of_week'] = data['timestamp'].dt.dayofweek
data['month'] = data['timestamp'].dt.month
data['is_weekend'] = data['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)

# Multiple rolling window statistics
for col in ['temperature', 'vibration', 'pressure']:
    # Short-term patterns
    data[f'{col}_rolling_mean_5'] = data[col].rolling(window=5, min_periods=1).mean()
    data[f'{col}_rolling_std_5'] = data[col].rolling(window=5, min_periods=1).std().fillna(0)
    
    # Medium-term patterns
    data[f'{col}_rolling_mean_24'] = data[col].rolling(window=24, min_periods=1).mean()
    data[f'{col}_rolling_std_24'] = data[col].rolling(window=24, min_periods=1).std().fillna(0)
    
    # Long-term patterns
    data[f'{col}_rolling_mean_72'] = data[col].rolling(window=72, min_periods=1).mean()
    
    # Rate of change features
    data[f'{col}_rate_of_change'] = data[col].diff().fillna(0)
    data[f'{col}_rate_of_change_rolling'] = data[f'{col}_rate_of_change'].rolling(window=5, min_periods=1).mean()
    
    # Lag features
    for lag in [1, 2, 3, 6, 12, 24]:
        data[f'{col}_lag_{lag}'] = data[col].shift(lag).fillna(method='bfill')
    
    # Deviation from expected
    data[f'{col}_deviation'] = data[col] - data[f'{col}_rolling_mean_24']

# More sophisticated time since last failure
data['failure_shifted'] = data['failure'].shift(1).fillna(0)
data['time_since_last_failure'] = data.groupby(data['failure_shifted'].cumsum()).cumcount()
data['time_to_next_failure'] = data.iloc[::-1].groupby(data.iloc[::-1]['failure'].cumsum()).cumcount().iloc[::-1]

# Indicator for recent failure (within last 48 hours)
data['recent_failure'] = (data['time_since_last_failure'] <= 48).astype(int)

# Interaction features
data['temp_vibration_interaction'] = data['temperature'] * data['vibration']
data['all_sensors_product'] = data['temperature'] * data['vibration'] * data['pressure']

# Drop unnecessary columns
data.drop(columns=['failure_shifted'], inplace=True)

# Drop rows with NaNs after feature engineering
data.dropna(inplace=True)

# Train-test split in a time-aware manner (not random)
train_size = int(0.8 * len(data))
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

# Select features and target
X_train = train_data.drop(columns=['timestamp', 'failure'])
y_train = train_data['failure']
X_test = test_data.drop(columns=['timestamp', 'failure'])
y_test = test_data['failure']

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save processed data
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv("/Users/sujithsuresh/Downloads/Predictive_Maintenance/data/X_train.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv("/Users/sujithsuresh/Downloads/Predictive_Maintenance/data/X_test.csv", index=False)
pd.DataFrame(y_train).to_csv("/Users/sujithsuresh/Downloads/Predictive_Maintenance/data/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("/Users/sujithsuresh/Downloads/Predictive_Maintenance/data/y_test.csv", index=False)

# Save feature names for reference
with open("/Users/sujithsuresh/Downloads/Predictive_Maintenance/data/feature_names.txt", "w") as f:
    f.write(",".join(X_train.columns))

print("Enhanced processed data saved successfully.")