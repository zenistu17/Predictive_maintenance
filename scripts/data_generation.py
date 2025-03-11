import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

# Set random seed for reproducibility
np.random.seed(42)

# Generate timestamps
timestamps = pd.date_range(start="2023-01-01", periods=1000, freq="H")

# Base values for sensor readings
temperature = np.random.normal(50, 5, 1000)
vibration = np.random.normal(5, 0.5, 1000)
pressure = np.random.normal(100, 10, 1000)

# Add realistic patterns
# 1. Daily temperature cycles
hour_of_day = np.array([t.hour for t in timestamps])
temperature += 5 * np.sin(hour_of_day * 2 * np.pi / 24)

# 2. Weekly maintenance pattern (lower vibration after maintenance)
day_of_week = np.array([t.dayofweek for t in timestamps])
maintenance_effect = np.where(day_of_week == 0, -0.5, 0)  # Maintenance on Mondays
vibration += maintenance_effect

# 3. Gradual degradation leading to failures
degradation = np.zeros(1000)
failure = np.zeros(1000, dtype=int)

# Create realistic failure patterns
failure_points = [150, 300, 400, 550, 650, 750, 900]  # More failure points
for point in failure_points:
    # Gradual degradation before failure
    degradation_start = max(0, point - 50)
    for i in range(degradation_start, point):
        degradation[i] = (i - degradation_start) / (point - degradation_start) * 3
    
    # Mark the failure
    failure[point] = 1
    
    # Recovery after failure
    recovery_end = min(999, point + 20)
    for i in range(point + 1, recovery_end):
        degradation[i] = 3 * (1 - (i - point) / (recovery_end - point))

# Add noise to degradation
degradation += np.random.normal(0, 0.1, 1000)

# Introduce sudden failures
sudden_failure_points = [250, 700]  # Additional sudden failure points
for point in sudden_failure_points:
    failure[point] = 1
    temperature[point] += 10  # Sudden spike in temperature
    vibration[point] += 2  # Sudden spike in vibration
    pressure[point] -= 20  # Sudden drop in pressure

# Apply degradation to sensor readings
temperature += degradation * 2
vibration += degradation * 0.4
pressure -= degradation * 5

# Add sensor noise
temperature += np.random.normal(0, 0.5, 1000)
vibration += np.random.normal(0, 0.1, 1000)
pressure += np.random.normal(0, 2, 1000)

# Add external factors
# Simulate weather effects (e.g., temperature drops at night)
weather_effect = np.sin(hour_of_day * 2 * np.pi / 24) * 2
temperature += weather_effect

# Simulate operational load (e.g., higher vibration during peak hours)
operational_load = np.where((hour_of_day >= 8) & (hour_of_day <= 18), 0.5, 0)
vibration += operational_load

# Smooth sensor readings
temperature = savgol_filter(temperature, 5, 2)
vibration = savgol_filter(vibration, 5, 2)
pressure = savgol_filter(pressure, 5, 2)

# Create data frame
data = pd.DataFrame({
    "timestamp": timestamps,
    "temperature": temperature,
    "vibration": vibration,
    "pressure": pressure,
    "failure": failure
})

# Save to CSV
data.to_csv("/Users/sujithsuresh/Downloads/Predictive_Maintenance/data/raw_data.csv", index=False)
print("Enhanced synthetic data generated and saved.")