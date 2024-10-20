import pandas as pd
import numpy as np

# Load driver health data
driver_data = pd.read_csv('driver_health_data.csv', parse_dates=['timestamp'])

# Load vehicle data
vehicle_data = pd.read_csv('vehicle_data.csv', parse_dates=['timestamp'])

# Merge datasets on timestamp
data = pd.merge_asof(driver_data.sort_values('timestamp'),
                     vehicle_data.sort_values('timestamp'),
                     on='timestamp',
                     direction='nearest')

# Handle missing values
data = data.dropna()

# Normalize features
from sklearn.preprocessing import StandardScaler

features = data.drop(['timestamp', 'accident_label'], axis=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Prepare the final dataset
data_scaled = pd.DataFrame(features_scaled, columns=features.columns)
data_scaled['accident_label'] = data['accident_label'].values
