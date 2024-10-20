# Example: Calculate HRV features from heart rate data
def calculate_hrv(heart_rate_series):
    # Implement HRV metrics, e.g., RMSSD (Root Mean Square of Successive Differences)
    hr_diff = np.diff(heart_rate_series)
    rmssd = np.sqrt(np.mean(hr_diff ** 2))
    return rmssd

data_scaled['hrv'] = calculate_hrv(data['heart_rate'].values)

# Example: Calculate acceleration from speed data
data['acceleration'] = data['vehicle_speed'].diff() / data['timestamp'].diff().dt.total_seconds()

# Example: Calculate steering angle variability
data['steering_variability'] = data['steering_angle'].rolling(window=5).std()
