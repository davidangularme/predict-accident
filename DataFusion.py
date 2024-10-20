# Select relevant features
feature_columns = ['hrv', 'eeg_alpha', 'eeg_beta', 'eye_closure_rate',
                   'acceleration', 'steering_variability', 'lane_position', 'brake_pressure']

# Prepare feature matrix X and target vector y
X = data_scaled[feature_columns]
y = data_scaled['accident_label']
