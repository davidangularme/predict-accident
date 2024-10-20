def process_real_time_data(driver_input, vehicle_input):
    # Preprocess inputs
    # Assume driver_input and vehicle_input are dictionaries with sensor readings
    input_data = {**driver_input, **vehicle_input}
    input_df = pd.DataFrame([input_data])

    # Normalize features using the same scaler from training
    input_features = input_df[feature_columns]
    input_features_scaled = scaler.transform(input_features)

    # Make prediction
    risk_score = model.predict_proba(input_features_scaled)[:, 1]
    return risk_score[0]

# Example usage in a real-time loop
import time

def alert_driver():
    # Implement the alert mechanism (e.g., sound an alarm)
    print("Alert! High risk of accident detected.")

while True:
    # Read data from sensors (placeholder functions)
    driver_input = read_driver_sensors()
    vehicle_input = read_vehicle_sensors()
    
    # Process data and get risk score
    risk_score = process_real_time_data(driver_input, vehicle_input)
    
    # Threshold to determine if alert is needed
    if risk_score > 0.7:
        alert_driver()
    
    # Wait for a short interval before next reading
    time.sleep(1)
