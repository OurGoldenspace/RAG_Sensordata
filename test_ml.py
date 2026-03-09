import joblib
import pandas as pd
import numpy as np
import os

def test_inference():
    print("🔍 Testing ML Model Integrity...")
    
    # 1. Load the assets
    model_path = "models/rf_anomaly_model.pkl"
    scaler_path = "models/scaler.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("❌ Error: Model or Scaler missing. Run 'python train_anomaly_model.py' first.")
        return

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 2. Define a "Failing" machine scenario (High Temp + High Torque)
    # Features: Air temp, Process temp, Speed, Torque, Tool wear
    failing_machine = np.array([[305.0, 315.0, 1200, 65.0, 220]])
    
    # 3. Define a "Healthy" machine scenario
    healthy_machine = np.array([[298.0, 308.0, 1500, 40.0, 10]])

    # 4. Run Inference
    failing_scaled = scaler.transform(failing_machine)
    healthy_scaled = scaler.transform(healthy_machine)

    fail_pred = model.predict(failing_scaled)[0]
    health_pred = model.predict(healthy_scaled)[0]

    # 5. Output Results
    print(f"\nTest 1 (High Stress): {'🚩 FAILURE DETECTED' if fail_pred == 1 else '✅ HEALTHY'}")
    print(f"Test 2 (Normal Ops): {'🚩 FAILURE DETECTED' if health_pred == 1 else '✅ HEALTHY'}")

    if fail_pred == 1 and health_pred == 0:
        print("\n✅ SUCCESS: The model is distinguishing between stress and normal states correctly.")
    else:
        print("\n⚠️ WARNING: Model logic might be inverted or biased. Check training data.")

if __name__ == "__main__":
    test_inference()