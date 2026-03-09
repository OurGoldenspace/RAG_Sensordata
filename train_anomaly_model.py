import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def train_model():
    print("--- Starting ML Pipeline for IoT Telemetry ---")
    
    # 1. Load the Data
    DATA_PATH = "data/ai4i2020.csv"
    if not os.path.exists(DATA_PATH):
        print(f"Error: Could not find {DATA_PATH}.")
        return

    df = pd.read_csv(DATA_PATH)
    
    # 2. Feature Engineering & Selection
    # We only want the AI to learn from the physical sensor readings, not the IDs.
    features = [
        'Air temperature [K]', 
        'Process temperature [K]', 
        'Rotational speed [rpm]', 
        'Torque [Nm]', 
        'Tool wear [min]'
    ]
    
    X = df[features]
    y = df['Machine failure'] # The target we want to predict

    # 3. Train/Test Split (80% for training, 20% for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Data Processing: Scale the features 
    # (Important for algorithms that are sensitive to data variance)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Train the Random Forest Model
    print("Training Random Forest Classifier on historical sensor data...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

# 1. Save Confusion Matrix
    y_pred = ml_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Failure Prediction Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('plots/confusion_matrix.png')

# 2. Save Feature Importance
    importances = ml_model.feature_importances_
    features = ['Air temp', 'Process temp', 'Speed', 'Torque', 'Tool wear']
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=features)
    plt.title('Key Drivers of Machine Failure')
    plt.savefig('plots/feature_importance.png')

    
    # 6. Evaluate the Model
    y_pred = model.predict(X_test_scaled)
    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 7. Save the Model and Scaler to disk
    # We create a 'models' directory to keep the repo clean
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/rf_anomaly_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    
    print("\nSuccess! Model and Scaler saved to the 'models/' directory.")
    print("Your Edge AI agent can now use these files for real-time inference.")

if __name__ == "__main__":
    train_model()