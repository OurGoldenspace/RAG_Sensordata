import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

def train_model():
    print("--- Starting ML Pipeline for IoT Telemetry ---")
    
    # 1. Load the Data
    DATA_PATH = "data/ai4i2020.csv"
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Could not find {DATA_PATH}.")
        return

    df = pd.read_csv(DATA_PATH)
    
    # 2. Feature Engineering & Selection
    features = [
        'Air temperature [K]', 
        'Process temperature [K]', 
        'Rotational speed [rpm]', 
        'Torque [Nm]', 
        'Tool wear [min]'
    ]
    
    X = df[features]
    y = df['Machine failure'] 

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Data Processing: Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Train the Random Forest Model
    print("🚀 Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    # --- PLOTTING SECTION (Corrected variable names) ---
    os.makedirs("plots", exist_ok=True)
    y_pred = model.predict(X_test_scaled) # Using the correct model variable

    # 1. Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Failure Prediction Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('plots/confusion_matrix.png')
    plt.close() # Close plot to save memory

    # 2. Save Feature Importance
    importances = model.feature_importances_
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=features)
    plt.title('Key Drivers of Machine Failure')
    plt.savefig('plots/feature_importance.png')
    plt.close()
    
    # 6. Evaluate the Model
    print("\n--- Model Evaluation Results ---")
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 7. Save the Model and Scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/rf_anomaly_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    
    print("\n🎉 Success! Model, Scaler, and Plots have been saved.")

if __name__ == "__main__":
    train_model()