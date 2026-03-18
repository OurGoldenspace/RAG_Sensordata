import subprocess
import os

def initialize_system():
    print("🛠️  Auto-initializing Diagnostic System...")
    
    if not os.path.exists("models/rf_anomaly_model.pkl"):
        print("Training ML model...")
        subprocess.run(["python", "train_model.py"])
        
    if not os.path.exists("chroma_db"):
        print("Building Vector Knowledge Base...")
        subprocess.run(["python", "ingest.py"])

    print("🚀 Launching Dashboard...")
    subprocess.run(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    initialize_system()