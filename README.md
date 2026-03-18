⚙️ Industrial IoT Diagnostic Agent
A Hybrid ML + GenAI approach to Predictive Maintenance.

This project is a local-first AI tool that doesn't just tell you when a machine is failing—it tells you how to fix it. It combines traditional Machine Learning for sensor analysis with Generative AI for technical reasoning.

🛠️ The Stack
Brain: Llama 3.2:3b (via Ollama)

Agent Logic: LangGraph & LangChain

ML Model: Random Forest (Scikit-Learn)

Database: ChromaDB (Vector Store)

Frontend: Streamlit

🚀 How It Works
Data Processing: Cleans and scales CNC sensor data (Temp, Speed, Torque).

ML Prediction: A Random Forest model classifies the machine status and gives a Confidence Score.

GenAI Agent: If a failure is detected, an AI Agent automatically searches the local knowledge base (RAG).

SOP Retrieval: The agent provides the exact repair steps from the technical manuals, with citations.

📂 Project Structure
data/ – CNC Telemetry CSV.

documents/ – Your repair manuals (.txt).

models/ – Saved ML models and Scalers.

app.py – The main dashboard.

ask_bot.py – The AI Agent logic and tools.

ingest.py – Builds the AI's "memory" from your documents.

train_model.py – Trains the failure prediction model.

⚡ Quick Start
1. Get the AI
Install Ollama and download the model:

Bash
ollama pull llama3.2:3b
2. Setup & Run
Bash
# Install libraries
pip install -r requirements.txt

# 1. Train the ML model
python train_model.py

# 2. Build the Knowledge Base
python ingest.py

# 3. Start the dashboard
streamlit run app.py
💡 Why This Matters
Privacy: Everything runs 100% locally. No factory data ever touches the cloud.

Hybrid AI: Uses ML for speed and GenAI for reasoning—the best of both worlds.

Actionable: Instead of a simple "Error Code," you get a full repair guide.

Shreyas Varma | University of New Brunswick

Capstone Project