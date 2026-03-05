Edge-AI Predictive Maintenance System
An autonomous, air-gapped diagnostic agent for industrial IoT telemetry. This system combines Machine Learning (Random Forest), Agentic Workflows (LangGraph), and Local RAG (ChromaDB) to provide real-time machine health assessments and repair protocols.

🚀 Key Features
Predictive Diagnostics: Custom-trained Random Forest model achieves 98.25% accuracy in predicting machine failures based on sensor data.

Agentic Orchestration: Built with LangGraph to autonomously sequence telemetry retrieval, ML inference, and manual lookups.

Local RAG: Ingests unstructured technical manuals into a ChromaDB vector store for instant SOP retrieval.

Edge-Ready: Designed to run entirely on local hardware (Ollama/Llama 3.1) for zero-latency, private industrial environments.

🛠️ Tech Stack
LLM: Llama 3.1 (via Ollama)

Orchestration: LangGraph / LangChain

Machine Learning: Scikit-Learn (Random Forest)

Vector Database: ChromaDB

UI: Streamlit

Dataset: AI4I 2020 Predictive Maintenance (UCI Repository)

🔧 Setup Instructions
Install Requirements:
pip install -r requirements.txt

Start Ollama:
Ensure Ollama is running and Llama 3.1 is pulled: ollama pull llama3.1

Train the ML Model:
python train_anomaly_model.py

Ingest Manuals:
python ingest.py

Run the App:
streamlit run app.py