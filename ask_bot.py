import pandas as pd
import joblib
import numpy as np
import logging
import os
import re

# LangChain & LangGraph Imports (2026 Standards)
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

# 1. Setup Logging & Directories
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    filename='logs/agent_activity.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 2. Load Dataset & ML Models
DATA_PATH = "data/ai4i2020.csv"
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    logging.error(f"Dataset not found at {DATA_PATH}")
    df = pd.DataFrame()

try:
    ml_model = joblib.load("models/rf_anomaly_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
except FileNotFoundError:
    logging.warning("ML models not found. Run training script first.")
    ml_model, scaler = None, None

# 3. Initialize RAG (Vector DB)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embeddings, 
    collection_name="machine_sops"
)

# Global memory for UI charts
query_history = []

# --- TOOLS ---

@tool
def get_machine_sensor_data(udi: int) -> str:
    """Fetches telemetry (Temp, Speed, Torque) for a specific machine UDI."""
    row = df[df['UDI'] == udi]
    if row.empty: return f"Error: UDI {udi} not found."
    
    data = {
        "Air": row['Air temperature [K]'].values[0],
        "Process": row['Process temperature [K]'].values[0],
        "Speed": row['Rotational speed [rpm]'].values[0],
        "Torque": row['Torque [Nm]'].values[0]
    }
    return f"Telemetry UDI {udi}: {data['Air']}K Air, {data['Process']}K Process, {data['Speed']}rpm, {data['Torque']}Nm."

@tool
def check_machine_failure_status(udi: int) -> str:
    """Predicts if a machine is failing using the Random Forest ML model."""
    if ml_model is None or scaler is None:
        return "ML Model not initialized."
        
    row = df[df['UDI'] == udi]
    if row.empty: return f"Error: UDI {udi} not found."
    
    features = row[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
    features_scaled = scaler.transform(features)
    prediction = ml_model.predict(features_scaled)[0]
    
    if prediction == 0:
        return f"UDI {udi} status: NORMAL. No anomalies detected."
    
    failures = [col for col in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF'] if row[col].values[0] == 1]
    mode = ', '.join(failures) if failures else 'Unknown Anomaly'
    return f"CRITICAL: Machine UDI {udi} has FAILED. Mode: {mode}"

@tool
def retrieve_troubleshooting_sop(query: str) -> str:
    """Retrieves repair steps. Input can be a UDI number or keywords like 'wear limit'."""
    search_term = str(query)
    
    # If the input is a UDI, find the specific failure mode first
    if search_term.isdigit():
        udi = int(search_term)
        row = df[df['UDI'] == udi]
        if not row.empty:
            failures = [col for col in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF'] if row[col].values[0] == 1]
            if failures: search_term = failures[0]

    docs = vector_db.similarity_search(search_term, k=1)
    if docs:
        source = docs[0].metadata.get('source', 'SOP_Manual.txt')
        return f"SOURCE: {source}\n\nCONTENT: {docs[0].page_content}"
    return f"No SOP found for '{search_term}'."

tools = [get_machine_sensor_data, check_machine_failure_status, retrieve_troubleshooting_sop]

# --- AGENT SETUP ---
llm = ChatOllama(model="llama3.2:3b", temperature=0)
# create_react_agent is the stable LangGraph way to bind tools
agent_executor = create_react_agent(llm, tools)

def get_recent_telemetry_history():
    """Returns data for the last 10 machines queried."""
    if not query_history: return pd.DataFrame()
    return df[df['UDI'].isin(query_history)].tail(10)

def query_telemetry_and_diagnose(user_question: str):
    """Main interface for Streamlit."""
    try:
        # 1. Update Trend History
        found_udis = re.findall(r'\d+', user_question)
        if found_udis:
            val = int(found_udis[0])
            if not query_history or query_history[-1] != val:
                query_history.append(val)

        # 2. Configure System Logic
        system_message = (
            "You are a Lead Reliability Engineer. Use tools to diagnose machines.\n"
            "1. Check sensor data and failure status for any UDI mentioned.\n"
            "2. If a failure is found, ALWAYS retrieve the SOP manual.\n"
            "3. If healthy, provide telemetry and stop.\n"
            "4. Always cite the source file from the manual (e.g., 'Per SOP_HDF.txt...')."
        )

        # 3. Invoke Agent
        inputs = {"messages": [SystemMessage(content=system_message), HumanMessage(content=user_question)]}
        result = agent_executor.invoke(inputs)
        
        # 4. Parse Results
        final_answer = result["messages"][-1].content
        
        # Find the SOP in the message history
        retrieved_sop = None
        for msg in reversed(result["messages"]):
            if isinstance(msg, ToolMessage) and msg.name == "retrieve_troubleshooting_sop":
                retrieved_sop = msg.content
                break
                
        return final_answer, retrieved_sop

    except Exception as e:
        logging.error(f"Execution Error: {str(e)}")
        return f"System Error: {str(e)}", None