import pandas as pd
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage
import joblib
import numpy as np

import logging
import os

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    filename='logs/agent_activity.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 1. Load the Dataset Safely
DATA_PATH = "data/ai4i2020.csv"
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: Could not find {DATA_PATH}.")
    df = pd.DataFrame()

# 2. Initialize Vector Database for RAG
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings, collection_name="machine_sops")
retriever = vector_db.as_retriever(search_kwargs={"k": 1})

# --- DEFINE DETERMINISTIC TOOLS ---

@tool
def get_machine_sensor_data(udi: int) -> str:
    """Use this tool to fetch the current sensor telemetry (temperature, speed, torque) for a specific machine UDI."""
    row = df[df['UDI'] == udi]
    if row.empty:
        return f"Error: Machine UDI {udi} not found in the database."
    
    air_temp = row['Air temperature [K]'].values[0]
    proc_temp = row['Process temperature [K]'].values[0]
    speed = row['Rotational speed [rpm]'].values[0]
    torque = row['Torque [Nm]'].values[0]
    
    return f"Telemetry for UDI {udi}: Air Temp: {air_temp}K, Process Temp: {proc_temp}K, Speed: {speed}rpm, Torque: {torque}Nm."

# Load the trained ML model and scaler globally
try:
    ml_model = joblib.load("models/rf_anomaly_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
except FileNotFoundError:
    print("Warning: ML model not found. Please run train_anomaly_model.py first.")

@tool
def check_machine_failure_status(udi: int) -> str:
    """Use this tool to predict if a machine is failing using the trained ML model."""
    row = df[df['UDI'] == udi]
    if row.empty:
        return f"Error: Machine UDI {udi} not found."
    
    # 1. Extract the raw sensor features
    features = row[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
    
    # 2. Scale the data using the saved scaler
    features_scaled = scaler.transform(features)
    
    # 3. Predict using the trained Random Forest model!
    prediction = ml_model.predict(features_scaled)[0]
    
    if prediction == 0:
        return f"Machine UDI {udi} is predicted to be operating normally by the ML model. No failures detected."
    
    # 4. If the model predicts a failure, check the telemetry to guess the mode 
    # (Since our binary model just predicts failure, we pull the specific flag for the RAG lookup)
    failures = [col for col in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF'] if row[col].values[0] == 1]
    failure_string = ', '.join(failures) if failures else 'Unknown Anomaly'
            
    return f"CRITICAL: ML Model predicts Machine UDI {udi} has FAILED. Probable failure modes: {failure_string}"



@tool
def retrieve_troubleshooting_sop(udi: int) -> str:
    """Use this tool ONLY if a machine has failed. Pass the machine UDI to retrieve the repair manual."""
    # 1. We look up the failure mode ourselves so the LLM doesn't have to!
    row = df[df['UDI'] == udi]
    if row.empty:
        return "Machine not found."
    
    failures = [col for col in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF'] if row[col].values[0] == 1]
    
    if not failures:
        return "No specific failure mode logged for retrieval."
        
    failure_mode = failures[0] # Grab the first failure mode
    
    # 2. Now we query the vector database using the string we found
    docs = retriever.invoke(failure_mode)
    return docs[0].page_content if docs else "No standard operating procedure found for this issue."

# Compile the tools into a list
tools = [get_machine_sensor_data, check_machine_failure_status, retrieve_troubleshooting_sop]

# --- INITIALIZE LLM & LANGGRAPH AGENT ---

# Llama 3.1 supports native tool calling, which LangGraph leverages perfectly
llm = ChatOllama(model="llama3.1", temperature=0)

# Build the LangGraph ReAct Agent
agent_executor = create_react_agent(llm, tools)

# --- MAIN EXECUTOR FUNCTION FOR APP.PY ---


def query_telemetry_and_diagnose(user_question: str):
    """Executes the LangGraph agent and parses the output for the Streamlit UI."""
    try:
        # The Locked-Down System Prompt
        system_prompt = """You are a Lead AI Reliability Engineer.
        You have access to EXACTLY three tools. You must use their exact names:
        1. `get_machine_sensor_data`: Fetches current telemetry (temp, speed, torque).
        2. `check_machine_failure_status`: Predicts if the machine failed using our ML model.
        3. `retrieve_troubleshooting_sop`: Retrieves the repair manual using ONLY the integer UDI.

        CRITICAL WORKFLOW AND TOOL RULES:
        - ALWAYS call `get_machine_sensor_data` and `check_machine_failure_status` first.
        - If the ML model predicts the machine is operating normally, DO NOT use `retrieve_troubleshooting_sop`. Just report that it is healthy and stop.
        - If the ML model predicts a failure, you MUST use `retrieve_troubleshooting_sop`.
        - DO NOT make up tool names (Never use 'get_sop' or 'check_telemetry_data').
        
        STRICT FORMATTING RULES:
        - NEVER use the string '<|python_tag|>' in your response.
        - To call a tool, use the format:
          Thought: [Your reasoning]
          Action: [Exact Tool Name]
          Action Input: [UDI Number]
        - To provide the final answer, use the format:
          Final Answer: [Your natural language report to the user]
        - ZERO CHATTER: Do not explain your steps; just use the format above.
        """
        
        # Run the graph
        response = agent_executor.invoke({
            "messages": [
                ("system", system_prompt),
                ("user", user_question)
            ]
        })
        
        # The final answer is always the content of the very last message in the graph state
        final_answer = response["messages"][-1].content
        
        # Extract the SOP by looking for the specific ToolMessage
        retrieved_sop = None
        for message in response["messages"]:
            if isinstance(message, ToolMessage) and message.name == "retrieve_troubleshooting_sop":
                # Ensure we don't display LangChain Pydantic validation errors in the UI
                if "Error invoking tool" not in message.content:
                    retrieved_sop = message.content
                
        return final_answer, retrieved_sop

    except Exception as e:
        return f"System Error: {str(e)}", None