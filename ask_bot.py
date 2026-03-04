import pandas as pd
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage

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

@tool
def check_machine_failure_status(udi: int) -> str:
    """Use this tool to check if a machine failed and get its specific failure mode (e.g., OSF, HDF, TWF)."""
    row = df[df['UDI'] == udi]
    if row.empty:
        return f"Error: Machine UDI {udi} not found."
    
    if row['Machine failure'].values[0] == 0:
        return f"Machine UDI {udi} is operating normally. No failures detected."
    
    failures = []
    for col in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']:
        if row[col].values[0] == 1:
            failures.append(col)
            
    return f"CRITICAL: Machine UDI {udi} FAILED. Failure modes detected: {', '.join(failures)}"

@tool
def retrieve_troubleshooting_sop(failure_mode: str) -> str:
    """Use this tool ONLY if a machine has failed. Pass the failure mode (e.g., 'OSF') to retrieve the repair manual."""
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
        # We pass the system instructions and the user question as a message history
        system_prompt = "You are a Lead AI Reliability Engineer. Answer the user's question by using your tools to check telemetry and failure status. If it failed, retrieve the SOP and provide a final diagnostic report."
        
        # Run the graph
        response = agent_executor.invoke({
            "messages": [
                ("system", system_prompt),
                ("user", user_question)
            ]
        })
        
        # The final answer is always the content of the very last message in the graph state
        final_answer = response["messages"][-1].content
        
        # Extract the SOP by looking for the specific ToolMessage in the conversation history
        retrieved_sop = None
        for message in response["messages"]:
            if isinstance(message, ToolMessage) and message.name == "retrieve_troubleshooting_sop":
                retrieved_sop = message.content
                
        return final_answer, retrieved_sop

    except Exception as e:
        return f"System Error: {str(e)}", None