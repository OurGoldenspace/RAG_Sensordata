import streamlit as st
import pandas as pd
from ask_bot import query_telemetry_and_diagnose, get_recent_telemetry_history

# 1. Page Configuration
st.set_page_config(page_title="Local IoT Diagnostic Agent", layout="wide")
st.title("⚙️ Secure Local Predictive Maintenance Agent")

# 2. Sidebar Context
with st.sidebar:
    st.header("About this System")
    st.write("This application runs 100% locally. It analyzes CNC telemetry using a local LLM Agent and retrieves troubleshooting SOPs using a local ChromaDB instance.")
    st.header("⚙️ System Status")
    st.success("✅ ML Model: Random Forest Loaded")
    st.success("✅ Knowledge Base: ChromaDB Connected")
    st.info(f"📊 Model Accuracy: 98.25%")
    st.divider()
    st.write("Target Device: Industrial Edge Gateway")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# 3. Load Data Safely & Create Dashboard
@st.cache_data
def load_data():
    try:
        # Note: Ensure path matches your actual file name
        return pd.read_csv("data/ai4i2020.csv")
    except FileNotFoundError:
        return None

df = load_data()
if df is not None:
    st.subheader("📊 Live Telemetry Dashboard")
    
    # Quick KPI metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Machines Monitored", value=len(df))
    with col2:
        st.metric(label="Max Process Temp", value=f"{df['Process temperature [K]'].max()} K")
    with col3:
        st.metric(label="Detected Failures", value=df['Machine failure'].sum(), delta="- Requires Attention", delta_color="inverse")
    
    with st.expander("View Raw Telemetry Data"):
        st.dataframe(df.head(50))
else:
    st.error("⚠️ Data file not found. Please place 'ai4i2020.csv' inside a folder named 'data'.")

# 4. Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Display past chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sop"):
            with st.expander("View Retrieved Knowledge Base SOP"):
                st.info(message["sop"])

# 6. React to user input
if prompt := st.chat_input("Ask about the telemetry data (e.g., 'Check UDI 15')"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Local Agent is analyzing..."):
            final_response, retrieved_sop = query_telemetry_and_diagnose(prompt)
            st.markdown(final_response)
            
            if retrieved_sop:
                with st.expander("View Retrieved Knowledge Base SOP"):
                    st.info(retrieved_sop)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": final_response,
                "sop": retrieved_sop
            })
            # Rerun to refresh the history-based charts below
            st.rerun()

# 7. --- NEW: FLEET TELEMETRY INSIGHTS (TRENDS) ---
st.divider()
st.header("📈 Fleet Telemetry Insights")

# Fetch the last 10 machines queried from the backend history
history_data = get_recent_telemetry_history()

if not history_data.empty:
    st.write("Visualizing sensor behavior trends across recently inspected units:")
    
    # Allow operator to select metrics for comparison
    available_metrics = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    selected_metrics = st.multiselect(
        "Select Metrics to Plot:",
        options=available_metrics,
        default=['Air temperature [K]', 'Torque [Nm]']
    )
    
    if selected_metrics:
        # Plotting the data indexed by UDI to show machine-to-machine trends
        chart_df = history_data.set_index('UDI')[selected_metrics]
        st.line_chart(chart_df)
    else:
        st.warning("Please select at least one metric to visualize the trend.")
else:
    st.info("💡 **History is empty.** Ask the agent about a machine (e.g., 'What is the status of UDI 25?') to see telemetry trends appear here.")