import streamlit as st
import pandas as pd
from ask_bot import query_telemetry_and_diagnose

# 1. Page Configuration
st.set_page_config(page_title="Local IoT Diagnostic Agent", layout="wide")
st.title("⚙️ Secure Local Predictive Maintenance Agent")

# 2. Sidebar Context
with st.sidebar:
    st.header("About this System")
    st.write("This application runs 100% locally. It analyzes CNC telemetry using a local LLM Pandas Agent and retrieves troubleshooting SOPs using a local ChromaDB instance.")
    st.markdown("---")
    st.write("**Privacy Status:** 🟢 Air-Gapped / No API Calls")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# 3. Load Data Safely & Create Dashboard
@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/ai4i2020.csv")
    except FileNotFoundError:
        return None

df = load_data()
if df is not None:
    # --- NEW DASHBOARD SECTION ---
    st.subheader("📊 Live Telemetry Dashboard")
    
    # Create 3 columns for quick KPI metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Machines Monitored", value=len(df))
    with col2:
        st.metric(label="Max Process Temp", value=f"{df['Process temperature [K]'].max()} K")
    with col3:
        st.metric(label="Detected Failures", value=df['Machine failure'].sum(), delta="- Requires Attention", delta_color="inverse")
    
    # Create a time-series line chart for the first 100 rows to simulate live data
    st.write("Recent Torque & Rotational Speed Trends (Sample Window)")
    chart_data = df[['Torque [Nm]', 'Rotational speed [rpm]']].head(100)
    # Normalize the data just for visual comparison on the same chart
    chart_data['Rotational speed [rpm] (Scaled)'] = chart_data['Rotational speed [rpm]'] / 30 
    st.line_chart(chart_data[['Torque [Nm]', 'Rotational speed [rpm] (Scaled)']])
    # -----------------------------

    with st.expander("View Raw Telemetry Data"):
        st.dataframe(df.head(50))
else:
    st.error("⚠️ Data file not found. Please place 'ai4i2020_predictive_maintenance_dataset.csv' inside a folder named 'data'.")

# 4. Initialize Chat History in Session State
# This is the magic that stops the app from "forgetting" your previous questions
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Display past chat messages on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If there was an SOP retrieved in the past, display it in an expander
        if message.get("sop"):
            with st.expander("View Retrieved Knowledge Base SOP"):
                st.info(message["sop"])

# 6. React to user input
# st.chat_input locks to the bottom of the screen and waits for the user to hit Enter
if prompt := st.chat_input("Ask about the telemetry data (e.g., 'What are the sensor readings for UDI 168? Did it fail?')"):
    
    # Immediately display the user's new question
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add the user's question to the session memory
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display the assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Local Agent is analyzing the CSV..."):
            
            # Call your logic script
            final_response, retrieved_sop = query_telemetry_and_diagnose(prompt)
            
            # Display the answer
            st.markdown(final_response)
            
            # Display the SOP if one was triggered
            if retrieved_sop:
                with st.expander("View Retrieved Knowledge Base SOP"):
                    st.info(retrieved_sop)
            
            # Add the assistant's answer and SOP to the session memory.
            st.session_state.messages.append({
                "role": "assistant", 
                "content": final_response,
                "sop": retrieved_sop
            })