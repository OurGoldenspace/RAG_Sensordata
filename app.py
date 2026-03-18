import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from ask_bot import query_telemetry_and_diagnose, get_recent_telemetry_history

# 1. Page Configuration
st.set_page_config(page_title="Local IoT Diagnostic Agent", layout="wide")
st.title("⚙️ Secure Local Predictive Maintenance Agent")

# 2. Analytics & Forecasting Logic (Defined BEFORE the UI)
def calculate_tool_forecast(df, udi):
    """Predicts Remaining Useful Life using Linear Regression trend analysis."""
    # Filtering for the specific machine's historical trend
    machine_data = df[df['UDI'] <= udi].tail(10) 
    if len(machine_data) < 5:
        return None, None

    # X = cycle index, y = tool wear level
    X = np.array(range(len(machine_data))).reshape(-1, 1)
    y = machine_data['Tool wear [min]'].values
    
    model = LinearRegression().fit(X, y)
    
    # Forecast next 20 cycles
    future_X = np.array(range(len(machine_data), len(machine_data) + 20)).reshape(-1, 1)
    forecast = model.predict(future_X)
    
    # RUL calculation: (Failure Limit 200m - Current Wear) / Wear Rate
    current_wear = y[-1]
    wear_rate = model.coef_[0] 
    rul = (200 - current_wear) / wear_rate if wear_rate > 0 else 999
        
    return forecast, max(0, int(rul))

# 3. Data Loading
@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/ai4i2020.csv")
    except FileNotFoundError:
        return None

df = load_data()

# 4. Sidebar: Control Panel
with st.sidebar:
    st.header("🛠️ System Control")
    st.write("Transitioning from reactive to proactive maintenance via Hybrid AI.")
    
    if df is not None:
        st.divider()
        st.subheader("🔮 Forecasting Inputs")
        target_udi = st.number_input("Select Machine UDI", value=12, min_value=1)
        
        st.divider()
        st.subheader("🔗 Data Insights")
        with st.expander("Feature Correlation Heatmap"):
            corr_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Machine failure']
            fig_corr, ax_corr = plt.subplots()
            sns.heatmap(df[corr_features].corr(), annot=True, cmap='coolwarm', ax=ax_corr)
            st.pyplot(fig_corr)

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# 5. Main Dashboard: Visualization & GenAI
if df is not None:
    # --- SECTION A: KPIs ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Fleet Size", len(df))
    col2.metric("Max Temperature", f"{df['Process temperature [K]'].max()} K")
    col3.metric("Failures Detected", df['Machine failure'].sum(), delta="- Action Required", delta_color="inverse")

    # --- SECTION B: Forecasting (The 'Predictive' Requirement) ---
    st.divider()
    st.subheader("🔮 Predictive Maintenance Horizon")
    forecast_values, rul = calculate_tool_forecast(df, target_udi)

    if forecast_values is not None:
        res_col, plot_col = st.columns([1, 2])
        with res_col:
            st.write(f"### Remaining Useful Life: **{rul} Cycles**")
            if rul < 25:
                st.error("⚠️ CRITICAL: Immediate Tool Replacement Required.")
            elif rul < 60:
                st.warning("⚡ SCHEDULE: Maintenance required this week.")
            else:
                st.success("✅ OPTIMAL: Tool is within safe operating limits.")
        
        with plot_col:
            fig_f, ax_f = plt.subplots(figsize=(10, 4))
            ax_f.plot(range(20), forecast_values, '--', color='orange', label="Forecasted Wear")
            ax_f.axhline(y=200, color='r', linestyle='-', label="Failure Threshold")
            ax_f.set_title(f"Wear Forecast for UDI {target_udi}")
            ax_f.legend()
            st.pyplot(fig_f)

    # --- SECTION C: GenAI Agent (The 'Generative' Requirement) ---
    st.divider()
    st.subheader("🤖 AI Diagnostic Assistant")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sop"):
                with st.expander("View Retrieved SOP"):
                    st.info(message["sop"])

    if prompt := st.chat_input("Ask: 'Is machine 12 failing and how do I fix it?'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("🔍 Agent investigating telemetry...", expanded=False) as status:
                final_answer, retrieved_sop = query_telemetry_and_diagnose(prompt)
                status.update(label="✅ Diagnostic Complete", state="complete")
            
            st.markdown(final_answer)
            if retrieved_sop:
                with st.expander("View Retrieved SOP"):
                    st.info(retrieved_sop)
            
            st.session_state.messages.append({"role": "assistant", "content": final_answer, "sop": retrieved_sop})

else:
    st.error("⚠️ CSV data not found. Please check your /data folder.")

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