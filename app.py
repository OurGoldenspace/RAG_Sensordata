import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# 1. UI Configuration
st.set_page_config(page_title="IIoT Maintenance Copilot", page_icon="⚙️", layout="wide")
st.title("⚙️ Industrial Maintenance Copilot")
st.caption("Verified Architecture: Memory + Agentic Routing + Source Attribution | Llama 3.2")

# 2. Load Engines & Retrievers (Cached)
@st.cache_resource
def load_system():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load Databases
    csv_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    pdf_db = Chroma(persist_directory="./chroma_db_pdf", embedding_function=embeddings)
    
    # Setup Retrievers
    csv_retriever = csv_db.as_retriever(search_kwargs={"k": 5})
    pdf_retriever = pdf_db.as_retriever(search_kwargs={"k": 5})
    
    # Initialize Local LLM (Temperature 0 for zero "bullshit")
    llm = ChatOllama(model="llama3.2", temperature=0)
    
    return llm, csv_retriever, pdf_retriever

llm, csv_retriever, pdf_retriever = load_system()

# 3. Step 1: Strict Rephraser (Memory)
rephrase_prompt = ChatPromptTemplate.from_messages([
    ("system", "Output ONLY a search query based on the history. No conversation. No questions."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])
rephrase_chain = rephrase_prompt | llm | StrOutputParser()

# 4. Step 2: Strict Router (Agentic Logic)
router_prompt = ChatPromptTemplate.from_template(
    "Classify this question as 'telemetry' (sensors/data), 'manual' (safety/fixes), or 'both'. Output ONLY the word: {question}"
)
router_chain = router_prompt | llm | StrOutputParser()

# 5. Response Generation Engine
def generate_response(user_input, chat_history):
    # A. Rephrase the question using history
    standalone_q = rephrase_chain.invoke({"input": user_input, "chat_history": chat_history}).strip()
    
    # B. Route the question
    route = router_chain.invoke({"question": standalone_q}).strip().lower()
    print(f"DEBUG: The AI chose route: {route}") 
    
    # Debug info in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ AI Debugger")
    st.sidebar.info(f"📍 **Route:** {route.upper()}")
    st.sidebar.code(f"Search Query:\n{standalone_q}")

    # C. Retrieve Context & Metadata
    docs = []
    if "telemetry" in route or "both" in route:
        docs.extend(csv_retriever.invoke(standalone_q))
    if "manual" in route or "both" in route:
        docs.extend(pdf_retriever.invoke(standalone_q))

    print(f"DEBUG: Found {len(docs)} matching documents.")
    for d in docs:
        print(f"DEBUG: Snippet: {d.page_content[:100]}...")
    
    # Extract Sources (Metadata)
    sources = []
    for d in docs:
        source_name = d.metadata.get("source", "Unknown")
        page = d.metadata.get("page", "N/A")
        row = d.metadata.get("row", "N/A")
        
        if row != "N/A":
            sources.append(f"📊 CSV Log (Row {row})")
        elif page != "N/A":
            sources.append(f"📄 Manual (Page {int(page) + 1})")
        else:
            sources.append(f"🔍 Source: {source_name}")
    
    unique_sources = list(set(sources))
    context_text = "\n\n".join([d.page_content for d in docs])

    # D. Final Answer (Cold Technical Engine)
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a technical diagnostic tool. Use the context to provide a direct answer. "
                   "1. Use only the context. 2. Do not ask questions. 3. No conversational filler. "
                   "If the answer is not in the context, say 'DATA_NOT_FOUND'.\n\n"
                   "Context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    final_chain = qa_prompt | llm | StrOutputParser()
    raw_answer = final_chain.invoke({
        "context": context_text,
        "chat_history": chat_history,
        "input": standalone_q
    })

    # E. Guardrail: Remove trailing follow-up questions
    clean_lines = [line for line in raw_answer.split('\n') if not line.strip().endswith('?')]
    final_answer = "\n".join(clean_lines).strip()

    if not final_answer or "DATA_NOT_FOUND" in final_answer:
        return "I could not find a specific answer in the local telemetry or the maintenance manual.", []
    
    return final_answer, unique_sources

# 6. Streamlit Chat Interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display Message History
for msg in st.session_state.chat_history:
    st.chat_message("human" if isinstance(msg, HumanMessage) else "ai").write(msg.content)

# Handle New User Input
if prompt := st.chat_input("Ask about machine health (e.g., 'What is the torque for a Power Failure?')"):
    st.chat_message("human").write(prompt)
    
    with st.chat_message("ai"):
        with st.spinner("Analyzing sensors and manuals..."):
            response, found_sources = generate_response(prompt, st.session_state.chat_history)
            st.write(response)
            
            # Show Source Attribution
            if found_sources:
                with st.expander("✅ Verified Data Sources"):
                    for s in found_sources:
                        st.caption(s)
    
    # Update Session Memory
    st.session_state.chat_history.extend([
        HumanMessage(content=prompt), 
        AIMessage(content=response)
    ])