import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Configure the Web Page
st.set_page_config(page_title="IIoT Hybrid Copilot", page_icon="⚙️")
st.title("⚙️ Industrial Hybrid Copilot")
st.caption("Agentic Routing: Llama 3.2 + Dual Vector Databases")

# 2. Cache the Dual-Engine
@st.cache_resource
def load_hybrid_engine():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load BOTH databases
    csv_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    pdf_db = Chroma(persist_directory="./chroma_db_pdf", embedding_function=embeddings)
    
    csv_retriever = csv_db.as_retriever(search_kwargs={"k": 3})
    pdf_retriever = pdf_db.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatOllama(model="llama3.2", temperature=0) 
    
    return llm, csv_retriever, pdf_retriever

llm, csv_retriever, pdf_retriever = load_hybrid_engine()

# 3. The Router Function
def intelligent_route_and_answer(question):
    # STEP 1: The Interceptor (Decide which database to use)
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a routing assistant. Classify the user's question into one of three categories: "
                   "'telemetry' (if asking about sensor data, historical failures, rpm, temperature), "
                   "'manual' (if asking about how to fix something, safety, procedures, or instructions), "
                   "or 'both' (if it involves both). ONLY output the single word: telemetry, manual, or both."),
        ("human", "{question}")
    ])
    
    router_chain = router_prompt | llm | StrOutputParser()
    route_decision = router_chain.invoke({"question": question}).strip().lower()
    
    st.sidebar.write(f"🧠 **AI Routing Decision:** `{route_decision}`")

    # STEP 2: Retrieve the right context
    context = []
    if "telemetry" in route_decision or "both" in route_decision:
        csv_docs = csv_retriever.invoke(question)
        context.extend([doc.page_content for doc in csv_docs])
    
    if "manual" in route_decision or "both" in route_decision:
        pdf_docs = pdf_retriever.invoke(question)
        context.extend([doc.page_content for doc in pdf_docs])
        
    combined_context = "\n\n".join(context)

    # STEP 3: Generate the Final Answer
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Industrial IoT maintenance assistant. "
                   "Use the provided context to answer the technician's question. "
                   "If the context doesn't contain the answer, say you don't know.\n\n"
                   "Context: {context}"),
        ("human", "{question}")
    ])
    
    qa_chain = qa_prompt | llm | StrOutputParser()
    return qa_chain.invoke({"context": combined_context, "question": question})

# 4. The Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a telemetry or maintenance question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Routing query and analyzing databases..."):
            answer = intelligent_route_and_answer(prompt)
            st.markdown(answer)
            
    st.session_state.messages.append({"role": "assistant", "content": answer})