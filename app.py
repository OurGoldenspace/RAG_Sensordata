import streamlit as st
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# --- ADVANCED SEARCH TOOLS ---
# 1. First, import the underlying Ranker to register it in Python's namespace
try:
    from flashrank import Ranker
except ImportError:
    st.error("Missing 'flashrank' library. Please run: pip install flashrank")

# 2. Import the Classic Bridge and Community tools
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever

# 3. 🔥 CRITICAL FIX: Rebuild the model NOW that 'Ranker' is imported
FlashrankRerank.model_rebuild()

# 1. UI Setup
st.set_page_config(page_title="IIoT Advanced Copilot", layout="wide")
st.title("⚙️ IIoT Advanced Copilot")
st.caption("Hybrid Search (Vector + BM25) + FlashRank Re-ranking | 2026 Edition")

@st.cache_resource
def load_advanced_engine():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatOllama(model="llama3.2", temperature=0)
    
    # Load Vector Stores
    csv_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    pdf_db = Chroma(persist_directory="./chroma_db_pdf", embedding_function=embeddings)
    
    # 1. Sanitize PDF data for BM25
    raw_data = pdf_db.get()
    if not raw_data['documents']:
        st.error("PDF Database is empty! Please run your ingest script first.")
        return llm, csv_db, None

    documents = raw_data['documents']
    # Metadata Fix: Ensure every chunk has a dict (even if empty) to satisfy Pydantic
    metadatas = [m if m is not None else {} for m in raw_data['metadatas']]

    # 2. Setup Hybrid Ensemble (Keyword + Semantic)
    bm25 = BM25Retriever.from_texts(documents, metadatas=metadatas)
    bm25.k = 10
    
    vector_retriever = pdf_db.as_retriever(search_kwargs={"k": 10})
    
    ensemble = EnsembleRetriever(
        retrievers=[bm25, vector_retriever], 
        weights=[0.5, 0.5]
    )
    
    # 3. Setup Re-ranker
    try:
        compressor = FlashrankRerank()
        final_pdf_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=ensemble
        )
    except Exception as e:
        st.warning(f"Flashrank initialization issue: {e}. Falling back to standard hybrid search.")
        final_pdf_retriever = ensemble
    
    return llm, csv_db, final_pdf_retriever

llm, csv_db, pdf_hybrid_retriever = load_advanced_engine()

# --- REPHRASE & ROUTE LOGIC ---
rephrase_prompt = ChatPromptTemplate.from_messages([
    ("system", "Convert the user input into a standalone technical search query. Output ONLY the query."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])
rephrase_chain = rephrase_prompt | llm | StrOutputParser()

router_prompt = ChatPromptTemplate.from_template(
    "Classify this question as 'telemetry' (sensor data), 'manual' (how-to/fixes), or 'both'. Output ONLY the word: {question}"
)
router_chain = router_prompt | llm | StrOutputParser()

# --- MAIN GENERATOR ---
def generate_response(user_input, chat_history):
    # A. Rephrase based on history
    standalone_q = rephrase_chain.invoke({"input": user_input, "chat_history": chat_history})
    
    # B. Route to correct DB
    route = router_chain.invoke({"question": standalone_q}).strip().lower()
    st.sidebar.info(f"📍 Routing: {route.upper()}")
    st.sidebar.code(f"Query: {standalone_q}")

    docs = []
    # C. Execute Retrieval
    if "telemetry" in route or "both" in route:
        docs.extend(csv_db.as_retriever(search_kwargs={"k": 5}).invoke(standalone_q))
    
    if "manual" in route or "both" in route:
        if pdf_hybrid_retriever:
            docs.extend(pdf_hybrid_retriever.invoke(standalone_q))

    context = "\n\n".join([d.page_content for d in docs])
    
    # D. Final Strict Answer Generation
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert IIoT maintenance assistant. Use the context to provide a direct, "
                   "factual answer. Do not ask follow-up questions. If not in context, say 'DATA_NOT_FOUND'.\n\n"
                   "Context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    response = (qa_prompt | llm | StrOutputParser()).invoke({
        "context": context, "chat_history": chat_history, "input": standalone_q
    })
    
    return response

# --- CHAT UI ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display conversation
for m in st.session_state.chat_history:
    role = "human" if isinstance(m, HumanMessage) else "ai"
    with st.chat_message(role):
        st.write(m.content)

# Interaction
if prompt := st.chat_input("Ask about machine health or repair steps..."):
    with st.chat_message("human"):
        st.write(prompt)
    
    with st.chat_message("ai"):
        with st.spinner("Retrieving data and re-ranking results..."):
            res = generate_response(prompt, st.session_state.chat_history)
            st.write(res)
            
    # Update History
    st.session_state.chat_history.extend([
        HumanMessage(content=prompt), 
        AIMessage(content=res)
    ])