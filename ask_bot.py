from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# THE FIX: We now import these from langchain_classic instead of langchain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

def ask_machine_bot(question):
    print("1. Waking up the local vector database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("2. Booting up Llama 3.2 (100% Local)...")
    llm = ChatOllama(model="llama3.2", temperature=0) 
    
    print("3. Assembling the RAG Chain...")
    system_prompt = (
        "You are an expert Industrial IoT maintenance assistant. "
        "Use the provided sensor log context to answer the technician's question. "
        "If you don't know the answer, say you don't know."
        "\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    print(f"\n--- INCOMING QUERY ---")
    print(f"Technician asks: {question}\n")
    
    response = rag_chain.invoke({"input": question})
    print(f"Copilot Answer: {response['answer']}\n")

if __name__ == "__main__":
    test_question = "Based on the historical logs, what are the typical rotational speeds and torque readings when a Heat Dissipation Failure occurs?"
    ask_machine_bot(test_question)