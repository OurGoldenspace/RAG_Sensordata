import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def build_vector_db():
    print("1. Loading sensor data from CSV...")
    # CSVLoader automatically turns each row into a LangChain Document
    loader = CSVLoader(file_path='./data/ai4i2020.csv')
    documents = loader.load()
    print(f"Loaded {len(documents)} rows of telemetry.")

    print("2. Initializing Hugging Face Embeddings (Local)...")
    # This downloads a small, fast, free embedding model to your machine
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("3. Building the Chroma Vector Store...")
    # This embeds the text and saves the database to a local folder
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db" 
    )

    print("Success! Database built and saved to ./chroma_db")

if __name__ == "__main__":
    build_vector_db()