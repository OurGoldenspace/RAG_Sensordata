import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def build_vector_db():
    print("🚀 Initializing SOP knowledge base from local documents...")
    
    # 1. Path Configuration
    docs_dir = "./documents"
    persist_directory = "./chroma_db"
    
    # Ensure the documents directory exists
    if not os.path.exists(docs_dir):
        print(f"❌ Error: {docs_dir} folder not found. Create it and add your SOP .txt files first!")
        return

    # 2. Load Documents (Extracts Filename as Metadata)
    print(f"📂 Loading manuals from {docs_dir}...")
    loader = DirectoryLoader(docs_dir, glob="./*.txt", loader_cls=TextLoader)
    raw_documents = loader.load()
    
    if not raw_documents:
        print("⚠️ No .txt files found in the documents folder. Ingestion aborted.")
        return

    # 3. Chunking (AI Engineering Best Practice)
    # Breaking text into smaller pieces helps the AI find specific answers faster
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    # 4. Initialize Local Embeddings via HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 5. Build and Persist Chroma Database
    # We use 'from_documents' to preserve metadata (like the source filename)
    vector_db = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="machine_sops"
    )
    
    print(f"✅ Success! Vector database built with {len(documents)} chunks.")
    print(f"💾 Saved locally to: {persist_directory}")

if __name__ == "__main__":
    build_vector_db()