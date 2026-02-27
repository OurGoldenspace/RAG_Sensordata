import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document # <-- REQUIRED FOR THE METADATA FIX

def build_semantic_pdf_db():
    pdf_path = "./data/manual.pdf"
    persist_dir = "./chroma_db_pdf"

    if not os.path.exists(pdf_path):
        print(f"❌ Error: Could not find {pdf_path}")
        return

    print("1. Loading the PDF Manual...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    # We keep the page text separate first so we can track page numbers roughly
    # Alternatively, for pure semantic, we combine:
    full_text = " ".join([page.page_content for page in pages])

    print("2. Initializing Semantic Chunker...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    text_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile" 
    )

    print("3. Slicing the manual based on meaning...")
    # This returns a list of Document objects
    semantic_docs = text_splitter.create_documents([full_text])
    print(f"✅ Success! Created {len(semantic_docs)} semantically coherent chunks.")

    print("4. Sanitizing Metadata for Pydantic V2...")
    docs_to_save = []
    
    for i, doc in enumerate(semantic_docs):
        # 1. Ensure metadata is a dictionary, not None
        clean_metadata = doc.metadata if doc.metadata is not None else {}
        
        # 2. Add fallback page/source info (Semantic Chunker loses original page numbers)
        if "page" not in clean_metadata:
            clean_metadata["page"] = "Multiple/Semantic"
        if "source" not in clean_metadata:
            clean_metadata["source"] = "manual.pdf"
        
        # 3. Create a fresh Document object with the clean metadata
        new_doc = Document(page_content=doc.page_content, metadata=clean_metadata)
        docs_to_save.append(new_doc)

    print("5. Saving to ChromaDB...")
    vector_db = Chroma.from_documents(
        documents=docs_to_save,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"🚀 Database build complete in {persist_dir}")

if __name__ == "__main__":
    # Remove old DB to prevent duplicate data
    import shutil
    if os.path.exists("./chroma_db_pdf"):
        shutil.rmtree("./chroma_db_pdf")
        print("🗑️ Cleaned old database.")
        
    build_semantic_pdf_db()