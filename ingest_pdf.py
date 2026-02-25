import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker # <-- NEW IMPORT
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def build_semantic_pdf_db():
    print("1. Loading the PDF Manual...")
    loader = PyPDFLoader("./data/manual.pdf")
    pages = loader.load()
    
    # We first combine all pages into one large text block for the Chunker to analyze
    full_text = " ".join([page.page_content for page in pages])

    print("2. Initializing Semantic Chunker (this uses the AI to find topic breaks)...")
    # This chunker needs an embedding model to "understand" the sentences
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # The 'percentile' threshold means it splits when a sentence is 
    # significantly different from the ones before it.
    text_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile" 
    )

    print("3. Slicing the manual based on meaning...")
    docs = text_splitter.create_documents([full_text])
    print(f"Success! The manual was split into {len(docs)} semantically coherent chunks.")

    print("4. Saving to the PDF Database...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db_pdf" 
    )
    print("PDF Database updated with Semantic Chunks.")

if __name__ == "__main__":
    # IMPORTANT: Delete your old ./chroma_db_pdf folder before running this 
    # so we don't double-index the data.
    build_semantic_pdf_db()