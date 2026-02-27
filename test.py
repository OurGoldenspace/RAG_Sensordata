try:
    from langchain_community.retrievers import BM25Retriever
    print("✅ Community Retrievers: OK")
    
    # Check if the main wrapper actually linked correctly
    import langchain
    print(f"✅ LangChain Version: {langchain.__version__}")
    
    from langchain.retrievers import EnsembleRetriever
    print("✅ Ensemble Retriever: OK")
except Exception as e:
    print(f"❌ Failed at: {e}")
    print("\n💡 FIX: Run 'pip install langchain-classic' to restore missing paths.")