import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def build_vector_db():
    print("Initializing SOP knowledge base locally...")
    
    # 1. Define the synthetic SOPs
    sops = [
        "SOP for Tool Wear Failure (TWF): Triggered by gradual tool degradation. Action: Inspect milling insert for abrasive wear. Replace tool if wear exceeds 0.2mm. Recalibrate Z-axis offset.",
        "SOP for Heat Dissipation Failure (HDF): Triggered when Air-to-Process temperature difference is less than 8.6 K. Action: Check coolant pump pressure. Clean heat exchanger fins. Verify ambient factory temperature.",
        "SOP for Power Failure (PWF): Triggered when Power (Torque * Rad/s) exceeds 9000 W or drops below 3500 W. Action: Inspect spindle drive amplifier. Check incoming 3-phase line voltage for drops.",
        "SOP for Overstrain Failure (OSF): Triggered when Torque exceeds 60 Nm and Rotational Speed drops below 1300 rpm. Action: Immediately halt spindle. Check for tool breakage. Inspect workpiece for hard spots.",
        "SOP for Random Failures (RNF): Triggered by general degradation not tied to a specific kinematic parameter. Action: Perform a full Level 1 diagnostic check on all electrical cabinets and mechanical couplings."
    ]

    # 2. Initialize Local Embeddings via HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 3. Create and persist the Chroma Database locally
    persist_directory = "./chroma_db"
    
    vector_db = Chroma.from_texts(
        texts=sops, 
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="machine_sops"
    )
    
    print(f"Vector database built entirely locally and saved to {persist_directory}!")

if __name__ == "__main__":
    build_vector_db()