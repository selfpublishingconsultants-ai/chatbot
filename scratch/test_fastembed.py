import os
import sys
from dotenv import load_dotenv

load_dotenv()

try:
    from langchain_community.embeddings import FastEmbedEmbeddings
    from langchain_community.vectorstores import FAISS
    
    print("[INFO] Initializing FastEmbedEmbeddings...")
    embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    VECTOR_STORE_PATH = "vector_store/self_publishing_consultant_faiss_index"
    
    print("[INFO] Loading local FAISS index...")
    if os.path.exists(VECTOR_STORE_PATH):
        vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        print("[OK] FAISS index loaded successfully with FastEmbedEmbeddings!")
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        print("[INFO] Invoking retriever query...")
        docs = retriever.invoke("What editing packages do you offer?")
        print(f"[OK] Retrieved {len(docs)} documents:")
        for i, doc in enumerate(docs):
            print(f"\n--- Document {i+1} ---")
            print(doc.page_content[:300])
    else:
        print(f"[ERROR] Vector store path not found: {VECTOR_STORE_PATH}")
except Exception as e:
    print(f"[ERROR] Test failed with: {e}")
    import traceback
    traceback.print_exc()
