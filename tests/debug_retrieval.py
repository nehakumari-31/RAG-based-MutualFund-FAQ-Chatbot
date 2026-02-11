import os
import sys
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

DB_DIR = os.path.join(PROJECT_ROOT, "vector_db")

def debug_retrieval(query, scheme_filter=None):
    print(f"\n--- Debugging Retrieval for: '{query}' (Filter: {scheme_filter}) ---")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    search_kwargs = {"k": 10}
    if scheme_filter:
        search_kwargs["filter"] = {"scheme": scheme_filter}
    
    # Get documents with scores
    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(query, **search_kwargs)
    
    if not docs_with_scores:
        print("❌ No documents found.")
        return

    for i, (doc, score) in enumerate(docs_with_scores, 1):
        is_web = doc.metadata.get('document_type') == 'Web'
        print(f"\n[{i}] Score: {score:.4f} | Scheme: {doc.metadata.get('scheme')} | Type: {doc.metadata.get('document_type')}")
        print(f"Source: {doc.metadata.get('description', 'Unknown')}")
        
        if is_web:
            print(f"FULL Content:\n{'-'*20}\n{doc.page_content}\n{'-'*20}")
        else:
            print(f"Content Snippet: {doc.page_content[:200]}...")
        
        # Check if NAV is in this chunk
        if "₹" in doc.page_content or "NAV" in doc.page_content:
            print("  ➡️ Contains NAV keywords or symbols")

if __name__ == "__main__":
    debug_retrieval("what is nav of elss", "hdfc_elss")
    debug_retrieval("current nav hdfc elss tax saver", "hdfc_elss")
