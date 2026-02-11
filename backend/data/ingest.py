import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Load env from phase1 root
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

# Configuration
# Path relative to project root
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Document Sources"))
DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../vector_db"))
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def get_scheme_name(filename):
    """Detects scheme name from filename for metadata classification."""
    fn = filename.lower()
    if "elss" in fn or "tax saver" in fn:
        return "hdfc_elss"
    elif "flexi cap" in fn:
        return "hdfc_flexi_cap"
    elif "large cap" in fn or "top 100" in fn:
        return "hdfc_large_cap"
    return "general"

def clean_text(text):
    """Basic cleaning for PDF extracted text."""
    import re
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace multiple spaces with a single space
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def ingest_docs():
    all_documents = []
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory {DATA_DIR} not found.")
        return

    print(f"Reading PDFs from: {DATA_DIR}")
    
    # Load all PDFs from Document Sources
    # Filter out Groww pages and blog posts (non-official sources)
    forbidden_terms = ["groww", "blog"]
    
    for file in os.listdir(DATA_DIR):
        fn_lower = file.lower()
        if file.endswith(".pdf") and not any(term in fn_lower for term in forbidden_terms):
            print(f"Processing {file}...")
            try:
                loader = PyPDFLoader(os.path.join(DATA_DIR, file))
                docs = loader.load()
                
                # Add metadata and clean text
                scheme = get_scheme_name(file)
                for doc in docs:
                    doc.page_content = clean_text(doc.page_content)
                    doc.metadata["scheme"] = scheme
                    doc.metadata["source"] = file
                
                all_documents.extend(docs)
            except Exception as e:
                print(f"Failed to process {file}: {e}")

    if not all_documents:
        print("No documents found to ingest.")
        return

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(all_documents)
    print(f"Created {len(splits)} chunks.")

    # Vector DB (Using Free Local HuggingFace Embeddings)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    # Ensure persistence across restarts (Streamlit deploys).
    vectorstore.persist()
    print(f"Successfully ingested documents into {DB_DIR}")

if __name__ == "__main__":
    ingest_docs()
