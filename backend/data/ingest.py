import os
import csv
import requests
import time
from pathlib import Path
from urllib.parse import urlparse, unquote
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load env from phase1 root
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

# Configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SOURCES_CSV = os.path.join(PROJECT_ROOT, "sources.csv")
DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, "downloaded_sources")
DB_DIR = os.path.join(PROJECT_ROOT, "vector_db")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def download_pdf(url, download_dir):
    """Download PDF from URL and return local file path."""
    try:
        # Create download directory if it doesn't exist
        os.makedirs(download_dir, exist_ok=True)
        
        # Extract filename from URL
        parsed_url = urlparse(url)
        filename = unquote(os.path.basename(parsed_url.path))
        
        # If filename doesn't end with .pdf, generate one from URL
        if not filename.endswith('.pdf'):
            filename = f"document_{hash(url)}.pdf"
        
        filepath = os.path.join(download_dir, filename)
        
        # Skip download if file already exists
        if os.path.exists(filepath):
            print(f"  ✓ Already downloaded: {filename}")
            return filepath
        
        # Download the file
        print(f"  ⬇ Downloading: {filename}")
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"  ✓ Downloaded: {filename}")
        return filepath
    
    except Exception as e:
        print(f"  ✗ Failed to download {url}: {e}")
        return None

def clean_text(text):
    """
    Clean extracted text while preserving semantic structure.
    Removes excessive noise but keeps numbers and labels intact.
    """
    import re
    
    # 1. Basic normalization
    text = text.replace('\xa0', ' ') # Remove non-breaking spaces
    
    # 2. Remove purely decorative characters but keep currency symbols
    text = re.sub(r'[^\x00-\x7F₹]+', ' ', text)
    
    # 3. Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    
    # 4. Handle lines and segments
    lines = [line.strip() for line in text.split('\n')]
    clean_lines = []
    for line in lines:
        if not line:
            continue
        # Skip lines that look like pure UI noise (very short navigation items etc)
        if len(line.split()) < 2 and len(line) < 5:
            continue
        clean_lines.append(line)
        
    return '\n'.join(clean_lines)

def load_sources_from_csv():
    """Load document sources from sources.csv."""
    sources = []
    
    if not os.path.exists(SOURCES_CSV):
        print(f"Error: sources.csv not found at {SOURCES_CSV}")
        return sources
    
    with open(SOURCES_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sources.append({
                'url': row['url'],
                'document_type': row['document_type'],
                'scheme': row['scheme'],
                'description': row['description']
            })
    
    print(f"Loaded {len(sources)} sources from sources.csv")
    return sources

def ingest_docs():
    """Main ingestion function that downloads and processes documents from URLs."""
    all_documents = []
    
    # Load sources from CSV
    sources = load_sources_from_csv()
    
    if not sources:
        print("No sources found to ingest.")
        return
    
    print(f"\n{'='*60}")
    print(f"Starting document ingestion from {len(sources)} sources")
    print(f"{'='*60}\n")
    
    # Process each source
    for idx, source in enumerate(sources, 1):
        url = source['url']
        scheme = source['scheme']
        doc_type = source['document_type']
        description = source['description']
        
        print(f"[{idx}/{len(sources)}] Processing: {description}")
        print(f"  URL: {url}")
        print(f"  Scheme: {scheme} | Type: {doc_type}")
        
        try:
            # Check if URL is a PDF or web page
            if url.endswith('.pdf') or 'pdf' in url.lower():
                # Download PDF
                filepath = download_pdf(url, DOWNLOAD_DIR)
                
                if filepath:
                    # Load PDF
                    loader = PyPDFLoader(filepath)
                    docs = loader.load()
                    
                    # Add metadata and clean text
                    for doc in docs:
                        doc.page_content = clean_text(doc.page_content)
                        doc.metadata["scheme"] = scheme
                        doc.metadata["document_type"] = doc_type
                        doc.metadata["source_url"] = url
                        doc.metadata["description"] = description
                    
                    all_documents.extend(docs)
                    print(f"  ✓ Loaded {len(docs)} pages from PDF\n")
            
            else:
                # Load web page content using direct Playwright for dynamic data (NAV, AUM, etc.)
                print(f"  ⬇ Loading dynamic web page content...")
                
                from playwright.sync_api import sync_playwright
                
                with sync_playwright() as p:
                    # Launch browser
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    
                    try:
                        # Navigate with generous timeout
                        page.goto(url, wait_until="networkidle", timeout=60000)
                        # Extra wait for dynamic charts/numbers
                        time.sleep(5)
                        
                        # Extract clean text from the body
                        text_content = page.evaluate("document.body.innerText")
                        clean_content = clean_text(text_content)
                        
                        docs = [Document(page_content=clean_content, metadata={
                            "scheme": scheme,
                            "document_type": doc_type,
                            "source_url": url,
                            "description": description
                        })]
                        
                        all_documents.extend(docs)
                        print(f"  ✓ Captured dynamic content from {url} ({len(clean_content)} chars)\n")
                        
                    except Exception as e:
                        print(f"  ✗ Failed to scrape {url}: {e}\n")
                    finally:
                        browser.close()
        
        except Exception as e:
            print(f"  ✗ Failed to process: {e}\n")
            continue
    
    if not all_documents:
        print("No documents were successfully loaded.")
        return
    
    print(f"\n{'='*60}")
    print(f"Successfully loaded {len(all_documents)} document pages")
    print(f"{'='*60}\n")
    
    # Chunking
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(all_documents)
    print(f"✓ Created {len(splits)} chunks\n")
    
    # Vector DB (Using Free Local HuggingFace Embeddings)
    print("Creating vector embeddings and storing in ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    # ChromaDB auto-persists in newer versions
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully ingested documents into {DB_DIR}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    ingest_docs()
