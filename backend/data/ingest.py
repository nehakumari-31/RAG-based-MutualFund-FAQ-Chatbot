import os
import sys
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
import shutil
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
        # RELAXED: NAV/AUM labels are short, so let's be less aggressive
        if len(line) < 2:
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
    # 1. Deduplication: Clear existing vector database
    if os.path.exists(DB_DIR):
        print(f"🧹 Clearing existing vector database at {DB_DIR}...")
        shutil.rmtree(DB_DIR, ignore_errors=True)
        
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
    from playwright.sync_api import sync_playwright
    import subprocess
    
    def ensure_playwright_browsers():
        """Ensure Playwright browsers are installed."""
        print("🔍 Checking Playwright browser installation...")
        try:
            # Simple check by trying to launch in a dummy context if possible, 
            # but usually it's better to just try and run the install command
            # if we get the specific error.
            pass 
        except Exception:
            pass

    with sync_playwright() as p:
        # 1.5. Ensure browser is available
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as e:
            if "playwright install" in str(e).lower() or "executable doesn't exist" in str(e).lower():
                print("⚠️ Playwright browser missing. Attempting to install chromium...")
                try:
                    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
                    browser = p.chromium.launch(headless=True)
                except Exception as install_err:
                    print(f"❌ Failed to auto-install Playwright browser: {install_err}")
                    print("Please run 'playwright install chromium' manually on your server.")
                    return
            else:
                print(f"❌ Failed to launch browser: {e}")
                return

        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        
        for idx, source in enumerate(sources, 1):
            url = source.get('url', '')
            doc_type = source.get('document_type', 'General')
            scheme = source.get('scheme', 'general')
            description = source.get('description', 'Unknown Source')
            
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
                        for doc in docs:
                            doc.page_content = clean_text(doc.page_content)
                            doc.metadata.update({
                                "scheme": scheme,
                                "document_type": doc_type,
                                "source": url,
                                "description": description,
                                "is_live": False
                            })
                        all_documents.extend(docs)
                        print(f"  ✓ Loaded {len(docs)} pages from PDF\n")
                
                else:
                    # Process Dynamic Web Page (Live)
                    print(f"  ⬇ Loading dynamic web page content...")
                    try:
                        page.goto(url, wait_until="networkidle", timeout=60000)
                        time.sleep(7) 
                        
                        raw_content = page.evaluate("document.body.innerText")
                        clean_content = clean_text(raw_content)
                        
                        print(f"  ✓ Captured dynamic content from {url} ({len(clean_content)} chars)")
                        if len(clean_content) > 0:
                            print(f"    Sample: {clean_content[:150]}...")
                        
                        # Verification
                        if "₹" in clean_content or "NAV" in clean_content:
                            print(f"    ➡️ Found potential NAV data!")
                            nav_idx = clean_content.find("NAV")
                            if nav_idx != -1:
                                print(f"    NAV Snippet: ...{clean_content[max(0, nav_idx-50):nav_idx+100]}...")
                        else:
                            print(f"    ⚠️ Warning: No 'NAV' or '₹' found in captured content.")
                        
                        doc = Document(page_content=clean_content, metadata={
                            "source": url,
                            "scheme": scheme,
                            "document_type": doc_type,
                            "description": description,
                            "is_live": True
                        })
                        all_documents.extend([doc])
                        print("")
                    except Exception as e:
                        print(f"  ✗ Failed to scrape {url}: {e}\n")
            except Exception as e:
                print(f"  ✗ Failed to process: {e}\n")
                continue
        
        browser.close()
    
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
