import os
import sys
import threading
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Add current dir to path for local imports
sys.path.append(os.path.dirname(__file__))
from router import get_router

# Load env from phase2 root
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env")))

# Configuration
DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../vector_db"))

_VECTOR_DB_LOCK = threading.Lock()
_VECTOR_DB_READY = False

# Embeddings cache for performance optimization
_EMBEDDINGS_CACHE = None
_EMBEDDINGS_LOCK = threading.Lock()

def get_embeddings():
    """Get cached embeddings instance to avoid reloading the model on every query."""
    global _EMBEDDINGS_CACHE
    
    if _EMBEDDINGS_CACHE is not None:
        return _EMBEDDINGS_CACHE
    
    with _EMBEDDINGS_LOCK:
        if _EMBEDDINGS_CACHE is None:
            print("🔄 Loading embeddings model (one-time initialization)...")
            start = time.time()
            _EMBEDDINGS_CACHE = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            elapsed = time.time() - start
            print(f"✓ Embeddings model loaded in {elapsed:.2f}s")
        return _EMBEDDINGS_CACHE


def _is_vector_db_ready() -> bool:
    if not os.path.isdir(DB_DIR):
        return False
    try:
        with os.scandir(DB_DIR) as it:
            return any(True for _ in it)
    except FileNotFoundError:
        return False


def ensure_vector_db() -> None:
    """Ensure persisted Chroma DB exists; build it from PDFs if missing.

    Streamlit deploys won't have `vector_db/` because it's gitignored, so we
    build it on-demand from `Document Sources/` on the first query.
    """
    global _VECTOR_DB_READY
    if _VECTOR_DB_READY and _is_vector_db_ready():
        return

    with _VECTOR_DB_LOCK:
        if _is_vector_db_ready():
            _VECTOR_DB_READY = True
            return

        # Build the vector DB from local PDFs.
        from backend.data.ingest import ingest_docs

        ingest_docs()
        _VECTOR_DB_READY = _is_vector_db_ready()

# Official HDFC Scheme Page Mapping
HDFC_SOURCE_LINKS = {
    "hdfc_large_cap": "https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct",
    "hdfc_flexi_cap": "https://www.hdfcfund.com/explore/mutual-funds/hdfc-flexi-cap-fund/direct",
    "hdfc_elss": "https://www.hdfcfund.com/explore/mutual-funds/hdfc-elss-tax-saver/direct",
    "general": "https://www.hdfcfund.com/investor-services/request-statement",
    "investor_education": "https://www.hdfcfund.com/information/investor-education",
    "sip_education": "https://www.hdfcfund.com/learn/blog/how-does-sip-work"
}

QA_PROMPT_TEMPLATE = """Information from Official HDFC Scheme Documents (SID/KIM/Notices):
--------------------------------------
{context}
--------------------------------------

Additional Official Process Knowledge:
- Capital Gains Statement: To download, visit the HDFC Mutual Fund 'Request Statement' page, scroll to 'Capital Gains Statement', click 'Click here' to log in (using PAN/Folio), and navigate to Reports > Capital Gains Statement. Alternatively, use the CAMS portal for a consolidated version.
- Account Statement: Can be requested via SMS (CAMS H SOA <Folio> <Password> to 56767) or via the portal's 'Request Statement' section.
- KYC Status: Can be checked on KRA websites (CVL, NDML, etc.) using PAN.

Chat History:
{chat_history}

Instructions for the Assistant:
1. You are a professional Groww Mutual Fund Assistant.
2. STRICT NO-ADVICE POLICY: If the user asks whether they should "buy", "sell", "invest", or "hold", YOU MUST POLITELY REFUSE to provide advice.
3. FACTS-ONLY RESPONSE: When refusing advice, you SHOULD still provide a concise, strictly factual summary of the mentioned fund(s) from the context (e.g., objective, lock-in period, riskometer). 
4. CONCISENESS: Your entire answer MUST be 3 sentences or less.
5. SOURCE LINE: You MUST end your response with the line: "Last updated from sources: [List official document names here]".
6. COMPLIANCE: Do NOT include URLs or links like 'https://...' directly in your answer text. They will be provided in a separate 'View Official Document' section.
7. Based ONLY on the provided context AND the Additional Official Process Knowledge, answer factual questions with data.
8. SOURCE GUARDRAIL: Do not use or reference unofficial sources.
9. If the answer is not in the context/knowledge, state: "I'm sorry, that specific data point is not available."

Current Question: {question}
Answer:"""

def get_rag_chain(scheme_filter=None):
    """Create a RAG chain using modern langchain API (no deprecated chains)."""
    ensure_vector_db()
    embeddings = get_embeddings()  # Use cached embeddings
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    
    search_kwargs = {"k": 10}
    if scheme_filter:
        search_kwargs["filter"] = {"scheme": scheme_filter}
    
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    # Create a simple chain using LCEL (LangChain Expression Language)
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    return retriever, llm, format_docs

class Phase4RAG:
    """Orchestrator for Phase 4 RAG with Memory, Routing, and Session tracking."""
    def __init__(self):
        self.router = get_router()
        self.sessions = {} # session_id -> {chat_history, last_scheme}
    
    def warmup(self):
        """Pre-load all expensive components to optimize first query performance."""
        print("\n🚀 Warming up chatbot components...")
        start_total = time.time()
        
        # 1. Pre-load embeddings model
        get_embeddings()
        
        # 2. Ensure vector database is ready
        print("🔄 Checking vector database...")
        start_db = time.time()
        ensure_vector_db()
        elapsed_db = time.time() - start_db
        print(f"✓ Vector database ready in {elapsed_db:.2f}s")
        
        # 3. Pre-warm router with a test query
        print("🔄 Initializing query router...")
        start_router = time.time()
        try:
            self.router.invoke({"query": "What is the expense ratio?"})
        except Exception:
            pass  # Ignore errors, just warming up the LLM
        elapsed_router = time.time() - start_router
        print(f"✓ Router initialized in {elapsed_router:.2f}s")
        
        elapsed_total = time.time() - start_total
        print(f"✅ Warmup complete in {elapsed_total:.2f}s - Ready for queries!\n")

    def get_session_state(self, session_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "chat_history": [],
                "last_scheme": "general"
            }
        return self.sessions[session_id]

    def query(self, user_query: str, session_id: str = "default"):
        state = self.get_session_state(session_id)
        
        # 1. Route the query
        route_res = self.router.invoke({"query": user_query})
        
        # 2. Logic for Scheme Detection & Inheritance
        if route_res.classification == "general":
            scheme_slug = "general"
        else: # scheme_specific
            candidate = route_res.scheme
            # Robust check for valid scheme from router
            if candidate and str(candidate).lower() not in ["none", "null", "undefined"]:
                scheme_slug = candidate
            else:
                scheme_slug = state["last_scheme"]
        
        # Only update last_scheme if we actually identified a specific fund
        if scheme_slug != "general":
            state["last_scheme"] = scheme_slug

        # 3. Get official links
        scheme_link = HDFC_SOURCE_LINKS.get(scheme_slug, HDFC_SOURCE_LINKS["general"])
        
        official_links = [
            {"label": "View Official Document", "url": scheme_link}
        ]
        
        # Special case: For 'min SIP' queries, add educational blog link
        query_lower = user_query.lower()
        if "min sip" in query_lower or "minimum sip" in query_lower:
            official_links.append({
                "label": "Learn: How Does SIP Work?", 
                "url": HDFC_SOURCE_LINKS["sip_education"]
            })

        # 4. Get RAG chain components
        retriever, llm, format_docs = get_rag_chain(
            scheme_filter=scheme_slug if scheme_slug != "general" else None
        )
        
        # 5. Retrieve relevant documents
        docs = retriever.invoke(user_query)
        context = format_docs(docs)
        
        # 6. Format chat history
        chat_history_str = "\n".join([
            f"Human: {msg['question']}\nAssistant: {msg['answer']}" 
            for msg in state["chat_history"][-3:]  # Last 3 exchanges
        ]) if state["chat_history"] else "No previous conversation."
        
        # 7. Generate answer using LLM
        prompt = QA_PROMPT_TEMPLATE.format(
            context=context,
            chat_history=chat_history_str,
            question=user_query
        )
        
        answer = llm.invoke(prompt).content
        
        # 8. Update chat history
        state["chat_history"].append({
            "question": user_query,
            "answer": answer
        })
        
        return {
            "answer": answer,
            "sources": list(set([doc.metadata.get("source", "Unknown") for doc in docs])),
            "official_links": official_links,
            "routing": {
                "classification": route_res.classification,
                "scheme": scheme_slug,
                "inherited": route_res.classification == "scheme_specific" and (not route_res.scheme or str(route_res.scheme).lower() in ["none", "null", "undefined"])
            }
        }

if __name__ == "__main__":
    rag = Phase4RAG()
    res1 = rag.query("What is the expense ratio of HDFC Large Cap Fund?", "user_1")
    print(f"\nQ1: {res1['answer']}\n")
    
    res2 = rag.query("What about its exit load?", "user_1")
    print(f"Q2 (Follow-up): {res2['answer']}")
