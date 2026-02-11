import os
import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.prompts import PromptTemplate
from dotenv import load_dotenv

# Add current dir to path for local imports
sys.path.append(os.path.dirname(__file__))
from router import get_router

# Load env from phase2 root
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env")))

# Configuration
DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../vector_db"))

from langchain_classic.memory import ConversationBufferMemory

# Official HDFC Scheme Page Mapping
HDFC_SOURCE_LINKS = {
    "hdfc_large_cap": "https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct",
    "hdfc_flexi_cap": "https://www.hdfcfund.com/explore/mutual-funds/hdfc-flexi-cap-fund/direct",
    "hdfc_elss": "https://www.hdfcfund.com/explore/mutual-funds/hdfc-elss-tax-saver/direct",
    "general": "https://www.hdfcfund.com/investor-services/request-statement",
    "investor_education": "https://www.hdfcfund.com/information/investor-education",
    "sip_education": "https://www.hdfcfund.com/learn/blog/how-does-sip-work"
}

CONDENSE_QUESTION_PROMPT = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

QA_PROMPT_TEMPLATE = """Information from Official HDFC Scheme Documents (SID/KIM/Notices):
--------------------------------------
{context}
--------------------------------------

Additional Official Process Knowledge:
- Capital Gains Statement: To download, visit the HDFC Mutual Fund 'Request Statement' page, scroll to 'Capital Gains Statement', click 'Click here' to log in (using PAN/Folio), and navigate to Reports > Capital Gains Statement. Alternatively, use the CAMS portal for a consolidated version.
- Account Statement: Can be requested via SMS (CAMS H SOA <Folio> <Password> to 56767) or via the portal's 'Request Statement' section.
- KYC Status: Can be checked on KRA websites (CVL, NDML, etc.) using PAN.

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

def get_rag_chain(scheme_filter=None, memory=None):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    
    qa_prompt = PromptTemplate(template=QA_PROMPT_TEMPLATE, input_variables=["context", "question"])
    condense_prompt = PromptTemplate(template=CONDENSE_QUESTION_PROMPT, input_variables=["chat_history", "question"])

    search_kwargs = {"k": 5}
    if scheme_filter:
        search_kwargs["filter"] = {"scheme": scheme_filter}

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs=search_kwargs),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        condense_question_prompt=condense_prompt,
        return_source_documents=True
    )
    return chain

class Phase4RAG:
    """Orchestrator for Phase 4 RAG with Memory, Routing, and Session tracking."""
    def __init__(self):
        self.router = get_router()
        self.sessions = {} # session_id -> {memory, last_scheme}

    def get_session_state(self, session_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "memory": ConversationBufferMemory(memory_key="chat_history", input_key="question", output_key="answer", return_messages=True),
                "last_scheme": "general"
            }
        return self.sessions[session_id]

    def query(self, user_query: str, session_id: str = "default"):
        state = self.get_session_state(session_id)
        
        # 1. Route the query
        route_res = self.router.invoke({"query": user_query})
        
        # 2. Logic for Scheme Detection & Inheritance
        # - If route is general: Use 'general', don't inherit.
        # - If route is scheme_specific: 
        #   - If scheme is found: Use found scheme.
        #   - If scheme is NOT found (follow-up): Inherit from state.
        
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

        # 4. Get chain with memory
        # If it's general, we don't apply a metadata filter so it can search all docs (KYC, Statements, etc.)
        chain = get_rag_chain(
            scheme_filter=scheme_slug if scheme_slug != "general" else None,
            memory=state["memory"]
        )

        response = chain.invoke({"question": user_query})
        
        return {
            "answer": response["answer"],
            "sources": list(set([doc.metadata.get("source") for doc in response.get("source_documents", [])])),
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
