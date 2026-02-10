import os
import sys

# Add phase2 directory to path for absolute imports starting with 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.engine.rag_chain import get_rag_chain, Phase4RAG
from typing import Optional
import uvicorn

app = FastAPI(title="HDFC Mutual Fund FAQ Chatbot - Phase 1")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []
    official_links: list[dict] = []
    routing: Optional[dict] = None

# Global orchestrator instance
phase4_rag = None

@app.on_event("startup")
def startup_event():
    global phase4_rag
    try:
        phase4_rag = Phase4RAG()
        print("Phase 4 RAG Orchestrator Loaded with Memory support.")
    except Exception as e:
        print(f"Error initializing Phase 4 RAG: {e}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global phase4_rag
    if not phase4_rag:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        result = phase4_rag.query(request.message, session_id=request.session_id)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
