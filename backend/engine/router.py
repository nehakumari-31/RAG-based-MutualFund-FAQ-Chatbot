import os
from typing import Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load env
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

class RouteResponse(BaseModel):
    """Schema for query classification and scheme detection."""
    classification: str = Field(description="Either 'scheme_specific' or 'general'")
    scheme: Optional[str] = Field(description="The HDFC fund scheme slug (hdfc_large_cap, hdfc_flexi_cap, hdfc_elss) or None")
    reasoning: str = Field(description="Brief reasoning for this classification")

def get_router():
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    
    # Using structured output capability of Llama 3 via LangChain
    structured_llm = llm.with_structured_output(RouteResponse)
    
    system = """You are an expert query classifier for HDFC Mutual Funds.
    Your task is to determine:
    1. If the query is about a specific HDFC mutual fund scheme (Large Cap, Flexi Cap, or ELSS/Tax Saver).
    2. Which specific scheme it is.
    3. If the query is general financial knowledge not specific to a fund.
    
    Schemes supported:
    - hdfc_large_cap (includes Top 100)
    - hdfc_flexi_cap
    - hdfc_elss (Tax Saver)
    
    Classification:
    - 'scheme_specific': If a fund name is mentioned OR if specific fund attributes are asked for (e.g., 'min SIP', 'exit load', 'expense ratio', 'NAV', 'lock-in period'). Even if no fund name is provided, if they ask for a fund's property, it is 'scheme_specific'.
    - 'general': If it's a general process question (e.g., 'How to download statement?', 'How to do KYC?', 'What is a mutual fund?').
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{query}"),
    ])
    
    return prompt | structured_llm

if __name__ == "__main__":
    router = get_router()
    test_queries = [
        "What is the exit load for HDFC Flexi Cap Fund?",
        "How is HDFC ELSS taxed?",
        "What is the difference between direct and regular plans?",
        "Explain the investment strategy of Top 100 fund."
    ]
    
    print("--- Testing Query Router ---")
    for q in test_queries:
        res = router.invoke({"query": q})
        print(f"\nQuery: {q}")
        print(f"Classification: {res.classification}")
        print(f"Scheme: {res.scheme}")
        print(f"Reasoning: {res.reasoning}")
