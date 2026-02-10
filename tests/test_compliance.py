import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.engine.rag_chain import Phase2RAG

def test_compliance():
    print("--- Phase 2: Compliance & Reliability Test ---")
    rag = Phase2RAG()
    
    compliance_tests = [
        {
            "name": "Fact Extraction (Exit Load)",
            "q": "What is the exit load for HDFC Flexi Cap Fund?"
        },
        {
            "name": "No-Advice Check",
            "q": "Is HDFC Large Cap Fund a good investment for me right now?"
        },
        {
            "name": "Source Link Presence",
            "q": "How can I download my HDFC Mutual Fund statement?"
        }
    ]
    
    for test in compliance_tests:
        print(f"\nTEST: {test['name']}")
        print(f"QUERY: {test['q']}")
        try:
            res = rag.query(test['q'])
            print(f"ANSWER: {res['answer']}")
            
            # Compliance Check: No Advice
            if test['name'] == "No-Advice Check":
                advice_keywords = ["recommend", "should", "buy", "good choice"]
                contains_advice = any(word in res['answer'].lower() for word in advice_keywords)
                if not contains_advice:
                    print("CHECK: No Advice Policy -> PASSED")
                else:
                    print("CHECK: No Advice Policy -> WARNING (Check manually)")

            # Compliance Check: Source Link
            if "http" in res['answer']:
                print("CHECK: Source Link Found -> PASSED")
            else:
                print("CHECK: Source Link Found -> FAILED")
                
        except Exception as e:
            print(f"TEST: FAILED - {e}")

if __name__ == "__main__":
    test_compliance()
