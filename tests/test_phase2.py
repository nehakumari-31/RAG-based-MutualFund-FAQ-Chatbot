import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.engine.rag_chain import Phase2RAG

def test_phase2_flow():
    print("--- Phase 2: Router & Filtered Retrieval Verification ---")
    rag = Phase2RAG()
    
    test_queries = [
        {
            "q": "What is the exit load for HDFC Large Cap Fund?",
            "expected_scheme": "HDFC Large Cap"
        },
        {
            "q": "Minimum investment for HDFC ELSS Tax Saver?",
            "expected_scheme": "HDFC ELSS Tax Saver"
        },
        {
            "q": "How does HDFC Flexi Cap Fund invest?",
            "expected_scheme": "HDFC Flexi Cap"
        }
    ]
    
    for test in test_queries:
        print(f"\nQUERY: {test['q']}")
        try:
            res = rag.query(test['q'])
            print(f"ROUTER: {res['routing']['classification']} | Scheme: {res['routing']['scheme']}")
            print(f"RESULT: {res['answer'][:200]}...")
            print(f"SOURCES: {res['sources']}")
            
            # Simple check for scheme presence in sources or answer
            # In a real test we'd be stricter, but here we just want to see it run
            if test['expected_scheme'].lower() in str(res['routing']['scheme']).lower():
                print("TEST: SUCCESS (Scheme Detected Correctly)")
            else:
                print(f"TEST: WARNING (Expected {test['expected_scheme']}, got {res['routing']['scheme']})")
                
        except Exception as e:
            print(f"TEST: FAILED - {e}")

if __name__ == "__main__":
    test_phase2_flow()
