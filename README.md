# HDFC Mutual Fund FAQ Chatbot

A production-ready RAG (Retrieval-Augmented Generation) chatbot for answering questions about HDFC Mutual Funds using official documents.

## Features

- **Smart Query Routing**: Automatically classifies queries as scheme-specific or general
- **Scheme Detection**: Identifies HDFC Large Cap, Flexi Cap, and ELSS funds
- **Conversational Memory**: Remembers context across follow-up questions
- **Official Sources Only**: Uses verified HDFC documents (KIMs, SIDs, Notices)
- **Compliance-First**: Strict no-advice policy with factual responses only
- **Clean UI**: Professional Groww-inspired interface

## Project Structure

```
RAG-based-MutualFund-FAQ-Chatbot/
├── backend/
│   ├── api/          # FastAPI backend
│   ├── engine/       # RAG chain, router, and retrieval logic
│   └── data/         # Document ingestion scripts
├── frontend/         # HTML/CSS/JS chatbot UI
├── tests/            # Performance and verification tests
├── vector_db/        # Chroma vector database
├── Document Sources/ # Official HDFC PDFs (29 documents)
├── requirements.txt  # Python dependencies
└── .env             # Environment variables (GROQ_API_KEY)
```

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**
   Create a `.env` file:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Ingest Documents** (if vector_db doesn't exist)
   ```bash
   cd backend/data
   python3 ingest.py
   ```

4. **Run the Streamlit App** (Recommended)
   ```bash
   python3 -m streamlit run app.py
   ```
   
   The app will open automatically at `http://localhost:8501`

**Alternative: Run FastAPI Backend + HTML Frontend**
   ```bash
   # Terminal 1: Start backend
   python3 -m uvicorn backend.api.main:app --reload --port 8000
   
   # Terminal 2: Open frontend/index.html in browser
   ```

## Streamlit Deployment

The chatbot is now available as a Streamlit app (`app.py`) with:
- 💬 Chat interface with conversational memory
- 🎨 Groww-inspired dark theme
- 📊 Example question buttons
- 🔗 Official HDFC document links
- ⚖️ Prominent compliance disclaimer

**Deploy to Streamlit Cloud:**
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add `GROQ_API_KEY` to Secrets
5. Deploy!

## API Endpoints

- `POST /chat` - Send a message and get a response
  ```json
  {
    "message": "What is the expense ratio of HDFC Large Cap?",
    "session_id": "optional-session-id"
  }
  ```

## Key Technologies

- **LLM**: Groq (llama-3.3-70b-versatile)
- **Embeddings**: HuggingFace (sentence-transformers/all-MiniLM-L6-v2)
- **Vector DB**: Chroma
- **Framework**: LangChain
- **Backend**: FastAPI
- **Frontend**: Vanilla HTML/CSS/JS

## Response Guidelines

- Maximum 3 sentences per answer
- Mandatory source attribution line
- No investment advice (buy/sell/hold)
- Official HDFC links provided separately
- Special handling for SIP queries (dual links: fund page + educational blog)

## Document Sources

The chatbot uses 29 official HDFC documents including:
- Key Information Memorandums (KIMs)
- Scheme Information Documents (SIDs)
- Expense change notices
- KYC FAQs
- Investor charters
- Annual riskometer disclosures

## License

MIT
