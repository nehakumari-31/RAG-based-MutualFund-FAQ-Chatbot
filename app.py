import streamlit as st
import os
from backend.engine.rag_chain import Phase4RAG
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="HDFC Mutual Fund Assistant",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Groww-inspired styling (matching HTML frontend)
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Root variables matching frontend */
    :root {
        --groww-green: #00D09C;
        --groww-green-dark: #00b085;
        --text-main: #222222;
        --text-secondary: #7C7E8C;
        --bg-light: #FFFFFF;
        --bg-shade: #F6F6F7;
        --border-color: #E5E7EB;
    }
    
    /* Main app background */
    .stApp {
        background-color: #F6F6F7;
        font-family: 'Inter', 'Roboto', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        color: #222222;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 8px;
        text-align: center;
    }
    
    .compliance-tag {
        display: inline-block;
        background: rgba(0, 208, 156, 0.1);
        color: #00b085;
        font-size: 12px;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .disclaimer {
        background: #EEF2FF;
        color: #4F46E5;
        font-size: 14px;
        font-weight: 600;
        padding: 8px 20px;
        border-radius: 100px;
        border: 1px solid #E0E7FF;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: white;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* User message (green background) */
    [data-testid="stChatMessageContent"]:has(+ div [data-testid="stMarkdownContainer"]) {
        background-color: #00D09C;
        color: white;
        border-radius: 12px;
        padding: 16px;
    }
    
    /* Assistant message (light gray background) */
    .stChatMessage[data-testid="chat-message-assistant"] {
        background-color: #F6F6F7;
        border-bottom-left-radius: 2px;
    }
    
    /* User message styling */
    .stChatMessage[data-testid="chat-message-user"] {
        background-color: #00D09C;
        color: white;
        border-bottom-right-radius: 2px;
    }
    
    .stChatMessage[data-testid="chat-message-user"] p {
        color: white !important;
        font-weight: 500;
    }
    
    /* Link buttons */
    .link-button {
        display: inline-block;
        color: #00b085;
        text-decoration: none;
        font-weight: 600;
        font-size: 13px;
        border: 1px solid #00D09C;
        padding: 6px 12px;
        border-radius: 6px;
        margin-right: 8px;
        margin-top: 8px;
        transition: all 0.2s;
    }
    
    .link-button:hover {
        background-color: #00D09C;
        color: white;
    }
    
    /* Example buttons */
    .stButton button {
        background-color: white;
        border: 1px solid #E5E7EB;
        color: #222222;
        font-weight: 500;
        border-radius: 100px;
        padding: 8px 16px;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        border-color: #00D09C;
        color: #00D09C;
        background-color: rgba(0, 208, 156, 0.05);
    }
    
    /* Chat input */
    .stChatInputContainer {
        background-color: #F6F6F7;
        border-radius: 12px;
        border: 1px solid transparent;
    }
    
    .stChatInputContainer:focus-within {
        border-color: #00D09C;
        background-color: white;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: white;
    }
    
    /* Footer text */
    .footer-text {
        font-size: 11px;
        color: #7C7E8C;
        text-align: center;
        margin-top: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize RAG system
@st.cache_resource
def initialize_rag():
    """Initialize RAG system with warmup for optimal first query performance."""
    with st.spinner("🚀 Loading AI models and initializing chatbot..."):
        rag = Phase4RAG()
        rag.warmup()  # Pre-load all expensive components
    return rag

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

# Initialize RAG
rag = initialize_rag()

# Header
st.markdown('<div class="main-header">HDFC Mutual Fund Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="compliance-tag">Official Source Verified</div>', unsafe_allow_html=True)

# Welcome message and example questions
if len(st.session_state.messages) == 0:
    st.markdown("### Welcome! How can I help you with HDFC Mutual Fund facts today?")
    st.markdown('<div class="disclaimer">Facts-only. No investment advice.</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Expense Ratio", use_container_width=True):
            st.session_state.example_query = "What is the expense ratio of HDFC Flexi Cap Fund?"
            st.rerun()
    
    with col2:
        if st.button("Tax Statement", use_container_width=True):
            st.session_state.example_query = "How to download capital gains statement?"
            st.rerun()
    
    with col3:
        if st.button("Exit Load", use_container_width=True):
            st.session_state.example_query = "What is the exit load for HDFC Large Cap?"
            st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display official links if available
        if "links" in message and message["links"]:
            links_html = ""
            for link in message["links"]:
                links_html += f'<a href="{link["url"]}" target="_blank" class="link-button">{link["label"]}</a>'
            st.markdown(links_html, unsafe_allow_html=True)

# Handle example query
if "example_query" in st.session_state:
    user_input = st.session_state.example_query
    del st.session_state.example_query
else:
    user_input = st.chat_input("Ask about HDFC Mutual Funds...")

# Process user input
if user_input:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get response from RAG
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = rag.query(user_input, st.session_state.session_id)
                
                # Display answer
                st.markdown(response["answer"])
                
                # Display official links
                if response.get("official_links"):
                    links_html = ""
                    for link in response["official_links"]:
                        links_html += f'<a href="{link["url"]}" target="_blank" class="link-button">{link["label"]}</a>'
                    st.markdown(links_html, unsafe_allow_html=True)
                
                # Add assistant message to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "links": response.get("official_links", [])
                })
                
            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Sidebar with info
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    This chatbot answers questions about:
    - HDFC Large Cap Fund
    - HDFC Flexi Cap Fund
    - HDFC ELSS Tax Saver
    
    **Sources:** Official HDFC documents (KIMs, SIDs, Notices)
    
    **Session ID:** `{}`
    """.format(st.session_state.session_id[:8] + "..."))
    
    if st.button("🔄 Clear Chat"):
        st.session_state.messages = []
        st.rerun()
