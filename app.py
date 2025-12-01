import streamlit as st
import os
from backend.database import ChroniclesDatabase
from backend.agent import ChroniclesAgent
import config

# Page configuration
st.set_page_config(
    page_title="Chronicles RAG Assistant",
    page_icon="📖",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'db_path' not in st.session_state:
    st.session_state.db_path = config.DEFAULT_DB_PATH

if 'top_k' not in st.session_state:
    st.session_state.top_k = config.DEFAULT_TOP_K

if 'database' not in st.session_state:
    st.session_state.database = None

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key"
    )
    
    # Database path input
    db_path = st.text_input(
        "Database Path",
        value=config.DEFAULT_DB_PATH,
        help="Path to your DuckDB vector database (relative to where you run streamlit)"
    )
    
    # Store in session state
    st.session_state.db_path = db_path
    
    # Top K results
    top_k = st.slider(
        "Results per Query",
        min_value=3,
        max_value=20,
        value=config.DEFAULT_TOP_K,
        help="Number of passages to retrieve per search"
    )
    
    # Store in session state
    st.session_state.top_k = top_k
    
    # Model selection
    model_choice = st.selectbox(
        "LLM Model",
        config.AVAILABLE_MODELS,
        index=0
    )
    
    # Max iterations
    max_iter = st.slider(
        "Max Tool Calls",
        min_value=1,
        max_value=5,
        value=config.DEFAULT_MAX_ITER,
        help="Maximum number of database queries per question"
    )
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    st.markdown("""
    ### About
    This RAG assistant answers questions about the book of Chronicles using:
    - Bible Project video transcripts
    - Biblical text from Chronicles
    - Semantic search with embeddings
    """)

# Main app
st.title("📖 Chronicles RAG Assistant")
st.markdown("Ask questions about the book of Chronicles and get AI-powered answers based on curated content.")

# Initialize Database
if not st.session_state.database or st.session_state.database.db_path != db_path:
    st.session_state.database = ChroniclesDatabase(db_path)

# Check database connection
if not st.session_state.database.test_connection():
    st.error(f"❌ Database not found at: `{db_path}`")
    st.info("Please update the database path in the sidebar.")
    if not os.path.exists(db_path):
        st.stop()
else:
    st.success(f"✅ Database connected: `{db_path}`")

# Check if API key is provided
if not api_key:
    st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue.")
    st.stop()

# Set the API key
os.environ["OPENAI_API_KEY"] = api_key

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources for assistant messages if available
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"📚 View Sources ({len(message['sources'])} passages retrieved)"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}** (Similarity: {source['similarity']:.3f})")
                    st.text_area(
                        f"Passage {i}",
                        source["text"],
                        height=150,
                        key=f"source_{id(message)}_{i}",
                        label_visibility="collapsed"
                    )
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask a question about Chronicles..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching database and generating answer..."):
            try:
                # Initialize Agent
                agent = ChroniclesAgent(
                    db=st.session_state.database,
                    model_name=model_choice,
                    max_iter=max_iter
                )
                
                # Get answer
                result = agent.ask(prompt)
                response = result["answer"]
                sources = result["sources"]
                
                st.markdown(response)
                
                # Display sources immediately
                if sources:
                    with st.expander(f"📚 View Sources ({len(sources)} passages retrieved)"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Source {i}** (Similarity: {source['similarity']:.3f})")
                            st.text_area(
                                f"Passage {i}",
                                source["text"],
                                height=150,
                                key=f"source_new_{i}",
                                label_visibility="collapsed"
                            )
                            if i < len(sources):
                                st.divider()
                
                # Add to history with sources
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources
                })
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })

# Example questions in an expander
with st.expander("💡 Example Questions"):
    examples = [
        "Give me an outline of the Bible Project's video of the book of Chronicles.",
        "What are the main themes in Chronicles?",
        "How does Chronicles portray King David?",
        "What is the structure of 1 Chronicles?",
        "How does Chronicles differ from Kings?"
    ]
    
    for example in examples:
        if st.button(example, key=example):
            # Simulate entering the question
            st.session_state.messages.append({"role": "user", "content": example})
            st.rerun()