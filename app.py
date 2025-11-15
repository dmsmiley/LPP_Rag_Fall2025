import streamlit as st
import os
import duckdb
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from sentence_transformers import SentenceTransformer

# Page configuration
st.set_page_config(
    page_title="Chronicles RAG Assistant",
    page_icon="📖",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'model' not in st.session_state:
    st.session_state.model = None

if 'db_path' not in st.session_state:
    st.session_state.db_path = None

if 'top_k' not in st.session_state:
    st.session_state.top_k = 10

if 'last_sources' not in st.session_state:
    st.session_state.last_sources = []

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
        value="Chronicles_VectorDB_DuckDB/chronicles_vector.duckdb",
        help="Path to your DuckDB vector database (relative to where you run streamlit)"
    )
    
    # Store in session state
    st.session_state.db_path = db_path
    
    # Top K results
    top_k = st.slider(
        "Results per Query",
        min_value=3,
        max_value=20,
        value=10,
        help="Number of passages to retrieve per search"
    )
    
    # Store in session state
    st.session_state.top_k = top_k
    
    # Model selection
    model_choice = st.selectbox(
        "LLM Model",
        ["gpt-4o-mini"],#, "gpt-4o", "gpt-3.5-turbo"],
        index=0
    )
    
    # Max iterations
    max_iter = st.slider(
        "Max Tool Calls",
        min_value=1,
        max_value=5,
        value=3,
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

# Get db_path from session state
db_path = st.session_state.get('db_path')

# Check if database exists and can be opened
if db_path and not os.path.exists(db_path):
    st.error(f"❌ Database not found at: `{db_path}`")
    st.info("Please update the database path in the sidebar.")
    st.stop()
elif db_path:
    # Test database connection
    try:
        test_con = duckdb.connect(db_path, read_only=True)
        test_con.close()
        st.success(f"✅ Database connected: `{db_path}`")
    except Exception as e:
        st.error(f"❌ Cannot open database: {str(e)}")
        st.info("Please check the database path in the sidebar.")
        st.stop()

# Check if API key is provided
if not api_key:
    st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue.")
    st.stop()

# Set the API key
os.environ["OPENAI_API_KEY"] = api_key

# Initialize the sentence transformer model (cached)
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

try:
    st.session_state.model = load_sentence_transformer()
except Exception as e:
    st.error(f"Error loading sentence transformer: {str(e)}")
    st.stop()

# Define the tool
@tool("Query Chronicles Database")
def query_chronicles_db(query: str) -> str:
    """Search the Chronicles database containing Bible Project video content and biblical text.
    
    Args:
        query: Search query about Chronicles
        
    Returns:
        Relevant passages from the database
    """
    try:
        # Get db_path from session state
        db_path = st.session_state.get('db_path')
        top_k = st.session_state.get('top_k', 10)
        
        if not db_path:
            return "Error: Database path not configured"
        
        con = duckdb.connect(db_path, read_only=True)
        
        # Use the cached model
        query_embedding = st.session_state.model.encode(query).tolist()
        
        results = con.execute("""
            SELECT text, array_cosine_similarity(embedding, ?::FLOAT[384]) as similarity
            FROM chr_rag_documents
            ORDER BY similarity DESC
            LIMIT ?
        """, [query_embedding, top_k]).fetchall()
        
        con.close()
        
        if results:
            # Store sources in session state for display
            sources = [{"text": row[0], "similarity": float(row[1])} for row in results]
            
            # Append to last_sources (accumulate all queries in this conversation turn)
            if 'last_sources' not in st.session_state:
                st.session_state.last_sources = []
            st.session_state.last_sources.extend(sources)
            
            passages = [row[0] for row in results]
            return "\n\n---\n\n".join([f"Passage {i+1}:\n{doc}" for i, doc in enumerate(passages)])
        else:
            return "No relevant passages found."
            
    except Exception as e:
        return f"Error querying database: {str(e)}\nAttempted path: {db_path}"

# Create the agent and crew
def create_agent(_llm_model, _max_iter):
    """Create the Chronicles agent (NOT cached - recreated each time)"""
    llm = LLM(model=_llm_model)
    
    agent = Agent(
        role='Chronicles Content Assistant',
        goal='Answer questions about Chronicles using the database',
        backstory='You are an expert who has access to a database with content about the book of Chronicles.',
        tools=[query_chronicles_db],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=_max_iter
    )
    
    return agent

def ask_chronicles_question(question: str, agent):
    """Query the Chronicles database with a question"""
    task = Task(
        description=question,
        agent=agent,
        expected_output='A comprehensive answer based on the database content'
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
        max_rpm=20
    )
    
    try:
        result = crew.kickoff()
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

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
                # Clear previous sources
                st.session_state.last_sources = []
                
                # Create agent
                agent = create_agent(model_choice, max_iter)
                
                # Get answer
                response = ask_chronicles_question(prompt, agent)
                
                st.markdown(response)
                
                # Display sources immediately
                if st.session_state.last_sources:
                    with st.expander(f"📚 View Sources ({len(st.session_state.last_sources)} passages retrieved)"):
                        for i, source in enumerate(st.session_state.last_sources, 1):
                            st.markdown(f"**Source {i}** (Similarity: {source['similarity']:.3f})")
                            st.text_area(
                                f"Passage {i}",
                                source["text"],
                                height=150,
                                key=f"source_new_{i}",
                                label_visibility="collapsed"
                            )
                            if i < len(st.session_state.last_sources):
                                st.divider()
                
                # Add to history with sources
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": st.session_state.last_sources.copy()
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