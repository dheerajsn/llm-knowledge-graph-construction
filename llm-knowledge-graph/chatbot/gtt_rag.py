import streamlit as st
from gtt_rag_engine import GTTRAGEngine
import os
from pathlib import Path

# Page config
st.set_page_config(
    page_title="GTT Documentation Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    border-left: 4px solid #1f77b4;
    background-color: #f0f7ff;
}
.source-info {
    font-size: 0.8rem;
    color: #666;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)

def initialize_rag_engine():
    """Initialize the RAG engine"""
    if 'rag_engine' not in st.session_state:
        openai_key = st.session_state.get('openai_api_key')
        if openai_key:
            st.session_state.rag_engine = GTTRAGEngine(openai_key)
            if st.session_state.rag_engine.load_index():
                st.session_state.index_loaded = True
            else:
                st.session_state.index_loaded = False
        else:
            st.session_state.index_loaded = False

def main():
    st.markdown('<h1 class="main-header">📚 GTT Documentation Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OpenAI API Key
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key"
        )
        
        if openai_key:
            st.session_state.openai_api_key = openai_key
            initialize_rag_engine()
        
        # Index status
        st.header("📊 System Status")
        
        index_dir = "./gtt_index_storage"
        if Path(index_dir).exists():
            st.success("✅ Index found")
            if st.session_state.get('index_loaded', False):
                st.success("✅ RAG engine loaded")
            else:
                st.warning("⚠️ RAG engine not loaded")
        else:
            st.error("❌ Index not found")
            st.info("Please run create_index.py first")
        
        # Instructions
        st.header("📖 Instructions")
        st.markdown("""
        1. Enter your OpenAI API key
        2. Make sure the index is created
        3. Ask questions about GTT documentation
        4. View relevant source chunks
        """)
    
    # Main chat interface
    if not st.session_state.get('index_loaded', False):
        st.warning("⚠️ Please configure your OpenAI API key and ensure the index is created.")
        st.info("Run `python create_index.py` to create the index first.")
        return
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! I'm your GTT Documentation Assistant. Ask me anything about the GTT application!"
        })
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about GTT documentation..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documentation..."):
                response = st.session_state.rag_engine.query(prompt)
                st.markdown(response)
                
                # Add assistant response
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Show relevant chunks (optional)
                with st.expander("📄 View Source Information"):
                    chunks = st.session_state.rag_engine.get_relevant_chunks(prompt)
                    
                    for i, chunk in enumerate(chunks, 1):
                        st.markdown(f"**Source {i}** (Score: {chunk['score']:.3f})")
                        st.markdown(f"Page {chunk['metadata'].get('page', 'Unknown')}")
                        st.text_area(
                            f"Content {i}",
                            chunk['content'],
                            height=100,
                            key=f"chunk_{i}_{len(st.session_state.messages)}"
                        )
                        st.markdown("---")
    
    # Quick questions
    st.markdown("### 🚀 Quick Questions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔧 How to configure GTT?"):
            st.session_state.messages.append({
                "role": "user", 
                "content": "How do I configure GTT settings?"
            })
            st.rerun()
    
    with col2:
        if st.button("⚡ What are GTT features?"):
            st.session_state.messages.append({
                "role": "user", 
                "content": "What are the main features of GTT application?"
            })
            st.rerun()
    
    with col3:
        if st.button("🔍 Troubleshooting help"):
            st.session_state.messages.append({
                "role": "user", 
                "content": "How to troubleshoot common GTT issues?"
            })
            st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()

#pip install llama-index streamlit pymupdf pillow pytesseract openai