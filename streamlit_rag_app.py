import streamlit as st
import tempfile
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
import pprint

# Page configuration
st.set_page_config(
    page_title="RAG Pipeline UI",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'documents' not in st.session_state:
    st.session_state.documents = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# Title and description
st.title("📚 RAG Pipeline - Document Q&A System")
st.markdown("Upload a PDF document, process it, and ask questions!")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    chunk_size = st.slider("Chunk Size", min_value=500, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
    num_results = st.slider("Number of Results", min_value=1, max_value=10, value=3, step=1)
    
    st.divider()
    
    # Model selection
    embedding_model = st.selectbox(
        "Embedding Model",
        options=["llama3.2", "llama3", "mistral", "nomic-embed"],
        index=0
    )
    
    st.divider()
    
    if st.button("🔄 Reset Session", type="secondary"):
        st.session_state.vector_store = None
        st.session_state.documents = None
        st.session_state.processing_complete = False
        st.rerun()

# Main content area
tab1, tab2, tab3 = st.tabs(["📄 Upload & Process", "🔍 Query", "📊 Document Info"])

with tab1:
    st.header("Step 1: Upload PDF Document")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document to process"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        st.info(f"File size: {uploaded_file.size / 1024:.2f} KB")
        
        if st.button("🚀 Process Document", type="primary"):
            with st.spinner("Processing document... This may take a moment."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Step 1: Load PDF
                    st.write("📖 Loading PDF...")
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    st.success(f"✅ Loaded {len(docs)} pages")
                    
                    # Step 2: Split documents
                    st.write("✂️ Splitting documents into chunks...")
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    documents = text_splitter.split_documents(docs)
                    st.success(f"✅ Created {len(documents)} chunks")
                    
                    # Step 3: Create embeddings and vector store
                    st.write("🔢 Creating embeddings and vector store...")
                    embeddings = OllamaEmbeddings(model=embedding_model)
                    
                    # Create vector store
                    vector_store = Chroma.from_documents(
                        documents=documents,
                        embedding=embeddings,
                        persist_directory=None  # In-memory for this demo
                    )
                    
                    # Store in session state
                    st.session_state.vector_store = vector_store
                    st.session_state.documents = documents
                    st.session_state.processing_complete = True
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
                    
                    st.success("✅ Document processing complete! You can now query the document.")
                    
                except Exception as e:
                    st.error(f"❌ Error processing document: {str(e)}")
                    if 'tmp_path' in locals():
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
    
    elif st.session_state.processing_complete:
        st.info("ℹ️ Document already processed. You can query it in the 'Query' tab or upload a new document.")

with tab2:
    st.header("Step 2: Query Your Document")
    
    if not st.session_state.processing_complete:
        st.warning("⚠️ Please upload and process a document first in the 'Upload & Process' tab.")
    else:
        st.success("✅ Ready to query!")
        
        # Query input
        query = st.text_input(
            "Enter your question:",
            placeholder="e.g., Who are the authors of attention is all you need?",
            key="query_input"
        )
        
        search_button = st.button("🔍 Search", type="primary")
        if search_button or (query and len(query.strip()) > 0):
            if query and len(query.strip()) > 0:
                try:
                    if st.session_state.vector_store is None:
                        st.error("❌ Vector store not initialized. Please process a document first.")
                    else:
                        with st.spinner("Searching..."):
                            # Perform similarity search
                            retrieved_results = st.session_state.vector_store.similarity_search(
                                query,
                                k=num_results
                            )
                    
                            if retrieved_results:
                                st.subheader(f"📋 Search Results ({len(retrieved_results)} found)")
                                
                                # Display results
                                for i, doc in enumerate(retrieved_results, 1):
                                    with st.expander(f"Result {i} - Page {doc.metadata.get('page', 'N/A')}", expanded=(i == 1)):
                                        st.markdown("**Content:**")
                                        st.write(doc.page_content)
                                        
                                        st.markdown("**Metadata:**")
                                        # Format metadata nicely
                                        metadata_dict = doc.metadata
                                        for key, value in metadata_dict.items():
                                            if key == 'source':
                                                st.text(f"{key}: {Path(value).name if value else 'N/A'}")
                                            else:
                                                st.text(f"{key}: {value}")
                                        
                                        st.divider()
                            else:
                                st.warning("No results found for your query.")
                            
                            # Show raw results option
                            if retrieved_results:
                                with st.expander("🔧 View Raw Results"):
                                    pp = pprint.PrettyPrinter(indent=2)
                                    for i, doc in enumerate(retrieved_results):
                                        st.text(f"\n--- Result {i+1} ---")
                                        st.text(f"Content: {doc.page_content[:200]}...")
                                        st.json(doc.metadata)
                
                except Exception as e:
                    st.error(f"❌ Error during search: {str(e)}")
            else:
                st.info("Please enter a query to search.")

with tab3:
    st.header("Document Information")
    
    if not st.session_state.processing_complete:
        st.warning("⚠️ No document processed yet. Please upload and process a document first.")
    else:
        st.success("✅ Document information available")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.documents:
                st.metric("Total Chunks", len(st.session_state.documents))
                st.metric("Average Chunk Size", f"{sum(len(doc.page_content) for doc in st.session_state.documents) // len(st.session_state.documents)} chars")
            else:
                st.metric("Total Chunks", 0)
        
        with col2:
            st.metric("Chunk Size Setting", chunk_size)
            st.metric("Chunk Overlap Setting", chunk_overlap)
        
        st.divider()
        
        # Show sample chunks
        st.subheader("📝 Sample Chunks")
        if st.session_state.documents:
            num_samples = st.slider("Number of sample chunks to display", 1, 10, 3)
            
            for i, doc in enumerate(st.session_state.documents[:num_samples], 1):
                with st.expander(f"Chunk {i} - Page {doc.metadata.get('page', 'N/A')}"):
                    st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                    st.caption(f"Length: {len(doc.page_content)} characters")

# Footer
st.divider()
st.caption("💡 Tip: Make sure Ollama is running with the embedding model installed. Run: `ollama pull llama3.2` or your chosen model")

