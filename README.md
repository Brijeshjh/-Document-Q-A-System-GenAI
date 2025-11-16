# RAG Pipeline - Document Q&A System

A Retrieval Augmented Generation (RAG) pipeline implementation with an interactive Streamlit web interface for document question-answering. This project enables users to upload PDF documents, process them into vector embeddings, and query them using semantic search.

## 🚀 Features

- **PDF Document Processing**: Upload and process PDF documents with automatic text extraction
- **Intelligent Text Chunking**: Configurable chunk size and overlap for optimal document segmentation
- **Vector Embeddings**: Uses Ollama embeddings (supports multiple models: llama3.2, llama3, mistral, nomic-embed)
- **Semantic Search**: Query documents using natural language with similarity-based retrieval
- **Interactive UI**: User-friendly Streamlit interface with real-time processing feedback
- **Document Analytics**: View document statistics, chunk information, and sample content

## 📋 Prerequisites

1. **Python 3.8+** installed on your system
2. **Ollama** installed and running ([Download here](https://ollama.ai))
3. **Ollama Embedding Model** pulled:
   ```bash
   ollama pull llama3.2
   # or
   ollama pull nomic-embed
   ```

## 🛠️ Installation

1. **Clone or download this repository**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Ollama is running**:
   ```bash
   ollama list
   ```

## 📖 Usage

### Running the Application

**Option 1: Using the run script**
```bash
python run_app.py
```

**Option 2: Direct Streamlit command**
```bash
streamlit run streamlit_rag_app.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the Application

1. **Upload & Process**:
   - Navigate to the "Upload & Process" tab
   - Upload a PDF file
   - Adjust chunk size and overlap settings in the sidebar (optional)
   - Click "Process Document" and wait for processing to complete

2. **Query Documents**:
   - Go to the "Query" tab
   - Enter your question in natural language
   - View retrieved results with relevant document chunks

3. **View Document Info**:
   - Check the "Document Info" tab for statistics
   - View sample chunks and document metrics

## ⚙️ Configuration

The sidebar allows you to configure:
- **Chunk Size**: 500-2000 characters (default: 1000)
- **Chunk Overlap**: 0-500 characters (default: 200)
- **Number of Results**: 1-10 results per query (default: 3)
- **Embedding Model**: Choose from available Ollama models

## 📁 Project Structure

```
PROJECTS/
├── streamlit_rag_app.py    # Main Streamlit application
├── run_app.py              # Helper script to run the app
├── ragpipeline.ipynb       # Jupyter notebook with pipeline development
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔧 Technical Stack

- **Streamlit**: Web application framework
- **LangChain**: Document processing and vector store management
- **ChromaDB**: Vector database for embeddings storage
- **Ollama**: Local LLM and embedding models
- **PyPDF**: PDF document parsing

## 📝 Notes

- The vector store is stored in session state (in-memory), so it resets when you refresh the page
- Large PDFs may take time to process depending on document size and system resources
- Ensure Ollama is running before processing documents
- The application supports multiple embedding models - choose based on your accuracy/performance needs

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is open source and available for educational and research purposes.

