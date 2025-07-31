# RAG System for PDF Processing and Query Answering

A Retrieval-Augmented Generation (RAG) system that processes PDF documents from URLs and answers queries using FAISS vector search combined with ChatGroq LLM.

## 🚀 Features

- **PDF Text Extraction**: Extract text content from PDF URLs
- **Vector Search**: FAISS-based semantic search for relevant document chunks
- **LLM Integration**: ChatGroq integration for intelligent query answering
- **Optimized Chunking**: Smart text chunking with sentence boundary detection
- **Concise Responses**: Formatted, policy-expert style answers

## 📋 Requirements

### Core Dependencies
- `faiss-cpu>=1.7.4` - Vector similarity search
- `PyPDF2>=3.0.1` - PDF text extraction
- `sentence-transformers>=2.2.2` - Text embeddings
- `langchain-groq>=0.1.0` - LLM integration
- `python-dotenv>=0.21.0` - Environment variable management
- `requests>=2.31.0` - HTTP requests
- `numpy>=1.24.0` - Numerical operations

### Environment Setup
Create a `.env` file in the project root with your GROQ API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Hackrx
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or install individually:
   ```bash
   pip install faiss-cpu PyPDF2 sentence-transformers langchain-groq python-dotenv requests numpy
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file and add your GROQ API key
   echo "GROQ_API_KEY=your_api_key_here" > .env
   ```

## 🎯 Usage

### Basic Usage
```python
python testing-pyt.py
```

### Custom Implementation
```python
from testing_pyt import FAISSRetriever, extract_pdf_from_url, process_pdf_queries

# Extract PDF content
pdf_url = "your_pdf_url_here"
text_content = extract_pdf_from_url(pdf_url)

# Create retriever
retriever = FAISSRetriever(text_content)

# Search for relevant chunks
query = "your question here"
results = retriever.retrieve(query, top_k=3)

# Process queries with LLM
queries_json = json.dumps({
    "queries": ["question 1", "question 2"]
})
answers = process_pdf_queries(pdf_url, queries_json, groq_api_key)
```

## 📁 Project Structure

```
Hackrx/
├── testing-pyt.py          # Main RAG system implementation
├── rag.py                  # Additional RAG functionality
├── RAG_BajajHackerx.ipynb  # Jupyter notebook version
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## 🔧 Key Components

### FAISSRetriever Class
- **Text Chunking**: Intelligent text splitting with overlap
- **Embeddings**: SentenceTransformer-based embeddings
- **Vector Search**: FAISS IndexFlatIP for cosine similarity
- **Retrieval**: Top-k similarity search with scoring

### PDF Processing
- **URL Support**: Direct PDF processing from URLs
- **Error Handling**: Robust error handling for network/parsing issues
- **Text Extraction**: PyPDF2-based text extraction

### LLM Integration
- **ChatGroq**: Optimized for fast, concise responses
- **Prompt Engineering**: Policy-expert style formatting
- **Response Cleaning**: Automatic answer formatting

## 🎯 Example Queries

The system is optimized for policy-related queries:

- **Coverage Questions**: "Does this policy cover maternity expenses?"
- **Time-based Queries**: "What is the waiting period for pre-existing diseases?"
- **Exclusions**: "What are the exclusions in this policy?"
- **Claims**: "What documents are required for claim submission?"

## ⚙️ Configuration

### Embedding Model
Default: `all-MiniLM-L6-v2`
- Fast inference
- Good balance of quality/speed
- 384-dimensional embeddings

### Chunking Parameters
- **Chunk Size**: 1000 characters
- **Overlap**: 100 characters
- **Boundary Detection**: Sentence-aware splitting

### LLM Settings
- **Model**: `llama-3.1-8b-instant`
- **Temperature**: 0.1 (deterministic)
- **Max Tokens**: 150 (concise responses)

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure all packages are installed
   pip install --upgrade faiss-cpu PyPDF2 sentence-transformers langchain-groq
   ```

2. **GROQ API Key**
   ```bash
   # Check environment variable
   echo $GROQ_API_KEY  # Linux/Mac
   echo %GROQ_API_KEY%  # Windows
   ```

3. **PDF Access Issues**
   - Ensure PDF URL is accessible
   - Check internet connectivity
   - Verify URL format and permissions

## 📊 Performance

- **Chunk Processing**: ~126 chunks for typical policy document
- **Embedding Generation**: ~2-3 seconds for full document
- **Query Response**: ~1-2 seconds per query
- **Memory Usage**: ~500MB for embeddings + model

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the HackRx challenge submission.

## 🙏 Acknowledgments

- **FAISS**: Facebook AI Similarity Search
- **SentenceTransformers**: Hugging Face
- **LangChain**: LangChain Community
- **ChatGroq**: Groq API