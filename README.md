# Bajaj Finserv 6.0 - Document Q&A System

A comprehensive document question-answering system built with FastAPI, vector databases, and LLMs for processing insurance policies, legal documents, and other structured documents.

## 🎯 Project Overview

This project provides an intelligent document processing and question-answering system that can:
- Extract text from various document formats (PDF, DOCX, EML)
- Process and chunk documents for efficient retrieval
- Store document embeddings in vector databases (Qdrant/OpenAI)
- Answer questions using advanced LLMs (Groq, OpenAI)
- Provide RESTful API endpoints for document processing

## 🏗️ Architecture

### Core Components

1. **Document Processing Pipeline**
   - `document_extractor.py` - Handles text extraction from multiple file formats
   - `embedding.py` - Manages document embeddings using sentence transformers
   - `vec_store/` - Vector database management utilities

2. **API Services**
   - `main.py` - Primary FastAPI service with Qdrant vector database
   - `myapp.py` - Alternative FastAPI service with OpenAI vector store
   - `vec_store/new.py` - Enhanced API with file-specific filtering

3. **Testing & Utilities**
   - `vector_search.py` - Client for testing API endpoints
   - `LLMtest.py` - Groq LLM testing utility
   - `req.py` - Simple API request testing

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for better performance)
- API keys for:
  - OpenAI
  - Groq
  - Qdrant
  - Hugging Face (optional)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Bajaj_Finserv_6.0-main
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   # API Keys
   GEMINI=your_gemini_api_key
   GROQ_API_KEY=your_groq_api_key
   chatgpt=your_openai_api_key
   api_key=your_qdrant_api_key
   HF_TOKEN=your_huggingface_token
   
   # Qdrant Configuration
   QDRANT_URL=https://your-qdrant-instance.cloud.qdrant.io:6333
   ```

## 🛠️ Setup Instructions

### 1. Vector Database Setup

#### Option A: Qdrant (Recommended)
```bash
# The system is pre-configured with a Qdrant instance
# Update the URL and API key in your .env file
```

#### Option B: OpenAI Vector Store
```bash
# Run the vector store setup
python vec_store/upload_vec.py
```

### 2. Document Processing

#### Upload Documents to Vector Store
```bash
# For Qdrant-based system
python document_extractor.py

# For OpenAI vector store
python vec_store/upload_vec.py
```

#### Check Uploaded Files
```bash
python vec_store/search.py
```

### 3. Start the API Server

#### Main Service (Qdrant-based)
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

#### Alternative Service (OpenAI-based)
```bash
uvicorn myapp:app --host 0.0.0.0 --port 8000
```

#### Enhanced Service (File-specific filtering)
```bash
uvicorn vec_store.new:app --host 0.0.0.0 --port 8000
```

## 📚 Usage Examples

### 1. API Endpoint Usage

#### Basic Question Answering
```python
import requests

url = "http://localhost:8000/api/v1/hackrx/run"
headers = {
    "Authorization": "Bearer 61b907454794627e4bc5798168e95ec604fd4af04338837ec0e6b3c81bdf06ce",
    "Content-Type": "application/json"
}

payload = {
    "documents": "https://example.com/document.pdf",
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?",
        "Does this policy cover maternity expenses?"
    ]
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

#### Using the Test Script
```bash
python vector_search.py
```

### 2. Document Processing

#### Extract Text from Documents
```bash
python document_extractor.py
```

#### Test LLM Integration
```bash
python LLMtest.py
```

### 3. Vector Store Management

#### Upload PDFs to Vector Store
```bash
python vec_store/upload_vec.py
```

#### List Files in Vector Store
```bash
python vec_store/search.py
```

#### Clean Up Vector Store
```bash
python vec_store/del_vect.py
```

## 🔧 Configuration

### Model Configuration

The system uses several AI models:
- **Embedding Model**: `BAAI/bge-large-en-v1.5` (Sentence Transformers)
- **LLM Models**: 
  - `llama-3.3-70b-versatile` (Groq)
  - `llama-3.1-8b-instant` (Groq)
  - OpenAI GPT models

### Chunking Configuration

Documents are chunked using:
- **Chunk Size**: 500 tokens
- **Chunk Overlap**: 0
- **Separators**: `["\n\n", "\n", ".", " ", ""]`

### API Configuration

- **Base Path**: `/api/v1`
- **Authentication**: Bearer token required
- **Rate Limiting**: Not implemented (can be added)

## 📁 Project Structure

```
Bajaj_Finserv_6.0-main/
├── main.py                 # Primary FastAPI service (Qdrant)
├── myapp.py               # Alternative FastAPI service (OpenAI)
├── document_extractor.py  # Document processing pipeline
├── embedding.py           # Embedding utilities
├── vector_search.py       # API testing client
├── LLMtest.py            # LLM testing utility
├── req.py                # Simple API testing
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── vec_store/            # Vector database utilities
│   ├── upload_vec.py     # Upload documents to vector store
│   ├── search.py         # List files in vector store
│   ├── new.py           # Enhanced API with filtering
│   └── del_vect.py      # Clean up vector store
├── qdrant/              # API request logs
└── *.pdf, *.txt         # Sample documents
```

## 🔍 API Endpoints

### POST `/api/v1/hackrx/run`

Process documents and answer questions.

**Request Body:**
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": ["Question 1", "Question 2", "Question 3"]
}
```

**Response:**
```json
{
  "answers": ["Answer 1", "Answer 2", "Answer 3"]
}
```

**Headers Required:**
```
Authorization: 
Content-Type: application/json
```

## 🧪 Testing

### 1. Test API Endpoint
```bash
python vector_search.py
```

### 2. Test LLM Integration
```bash
python LLMtest.py
```

### 3. Test Document Processing
```bash
python document_extractor.py
```

### 4. Test Vector Store Operations
```bash
# Upload documents
python vec_store/upload_vec.py

# List files
python vec_store/search.py

# Clean up (optional)
python vec_store/del_vect.py
```

## 📊 Performance Optimization

### GPU Acceleration
- The system automatically uses CUDA if available
- Embedding generation is significantly faster with GPU

### Batch Processing
- Documents are processed in batches for efficiency
- Vector operations are optimized for large datasets

### Caching
- Model loading is done once at startup
- Embeddings are cached in vector database

## 🔒 Security

### Authentication
- Bearer token authentication required for all API endpoints


### Data Privacy
- Documents are processed locally
- No sensitive data is stored permanently
- Vector embeddings are stored securely in cloud databases

## 🐛 Troubleshooting

### Common Issues

1. **CUDA not available**
   - Install CUDA toolkit
   - Or use CPU-only mode (slower but functional)

2. **API key errors**
   - Verify all API keys in `.env` file
   - Check API key permissions

3. **Memory issues**
   - Reduce batch size in embedding generation
   - Use smaller models if needed

4. **Vector store connection issues**
   - Verify Qdrant/OpenAI credentials
   - Check network connectivity

### Logs
- API requests are logged in `qdrant/` directory
- Check console output for detailed error messages

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is proprietary and confidential. All rights reserved.

## 📞 Support

For technical support or questions:
- Check the troubleshooting section
- Review API logs in `qdrant/` directory
- Contact the development team

## 🔄 Version History

- **v6.0** - Current version with enhanced document processing
- **v5.0** - Added OpenAI vector store support
- **v4.0** - Implemented Qdrant vector database
- **v3.0** - Added Groq LLM integration
- **v2.0** - Basic document processing
- **v1.0** - Initial release

---

**Note**: This system is designed for processing insurance policies, legal documents, and other structured documents. Ensure compliance with data privacy regulations when processing sensitive documents. 
