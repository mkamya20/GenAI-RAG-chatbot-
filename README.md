# Gravity Spy - Document Search & Chatbot



## Project Structure

```
Parser/
├── data/
│   └── pdfs/              # Place your PDF files here
├── outputs/
│   └── chunks.jsonl      # Processed chunks (backup)
├── chroma_db/            # ChromaDB vector database
├── frontend.html         # Web interface
├── fastapi_example.py    # FastAPI backend server
├── ingest_pdfs.py        # PDF processing script
├── retrieve_example.py   # Example retrieval script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- PDF files to process (optional, for initial setup)

## Installation

### 1. Clone or Navigate to the Project Directory

```bash
cd "C:...\Parser"
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The `requirements.txt` file includes basic dependencies. You may also need to install:

```bash
pip install fastapi uvicorn openai chromadb python-dotenv
```

### 3. Set Up Environment Variables (Optional - for Azure OpenAI)

If you want to use the AI chatbot feature, create a `.env` file in the project root:

```env
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
```

**Note**: The chatbot will work without Azure OpenAI, but will only return raw document chunks instead of generated answers.

## Usage

### Step 1: Process PDF Documents

Before you can search or chat, you need to process your PDF files:

1. **Place PDF files** in the `data/pdfs/` directory

2. **Run the ingestion script**:

```bash
python ingest_pdfs.py ingest --input_dir data/pdfs --output_path outputs/chunks.jsonl
```

Or use default settings:

```bash
python ingest_pdfs.py ingest
```

This will:
- Extract text from all PDFs in `data/pdfs/`
- Split text into chunks (default: 1000 characters with 200 character overlap)
- Generate embeddings using sentence-transformers
- Store chunks in ChromaDB vector database
- Save a backup JSONL file in `outputs/chunks.jsonl`

**Customization options**:
```bash
python ingest_pdfs.py ingest \
    --input_dir data/pdfs \
    --output_path outputs/chunks.jsonl \
    --chunk_size 1000 \
    --chunk_overlap 200 \
    --embedding_model all-MiniLM-L6-v2
```

### Step 2: Start the Backend Server

Start the FastAPI backend server:

```bash
python fastapi_example.py
```

Or using uvicorn directly:

```bash
uvicorn fastapi_example:app --host 127.0.0.1 --port 8000 --reload
```

The server will start on `http://127.0.0.1:8000`

**API Documentation**: Once the server is running, visit:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### Step 3: Open the Frontend

1. **Open `frontend.html`** in your web browser:
   - Double-click the file, or
   - Right-click → Open with → Your preferred browser

2. **Verify Connection**: The frontend will automatically check if the backend is running. You should see a green status indicator showing "Connected" with the number of chunks available.

## Using the Application

### Chat Interface

1. Click on the **💬 Chat** tab
2. Type your question in the input field
3. Press Enter or click "Send"
4. The system will:
   - Search for relevant document chunks
   - Generate an answer using Azure OpenAI (if configured)
   - Display the answer with source citations

**Example queries**:
- "What is the purpose of the LIGO detector?"
- "Explain the calibration methods used"
- "What are the key findings in the white paper?"

### Search Interface

1. Click on the **🔍 Search** tab
2. Enter your search query
3. Press Enter or click "Search"
4. View relevant document chunks with:
   - Source filename
   - Page numbers
   - Matching text excerpts

**Example searches**:
- "laser interferometer design"
- "detector sensitivity"
- "gravitational wave detection"

## API Endpoints

The FastAPI backend provides the following endpoints:

### Health Check
```
GET /api/health
```
Returns server status and database information.

### Chat (RAG-based Q&A)
```
POST /api/chat
Body: {
    "query": "Your question here",
    "top_k": 5,
    "use_rag": true
}
```

### Semantic Search
```
POST /api/search
Body: {
    "query": "Search query",
    "top_k": 5,
    "filter_filename": "optional_filename.pdf"
}
```

### Document Management
```
GET /api/documents              # List all documents
GET /api/documents/{filename}   # Get document details
POST /api/upload                # Upload and process a PDF
DELETE /api/documents/{filename} # Delete a document
```

## Command-Line Usage

### Search Documents (CLI)

You can also search from the command line:

```bash
python ingest_pdfs.py search --query "your search query" --top_k 5
```

### Example Queries

See `EXAMPLE_QUERIES.md` for a comprehensive list of example queries organized by document type.

## Troubleshooting

### Backend Not Starting

- **Check Python version**: Ensure you have Python 3.8+
- **Install dependencies**: Run `pip install -r requirements.txt` again
- **Check port**: Ensure port 8000 is not in use by another application

### Frontend Can't Connect

- **Verify backend is running**: Check `http://127.0.0.1:8000/api/health` in your browser
- **Check CORS settings**: The backend allows all origins by default
- **Check browser console**: Open Developer Tools (F12) to see error messages

### No Search Results

- **Process documents first**: Run `python ingest_pdfs.py ingest`
- **Check ChromaDB**: Ensure `chroma_db/` directory exists and contains data
- **Verify PDFs**: Make sure PDFs in `data/pdfs/` contain extractable text

### Azure OpenAI Errors

- **Check .env file**: Ensure all required environment variables are set
- **Verify API key**: Test your Azure OpenAI credentials
- **Check deployment name**: Ensure the deployment name matches your Azure resource
- **Note**: The system works without Azure OpenAI, but only returns raw chunks

## Advanced Configuration

### Changing Embedding Model

Edit the `embedding_model` parameter in `ingest_pdfs.py` or pass it as an argument:

```bash
python ingest_pdfs.py ingest --embedding_model all-mpnet-base-v2
```

Popular models:
- `all-MiniLM-L6-v2` (default, fast, 384 dimensions)
- `all-mpnet-base-v2` (slower, more accurate, 768 dimensions)
- `all-MiniLM-L12-v2` (balanced)

### Adjusting Chunk Size

Smaller chunks = more precise but may miss context
Larger chunks = more context but less precise

```bash
python ingest_pdfs.py ingest --chunk_size 500 --chunk_overlap 100
```

### Custom API Port

Edit `fastapi_example.py` or use uvicorn:

```bash
uvicorn fastapi_example:app --host 127.0.0.1 --port 8080
```

Then update `API_BASE_URL` in `frontend.html` (line 299) to match.

## Dependencies

### Core Dependencies
- `pypdf` - PDF text extraction
- `langchain-text-splitters` - Text chunking
- `sentence-transformers` - Embedding generation
- `numpy` - Numerical operations
- `chromadb` - Vector database
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `openai` - Azure OpenAI integration
- `python-dotenv` - Environment variable management



**Happy Searching! 🔭**

