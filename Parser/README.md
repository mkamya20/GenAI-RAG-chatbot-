# Gravity Spy - Document Search & Chatbot

A RAG-based chatbot for the Gravity Spy citizen science project. The chatbot answers questions about LIGO and gravitational wave detection using scientific documents and Zooniverse Talk posts as its knowledge base.

## Architecture

The application has two modes of operation:

**Runtime (bot):** A FastAPI server that queries a pre-built ChromaDB index and generates answers via Azure OpenAI. PDF source links redirect to S3. The index is downloaded from S3 at startup if not already present.

**Offline (loader):** A CLI pipeline that ingests PDFs and CSV talk posts, builds the ChromaDB index, and uploads everything to S3.

```
S3 Bucket
├── gravity-spy-wiki-bot/
│   ├── pdfs/                 # PDFs served via redirect
│   └── chroma_db.tar.gz      # Pre-built index archive

Bot (runtime)                  Loader (offline)
├── data/                      ├── data/
│   ├── chroma_db/ (from S3)   │   ├── pdfs/
│   └── init.log               │   ├── csvs/
├── app.py                     │   └── chroma_db/ (built locally)
├── routers/                   ├── cli.py
├── templates/                 └── upload_to_s3.py
└── static/
```

## Project Structure

```
├── routers/
│   ├── __init__.py
│   ├── utils.py              # Shared router utilities
│   ├── health.py             # Health and info endpoints
│   ├── chat.py               # Chat and search endpoints
│   └── pdfs.py               # PDF endpoints (redirects to S3)
├── static/
│   ├── css/
│   │   ├── chat.css          # Full page chat styles
│   │   └── embed.css         # Embeddable widget styles
│   └── js/
│       ├── chat.js           # Full page chat logic
│       └── embed.js          # Embeddable widget logic
├── templates/
│   ├── chat.html             # Full chat interface
│   ├── embed.html            # Embeddable chat widget
│   └── demo.html             # Embed demo page
├── tests/
│   └── ...
├── app.py                    # FastAPI application setup
├── config.py                 # Application configuration
├── logging_config.py         # Logging setup
├── azure_client.py           # Azure OpenAI client wrapper
├── vector_store.py           # ChromaDB operations
├── processor.py              # Shared chunking and embedding utilities
├── pdf_processor.py          # PDF processing library
├── talk_post_processor.py    # Talk post processing library
├── models.py                 # Pydantic request/response models
├── cli.py                    # CLI for index building and management
├── upload_to_s3.py           # Upload PDFs and index to S3
├── docker-entrypoint.sh      # Container entrypoint
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Prerequisites

- Python 3.10+
- Azure OpenAI credentials (for embeddings and chat)
- S3-compatible storage with a public endpoint (for PDFs and index)

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file:

```env
# Azure OpenAI
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT=GravitySpy-gpt-4o-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=GravitySpy-text-embedding-3-small

# S3 (for upload_to_s3.py)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_ENDPOINT_URL=https://s3.gswiki.ischool.syr.edu
AWS_BUCKET_NAME=images
```

## Building the Index (Offline)

Place source data in `data/pdfs/` and `data/csvs/`, then build and upload:

```bash
# Build the ChromaDB index
python cli.py pdfs --input-dir data/pdfs
python cli.py csv --csv-path data/csvs/talk_posts.csv

# Check the index
python cli.py status --verbose

# Upload PDFs and index archive to S3
python upload_to_s3.py
python upload_to_s3.py --dry-run    # Preview without uploading
```

## Running the Bot

### Local Development

```bash
python app.py
```

The bot checks for `data/chroma_db/`. If present, it starts immediately. If not, the entrypoint downloads and extracts the index from S3.

Access the application:
- Demo page: `http://localhost:8000`
- Embeddable widget: `http://localhost:8000/embed`
- Full chat interface: `http://localhost:8000/chat`
- API docs: `http://localhost:8000/docs`

### Docker

```bash
docker compose up --build -d
```

On first startup, the entrypoint downloads `chroma_db.tar.gz` from S3 and extracts it. Subsequent startups use the persisted index.

```bash
# View logs
docker compose logs

# Check initialization
cat ./data/init.log

# Stop
docker compose down
```

## Embedding the Chatbot

The `/embed` endpoint serves a zero-chrome chat widget suitable for iframing:

```html
<iframe
  src="https://your-domain/embed"
  width="400"
  height="600"
  style="border: none; border-radius: 12px;"
  title="LIGO AI Assistant"
></iframe>
```

The demo page at `/` shows this in action.

## Example Queries

The chatbot answers questions about LIGO detector technology, gravitational wave science, and the Gravity Spy citizen science project. Try these in the chat interface or via CLI:

- "What is the purpose of the LIGO detector?"
- "How does power recycling work?"
- "What are the main sources of noise in Advanced LIGO?"
- "Explain the Q-transform and how it's used for glitch classification"
- "What types of glitches does Gravity Spy classify?"
- "How are seismic disturbances mitigated in the detector?"
- "What upgrades are planned for future observing runs?"

Via CLI:

```bash
python cli.py search --query "thermal noise and quantum noise limits" --top-k 5
```

For better results, be specific ("laser interferometer calibration methods" rather than just "calibration") and try different phrasings if the first attempt doesn't surface what you need.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/init-log` | Container initialization log |
| `POST` | `/api/chat` | RAG-based Q&A |
| `POST` | `/api/search` | Direct semantic search |
| `GET` | `/api/pdfs` | List indexed PDFs |
| `GET` | `/api/pdfs/{filename}` | PDF metadata |
| `GET` | `/api/pdfs/{filename}/download` | Redirect to PDF on S3 |

## CLI Reference

| Command | Description |
|---------|-------------|
| `pdfs --input-dir <dir>` | Ingest all PDFs from a directory |
| `pdf --file <path>` | Ingest a single PDF |
| `csv --csv-path <path>` | Ingest talk posts from CSV |
| `add-post` | Add a single talk post |
| `search --query <text>` | Search the index |
| `status` | Show index status |
| `delete --filename <name>` | Delete a document |
| `clear --source-type <type>` | Clear all chunks of a type |

Common flags: `--dry-run`, `--replace`, `--force`, `--batch-size`, `--delay`

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_DB_PATH` | `data/chroma_db` | Vector database path |
| `PDF_DIR` | `data/pdfs` | PDF source directory |
| `CSV_DIR` | `data/csvs` | CSV source directory |
| `S3_SEED_DATA_BASE` | `https://s3.gswiki.ischool.syr.edu/images` | S3 public base URL |
| `S3_SEED_DATA_PREFIX` | `gravity-spy-wiki-bot` | S3 key prefix |

## Testing

```bash
pytest
pytest -v                              # Verbose
pytest tests/test_vector_store.py      # Specific file
pytest --cov=. --cov-report=term-missing  # With coverage
```

Tests use mocked embeddings and an isolated collection. Integration tests in `test_azure_integration.py` hit the real API and auto-skip when credentials are absent.