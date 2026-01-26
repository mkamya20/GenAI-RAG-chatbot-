# Docker Setup Guide

This guide explains how to build and run the Gravity Spy chatbot using Docker.

## Prerequisites

- Docker Desktop installed and running
- `.env` file with Azure OpenAI credentials (optional, can use environment variables)

## Quick Start

### Using Docker Compose (Recommended)

1. **Ensure Docker Desktop is running**

2. **Build and start the container:**
   ```bash
   docker-compose up --build
   ```

3. **Access the application:**
   - Frontend: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/api/health

4. **Stop the container:**
   ```bash
   docker-compose down
   ```

### Using Docker Commands

1. **Build the image:**
   ```bash
   docker build -t gravity-spy-app .
   ```

2. **Run the container:**
   ```bash
   docker run -d \
     -p 8000:8000 \
     -v ./chroma_db:/app/chroma_db \
     -v ./data/pdfs:/app/data/pdfs \
     -v ./outputs:/app/outputs \
     --env-file .env \
     --name gravity-spy-chatbot \
     gravity-spy-app
   ```

3. **View logs:**
   ```bash
   docker logs -f gravity-spy-chatbot
   ```

4. **Stop the container:**
   ```bash
   docker stop gravity-spy-chatbot
   docker rm gravity-spy-chatbot
   ```

## Environment Variables

Create a `.env` file in the project root with:

```env
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
```

Or set them directly in `docker-compose.yml`.

## Volumes

The following directories are mounted as volumes to persist data:

- `./chroma_db` → `/app/chroma_db` - Vector database
- `./data/pdfs` → `/app/data/pdfs` - PDF files
- `./outputs` → `/app/outputs` - Output files
- `./data/temp` → `/app/data/temp` - Temporary uploads

## First Run

On the first run, the embedding model will be downloaded (~80MB). This may take a few minutes.

## Troubleshooting

### Container won't start
- Check if Docker Desktop is running
- Check logs: `docker-compose logs`
- Verify port 8000 is not in use

### ChromaDB data not persisting
- Ensure volumes are properly mounted
- Check file permissions

### Model download issues
- Check internet connection
- First run downloads models automatically

## Production Deployment

For production:
1. Use a reverse proxy (nginx) in front of the container
2. Set specific CORS origins instead of "*"
3. Use Docker secrets for sensitive environment variables
4. Set up proper logging and monitoring
5. Use a process manager or orchestration tool (Kubernetes, Docker Swarm)
