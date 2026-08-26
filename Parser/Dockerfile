# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY *.py .
COPY routers/ routers/
COPY templates/ templates/
COPY static/ static/
COPY docker-entrypoint.sh .

# Create data directory (index downloaded from S3 at runtime)
RUN mkdir -p data/chroma_db

# Make docker-entrypoint executable
RUN chmod +x docker-entrypoint.sh

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run via docker-entrypoint
CMD ["./docker-entrypoint.sh"]