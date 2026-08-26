#!/bin/bash
# docker-entrypoint.sh
#
# 1. Download pre-built ChromaDB index from S3 if not present
# 2. Start the application
#
# All paths are relative to the working directory, so this works both
# inside the Docker container (WORKDIR /app) and in a dev environment
# (project root).

set -e

DATA_DIR="./data"
INIT_LOG="${DATA_DIR}/init.log"

# S3 public URL for the pre-built index archive
S3_SEED_DATA_BASE="${S3_SEED_DATA_BASE:-https://s3.gswiki.ischool.syr.edu/images}"
S3_SEED_DATA_PREFIX="${S3_SEED_DATA_PREFIX:-gravity-spy-wiki-bot}"
CHROMA_ARCHIVE_URL="${S3_SEED_DATA_BASE}/${S3_SEED_DATA_PREFIX}/chroma_db.tar.gz"

# Ensure data directory exists
mkdir -p "${DATA_DIR}"

# -------------------------------------------------------------------------
# Download ChromaDB index if not present
# -------------------------------------------------------------------------
if [ ! -d "${DATA_DIR}/chroma_db" ] || [ -z "$(ls -A "${DATA_DIR}/chroma_db" 2>/dev/null)" ]; then
    echo "$(date): No ChromaDB index found. Downloading from S3..." | tee -a "$INIT_LOG"

    curl -sf -o "${DATA_DIR}/chroma_db.tar.gz" "$CHROMA_ARCHIVE_URL" || {
        echo "$(date): ERROR - Failed to download ${CHROMA_ARCHIVE_URL}" | tee -a "$INIT_LOG"
        exit 1
    }

    echo "$(date): Extracting index..." | tee -a "$INIT_LOG"
    tar xzf "${DATA_DIR}/chroma_db.tar.gz" -C "${DATA_DIR}"
    rm "${DATA_DIR}/chroma_db.tar.gz"

    echo "$(date): ChromaDB index ready." | tee -a "$INIT_LOG"
else
    echo "$(date): ChromaDB index exists, skipping download." | tee -a "$INIT_LOG"
fi

exec python app.py