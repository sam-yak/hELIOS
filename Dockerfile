# --- Stage 1: Build the vector database ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY materials_database.json .
COPY ingest_v2.py .

# Build the ChromaDB vector database at build time
RUN python ingest_v2.py

# --- Stage 2: Production image ---
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy the pre-built database from builder stage
COPY --from=builder /app/db ./db

# Create logs directory
RUN mkdir -p logs

# Expose port (Railway uses PORT env var)
EXPOSE 8000

# Use shell form so $PORT is expanded at runtime
# Railway sets PORT automatically; fallback to 8000
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
