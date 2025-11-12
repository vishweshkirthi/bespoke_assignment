# Docker Deployment Guide

## Quick Start

### Using Docker Compose (Recommended)

1. **Build and run the container**:
   ```bash
   cd docker
   docker-compose up --build
   ```

2. **Access the application**:
   - Web UI: http://localhost:8000/ui
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

3. **Stop the container**:
   ```bash
   docker-compose down
   ```

### Using Docker directly

1. **Build the image**:
   ```bash
   cd docker
   docker build -f Dockerfile -t fasttext-classifier ..
   ```

2. **Run the container**:
   ```bash
   docker run -p 8000:8000 \
     -v "$(pwd)/../data:/app/data" \
     -v "$(pwd)/../models:/app/models" \
     fasttext-classifier
   ```

## Volume Mounts

The container uses volume mounts for data persistence:

- **`../data:/app/data`** - Training data and datasets
- **`../models:/app/models`** - Trained FastText models
- **`./logs:/app/logs`** - Application logs (optional)

## Environment Variables

- **`PYTHONPATH=/app`** - Python module path
- **`LOG_LEVEL=INFO`** - Logging level
- **`PYTHONDONTWRITEBYTECODE=1`** - Don't write .pyc files
- **`PYTHONUNBUFFERED=1`** - Unbuffered output

## Health Check

The container includes a health check that verifies the API is responding:
- **Interval**: 30 seconds
- **Timeout**: 10 seconds  
- **Retries**: 3
- **Start period**: 40 seconds
