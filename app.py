#!/usr/bin/env python3
"""
FastAPI service for document quality classification
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
import fasttext
import os
from typing import Optional
import logging
import sys

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocessing import preprocess_text, validate_text_requirements

# Import separated API modules
from score_api import router as score_router
from train_api import router as train_router
from similarity_api import router as similarity_router
import score_api
import train_api
import similarity_api

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Quality Classifier", version="1.0.0")

# Global variables
current_model: Optional[fasttext.FastText] = None
model_path = "models/document_quality_model.bin"
negative_file_path = "data/processed/train_negative.txt"

@app.on_event("startup")
async def startup_event():
    """Load existing model if available"""
    global current_model
    
    if os.path.exists(model_path):
        try:
            current_model = fasttext.load_model(model_path)
            logger.info(f"Loaded existing model from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load existing model: {e}")
    
    # Set up shared state for separated API modules
    score_api.set_model(current_model)
    train_api.set_globals(current_model, model_path, negative_file_path)
    similarity_api.set_model(current_model)

def update_shared_model():
    """Update the model in all APIs when it changes"""
    global current_model, model_path
    current_model = train_api.get_current_model()
    score_api.set_model(current_model)
    similarity_api.set_model(current_model)

# Add callback to train_api to update shared model
train_api.model_update_callback = update_shared_model

# Include the separated API routers
app.include_router(score_router)
app.include_router(train_router)
app.include_router(similarity_router)

@app.get("/")
async def root():
    """Redirect to UI"""
    return RedirectResponse(url="/ui")

@app.get("/api")
async def api_info():
    """API health check and endpoints info"""
    model_status = "loaded" if current_model else "not loaded"
    return {
        "message": "Document Quality Classifier API",
        "model_status": model_status,
        "endpoints": {
            "POST /train": "Upload positive examples file to train model",
            "POST /score": "Score a text for quality",
            "GET /models": "List available models",
            "POST /model/select": "Select a model to load",
            "GET /model/status": "Get current model status",
            "POST /score/batch": "Score batch of texts from file",
            "WebSocket /ws": "Training progress updates",
            "POST /similarity": "Find nearest neighbors for a word",
            "GET /similarity/{word}": "Find similar words (GET method)",
            "GET /model/vocabulary/size": "Get vocabulary size",
            "GET /model/vocabulary/sample": "Get sample of vocabulary words"
        }
    }

@app.get("/ui", response_class=HTMLResponse)
async def get_ui():
    """Serve the UI"""
    with open("main.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)