#!/usr/bin/env python3
"""
FastAPI service for document quality classification
"""

from fastapi import FastAPI, Request, Response
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
from session_manager import session_manager

# Import separated API modules
from api.score_api import router as score_router
from api.train_api import router as train_router
from api.similarity_api import router as similarity_router
import api.score_api as score_api
import api.train_api as train_api
import api.similarity_api as similarity_api

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Quality Classifier", version="1.0.0")

# Global variables for backward compatibility and default model
default_model_path = "models/document_quality_model.bin"
negative_file_path = "data/processed/train_negative.txt"

@app.on_event("startup")
async def startup_event():
    """Initialize session manager and load default model if available"""
    
    # Load default model into cache if it exists
    if os.path.exists(default_model_path):
        try:
            # Pre-load the default model into session manager cache
            if default_model_path not in session_manager.model_cache:
                default_model = fasttext.load_model(default_model_path)
                session_manager.model_cache[default_model_path] = default_model
                logger.info(f"Pre-loaded default model: {default_model_path}")
        except Exception as e:
            logger.warning(f"Could not load default model: {e}")
    
    # Set up API modules with session manager
    score_api.set_session_manager(session_manager)
    train_api.set_session_manager(session_manager, negative_file_path)
    similarity_api.set_session_manager(session_manager)
    
    logger.info("Session-based architecture initialized")

# Session management middleware
@app.middleware("http")
async def session_middleware(request: Request, call_next):
    """Add session management to all requests using tab-specific session IDs"""
    # Get session ID from header (sent by frontend)
    session_id = request.headers.get("x-session-id")
    
    # Get or create session
    session_id, session_data = session_manager.get_or_create_session(session_id)
    
    # Add session info to request state
    request.state.session_id = session_id
    request.state.session_data = session_data
    
    # Process request
    response = await call_next(request)
    
    # Add session ID to response headers for debugging
    response.headers["X-Session-ID"] = session_id
    
    return response

# Include the separated API routers
app.include_router(score_router)
app.include_router(train_router)
app.include_router(similarity_router)

@app.get("/")
async def root():
    """Redirect to UI"""
    return RedirectResponse(url="/ui")

@app.get("/api")
async def api_info(request: Request):
    """API health check and endpoints info"""
    session_id = request.state.session_id
    session_model = session_manager.get_model(session_id)
    model_status = "loaded" if session_model else "not loaded"
    
    stats = session_manager.get_stats()
    
    return {
        "message": "Document Quality Classifier API (Session-based)",
        "session_id": session_id,
        "model_status": model_status,
        "session_stats": stats,
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
            "GET /model/vocabulary/sample": "Get sample of vocabulary words",
            "GET /session/stats": "Get session statistics"
        }
    }

@app.get("/session/stats")
async def session_stats(request: Request):
    """Get session statistics"""
    session_id = request.state.session_id
    session_data = request.state.session_data
    stats = session_manager.get_stats()
    
    return {
        "session_id": session_id,
        "session_created_at": session_data['created_at'].isoformat(),
        "session_last_accessed": session_data['last_accessed'].isoformat(),
        "session_model_loaded": session_data['model'] is not None,
        "session_model_path": session_data.get('model_path'),
        "session_training_jobs": len(session_data['training_jobs']),
        "session_websocket_connections": len(session_data['websocket_connections']),
        "global_stats": stats
    }

@app.get("/debug/sessions")
async def debug_sessions():
    """Debug endpoint to see all active sessions"""
    with session_manager.lock:
        sessions_info = {}
        for sid, sdata in session_manager.sessions.items():
            sessions_info[sid] = {
                "created_at": sdata['created_at'].isoformat(),
                "last_accessed": sdata['last_accessed'].isoformat(),
                "has_model": sdata['model'] is not None,
                "model_path": sdata.get('model_path')
            }
    
    return {
        "total_sessions": len(session_manager.sessions),
        "sessions": sessions_info,
        "model_cache_size": len(session_manager.model_cache),
        "cached_models": list(session_manager.model_cache.keys())
    }

@app.get("/ui", response_class=HTMLResponse)
async def get_ui():
    """Serve the UI"""
    with open("main.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)