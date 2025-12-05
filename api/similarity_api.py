#!/usr/bin/env python3
"""
FastAPI router for word similarity using FastText get_nearest_neighbors
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Tuple, Optional
import fasttext
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Session manager will be set from main app
session_manager = None

class SimilarityRequest(BaseModel):
    word: str
    k: int = 10

class SimilarityResponse(BaseModel):
    word: str
    neighbors: List[Tuple[str, float]]
    message: str

def set_session_manager(sm):
    """Set the session manager for session-based similarity calculations"""
    global session_manager
    session_manager = sm

@router.post("/similarity", response_model=SimilarityResponse)
async def get_similar_words(request_data: SimilarityRequest, request: Request):
    """
    Get k nearest neighbors for a given word using session-specific FastText model
    
    Args:
        request_data: SimilarityRequest containing word and k (number of neighbors)
        request: FastAPI request object for session access
    
    Returns:
        SimilarityResponse with word, neighbors list, and message
    """
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not available")
    
    session_id = request.state.session_id
    current_model = session_manager.get_model(session_id)
    
    if not current_model:
        raise HTTPException(status_code=400, detail="No model loaded for your session. Please train or load a model first.")
    
    try:
        # Use FastText's get_nearest_neighbors method
        neighbors = current_model.get_nearest_neighbors(request_data.word, k=request_data.k)
        
        # FastText returns [(similarity_score, word), ...] but we need [(word, similarity_score), ...]
        # Also filter out the input word itself from the results
        formatted_neighbors = [(word, similarity_score) for similarity_score, word in neighbors 
                              if word.lower().strip() != request_data.word.lower().strip()]
        
        return SimilarityResponse(
            word=request_data.word,
            neighbors=formatted_neighbors,
            message=f"Found {len(formatted_neighbors)} nearest neighbors for '{request_data.word}'"
        )
        
    except Exception as e:
        logger.error(f"Error finding similar words for '{request_data.word}': {e}")
        raise HTTPException(status_code=500, detail=f"Error finding similar words: {str(e)}")

@router.get("/similarity/{word}")
async def get_similar_words_get(word: str, request: Request, k: int = 10):
    """
    GET endpoint to find similar words (alternative to POST)
    
    Args:
        word: Input word to find neighbors for
        request: FastAPI request object for session access
        k: Number of neighbors to return (default: 10)
    
    Returns:
        SimilarityResponse with word, neighbors list, and message
    """
    request_data = SimilarityRequest(word=word, k=k)
    return await get_similar_words(request_data, request)

@router.get("/model/vocabulary/size")
async def get_vocabulary_size(request: Request):
    """
    Get the size of the session-specific model's vocabulary
    
    Returns:
        Dictionary with vocabulary size information
    """
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not available")
    
    session_id = request.state.session_id
    current_model = session_manager.get_model(session_id)
    
    if not current_model:
        raise HTTPException(status_code=400, detail="No model loaded for your session.")
    
    try:
        # Get vocabulary from FastText model
        words = current_model.get_words()
        vocab_size = len(words)
        
        return {
            "vocabulary_size": vocab_size,
            "message": f"Model vocabulary contains {vocab_size} words"
        }
        
    except Exception as e:
        logger.error(f"Error getting vocabulary size: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting vocabulary: {str(e)}")

@router.get("/model/vocabulary/sample")
async def get_vocabulary_sample(request: Request, limit: int = 50):
    """
    Get a sample of words from the session-specific model's vocabulary
    
    Args:
        request: FastAPI request object for session access
        limit: Number of words to return (default: 50)
    
    Returns:
        Dictionary with sample words
    """
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not available")
    
    session_id = request.state.session_id
    current_model = session_manager.get_model(session_id)
    
    if not current_model:
        raise HTTPException(status_code=400, detail="No model loaded for your session.")
    
    try:
        words = current_model.get_words()
        sample_words = words[:limit]
        
        return {
            "sample_words": sample_words,
            "total_vocabulary_size": len(words),
            "sample_size": len(sample_words),
            "message": f"Showing {len(sample_words)} words from vocabulary of {len(words)}"
        }
        
    except Exception as e:
        logger.error(f"Error getting vocabulary sample: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting vocabulary sample: {str(e)}")