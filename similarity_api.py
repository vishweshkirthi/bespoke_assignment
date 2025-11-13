#!/usr/bin/env python3
"""
FastAPI router for word similarity using FastText get_nearest_neighbors
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Optional
import fasttext
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Global variable to hold the current model
current_model: Optional[fasttext.FastText] = None

class SimilarityRequest(BaseModel):
    word: str
    k: int = 10

class SimilarityResponse(BaseModel):
    word: str
    neighbors: List[Tuple[str, float]]
    message: str

def set_model(model: Optional[fasttext.FastText]):
    """Set the current model for similarity calculations"""
    global current_model
    current_model = model

@router.post("/similarity", response_model=SimilarityResponse)
async def get_similar_words(request: SimilarityRequest):
    """
    Get k nearest neighbors for a given word using FastText model
    
    Args:
        request: SimilarityRequest containing word and k (number of neighbors)
    
    Returns:
        SimilarityResponse with word, neighbors list, and message
    """
    if not current_model:
        raise HTTPException(status_code=400, detail="No model loaded. Please train or load a model first.")
    
    try:
        # Use FastText's get_nearest_neighbors method
        neighbors = current_model.get_nearest_neighbors(request.word, k=request.k)
        
        # FastText returns [(similarity_score, word), ...] but we need [(word, similarity_score), ...]
        # Also filter out the input word itself from the results
        formatted_neighbors = [(word, similarity_score) for similarity_score, word in neighbors 
                              if word.lower().strip() != request.word.lower().strip()]
        
        return SimilarityResponse(
            word=request.word,
            neighbors=formatted_neighbors,
            message=f"Found {len(formatted_neighbors)} nearest neighbors for '{request.word}'"
        )
        
    except Exception as e:
        logger.error(f"Error finding similar words for '{request.word}': {e}")
        raise HTTPException(status_code=500, detail=f"Error finding similar words: {str(e)}")

@router.get("/similarity/{word}")
async def get_similar_words_get(word: str, k: int = 10):
    """
    GET endpoint to find similar words (alternative to POST)
    
    Args:
        word: Input word to find neighbors for
        k: Number of neighbors to return (default: 10)
    
    Returns:
        SimilarityResponse with word, neighbors list, and message
    """
    request = SimilarityRequest(word=word, k=k)
    return await get_similar_words(request)

@router.get("/model/vocabulary/size")
async def get_vocabulary_size():
    """
    Get the size of the model's vocabulary
    
    Returns:
        Dictionary with vocabulary size information
    """
    if not current_model:
        raise HTTPException(status_code=400, detail="No model loaded.")
    
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
async def get_vocabulary_sample(limit: int = 50):
    """
    Get a sample of words from the model's vocabulary
    
    Args:
        limit: Number of words to return (default: 50)
    
    Returns:
        Dictionary with sample words
    """
    if not current_model:
        raise HTTPException(status_code=400, detail="No model loaded.")
    
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