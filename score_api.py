#!/usr/bin/env python3
"""
Score API - Document quality scoring endpoints
"""

from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import Optional, List
import logging
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocessing import preprocess_text, validate_text_requirements

# Setup logging
logger = logging.getLogger(__name__)

router = APIRouter()

class TextScore(BaseModel):
    text: str

class ScoreResponse(BaseModel):
    text: str
    predicted_label: str
    confidence: float
    is_high_quality: bool

class BatchScoreResponse(BaseModel):
    total_documents: int
    predictions: List[dict]
    metrics: Optional[dict] = None

# This will be set from main app
current_model = None

def set_model(model):
    """Set the current model for scoring"""
    global current_model
    current_model = model

@router.post("/score", response_model=ScoreResponse)
async def score_text(request: TextScore):
    """
    Score a text for quality.
    
    Returns prediction label, confidence score, and quality decision.
    """
    
    if not current_model:
        raise HTTPException(status_code=503, detail="No model available. Please train a model first using /train endpoint.")
    
    text = request.text.strip()
    
    # Validate text requirements
    is_valid, error_message = validate_text_requirements(text)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_message)
    
    # Preprocess text for prediction
    processed_text = preprocess_text(text)
    if not processed_text:
        raise HTTPException(status_code=400, detail="Text becomes empty after preprocessing")
    
    try:
        # Get prediction using preprocessed text
        prediction = current_model.predict(processed_text, k=1)
        
        predicted_label = prediction[0][0].replace('__label__', '')
        confidence = float(prediction[1][0])
        
        # Determine if high quality (you can adjust threshold)
        threshold = 0.5
        is_high_quality = predicted_label == 'high' and confidence >= threshold
        
        return ScoreResponse(
            text=request.text,
            predicted_label=predicted_label,
            confidence=confidence,
            is_high_quality=is_high_quality
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")

@router.post("/score/batch", response_model=BatchScoreResponse)
async def score_batch(file: UploadFile = File(...)):
    """
    Score a batch of texts from uploaded file
    
    Expected file format: Each line should be "__label__[high|low] [text]" or just "[text]"
    """
    
    if not current_model:
        raise HTTPException(status_code=503, detail="No model available. Please train or select a model first.")
    
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="File must be a .txt file")
    
    try:
        # Read file content
        content = await file.read()
        text_content = content.decode('utf-8')
        
        predictions = []
        true_labels = []
        predicted_labels = []
        
        for line_num, line in enumerate(text_content.split('\n'), 1):
            line = line.strip()
            if not line:
                continue
                
            # Extract true label if present
            true_label = None
            if line.startswith('__label__'):
                parts = line.split(' ', 1)
                if len(parts) > 1:
                    true_label = parts[0].replace('__label__', '')
                    text = parts[1]
                else:
                    continue
            else:
                text = line
            
            # Validate text requirements
            is_valid, error_message = validate_text_requirements(text)
            if not is_valid:
                predictions.append({
                    "line_number": line_num,
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "error": error_message,
                    "true_label": true_label
                })
                continue
            
            # Preprocess and predict
            processed_text = preprocess_text(text)
            if not processed_text:
                predictions.append({
                    "line_number": line_num,
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "error": "Text becomes empty after preprocessing",
                    "true_label": true_label
                })
                continue
            
            # Get prediction
            prediction = current_model.predict(processed_text, k=1)
            predicted_label = prediction[0][0].replace('__label__', '')
            confidence = float(prediction[1][0])
            
            threshold = 0.5
            is_high_quality = predicted_label == 'high' and confidence >= threshold
            
            prediction_result = {
                "line_number": line_num,
                "text": text[:100] + "..." if len(text) > 100 else text,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "is_high_quality": is_high_quality,
                "true_label": true_label
            }
            
            predictions.append(prediction_result)
            
            # Collect labels for metrics calculation
            if true_label:
                true_labels.append(true_label)
                predicted_labels.append(predicted_label)
        
        # Calculate metrics if we have true labels
        metrics = None
        if true_labels and len(true_labels) == len(predicted_labels):
            # Calculate confusion matrix
            tp = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 'high' and p == 'high')
            fp = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 'low' and p == 'high')
            tn = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 'low' and p == 'low')
            fn = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 'high' and p == 'low')
            
            accuracy = (tp + tn) / len(true_labels) if len(true_labels) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics = {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "confusion_matrix": {
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn
                },
                "total_labeled": len(true_labels)
            }
        
        return BatchScoreResponse(
            total_documents=len(predictions),
            predictions=predictions,
            metrics=metrics
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch scoring failed: {str(e)}")