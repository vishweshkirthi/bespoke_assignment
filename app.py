#!/usr/bin/env python3
"""
FastAPI service for document quality classification
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import fasttext
import os
import tempfile
import random
from typing import Optional, List
import logging
import json
import asyncio
import sys
import uuid
from datetime import datetime

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocessing import preprocess_text, validate_text_requirements

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Quality Classifier", version="1.0.0")

# Global variables
current_model: Optional[fasttext.FastText] = None
model_path = "models/document_quality_model.bin"
negative_file_path = "data/processed/train_negative.txt"

# WebSocket connections for training progress
active_connections: List[WebSocket] = []

# Training status
training_status = {
    "is_training": False,
    "progress": 0,
    "message": "",
    "total_examples": 0
}

class TextScore(BaseModel):
    text: str

class ScoreResponse(BaseModel):
    text: str
    predicted_label: str
    confidence: float
    is_high_quality: bool

class TrainResponse(BaseModel):
    message: str
    training_examples: int
    model_saved: str
    autotune_used: bool = False

class ModelInfo(BaseModel):
    id: str
    filename: str
    display_name: str
    created_at: str
    size_mb: float

class ModelListResponse(BaseModel):
    models: List[ModelInfo]
    current_model: Optional[str]

class ModelSelectRequest(BaseModel):
    model_id: str

class BatchScoreResponse(BaseModel):
    total_documents: int
    predictions: List[dict]
    metrics: Optional[dict] = None


def combine_datasets(positive_file_path: str, negative_file_path: str, output_path: str):
    """Combine positive and negative files for training with preprocessing"""
    
    # Read and preprocess positive examples
    positive_lines = []
    with open(positive_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('__label__'):
                parts = line.split(' ', 1)
                if len(parts) > 1:
                    label = parts[0]
                    text = preprocess_text(parts[1])
                    if text:  # Only include non-empty processed text
                        positive_lines.append(f"{label} {text}\n")
    
    # Read and preprocess negative examples
    negative_lines = []
    with open(negative_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('__label__'):
                parts = line.split(' ', 1)
                if len(parts) > 1:
                    label = parts[0]
                    text = preprocess_text(parts[1])
                    if text:  # Only include non-empty processed text
                        negative_lines.append(f"{label} {text}\n")
    
    logger.info(f"Combining {len(positive_lines)} positive and {len(negative_lines)} negative examples (after preprocessing)")
    
    # Combine and shuffle
    all_lines = positive_lines + negative_lines
    random.shuffle(all_lines)
    
    # Write combined file with UTF-8 encoding
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(all_lines)
    
    return len(all_lines)

async def broadcast_training_status():
    """Broadcast training status to all connected WebSockets"""
    if active_connections:
        message = json.dumps(training_status)
        disconnected = []
        for connection in active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            active_connections.remove(conn)

async def train_model_background(positive_file_path: str, validation_file_path: str = None):
    """Train model in background with progress updates"""
    global current_model, training_status
    
    try:
        # Update training status
        training_status["is_training"] = True
        training_status["progress"] = 0
        training_status["message"] = "Preparing data..."
        await broadcast_training_status()
        
        # Create temporary combined training file
        combined_file = "temp_combined_train.txt"
        total_examples = combine_datasets(positive_file_path, negative_file_path, combined_file)
        
        training_status["total_examples"] = total_examples
        training_status["message"] = "Starting training..."
        training_status["progress"] = 10
        await broadcast_training_status()
        
        logger.info("Starting FastText training...")
        
        # Train the model with or without autotune
        if validation_file_path:
            training_status["message"] = "Training with autotune (this may take several minutes)..."
            training_status["progress"] = -1  # Indeterminate progress
            await broadcast_training_status()
            
            model = fasttext.train_supervised(
                input=combined_file,
                autotuneValidationFile=validation_file_path,
                autotuneDuration=300,  # 5 minutes max
                verbose=0
            )
            training_status["autotune_used"] = True
        else:
            training_status["message"] = "Training model..."
            training_status["progress"] = -1  # Indeterminate progress
            await broadcast_training_status()
            
            model = fasttext.train_supervised(
                input=combined_file,
                lr=0.1,
                epoch=20,
                wordNgrams=2,
                dim=100,
                ws=5,
                minCount=5,
                minn=3,
                maxn=6,
                neg=5,
                loss='softmax',
                verbose=0  # Reduce verbosity for API
            )
            training_status["autotune_used"] = False
        
        training_status["message"] = "Saving model..."
        training_status["progress"] = 80
        await broadcast_training_status()
        
        # Use the UUID and filename from training status (generated when training started)
        model_uuid = training_status.get("model_uuid", str(uuid.uuid4()))
        model_filename = training_status.get("model_filename", f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_uuid[:8]}.bin")
        full_model_path = f"models/{model_filename}"
        
        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)
        
        # Save the model with UUID-based name
        model.save_model(full_model_path)
        
        # Update global model and path
        current_model = model
        model_path = full_model_path
        
        # Save model metadata
        metadata = {
            "id": model_uuid,
            "filename": model_filename,
            "created_at": datetime.now().isoformat(),
            "training_examples": training_status.get("total_examples", 0),
            "autotune_used": validation_file_path is not None,
            "parameters": {
                "lr": 0.1,
                "epoch": 20,
                "wordNgrams": 2,
                "dim": 100,
                "ws": 5,
                "minCount": 5,
                "minn": 3,
                "maxn": 6,
                "neg": 5,
                "loss": "softmax"
            } if not validation_file_path else "autotune_optimized"
        }
        
        metadata_path = f"models/{model_filename}.metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        training_status["message"] = "Training completed!"
        training_status["progress"] = 100
        training_status["is_training"] = False
        await broadcast_training_status()
        
        # Cleanup
        os.remove(combined_file)
        os.remove(positive_file_path)  # Remove uploaded file
        if validation_file_path:
            os.remove(validation_file_path)  # Remove uploaded validation file
        
        logger.info(f"Training completed! Model saved to {model_path}")
        
        # Reset status after a delay
        await asyncio.sleep(3)
        training_status["message"] = ""
        training_status["progress"] = 0
        await broadcast_training_status()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        training_status["is_training"] = False
        training_status["message"] = f"Training failed: {str(e)}"
        training_status["progress"] = 0
        await broadcast_training_status()

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

@app.get("/")
async def root():
    """Health check endpoint"""
    model_status = "loaded" if current_model else "not loaded"
    return {
        "message": "Document Quality Classifier API",
        "model_status": model_status,
        "endpoints": {
            "POST /train": "Upload positive examples file to train model",
            "POST /score": "Score a text for quality"
        }
    }

@app.post("/train", response_model=TrainResponse)
async def train_model(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    validation_file: UploadFile = File(None)
):
    """
    Upload positive examples file and train the model.
    
    Expected file format: Each line should be "__label__high [text]"
    Negative examples are automatically loaded from data/train_negative.txt
    
    Optional validation_file: If provided, FastText autotune will be used to optimize hyperparameters
    """
    
    # Validate files
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Training file must be a .txt file")
    
    if validation_file and not validation_file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Validation file must be a .txt file")
    
    # Check if negative file exists
    if not os.path.exists(negative_file_path):
        raise HTTPException(
            status_code=500, 
            detail=f"Negative examples file not found: {negative_file_path}. Please ensure data/train_negative.txt exists."
        )
    
    try:
        # Save uploaded training file temporarily
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_file_path = tmp_file.name
        
        # Save and preprocess validation file if provided
        validation_file_path = None
        if validation_file:
            # Read and preprocess validation file
            val_content = await validation_file.read()
            val_text = val_content.decode('utf-8')
            
            processed_val_lines = []
            for line in val_text.split('\n'):
                line = line.strip()
                if line.startswith('__label__'):
                    parts = line.split(' ', 1)
                    if len(parts) > 1:
                        label = parts[0]
                        text = preprocess_text(parts[1])
                        if text:
                            processed_val_lines.append(f"{label} {text}\n")
            
            # Save preprocessed validation file
            with tempfile.NamedTemporaryFile(mode='w', suffix='_val.txt', delete=False, encoding='utf-8') as tmp_val_file:
                tmp_val_file.writelines(processed_val_lines)
                validation_file_path = tmp_val_file.name
        
        # Count lines in uploaded file
        with open(temp_file_path, 'r', encoding='utf-8') as f:
            positive_count = len(f.readlines())
        
        # Count negative examples
        with open(negative_file_path, 'r', encoding='utf-8') as f:
            negative_count = len(f.readlines())
        
        total_examples = positive_count + negative_count
        
        # Start training in background
        asyncio.create_task(train_model_background(temp_file_path, validation_file_path))
        
        autotune_message = " with autotune" if validation_file else ""
        
        # Generate UUID for the model that will be created
        model_uuid = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        autotune_suffix = "_autotune" if validation_file else ""
        model_filename = f"model_{timestamp}_{model_uuid[:8]}{autotune_suffix}.bin"
        
        # Store the UUID in training status for the background task
        training_status["model_uuid"] = model_uuid
        training_status["model_filename"] = model_filename
        
        return TrainResponse(
            message=f"Training{autotune_message} started in background. Model UUID: {model_uuid}",
            training_examples=total_examples,
            model_saved=f"models/{model_filename}",
            autotune_used=bool(validation_file)
        )
        
    except Exception as e:
        # Cleanup on error
        if 'temp_file_path' in locals():
            try:
                os.remove(temp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/score", response_model=ScoreResponse)
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

@app.post("/score/batch", response_model=BatchScoreResponse)
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

@app.get("/model/status")
async def model_status():
    """Get current model status"""
    current_model_name = os.path.basename(model_path) if current_model else None
    
    if current_model:
        return {
            "status": "loaded",
            "model_path": model_path,
            "model_name": current_model_name,
            "model_exists": os.path.exists(model_path)
        }
    else:
        return {
            "status": "not_loaded",
            "model_path": model_path,
            "model_name": current_model_name,
            "model_exists": os.path.exists(model_path)
        }

def get_model_info(filename: str, models_dir: str) -> ModelInfo:
    """Get model information including metadata"""
    file_path = os.path.join(models_dir, filename)
    metadata_path = f"{file_path}.metadata.json"
    
    # Get file stats
    stat = os.stat(file_path)
    size_mb = round(stat.st_size / (1024 * 1024), 2)
    created_at = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    
    # Try to load metadata
    model_id = filename.replace('.bin', '')
    display_name = filename
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                model_id = metadata.get('id', model_id)
                created_at = metadata.get('created_at', created_at)
                if 'T' in created_at:  # ISO format
                    created_at = datetime.fromisoformat(created_at.replace('Z', '')).strftime("%Y-%m-%d %H:%M:%S")
                
                # Create display name with info
                autotune = " (Autotune)" if metadata.get('autotune_used') else ""
                examples = metadata.get('training_examples', 'Unknown')
                display_name = f"{filename.replace('.bin', '')} - {examples} examples{autotune}"
                
        except Exception as e:
            logger.warning(f"Could not load metadata for {filename}: {e}")
    
    return ModelInfo(
        id=model_id,
        filename=filename,
        display_name=display_name,
        created_at=created_at,
        size_mb=size_mb
    )

@app.get("/models", response_model=ModelListResponse)
async def list_models():
    """Get list of available models with metadata"""
    models_dir = "models"
    models = []
    
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.bin'):
                try:
                    model_info = get_model_info(file, models_dir)
                    models.append(model_info)
                except Exception as e:
                    logger.warning(f"Could not get info for model {file}: {e}")
    
    # Sort by creation date (newest first)
    models.sort(key=lambda x: x.created_at, reverse=True)
    
    current_model_id = None
    if current_model and model_path:
        current_filename = os.path.basename(model_path)
        for model in models:
            if model.filename == current_filename:
                current_model_id = model.id
                break
    
    return ModelListResponse(
        models=models,
        current_model=current_model_id
    )

@app.post("/model/select")
async def select_model(request: ModelSelectRequest):
    """Select a model to load by ID"""
    global current_model, model_path
    
    # Find model by ID
    models_dir = "models"
    target_filename = None
    
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.bin'):
                try:
                    model_info = get_model_info(file, models_dir)
                    if model_info.id == request.model_id:
                        target_filename = file
                        break
                except Exception:
                    continue
    
    if not target_filename:
        raise HTTPException(status_code=404, detail=f"Model with ID {request.model_id} not found")
    
    new_model_path = f"models/{target_filename}"
    
    try:
        # Load the new model
        new_model = fasttext.load_model(new_model_path)
        
        # Update global variables
        current_model = new_model
        model_path = new_model_path
        
        logger.info(f"Switched to model: {target_filename} (ID: {request.model_id})")
        
        return {
            "message": f"Successfully loaded model: {target_filename}",
            "model_id": request.model_id,
            "filename": target_filename,
            "model_path": new_model_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for training progress"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        # Send current status immediately
        await websocket.send_text(json.dumps(training_status))
        
        # Keep connection alive and handle ping/pong
        while True:
            try:
                await websocket.receive_text()
            except:
                break
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)

@app.get("/ui", response_class=HTMLResponse)
async def get_ui():
    """Serve the UI"""
    with open("main.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)