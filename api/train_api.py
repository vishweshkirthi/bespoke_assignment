#!/usr/bin/env python3
"""
Train API - Model training endpoints
"""

from fastapi import APIRouter, HTTPException, File, UploadFile, BackgroundTasks, WebSocket, WebSocketDisconnect, Form
from pydantic import BaseModel
from typing import Optional, List
import fasttext
import os
import tempfile
import random
import logging
import json
import asyncio
import uuid
from datetime import datetime
import sys

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocessing import preprocess_text

# Setup logging
logger = logging.getLogger(__name__)

router = APIRouter()

class TrainResponse(BaseModel):
    message: str
    training_examples: int
    model_saved: str
    autotune_used: bool = False
    hyperparams_used: Optional[dict] = None

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

# Global variables that will be set from main app
current_model = None
model_path = None
negative_file_path = None

# Callback for notifying main app when model changes
model_update_callback = None

# WebSocket connections for training progress
active_connections: List[WebSocket] = []

# Training status
training_status = {
    "is_training": False,
    "progress": 0,
    "message": "",
    "total_examples": 0
}

def set_globals(model, path, neg_file_path):
    """Set global variables from main app"""
    global current_model, model_path, negative_file_path
    current_model = model
    model_path = path
    negative_file_path = neg_file_path

def get_current_model():
    """Get current model reference"""
    return current_model

def update_current_model(new_model, new_path):
    """Update current model"""
    global current_model, model_path
    current_model = new_model
    model_path = new_path

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

async def train_model_background(positive_file_path: str, validation_file_path: str = None, hyperparams: dict = None):
    """Train model in background with progress updates"""
    global current_model, model_path, training_status
    
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
        
        # Prepare training parameters
        train_params = {
            'input': combined_file,
            'lr': 0.1,
            'epoch': 20,
            'wordNgrams': 2,
            'dim': 100,
            'ws': 5,
            'minCount': 5,
            'minn': 3,
            'maxn': 6,
            'neg': 5,
            'loss': 'softmax',
            'verbose': 0
        }
        
        # Override with custom hyperparameters if provided
        if hyperparams:
            train_params.update(hyperparams)
            logger.info(f"Using custom hyperparameters: {hyperparams}")
        
        # Store hyperparameters in training status
        training_status["hyperparams_used"] = train_params.copy()
        
        # Train the model with or without autotune
        if validation_file_path:
            training_status["message"] = "Training with autotune (this may take several minutes)..."
            training_status["progress"] = -1  # Indeterminate progress
            await broadcast_training_status()
            
            # For autotune, use base parameters but let FastText optimize them
            autotune_params = {
                'input': combined_file,
                'autotuneValidationFile': validation_file_path,
                'autotuneDuration': 300,  # 5 minutes max
                'verbose': 0
            }
            
            model = fasttext.train_supervised(**autotune_params)
            training_status["autotune_used"] = True
            
            # Get the autotuned hyperparameters
            autotuned_params = {}
            try:
                if hasattr(model, 'f'):
                    args = model.f.getArgs()
                    autotuned_params = {
                        "lr": args.lr,
                        "dim": args.dim,
                        "ws": args.ws,
                        "epoch": args.epoch,
                        "minn": args.minn,
                        "maxn": args.maxn,
                        "neg": args.neg,
                        "loss": str(args.loss),
                        "bucket": args.bucket,
                        "minCount": args.minCount,
                        "thread": args.thread
                    }
                    training_status["hyperparams_used"] = autotuned_params
                    logger.info(f"Autotuned hyperparameters: {autotuned_params}")
            except Exception as e:
                logger.warning(f"Could not extract autotuned parameters: {e}")
                
        else:
            training_status["message"] = "Training model..."
            training_status["progress"] = -1  # Indeterminate progress
            await broadcast_training_status()
            
            model = fasttext.train_supervised(**train_params)
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
        
        # Notify main app of model change
        if model_update_callback:
            model_update_callback()
        
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

@router.post("/train", response_model=TrainResponse)
async def train_model(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    validation_file: UploadFile = File(None),
    hyperparams: str = Form(None)
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
        
        # Parse hyperparameters if provided
        parsed_hyperparams = None
        if hyperparams:
            try:
                parsed_hyperparams = json.loads(hyperparams)
                logger.info(f"Received hyperparameters: {parsed_hyperparams}")
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid hyperparameters JSON: {e}")
                parsed_hyperparams = None
        
        # Start training in background
        asyncio.create_task(train_model_background(temp_file_path, validation_file_path, parsed_hyperparams))
        
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
            autotune_used=bool(validation_file),
            hyperparams_used=parsed_hyperparams
        )
        
    except Exception as e:
        # Cleanup on error
        if 'temp_file_path' in locals():
            try:
                os.remove(temp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

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

@router.get("/models", response_model=ModelListResponse)
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

@router.post("/model/select")
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
        
        # Notify main app of model change
        if model_update_callback:
            model_update_callback()
        
        logger.info(f"Switched to model: {target_filename} (ID: {request.model_id})")
        
        return {
            "message": f"Successfully loaded model: {target_filename}",
            "model_id": request.model_id,
            "filename": target_filename,
            "model_path": new_model_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@router.get("/model/status")
async def model_status():
    """Get current model status"""
    current_model_name = os.path.basename(model_path) if model_path and current_model else None
    
    if current_model:
        # Extract hyperparameters from the loaded model
        hyperparams = {}
        try:
            if hasattr(current_model, 'f'):
                args = current_model.f.getArgs()
                
                hyperparams = {
                    "lr": args.lr,
                    "dim": args.dim,
                    "ws": args.ws,
                    "epoch": args.epoch,
                    "minn": args.minn,
                    "maxn": args.maxn,
                    "neg": args.neg,
                    "loss": str(args.loss),
                    "bucket": args.bucket,
                    "minCount": args.minCount,
                    "thread": args.thread
                }
            else:
                hyperparams = None
                
        except Exception as e:
            logger.warning(f"Could not extract hyperparameters from model: {e}")
            hyperparams = None
            
        return {
            "status": "loaded",
            "model_path": model_path,
            "model_name": current_model_name,
            "model_exists": os.path.exists(model_path) if model_path else False,
            "hyperparams": hyperparams
        }
    else:
        return {
            "status": "not_loaded",
            "model_path": model_path,
            "model_name": current_model_name,
            "model_exists": os.path.exists(model_path) if model_path else False,
            "hyperparams": None
        }

@router.websocket("/ws")
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