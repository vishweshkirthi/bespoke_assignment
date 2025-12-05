#!/usr/bin/env python3
"""
Train API - Model training endpoints
"""

from fastapi import APIRouter, HTTPException, File, UploadFile, BackgroundTasks, WebSocket, WebSocketDisconnect, Form, Request
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
from concurrent.futures import ThreadPoolExecutor
import threading

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

# Session manager and global variables that will be set from main app
session_manager = None
negative_file_path = None

# Thread pool for training operations (4 concurrent workers)
training_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="training")

# Training queue to handle sequential requests
training_queue = asyncio.Queue()

# Global WebSocket connections (will be managed per session)
# active_connections: List[WebSocket] = []  # Now managed per session

# Training status
training_status = {
    "is_training": False,
    "progress": 0,
    "message": "",
    "total_examples": 0,
    "queue_position": 0,
    "queue_size": 0
}

# Track active training jobs
_active_training_jobs = 0
_max_concurrent_jobs = 4

# Track individual training job details
_training_jobs = {}  # {job_id: {uuid, status, progress, message, started_at, completed_at}}

# Lock for thread-safe model updates
model_lock = threading.RLock()

def set_session_manager(sm, neg_file_path):
    """Set session manager and globals from main app"""
    global session_manager, negative_file_path
    session_manager = sm
    negative_file_path = neg_file_path

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

async def broadcast_training_status_to_session(session_id: str, status_data: dict):
    """Broadcast training status to WebSockets for a specific session"""
    if not session_manager:
        return
        
    connections = session_manager.get_websockets(session_id)
    if connections:
        message = json.dumps(status_data)
        disconnected = []
        for connection in connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            session_manager.remove_websocket(session_id, conn)

def _train_fasttext_sync(combined_file: str, validation_file_path: str = None, hyperparams: dict = None):
    """Synchronous FastText training function to run in thread pool"""
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
    
    # Train the model with or without autotune
    if validation_file_path:
        # For autotune, use base parameters but let FastText optimize them
        autotune_params = {
            'input': combined_file,
            'autotuneValidationFile': validation_file_path,
            'autotuneDuration': 300,  # 5 minutes max
            'verbose': 0
        }
        
        model = fasttext.train_supervised(**autotune_params)
        autotune_used = True
        
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
        except Exception as e:
            logger.warning(f"Could not extract autotuned parameters: {e}")
        
        final_params = autotuned_params if autotuned_params else "autotune_optimized"
    else:
        model = fasttext.train_supervised(**train_params)
        autotune_used = False
        final_params = train_params.copy()
    
    return model, autotune_used, final_params

async def train_model_background(session_id: str, positive_file_path: str, validation_file_path: str = None, hyperparams: dict = None, model_uuid: str = None, model_filename: str = None):
    """Train model in background with session-specific progress updates"""
    if not session_manager:
        logger.error("Session manager not available")
        return
    
    try:
        # Create session-specific training status
        session_training_status = {
            "is_training": True,
            "progress": 0,
            "message": "Preparing data...",
            "model_uuid": model_uuid,
            "session_id": session_id,
            "queue_size": training_queue.qsize(),
            "active_jobs": _active_training_jobs
        }
        await broadcast_training_status_to_session(session_id, session_training_status)
        
        # Create temporary combined training file with unique name
        combined_file = f"temp_combined_train_{model_uuid[:8]}.txt"
        total_examples = combine_datasets(positive_file_path, negative_file_path, combined_file)
        
        session_training_status["total_examples"] = total_examples
        session_training_status["message"] = "Starting training..."
        session_training_status["progress"] = 10
        await broadcast_training_status_to_session(session_id, session_training_status)
        
        logger.info("Starting FastText training...")
        
        # Store hyperparameters in training status
        initial_params = {
            'lr': 0.1,
            'epoch': 20,
            'wordNgrams': 2,
            'dim': 100,
            'ws': 5,
            'minCount': 5,
            'minn': 3,
            'maxn': 6,
            'neg': 5,
            'loss': 'softmax'
        }
        
        if hyperparams:
            initial_params.update(hyperparams)
            logger.info(f"Using custom hyperparameters: {hyperparams}")
        
        session_training_status["hyperparams_used"] = initial_params.copy()
        
        # Train the model asynchronously using thread pool
        if validation_file_path:
            session_training_status["message"] = "Training with autotune (this may take several minutes)..."
            session_training_status["progress"] = -1  # Indeterminate progress
            await broadcast_training_status_to_session(session_id, session_training_status)
        else:
            session_training_status["message"] = "Training model..."
            session_training_status["progress"] = -1  # Indeterminate progress
            await broadcast_training_status_to_session(session_id, session_training_status)
        
        # Run training in thread pool to prevent blocking
        loop = asyncio.get_event_loop()
        model, autotune_used, final_params = await loop.run_in_executor(
            training_executor,
            _train_fasttext_sync,
            combined_file,
            validation_file_path,
            hyperparams
        )
        
        session_training_status["autotune_used"] = autotune_used
        session_training_status["hyperparams_used"] = final_params
        
        if autotune_used and isinstance(final_params, dict):
            logger.info(f"Autotuned hyperparameters: {final_params}")
        
        session_training_status["message"] = "Saving model..."
        session_training_status["progress"] = 80
        await broadcast_training_status_to_session(session_id, session_training_status)
        
        # Use the provided UUID and filename, or generate new ones if not provided
        if not model_uuid:
            model_uuid = session_training_status.get("model_uuid", str(uuid.uuid4()))
        if not model_filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = session_training_status.get("model_filename", f"model_{timestamp}_{model_uuid[:8]}.bin")
        full_model_path = f"models/{model_filename}"
        
        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)
        
        # Save the model with UUID-based name
        model.save_model(full_model_path)
        
        # Update session-specific model
        session_manager.set_model(session_id, full_model_path)
        
        # Save model metadata
        metadata = {
            "id": model_uuid,
            "filename": model_filename,
            "created_at": datetime.now().isoformat(),
            "training_examples": session_training_status.get("total_examples", 0),
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
        
        session_training_status["message"] = "Training completed!"
        session_training_status["progress"] = 100
        session_training_status["is_training"] = False
        await broadcast_training_status_to_session(session_id, session_training_status)
        
        logger.info(f"Training completed for session {session_id[:8]}! Model saved to {full_model_path}")
        
        # Reset status for this specific training session
        await asyncio.sleep(3)
        
        # Send completion status for this specific model to session
        completion_status = {
            "is_training": False,
            "progress": 100,
            "message": "Training completed!",
            "model_uuid": model_uuid,
            "queue_size": training_queue.qsize(),
            "active_jobs": _active_training_jobs,
            "session_id": session_id
        }
        
        # Broadcast completion to session
        await broadcast_training_status_to_session(session_id, completion_status)
        
    except Exception as e:
        logger.error(f"Training failed for session {session_id[:8]}: {e}")
        error_status = {
            "is_training": False,
            "message": f"Training failed: {str(e)}",
            "progress": 0,
            "session_id": session_id
        }
        await broadcast_training_status_to_session(session_id, error_status)
    
    finally:
        # Cleanup temporary files with error handling
        try:
            if os.path.exists(combined_file):
                os.remove(combined_file)
        except Exception as e:
            logger.warning(f"Could not remove combined file {combined_file}: {e}")
        
        try:
            if os.path.exists(positive_file_path):
                os.remove(positive_file_path)
        except Exception as e:
            logger.warning(f"Could not remove positive file {positive_file_path}: {e}")
        
        try:
            if validation_file_path and os.path.exists(validation_file_path):
                os.remove(validation_file_path)
        except Exception as e:
            logger.warning(f"Could not remove validation file {validation_file_path}: {e}")

async def process_single_training_request(training_request):
    """Process a single training request and manage concurrent job count"""
    global _active_training_jobs, _training_jobs
    
    job_id = training_request["model_uuid"]
    session_id = training_request["session_id"]
    
    # Add job to global tracking
    _training_jobs[job_id] = {
        "uuid": job_id,
        "status": "starting",
        "progress": 0,
        "message": "Initializing training...",
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "filename": training_request["model_filename"],
        "session_id": session_id
    }
    
    # Add job to session tracking if session manager is available
    if session_manager:
        session_manager.add_training_job(session_id, job_id, _training_jobs[job_id].copy())
    
    try:
        # Update job status
        _training_jobs[job_id]["status"] = "training"
        _training_jobs[job_id]["message"] = "Training in progress..."
        
        # Update session job tracking
        if session_manager:
            session_manager.update_training_job(session_id, job_id, {
                "status": "training",
                "message": "Training in progress..."
            })
        
        # Process the training request with session ID
        await train_model_background(
            session_id,
            training_request["positive_file_path"],
            training_request["validation_file_path"],
            training_request["hyperparams"],
            training_request["model_uuid"],
            training_request["model_filename"]
        )
        
        # Mark as completed
        _training_jobs[job_id]["status"] = "completed"
        _training_jobs[job_id]["progress"] = 100
        _training_jobs[job_id]["message"] = "Training completed successfully"
        _training_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
        # Update session job tracking
        if session_manager:
            session_manager.update_training_job(session_id, job_id, {
                "status": "completed",
                "progress": 100,
                "message": "Training completed successfully",
                "completed_at": datetime.now().isoformat()
            })
        
        logger.info(f"Training completed for session {session_id[:8]} model {training_request['model_uuid'][:8]}")
        
    except Exception as e:
        # Mark as failed
        _training_jobs[job_id]["status"] = "failed"
        _training_jobs[job_id]["message"] = f"Training failed: {str(e)}"
        _training_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
        # Update session job tracking
        if session_manager:
            session_manager.update_training_job(session_id, job_id, {
                "status": "failed",
                "message": f"Training failed: {str(e)}",
                "completed_at": datetime.now().isoformat()
            })
        
        logger.error(f"Error processing training request for session {session_id[:8]} model {training_request.get('model_uuid', 'unknown')}: {e}")
    
    finally:
        # Decrement active job count
        _active_training_jobs -= 1
        
        # Check if there are queued requests to process
        if not training_queue.empty() and _active_training_jobs < _max_concurrent_jobs:
            try:
                # Get next request from queue
                next_request = await training_queue.get()
                _active_training_jobs += 1
                
                # Process the next request
                asyncio.create_task(process_single_training_request(next_request))
                training_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error starting next training from queue: {e}")
        
        # Clean up job from tracking after delay (keep for 5 minutes for viewing)
        await asyncio.sleep(300)  # 5 minutes
        if job_id in _training_jobs:
            del _training_jobs[job_id]
        
        # Clean up from session tracking as well
        if session_manager:
            session_manager.remove_training_job(session_id, job_id)

@router.post("/train", response_model=TrainResponse)
async def train_model(
    request: Request,
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    validation_file: UploadFile = File(None),
    hyperparams: str = Form(None)
):
    """
    Upload positive examples file and train the model for the current session.
    
    Expected file format: Each line should be "__label__high [text]"
    Negative examples are automatically loaded from data/train_negative.txt
    
    Optional validation_file: If provided, FastText autotune will be used to optimize hyperparameters
    """
    
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not available")
    
    session_id = request.state.session_id
    
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
        
        # Generate UUID for the model that will be created
        model_uuid = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        autotune_suffix = "_autotune" if validation_file else ""
        model_filename = f"model_{timestamp}_{model_uuid[:8]}{autotune_suffix}.bin"
        
        # Add training request to queue with session ID
        training_request = {
            "session_id": session_id,
            "positive_file_path": temp_file_path,
            "validation_file_path": validation_file_path,
            "hyperparams": parsed_hyperparams,
            "model_uuid": model_uuid,
            "model_filename": model_filename
        }
        
        await training_queue.put(training_request)
        
        # Start training immediately if under concurrent limit, otherwise queue
        global _active_training_jobs
        if _active_training_jobs < _max_concurrent_jobs:
            _active_training_jobs += 1
            asyncio.create_task(process_single_training_request(training_request))
        # If at capacity, the request stays in queue and will be processed when a slot frees up
        
        autotune_message = " with autotune" if validation_file else ""
        
        if _active_training_jobs <= _max_concurrent_jobs:
            status_message = f"Training{autotune_message} started. Model UUID: {model_uuid}"
        else:
            queue_position = training_queue.qsize()
            status_message = f"Training{autotune_message} queued (position {queue_position}). Model UUID: {model_uuid}"
        
        return TrainResponse(
            message=status_message,
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
async def list_models(request: Request):
    """Get list of available models with metadata"""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not available")
    
    session_id = request.state.session_id
    current_model = session_manager.get_model(session_id)
    model_path = session_manager.get_model_path(session_id)
    
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
async def select_model(model_request: ModelSelectRequest, request: Request):
    """Select a model to load by ID for current session"""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not available")
    
    session_id = request.state.session_id
    
    # Find model by ID
    models_dir = "models"
    target_filename = None
    
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.bin'):
                try:
                    model_info = get_model_info(file, models_dir)
                    if model_info.id == model_request.model_id:
                        target_filename = file
                        break
                except Exception:
                    continue
    
    if not target_filename:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_request.model_id} not found")
    
    new_model_path = f"models/{target_filename}"
    
    try:
        # Set model for session using session manager
        success = session_manager.set_model(session_id, new_model_path)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load model for session")
        
        logger.info(f"Session {session_id[:8]} switched to model: {target_filename} (ID: {model_request.model_id})")
        
        return {
            "message": f"Successfully loaded model for session: {target_filename}",
            "model_id": model_request.model_id,
            "filename": target_filename,
            "model_path": new_model_path,
            "session_id": session_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@router.get("/training/queue")
async def training_queue_status(request: Request):
    """Get current training queue status"""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not available")
    
    session_id = request.state.session_id
    session_jobs = session_manager.get_training_jobs(session_id)
    
    # Count active jobs for this session
    active_session_jobs = len([job for job in session_jobs.values() if job.get('status') in ['starting', 'training']])
    
    global _active_training_jobs
    return {
        "queue_size": training_queue.qsize(),
        "active_jobs": _active_training_jobs,
        "session_active_jobs": active_session_jobs,
        "max_concurrent_jobs": _max_concurrent_jobs,
        "is_training": _active_training_jobs > 0,
        "session_is_training": active_session_jobs > 0,
        "accepts_new_requests": True,  # Always accept new requests
        "slots_available": _max_concurrent_jobs - _active_training_jobs,
        "session_id": session_id
    }

@router.get("/training/jobs")
async def get_training_jobs(request: Request):
    """Get current and recent training jobs for the session"""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not available")
    
    session_id = request.state.session_id
    session_jobs = session_manager.get_training_jobs(session_id)
    
    jobs_list = []
    for job_id, job_info in session_jobs.items():
        jobs_list.append({
            "uuid": job_info["uuid"],
            "short_uuid": job_info["uuid"][:8],
            "filename": job_info["filename"],
            "status": job_info["status"],
            "progress": job_info["progress"],
            "message": job_info["message"],
            "started_at": job_info["started_at"],
            "completed_at": job_info["completed_at"]
        })
    
    # Sort by started_at (newest first)
    jobs_list.sort(key=lambda x: x["started_at"], reverse=True)
    
    return {
        "jobs": jobs_list,
        "total_jobs": len(jobs_list),
        "active_count": len([j for j in jobs_list if j["status"] in ["starting", "training"]]),
        "completed_count": len([j for j in jobs_list if j["status"] == "completed"]),
        "failed_count": len([j for j in jobs_list if j["status"] == "failed"])
    }


@router.get("/model/status")
async def model_status(request: Request):
    """Get current model status for session"""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not available")
    
    session_id = request.state.session_id
    current_model = session_manager.get_model(session_id)
    model_path = session_manager.get_model_path(session_id)
    
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
            logger.warning(f"Could not extract hyperparameters from session {session_id[:8]} model: {e}")
            hyperparams = None
            
        return {
            "status": "loaded",
            "model_path": model_path,
            "model_name": current_model_name,
            "model_exists": os.path.exists(model_path) if model_path else False,
            "hyperparams": hyperparams,
            "session_id": session_id
        }
    else:
        return {
            "status": "not_loaded",
            "model_path": model_path,
            "model_name": current_model_name,
            "model_exists": os.path.exists(model_path) if model_path else False,
            "hyperparams": None,
            "session_id": session_id
        }

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for session-specific training progress"""
    await websocket.accept()
    
    # Extract session ID from query parameters or headers
    session_id = None
    try:
        # Try to get session_id from query parameters
        query_params = dict(websocket.query_params)
        session_id = query_params.get('session_id')
        
        if not session_id:
            # Try to get from headers (if sent by client)
            session_id = websocket.headers.get('x-session-id')
        
        if not session_id or not session_manager:
            await websocket.close(code=4000, reason="No session ID provided or session manager unavailable")
            return
        
        # Verify session exists, create if needed
        session_id, session_data = session_manager.get_or_create_session(session_id)
        
        # Add websocket to session
        session_manager.add_websocket(session_id, websocket)
        
        # Send initial status
        initial_status = {
            "session_id": session_id,
            "is_training": False,
            "message": "Connected to session-specific training updates",
            "progress": 0,
            "queue_size": training_queue.qsize(),
            "active_jobs": _active_training_jobs
        }
        await websocket.send_text(json.dumps(initial_status))
        
        # Keep connection alive and handle ping/pong
        while True:
            try:
                message = await websocket.receive_text()
                # Echo back or handle specific messages if needed
                if message == "ping":
                    await websocket.send_text("pong")
            except:
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if session_id and session_manager:
            session_manager.remove_websocket(session_id, websocket)