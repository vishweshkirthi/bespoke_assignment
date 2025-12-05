#!/usr/bin/env python3
"""
Session Management for per-user model isolation
"""

import uuid
import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import fasttext
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages user sessions and their isolated model states"""
    
    def __init__(self, session_timeout_minutes: int = 120, cleanup_interval_minutes: int = 30):
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.cleanup_interval = timedelta(minutes=cleanup_interval_minutes)
        logger.info(f"Session manager initialized: timeout={session_timeout_minutes}min, cleanup_interval={cleanup_interval_minutes}min")
        
        # Session storage: {session_id: session_data}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        # Model cache to avoid loading same model multiple times: {model_path: model_instance}
        self.model_cache: Dict[str, fasttext.FastText] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def create_session(self, provided_session_id: str = None) -> str:
        """Create a new session and return session ID"""
        # Use provided session ID if given, otherwise generate new UUID
        session_id = provided_session_id if provided_session_id else str(uuid.uuid4())
        
        with self.lock:
            self.sessions[session_id] = {
                'id': session_id,
                'created_at': datetime.now(),
                'last_accessed': datetime.now(),
                'model': None,
                'model_path': None,
                'training_jobs': {},  # session-specific training jobs
                'websocket_connections': []  # session-specific websocket connections
            }
        
        logger.info(f"Created new session: {session_id} (provided: {provided_session_id is not None})")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data and update last accessed time"""
        if not session_id:
            logger.info("get_session: session_id is None or empty")
            return None
            
        with self.lock:
            logger.info(f"get_session: looking for {session_id} in {list(self.sessions.keys())}")
            if session_id in self.sessions:
                self.sessions[session_id]['last_accessed'] = datetime.now()
                logger.info(f"get_session: found session {session_id}")
                return self.sessions[session_id]
            logger.info(f"get_session: session {session_id} not found")
            return None
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> tuple[str, Dict[str, Any]]:
        """Get existing session or create new one"""
        logger.info(f"get_or_create_session called with session_id: {session_id}")
        
        if session_id:
            session = self.get_session(session_id)
            if session:
                logger.info(f"Found existing session: {session_id}")
                return session_id, session
            else:
                logger.info(f"Session {session_id} not found in {list(self.sessions.keys())}")
                # Create new session using the provided session_id
                new_session_id = self.create_session(session_id)
                logger.info(f"Created new session with provided ID: {new_session_id}")
                return new_session_id, self.get_session(new_session_id)
        
        # Create new session with generated UUID
        new_session_id = self.create_session()
        logger.info(f"Created new session with generated ID: {new_session_id}")
        return new_session_id, self.get_session(new_session_id)
    
    def set_model(self, session_id: str, model_path: str) -> bool:
        """Set model for a session, using cache if possible"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        try:
            with self.lock:
                # Check if model is already cached
                if model_path in self.model_cache:
                    model = self.model_cache[model_path]
                    logger.info(f"Using cached model for session {session_id[:8]}: {model_path}")
                else:
                    # Load new model and cache it
                    model = fasttext.load_model(model_path)
                    self.model_cache[model_path] = model
                    logger.info(f"Loaded and cached new model for session {session_id[:8]}: {model_path}")
                
                # Set model for session
                session['model'] = model
                session['model_path'] = model_path
                
            return True
        except Exception as e:
            logger.error(f"Failed to set model for session {session_id[:8]}: {e}")
            return False
    
    def get_model(self, session_id: str) -> Optional[fasttext.FastText]:
        """Get model for a session"""
        session = self.get_session(session_id)
        if session:
            return session.get('model')
        return None
    
    def get_model_path(self, session_id: str) -> Optional[str]:
        """Get model path for a session"""
        session = self.get_session(session_id)
        if session:
            return session.get('model_path')
        return None
    
    def add_websocket(self, session_id: str, websocket):
        """Add websocket connection to session"""
        session = self.get_session(session_id)
        if session:
            with self.lock:
                if websocket not in session['websocket_connections']:
                    session['websocket_connections'].append(websocket)
                    logger.info(f"Added websocket to session {session_id[:8]}")
    
    def remove_websocket(self, session_id: str, websocket):
        """Remove websocket connection from session"""
        session = self.get_session(session_id)
        if session:
            with self.lock:
                if websocket in session['websocket_connections']:
                    session['websocket_connections'].remove(websocket)
                    logger.info(f"Removed websocket from session {session_id[:8]}")
    
    def get_websockets(self, session_id: str) -> list:
        """Get all websocket connections for a session"""
        session = self.get_session(session_id)
        if session:
            return session['websocket_connections'].copy()
        return []
    
    def add_training_job(self, session_id: str, job_id: str, job_data: Dict[str, Any]):
        """Add training job to session"""
        session = self.get_session(session_id)
        if session:
            with self.lock:
                session['training_jobs'][job_id] = job_data
                logger.info(f"Added training job {job_id[:8]} to session {session_id[:8]}")
    
    def get_training_jobs(self, session_id: str) -> Dict[str, Any]:
        """Get all training jobs for a session"""
        session = self.get_session(session_id)
        if session:
            return session['training_jobs'].copy()
        return {}
    
    def update_training_job(self, session_id: str, job_id: str, updates: Dict[str, Any]):
        """Update training job data"""
        session = self.get_session(session_id)
        if session and job_id in session['training_jobs']:
            with self.lock:
                session['training_jobs'][job_id].update(updates)
    
    def remove_training_job(self, session_id: str, job_id: str):
        """Remove training job from session"""
        session = self.get_session(session_id)
        if session and job_id in session['training_jobs']:
            with self.lock:
                del session['training_jobs'][job_id]
                logger.info(f"Removed training job {job_id[:8]} from session {session_id[:8]}")
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions and clean up resources"""
        current_time = datetime.now()
        expired_sessions = []
        
        with self.lock:
            logger.info(f"Cleanup check: {len(self.sessions)} active sessions")
            for session_id, session in self.sessions.items():
                time_since_access = current_time - session['last_accessed']
                logger.info(f"Session {session_id[:8]}: last accessed {time_since_access} ago")
                if time_since_access > self.session_timeout:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                logger.info(f"Cleaning up expired session: {session_id[:8]}")
                # Close any open websocket connections
                for ws in self.sessions[session_id]['websocket_connections']:
                    try:
                        ws.close()
                    except:
                        pass
                
                del self.sessions[session_id]
        
        # Clean up unused models from cache
        self._cleanup_unused_models()
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        else:
            logger.info("No expired sessions to clean up")
    
    def _cleanup_unused_models(self):
        """Remove models from cache that are not used by any active session"""
        with self.lock:
            used_models = set()
            for session in self.sessions.values():
                if session.get('model_path'):
                    used_models.add(session['model_path'])
            
            unused_models = []
            for model_path in list(self.model_cache.keys()):
                if model_path not in used_models:
                    unused_models.append(model_path)
            
            for model_path in unused_models:
                del self.model_cache[model_path]
                logger.info(f"Removed unused model from cache: {model_path}")
    
    def _start_cleanup_thread(self):
        """Start background thread for session cleanup"""
        def cleanup_worker():
            while True:
                time.sleep(self.cleanup_interval.total_seconds())
                try:
                    self.cleanup_expired_sessions()
                except Exception as e:
                    logger.error(f"Error during session cleanup: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logger.info("Started session cleanup thread")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session manager statistics"""
        with self.lock:
            active_sessions = len(self.sessions)
            cached_models = len(self.model_cache)
            total_training_jobs = sum(len(s['training_jobs']) for s in self.sessions.values())
            total_websockets = sum(len(s['websocket_connections']) for s in self.sessions.values())
            
            return {
                'active_sessions': active_sessions,
                'cached_models': cached_models,
                'total_training_jobs': total_training_jobs,
                'total_websockets': total_websockets,
                'session_timeout_minutes': self.session_timeout.total_seconds() / 60,
                'cleanup_interval_minutes': self.cleanup_interval.total_seconds() / 60
            }

# Global session manager instance
session_manager = SessionManager()