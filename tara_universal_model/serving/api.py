"""
FastAPI server for TARA Universal Model serving.
Provides REST API for model inference with health checks and monitoring.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from ..serving.model import TARAUniversalModel, ChatResponse
from ..utils.config import get_config, TARAConfig
from ..tts_integration import get_tts_manager, synthesize_tara_speech
from .gguf_model import TARAGGUFModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
tara_model: Optional[TARAUniversalModel] = None
tara_gguf: Optional[TARAGGUFModel] = None

# Rate limiting storage (in production, use Redis)
request_counts = {}
RATE_LIMIT_WINDOW = 3600  # 1 hour
RATE_LIMIT_REQUESTS = 100

class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    domain: Optional[str] = Field(None, description="Preferred domain")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    user_id: Optional[str] = Field(None, description="User identifier for rate limiting")

class ChatResponseModel(BaseModel):
    """Chat response model."""
    message: str
    confidence_score: float
    emotion_detected: Dict[str, Any]
    domain_used: str
    safety_check_passed: bool
    processing_time: float
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    model_loaded: bool
    supported_domains: List[str]
    version: str

class ModelStatus(BaseModel):
    """Model status response."""
    model_name: str
    domains_loaded: List[str]
    memory_usage: Optional[float]
    uptime: str
    total_requests: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI."""
    global tara_model, tara_gguf
    
    # Startup
    logger.info("Starting TARA Universal Model API server...")
    config = get_config()
    
    try:
        # Initialize TARA model
        tara_model = TARAUniversalModel(config)
        
        # Load base model
        logger.info("Loading base model...")
        tara_model.load_base_model()
        
        # Load domain adapters if available
        for domain in config.serving_config.preload_domains:
            try:
                tara_model.load_domain_adapter(domain)
                logger.info(f"Loaded {domain} domain adapter")
            except Exception as e:
                logger.warning(f"Could not load {domain} adapter: {e}")
        
        # Initialize GGUF model
        try:
            tara_gguf = TARAGGUFModel(config)
            logger.info("GGUF models initialized successfully")
        except Exception as e:
            logger.warning(f"GGUF models not available: {e}")
            tara_gguf = None
        
        # Store startup time for uptime calculation
        app.state.startup_time = datetime.now()
        app.state.request_count = 0
        
        logger.info("TARA Universal Model API server started successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize TARA model: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down TARA Universal Model API server...")

# Create FastAPI app
app = FastAPI(
    title="TARA Universal Model API",
    description="Privacy-first conversational AI with emotional intelligence and professional domain expertise",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure as needed
)

async def rate_limit_check(request: Request, user_id: Optional[str] = None) -> None:
    """Check rate limiting for requests."""
    config = get_config()
    
    if not config.serving_config.rate_limit_enabled:
        return
    
    # Use IP address if no user_id provided
    client_id = user_id or request.client.host
    current_time = time.time()
    
    # Clean old entries
    cutoff_time = current_time - RATE_LIMIT_WINDOW
    request_counts[client_id] = [
        timestamp for timestamp in request_counts.get(client_id, [])
        if timestamp > cutoff_time
    ]
    
    # Check rate limit
    if len(request_counts.get(client_id, [])) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    
    # Add current request
    if client_id not in request_counts:
        request_counts[client_id] = []
    request_counts[client_id].append(current_time)

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "TARA Universal Model API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global tara_model
    
    config = get_config()
    
    return HealthResponse(
        status="healthy" if tara_model else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=tara_model is not None,
        supported_domains=config.supported_domains,
        version="1.0.0"
    )

@app.get("/status", response_model=ModelStatus)
async def model_status():
    """Get detailed model status."""
    global tara_model
    
    if not tara_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Calculate uptime
    uptime_seconds = (datetime.now() - app.state.startup_time).total_seconds()
    uptime = str(timedelta(seconds=int(uptime_seconds)))
    
    return ModelStatus(
        model_name=tara_model.config.base_model_name,
        domains_loaded=list(tara_model.domain_adapters.keys()),
        memory_usage=None,  # Could implement memory monitoring
        uptime=uptime,
        total_requests=app.state.request_count
    )

@app.post("/chat", response_model=ChatResponseModel)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    _: None = Depends(lambda r=None, u=None: rate_limit_check(http_request, request.user_id))
):
    """Main chat endpoint."""
    global tara_model
    
    if not tara_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Process the chat request
        response = tara_model.process_user_input(request.message)
        
        # Increment request counter
        app.state.request_count += 1
        
        # Log request (background task)
        background_tasks.add_task(
            log_request,
            request.message,
            response,
            request.user_id
        )
        
        return ChatResponseModel(
            message=response.message,
            confidence_score=response.confidence_score,
            emotion_detected=response.emotion_detected,
            domain_used=response.domain_used,
            safety_check_passed=response.safety_check_passed,
            processing_time=response.processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/clear-conversation")
async def clear_conversation():
    """Clear conversation history."""
    global tara_model
    
    if not tara_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    tara_model.clear_conversation()
    return {"message": "Conversation cleared successfully"}

@app.get("/conversation-summary")
async def get_conversation_summary():
    """Get conversation summary."""
    global tara_model
    
    if not tara_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    summary = tara_model.get_conversation_summary()
    return summary

@app.post("/switch-domain")
async def switch_domain(domain: str):
    """Switch to a specific domain."""
    global tara_model
    
    if not tara_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    config = get_config()
    if domain not in config.supported_domains:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported domain. Choose from: {config.supported_domains}"
        )
    
    tara_model.current_domain = domain
    return {"message": f"Switched to {domain} domain"}

@app.get("/domains")
async def list_domains():
    """List available domains."""
    config = get_config()
    
    return {
        "supported_domains": config.supported_domains,
        "current_domain": tara_model.current_domain if tara_model else None
    }

@app.get("/metrics")
async def get_metrics():
    """Get API metrics."""
    return {
        "total_requests": app.state.request_count,
        "uptime": str(datetime.now() - app.state.startup_time),
        "model_loaded": tara_model is not None,
        "active_rate_limits": len(request_counts),
        "timestamp": datetime.now().isoformat()
    }

async def log_request(message: str, response: ChatResponse, user_id: Optional[str]):
    """Log request for monitoring (background task)."""
    config = get_config()
    
    # Only log if enabled and anonymized
    if config.security_config.log_conversations and config.security_config.anonymous_logging:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": "anonymous" if config.security_config.anonymous_logging else user_id,
            "domain": response.domain_used,
            "emotion": response.emotion_detected.get("primary_emotion"),
            "processing_time": response.processing_time,
            "safety_passed": response.safety_check_passed
        }
        
        # In production, send to monitoring system
        logger.info(f"Request logged: {log_entry}")

# TTS Endpoints
@app.post("/tts/synthesize")
async def synthesize_speech_endpoint(request: dict):
    """
    Synthesize speech from text.
    
    Request body:
    {
        "text": "Text to synthesize",
        "domain": "universal",  # optional
        "emotion": "happy",     # optional
        "system": "edge_tts"    # optional
    }
    """
    try:
        text = request.get("text", "")
        domain = request.get("domain", "universal")
        emotion = request.get("emotion")
        system = request.get("system")
        
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Synthesize speech
        success, audio_data, metadata = await synthesize_tara_speech(
            text=text,
            domain=domain,
            emotion=emotion
        )
        
        if not success:
            raise HTTPException(
                status_code=500, 
                detail=f"TTS synthesis failed: {metadata.get('error', 'Unknown error')}"
            )
        
        # Return audio data as base64
        import base64
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        
        return {
            "success": True,
            "audio_data": audio_b64,
            "metadata": metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tts/status")
async def tts_status():
    """Get TTS system status."""
    try:
        tts_manager = get_tts_manager()
        status = tts_manager.get_system_status()
        
        return {
            "success": True,
            "tts_status": status
        }
        
    except Exception as e:
        logger.error(f"TTS status error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/chat_with_voice")
async def chat_with_voice_endpoint(request: dict):
    """
    Chat with TARA and get both text and voice response.
    
    Request body:
    {
        "message": "User message",
        "domain": "universal",     # optional
        "include_voice": true,     # optional
        "voice_emotion": "happy"   # optional
    }
    """
    try:
        message = request.get("message", "")
        domain = request.get("domain", "universal")
        include_voice = request.get("include_voice", True)
        voice_emotion = request.get("voice_emotion")
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Get text response from TARA
        global tara_model
        if not tara_model:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        start_time = time.time()
        response = tara_model.process_user_input(message)
        processing_time = time.time() - start_time
        
        result = {
            "success": True,
            "text_response": response.message,
            "domain": response.domain_used,
            "confidence_score": response.confidence_score,
            "emotion_detected": response.emotion_detected,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add voice synthesis if requested
        if include_voice and response.message:
            success, audio_data, tts_metadata = await synthesize_tara_speech(
                text=response.message,
                domain=response.domain_used,
                emotion=voice_emotion
            )
            
            if success:
                import base64
                audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                result.update({
                    "voice_response": audio_b64,
                    "tts_metadata": tts_metadata
                })
            else:
                result["tts_error"] = tts_metadata.get("error", "Voice synthesis failed")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat with voice error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# GGUF Endpoints
@app.post("/gguf/chat")
async def gguf_chat(request: ChatRequest):
    """Chat with TARA using GGUF models."""
    if not tara_gguf:
        raise HTTPException(status_code=503, detail="GGUF models not available")
    
    try:
        response = tara_gguf.chat(
            message=request.message,
            domain=request.domain,
            model_preference=getattr(request, 'model_preference', None),
            max_tokens=getattr(request, 'max_tokens', 512),
            temperature=getattr(request, 'temperature', 0.7)
        )
        
        if "error" in response:
            raise HTTPException(status_code=500, detail=response["error"])
        
        return {
            "response": response["response"],
            "model": response["model"],
            "domain": response["domain"],
            "tokens_used": response.get("tokens_used", 0),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"GGUF chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gguf/models")
async def get_gguf_models():
    """Get information about available GGUF models."""
    if not tara_gguf:
        raise HTTPException(status_code=503, detail="GGUF models not available")
    
    try:
        return tara_gguf.get_model_info()
    except Exception as e:
        logger.error(f"Error getting GGUF model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gguf/chat_with_voice")
async def gguf_chat_with_voice(request: ChatRequest):
    """Chat with TARA using GGUF models and get voice response."""
    if not tara_gguf:
        raise HTTPException(status_code=503, detail="GGUF models not available")
    
    try:
        # Get text response from GGUF model
        chat_response = tara_gguf.chat(
            message=request.message,
            domain=request.domain,
            model_preference=getattr(request, 'model_preference', None),
            max_tokens=getattr(request, 'max_tokens', 512),
            temperature=getattr(request, 'temperature', 0.7)
        )
        
        if "error" in chat_response:
            raise HTTPException(status_code=500, detail=chat_response["error"])
        
        # Generate voice response
        voice_response = await synthesize_voice(
            text=chat_response["response"],
            domain=request.domain
        )
        
        return {
            "text_response": chat_response["response"],
            "model": chat_response["model"],
            "domain": request.domain,
            "tokens_used": chat_response.get("tokens_used", 0),
            "audio_file": voice_response["audio_file"],
            "voice_model": voice_response["voice_model"],
            "generation_time": voice_response["generation_time"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"GGUF chat with voice error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

def run_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
    """Run the FastAPI server."""
    config = get_config()
    
    uvicorn.run(
        "tara_universal_model.serving.api:app",
        host=host,
        port=port,
        workers=workers,
        log_level=config.serving_config.log_level.lower(),
        reload=False
    )

if __name__ == "__main__":
    run_server() 