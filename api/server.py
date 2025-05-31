"""
FastAPI server for the Hindi tokenizer API.
"""

import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from hindi_tokenizer.core import HindiTokenizer
from hindi_tokenizer import __version__
from api.models import (
    TokenizeRequest, TokenizeResponse,
    DetokenizeRequest, DetokenizeResponse,
    ModelInfoResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Hindi Tokenizer API",
    description="API for Hindi SentencePiece tokenizer",
    version=__version__,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global tokenizer instance
tokenizer = None
model_info = {
    "model_name": "",
    "vocab_size": 0,
    "model_type": "",
    "version": __version__
}


@app.on_event("startup")
async def startup_event():
    """Load the tokenizer model on startup."""
    global tokenizer, model_info
    
    model_path = os.environ.get("TOKENIZER_MODEL_PATH")
    if not model_path:
        logger.warning("TOKENIZER_MODEL_PATH environment variable not set. API will start but tokenizer is not loaded.")
        return
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    try:
        tokenizer = HindiTokenizer(model_path=model_path)
        logger.info(f"Tokenizer model loaded from {model_path}")
        
        # Update model info
        model_info["model_name"] = os.path.basename(model_path)
        # In a real implementation, we would extract these from the model
        model_info["vocab_size"] = 32000  # Placeholder
        model_info["model_type"] = "unigram"  # Placeholder
    except Exception as e:
        logger.error(f"Failed to load tokenizer model: {e}")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "name": "Hindi Tokenizer API",
        "version": __version__,
        "status": "active",
        "model_loaded": tokenizer is not None
    }


@app.get("/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model."""
    if not tokenizer:
        raise HTTPException(status_code=503, detail="Tokenizer model not loaded")
    return model_info


@app.post("/tokenize", response_model=TokenizeResponse)
async def tokenize(request: TokenizeRequest):
    """Tokenize Hindi text."""
    if not tokenizer:
        raise HTTPException(status_code=503, detail="Tokenizer model not loaded")
    
    try:
        tokens = tokenizer.encode(request.text)
        response = {"text": request.text, "tokens": tokens}
        
        if request.return_pieces:
            pieces = tokenizer.encode_as_pieces(request.text)
            response["pieces"] = pieces
        
        return response
    except Exception as e:
        logger.error(f"Tokenization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detokenize", response_model=DetokenizeResponse)
async def detokenize(request: DetokenizeRequest):
    """Detokenize token IDs to Hindi text."""
    if not tokenizer:
        raise HTTPException(status_code=503, detail="Tokenizer model not loaded")
    
    try:
        text = tokenizer.decode(request.tokens)
        return {"text": text, "tokens": request.tokens}
    except Exception as e:
        logger.error(f"Detokenization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def start_server(host="0.0.0.0", port=8000, model_path=None):
    """Start the API server."""
    if model_path:
        os.environ["TOKENIZER_MODEL_PATH"] = model_path
    
    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hindi Tokenizer API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--model", help="Path to the tokenizer model file")
    
    args = parser.parse_args()
    start_server(host=args.host, port=args.port, model_path=args.model)
