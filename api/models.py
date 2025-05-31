"""
API data models for the Hindi tokenizer API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class TokenizeRequest(BaseModel):
    """Request model for tokenization."""
    text: str = Field(..., description="Hindi text to tokenize")
    return_pieces: bool = Field(False, description="Whether to return token pieces instead of IDs")


class TokenizeResponse(BaseModel):
    """Response model for tokenization."""
    tokens: List[int] = Field([], description="List of token IDs")
    pieces: Optional[List[str]] = Field(None, description="List of token pieces")
    text: str = Field(..., description="Original text")


class DetokenizeRequest(BaseModel):
    """Request model for detokenization."""
    tokens: List[int] = Field(..., description="List of token IDs to detokenize")


class DetokenizeResponse(BaseModel):
    """Response model for detokenization."""
    text: str = Field(..., description="Detokenized text")
    tokens: List[int] = Field(..., description="Original tokens")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    model_name: str = Field(..., description="Name of the model")
    vocab_size: int = Field(..., description="Size of the vocabulary")
    model_type: str = Field(..., description="Type of the model (unigram or BPE)")
    version: str = Field(..., description="Version of the tokenizer")
