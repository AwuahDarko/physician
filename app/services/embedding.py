"""
ML Embedding models service
"""
import os
from app.config import Config

# Set timeout before importing models
os.environ['HUGGINGFACE_HUB_READ_TIMEOUT'] = str(Config.HUGGINGFACE_TIMEOUT)

from sentence_transformers import SentenceTransformer, CrossEncoder

# Global model instances
_embedding_model = None
_cross_encoder = None


def init_models():
    """Initialize ML models"""
    global _embedding_model, _cross_encoder
    
    # Initialize embedding model (medical-domain)
    print("Loading medical embedding model (this may take a moment on first run)...")
    try:
        _embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        print("✓ Medical embedding model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading embedding model: {e}")
        _embedding_model = None
    
    # Initialize cross-encoder for re-ranking
    print("Loading cross-encoder model for re-ranking...")
    try:
        _cross_encoder = CrossEncoder(Config.CROSS_ENCODER_MODEL)
        print("✓ Cross-encoder model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading cross-encoder: {e}")
        _cross_encoder = None
    
    return _embedding_model is not None


def get_embedding_model():
    """Get the embedding model instance"""
    return _embedding_model


def get_cross_encoder():
    """Get the cross-encoder instance"""
    return _cross_encoder


def encode_text(text):
    """Encode text using the embedding model"""
    if _embedding_model is None:
        return None
    return _embedding_model.encode(text)


def rerank_pairs(pairs):
    """Rerank text pairs using cross-encoder"""
    if _cross_encoder is None:
        return None
    return _cross_encoder.predict(pairs)

