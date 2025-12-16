"""
Application configuration
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Base configuration"""
    
    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Database
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_USER = os.environ.get('DB_USER', 'root')
    DB_PASSWORD = os.environ.get('DB_PASSWORD', '')
    DB_NAME = os.environ.get('DB_NAME', 'medical_case_db')
    DB_PORT = int(os.environ.get('DB_PORT', 3307))
    
    # ML Models
    EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'pritamdeka/S-PubMedBert-MS-MARCO')
    CROSS_ENCODER_MODEL = os.environ.get('CROSS_ENCODER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
    LLM_MODEL = os.environ.get('LLM_MODEL', 'gemma:2b')
    
    # Retrieval settings
    RERANK_CANDIDATES = int(os.environ.get('RERANK_CANDIDATES', 10))
    TOP_K_RESULTS = int(os.environ.get('TOP_K_RESULTS', 5))
    SEMANTIC_WEIGHT = float(os.environ.get('SEMANTIC_WEIGHT', 0.7))
    BM25_WEIGHT = float(os.environ.get('BM25_WEIGHT', 0.3))
    
    # Hugging Face
    HUGGINGFACE_TIMEOUT = int(os.environ.get('HUGGINGFACE_HUB_READ_TIMEOUT', 120))
    
    @classmethod
    def get_db_config(cls):
        """Get database configuration dictionary"""
        return {
            'host': cls.DB_HOST,
            'user': cls.DB_USER,
            'password': cls.DB_PASSWORD,
            'database': cls.DB_NAME,
            'port': cls.DB_PORT
        }


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False


# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

