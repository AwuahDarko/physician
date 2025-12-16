"""
Medical RAG Application Factory
"""
from flask import Flask
from app.config import Config


def create_app(config_class=Config):
    """Application factory"""
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config_class)
    app.secret_key = config_class.SECRET_KEY
    
    # Register blueprints
    from app.routes.auth import auth_bp
    from app.routes.main import main_bp
    from app.routes.api import api_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)
    
    return app


def init_app():
    """Initialize the application with all required setup"""
    print("=" * 60)
    print("Starting Medical RAG Application...")
    print("=" * 60)
    
    # Initialize ML models
    from app.services.embedding import init_models
    if not init_models():
        print("⚠ Warning: Embedding model not loaded. Please check your setup.")
    
    # Initialize database tables
    print("\nInitializing database tables...")
    from app.database import init_tables
    init_tables()
    
    # Load diseases
    print("\nLoading diseases from database...")
    from app.services.retrieval import load_diseases_from_db
    if load_diseases_from_db():
        print("\n✓ Application ready!")
        print("Make sure Ollama is running: ollama serve")
        print("=" * 60)
        return True
    else:
        print("\n✗ Failed to load diseases. Check your database connection.")
        print("=" * 60)
        return False

