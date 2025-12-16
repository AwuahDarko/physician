#!/usr/bin/env python3
"""
Entry point for the Medical RAG Application
"""
from app import create_app, init_app

# Create the Flask application
app = create_app()

if __name__ == '__main__':
    # Initialize the application (load models, setup database, etc.)
    if init_app():
        # Run the development server
        app.run(debug=True, port=5000)

