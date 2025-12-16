"""
Database connection utilities
"""
import mysql.connector
from app.config import Config


def get_db_connection():
    """Get a database connection"""
    return mysql.connector.connect(**Config.get_db_config())


def init_tables():
    """Initialize all database tables"""
    from app.models.user import init_users_table
    from app.services.history import init_history_table
    
    init_users_table()
    init_history_table()

