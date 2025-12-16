"""
User model and authentication utilities
"""
from functools import wraps
from flask import session, redirect, url_for, flash
from app.database import get_db_connection


def init_users_table():
    """Create users table if it doesn't exist"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(80) UNIQUE NOT NULL,
                email VARCHAR(120) UNIQUE NOT NULL,
                password_hash VARCHAR(256) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()
        print("✓ Users table ready")
        return True
    except Exception as e:
        print(f"✗ Error creating users table: {e}")
        return False


def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function


def get_current_user():
    """Get current logged in user"""
    if 'user_id' in session:
        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT id, username, email, created_at FROM users WHERE id = %s",
                (session['user_id'],)
            )
            user = cursor.fetchone()
            cursor.close()
            conn.close()
            return user
        except Exception:
            return None
    return None


def get_user_by_username_or_email(identifier):
    """Get user by username or email"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM users WHERE username = %s OR email = %s",
            (identifier, identifier)
        )
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        return user
    except Exception:
        return None


def user_exists(username, email):
    """Check if username or email already exists"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT id FROM users WHERE username = %s OR email = %s",
            (username, email)
        )
        existing = cursor.fetchone()
        cursor.close()
        conn.close()
        return existing is not None
    except Exception:
        return False


def create_user(username, email, password_hash):
    """Create a new user"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
            (username, email, password_hash)
        )
        conn.commit()
        user_id = cursor.lastrowid
        cursor.close()
        conn.close()
        return user_id
    except Exception as e:
        print(f"Error creating user: {e}")
        return None

