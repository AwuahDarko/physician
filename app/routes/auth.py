"""
Authentication routes
"""
from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash

from app.models.user import get_user_by_username_or_email, user_exists, create_user

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Login page and handler"""
    if 'user_id' in session:
        return redirect(url_for('main.index'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not username or not password:
            flash('Please fill in all fields.', 'error')
            return render_template('login.html')
        
        try:
            user = get_user_by_username_or_email(username)
            
            if user and check_password_hash(user['password_hash'], password):
                session['user_id'] = user['id']
                session['username'] = user['username']
                flash(f'Welcome back, {user["username"]}!', 'success')
                return redirect(url_for('main.index'))
            else:
                flash('Invalid username or password.', 'error')
        except Exception as e:
            flash('An error occurred. Please try again.', 'error')
            print(f"Login error: {e}")
    
    return render_template('login.html')


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page and handler"""
    if 'user_id' in session:
        return redirect(url_for('main.index'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Validation
        errors = []
        if not username or len(username) < 3:
            errors.append('Username must be at least 3 characters.')
        if not email or '@' not in email:
            errors.append('Please enter a valid email address.')
        if not password or len(password) < 6:
            errors.append('Password must be at least 6 characters.')
        if password != confirm_password:
            errors.append('Passwords do not match.')
        
        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('register.html')
        
        try:
            if user_exists(username, email):
                flash('Username or email already registered.', 'error')
                return render_template('register.html')
            
            # Create new user
            password_hash = generate_password_hash(password)
            user_id = create_user(username, email, password_hash)
            
            if user_id:
                # Auto-login after registration
                session['user_id'] = user_id
                session['username'] = username
                flash(f'Account created successfully! Welcome, {username}!', 'success')
                return redirect(url_for('main.index'))
            else:
                flash('Failed to create account. Please try again.', 'error')
            
        except Exception as e:
            flash('An error occurred. Please try again.', 'error')
            print(f"Registration error: {e}")
    
    return render_template('register.html')


@auth_bp.route('/logout')
def logout():
    """Logout handler"""
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('auth.login'))

