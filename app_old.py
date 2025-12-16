import mysql.connector
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import ollama
import json
from datetime import datetime
import os
import re


# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production-12345')

# MySQL database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',  # Change this
    'password': '',  # Change this
    'database': 'medical_case_db',  # Change this
    'port': 3307
}
# Increase timeout for model download
os.environ['HUGGINGFACE_HUB_READ_TIMEOUT'] = '120'


# ============== Authentication Functions ==============

def init_users_table():
    """Create users table if it doesn't exist"""
    try:
        conn = mysql.connector.connect(**db_config)
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
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def get_current_user():
    """Get current logged in user"""
    if 'user_id' in session:
        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT id, username, email, created_at FROM users WHERE id = %s", (session['user_id'],))
            user = cursor.fetchone()
            cursor.close()
            conn.close()
            return user
        except:
            return None
    return None


# ============== Search History Functions ==============

def init_history_table():
    """Create search_history table if it doesn't exist"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                symptoms TEXT NOT NULL,
                geography VARCHAR(50),
                prediction LONGTEXT,
                retrieved_diseases JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()
        print("✓ Search history table ready")
        return True
    except Exception as e:
        print(f"✗ Error creating search_history table: {e}")
        return False


def save_search_history(user_id, symptoms, geography, prediction, retrieved_diseases):
    """Save a search to history"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        # Convert retrieved_diseases to JSON string, removing embeddings
        diseases_for_storage = []
        for d in retrieved_diseases:
            diseases_for_storage.append({
                'id': d.get('id'),
                'disease': d.get('disease'),
                'symptoms': d.get('symptoms'),
                'exam_and_tests': d.get('exam_and_tests'),
                'similarity_score': d.get('similarity_score'),
                'confidence_level': d.get('confidence_level'),
                'confidence_score': d.get('confidence_score')
            })
        
        cursor.execute("""
            INSERT INTO search_history (user_id, symptoms, geography, prediction, retrieved_diseases)
            VALUES (%s, %s, %s, %s, %s)
        """, (user_id, symptoms, geography, prediction, json.dumps(diseases_for_storage)))
        
        conn.commit()
        history_id = cursor.lastrowid
        cursor.close()
        conn.close()
        return history_id
    except Exception as e:
        print(f"✗ Error saving search history: {e}")
        return None


def get_user_history(user_id, limit=20):
    """Get user's search history"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, symptoms, geography, prediction, retrieved_diseases, created_at
            FROM search_history
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """, (user_id, limit))
        history = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Parse JSON fields
        for item in history:
            if item['retrieved_diseases']:
                item['retrieved_diseases'] = json.loads(item['retrieved_diseases'])
            if item['created_at']:
                item['created_at'] = item['created_at'].isoformat()
        
        return history
    except Exception as e:
        print(f"✗ Error getting search history: {e}")
        return []


def get_history_item(history_id, user_id):
    """Get a specific history item"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, symptoms, geography, prediction, retrieved_diseases, created_at
            FROM search_history
            WHERE id = %s AND user_id = %s
        """, (history_id, user_id))
        item = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if item:
            if item['retrieved_diseases']:
                item['retrieved_diseases'] = json.loads(item['retrieved_diseases'])
            if item['created_at']:
                item['created_at'] = item['created_at'].isoformat()
        
        return item
    except Exception as e:
        print(f"✗ Error getting history item: {e}")
        return None


def delete_history_item(history_id, user_id):
    """Delete a specific history item"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM search_history
            WHERE id = %s AND user_id = %s
        """, (history_id, user_id))
        conn.commit()
        affected = cursor.rowcount
        cursor.close()
        conn.close()
        return affected > 0
    except Exception as e:
        print(f"✗ Error deleting history item: {e}")
        return False

# Initialize embedding model (medical-domain)
print("Loading medical embedding model (this may take a moment on first run)...")
try:
    embedding_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
    print("✓ Medical embedding model loaded successfully")
except Exception as e:
    print(f"✗ Error loading embedding model: {e}")
    embedding_model = None

# Initialize cross-encoder for re-ranking
print("Loading cross-encoder model for re-ranking...")
try:
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    print("✓ Cross-encoder model loaded successfully")
except Exception as e:
    print(f"✗ Error loading cross-encoder: {e}")
    cross_encoder = None

# Store diseases with embeddings in memory
diseases_db = []
# BM25 index for keyword search
bm25_index = None
tokenized_symptoms = []

def tokenize_text(text: str) -> list:
    """Tokenize text for BM25 indexing"""
    # Convert to lowercase and split on non-alphanumeric characters
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def load_diseases_from_db():
    """Load all diseases from MySQL database, create embeddings, and build BM25 index"""
    global diseases_db, bm25_index, tokenized_symptoms
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        # Adjust column names to match your table
        cursor.execute("SELECT id, article_title as disease, symptoms, exams_and_tests as exam_and_tests,geography, prevalence_score  FROM medical_articles_new;")
        results = cursor.fetchall()
        
        diseases_db = []
        tokenized_symptoms = []
        
        for row in results:
            # Create embedding for symptoms using semantic search
            if embedding_model:
                symptom_embedding = embedding_model.encode(row['symptoms'])
            else:
                symptom_embedding = None
            
            # Tokenize symptoms for BM25
            tokens = tokenize_text(row['symptoms'])
            tokenized_symptoms.append(tokens)
            
            diseases_db.append({
                'id': row['id'],
                'disease': row['disease'],
                'symptoms': row['symptoms'],
                'exam_and_tests': row['exam_and_tests'],
                'embedding': symptom_embedding,
                'geography': row.get('geography'),
                'prevalence_score': row.get('prevalence_score', 0.5)
            })
        
        # Build BM25 index
        if tokenized_symptoms:
            bm25_index = BM25Okapi(tokenized_symptoms)
            print(f"✓ BM25 index built with {len(tokenized_symptoms)} documents")
        
        cursor.close()
        conn.close()
        print(f"✓ Loaded {len(diseases_db)} diseases from database")
        return True
    except Exception as e:
        print(f"✗ Error loading diseases: {e}")
        return False


def retrieve_similar_diseases(user_symptoms: str, top_k=5, user_geography=None):
    """
    Hybrid retrieval using:
    1. Medical semantic embeddings (70% weight)
    2. BM25 keyword search (30% weight)
    3. Cross-encoder re-ranking for final results
    """
    
    if not embedding_model or not diseases_db:
        print("✗ Cannot retrieve diseases - embedding model or database not loaded")
        return []
    
    try:
        # Convert user symptoms to embedding
        user_embedding = embedding_model.encode(user_symptoms)
        
        # Get BM25 scores for keyword matching
        user_tokens = tokenize_text(user_symptoms)
        bm25_scores = []
        if bm25_index:
            bm25_scores = bm25_index.get_scores(user_tokens)
            # Normalize BM25 scores to 0-1 range
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
            bm25_scores = [s / max_bm25 for s in bm25_scores]
        else:
            bm25_scores = [0] * len(diseases_db)
        
        # Calculate hybrid similarity scores
        similarities = []
        for i, disease in enumerate(diseases_db):
            if disease['embedding'] is not None:
                # Semantic similarity (cosine)
                semantic_similarity = np.dot(user_embedding, disease['embedding']) / (
                    np.linalg.norm(user_embedding) * np.linalg.norm(disease['embedding']) + 1e-8
                )
                
                # BM25 keyword score
                bm25_score = bm25_scores[i]
                
                # Hybrid score: 70% semantic + 30% BM25
                hybrid_score = (0.7 * semantic_similarity) + (0.3 * bm25_score)
                
                # Apply geographic weighting if user geography is provided
                geographic_boost = 1.0
                if user_geography and disease.get('geography'):
                    if user_geography.lower() in disease['geography'].lower():
                        geographic_boost = 1.3  # 30% boost for diseases common in the region
                        print(f"  Geographic boost applied to {disease['disease']}")
                
                # Apply prevalence score weighting
                prevalence = disease.get('prevalence_score', 0.5)
                weighted_score = hybrid_score * (0.7 + (prevalence * 0.3)) * geographic_boost
                
                similarities.append({
                    'id': disease['id'],
                    'disease': disease['disease'],
                    'symptoms': disease['symptoms'],
                    'exam_and_tests': disease['exam_and_tests'],
                    'semantic_score': float(semantic_similarity),
                    'bm25_score': float(bm25_score),
                    'hybrid_score': float(hybrid_score),
                    'similarity_score': float(weighted_score),  # Final weighted score
                    'weighted_similarity': float(weighted_score),
                    'geography': disease.get('geography'),
                    'prevalence_score': prevalence
                })
         
        # Sort by weighted similarity and get top candidates for re-ranking
        similarities.sort(key=lambda x: x['weighted_similarity'], reverse=True)
        top_candidates = similarities[:10]  # Get top 10 for re-ranking (faster)
        
        print(f"✓ Hybrid retrieval found {len(similarities)} potential matches")
        print(f"  Top 10 candidates selected for re-ranking...")
        
        # Cross-encoder re-ranking
        if cross_encoder and len(top_candidates) > 0:
            print("  Applying cross-encoder re-ranking...")
            # Create pairs for cross-encoder
            pairs = [(user_symptoms, candidate['symptoms']) for candidate in top_candidates]
            
            # Get cross-encoder scores
            rerank_scores = cross_encoder.predict(pairs)
            
            # Add rerank scores and re-sort
            for j, candidate in enumerate(top_candidates):
                candidate['rerank_score'] = float(rerank_scores[j])
                # Final score combines hybrid and rerank scores
                candidate['final_score'] = (0.5 * candidate['weighted_similarity']) + (0.5 * candidate['rerank_score'])
            
            # Sort by final score after re-ranking
            top_candidates.sort(key=lambda x: x['final_score'], reverse=True)
            print("  ✓ Cross-encoder re-ranking complete")
        else:
            # If no cross-encoder, use hybrid score as final
            for candidate in top_candidates:
                candidate['rerank_score'] = 0.0
                candidate['final_score'] = candidate['weighted_similarity']
        
        # Get final top_k results
        final_results = top_candidates[:top_k]
        
        print(f"✓ Final {len(final_results)} matches after re-ranking:")
        for i, match in enumerate(final_results, 1):
            print(f"  {i}. {match['disease']} (semantic: {match['semantic_score']:.2%}, bm25: {match['bm25_score']:.2%}, rerank: {match.get('rerank_score', 0):.2f}, final: {match['final_score']:.2%})")
        
        return final_results
    
    except Exception as e:
        print(f"✗ Error retrieving diseases: {e}")
        import traceback
        traceback.print_exc()
        return []



def generate_prediction_with_gemma(user_symptoms: str, retrieved_diseases: list):
    """Use Gemma 2 to generate disease prediction"""
    
    # Calculate overall confidence based on similarity scores
    if retrieved_diseases:
        avg_similarity = sum(d['similarity_score'] for d in retrieved_diseases) / len(retrieved_diseases)
        max_similarity = max(d['similarity_score'] for d in retrieved_diseases)
        
        # Confidence scoring logic:
        # - If best match is >0.8 and average is good: HIGH confidence
        # - If best match is 0.6-0.8: MEDIUM confidence
        # - If best match is <0.6: LOW confidence
        if max_similarity > 0.8 and avg_similarity > 0.7:
            confidence_level = "HIGH"
            confidence_score = min(0.95, max_similarity * 1.1)  # Cap at 0.95
        elif max_similarity > 0.6 and avg_similarity > 0.5:
            confidence_level = "MEDIUM"
            confidence_score = (max_similarity + avg_similarity) / 2
        else:
            confidence_level = "LOW"
            confidence_score = max_similarity * 0.8
    else:
        confidence_level = "LOW"
        confidence_score = 0.0
    
    # Format retrieved information as context
    context = f"MEDICAL DATABASE MATCHES (Confidence: {confidence_level} - {confidence_score:.1%}):\n"
    for i, disease in enumerate(retrieved_diseases, 1):
        context += f"\n{i}. Disease: {disease['disease']}\n"
        context += f"   Similar symptoms in DB: {disease['symptoms']}\n"
        context += f"   Recommended tests: {disease['exam_and_tests']}\n"
        context += f"   Similarity match: {disease['similarity_score']:.2%}\n"
    
    # Create prompt for Gemma 2
    prompt = f"""You are a medical information assistant. Based on a patient's reported symptoms and matching information from a medical database, provide health information.

PATIENT REPORTED SYMPTOMS:
{user_symptoms}

{context}

Please provide:
1. The most likely diseases that match the reported symptoms (ranked by likelihood)
2. For each disease, list the recommended laboratory tests and examinations
3. A brief explanation of why each disease matches the symptoms
4. Important disclaimer about seeking professional medical advice

Keep the response informative but clear. This is for informational purposes only."""

    try:
        # Call local Gemma 2 model via Ollama
        print("Generating prediction with Gemma 2...")
        response = ollama.generate(
            model='gemma:2b',  # or 'gemma:7b' if you have the larger version
            prompt=prompt,
            stream=False,
            # timeout=120.0
        )
        print("✓ Prediction generated successfully")
        return response['response']
    except Exception as e:
        print(f"✗ Error generating prediction: {e}")
        return f"Error generating prediction: {str(e)}"
    

# ============== Authentication Routes ==============

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page and handler"""
    if 'user_id' in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not username or not password:
            flash('Please fill in all fields.', 'error')
            return render_template('login.html')
        
        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE username = %s OR email = %s", (username, username))
            user = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if user and check_password_hash(user['password_hash'], password):
                session['user_id'] = user['id']
                session['username'] = user['username']
                flash(f'Welcome back, {user["username"]}!', 'success')
                return redirect(url_for('index'))
            else:
                flash('Invalid username or password.', 'error')
        except Exception as e:
            flash('An error occurred. Please try again.', 'error')
            print(f"Login error: {e}")
    
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page and handler"""
    if 'user_id' in session:
        return redirect(url_for('index'))
    
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
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor(dictionary=True)
            
            # Check if username or email already exists
            cursor.execute("SELECT id FROM users WHERE username = %s OR email = %s", (username, email))
            existing = cursor.fetchone()
            
            if existing:
                flash('Username or email already registered.', 'error')
                cursor.close()
                conn.close()
                return render_template('register.html')
            
            # Create new user
            password_hash = generate_password_hash(password)
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                (username, email, password_hash)
            )
            conn.commit()
            
            # Auto-login after registration
            user_id = cursor.lastrowid
            session['user_id'] = user_id
            session['username'] = username
            
            cursor.close()
            conn.close()
            
            flash(f'Account created successfully! Welcome, {username}!', 'success')
            return redirect(url_for('index'))
            
        except Exception as e:
            flash('An error occurred. Please try again.', 'error')
            print(f"Registration error: {e}")
    
    return render_template('register.html')


@app.route('/logout')
def logout():
    """Logout handler"""
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))


# ============== Main Routes ==============

@app.route('/')
@login_required
def index():
    """Render the main page"""
    user = get_current_user()
    return render_template('index.html', user=user)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for disease prediction"""
    try:
        data = request.get_json()
        user_symptoms = data.get('symptoms', '').strip()
        user_geography = data.get('geography', None)  # Get geography if provided
        
        if not user_symptoms:
            return jsonify({'error': 'Please provide symptoms'}), 400
        
        print(f"\n--- New prediction request ---")
        print(f"User symptoms: {user_symptoms}")
        if user_geography:
            print(f"User geography: {user_geography}")
        
        # Step 1: Retrieve similar diseases with optional geographic weighting
        retrieved_diseases = retrieve_similar_diseases(user_symptoms, top_k=5, user_geography=user_geography)
        
        if not retrieved_diseases:
            return jsonify({'error': 'No matching diseases found in database. The system may not have data for these symptoms.'}), 404
        
        # Add confidence score to each disease based on similarity
        for disease in retrieved_diseases:
            # Use the similarity_score from the retrieved disease
            similarity = disease.get('similarity_score', 0)
            
            # Determine confidence level for this specific disease
            if similarity > 0.8:
                disease['confidence_level'] = 'HIGH'
            elif similarity > 0.6:
                disease['confidence_level'] = 'MEDIUM'
            else:
                disease['confidence_level'] = 'LOW'
            
            disease['confidence_score'] = float(similarity)
            
            print(f"Disease: {disease['disease']}, Similarity: {similarity}, Confidence: {disease['confidence_level']}")
        
        # Step 2: Generate prediction using Gemma 2
        prediction = generate_prediction_with_gemma(user_symptoms, retrieved_diseases)
        
        # Debug: Verify we're sending the data
        print(f"✓ Sending {len(retrieved_diseases)} diseases in response")
        for disease in retrieved_diseases:
            print(f"  - {disease['disease']}: confidence={disease.get('confidence_level')}, score={disease.get('confidence_score')}")
        
        # Save to history if user is logged in
        history_id = None
        if 'user_id' in session:
            history_id = save_search_history(
                session['user_id'],
                user_symptoms,
                user_geography,
                prediction,
                retrieved_diseases
            )
            if history_id:
                print(f"✓ Saved to history (id: {history_id})")
        
        return jsonify({
            'status': 'success',
            'user_symptoms': user_symptoms,
            'user_geography': user_geography,
            'prediction': prediction,
            'retrieved_diseases': retrieved_diseases,
            'history_id': history_id,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"✗ Error in predict: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============== History Routes ==============

@app.route('/history', methods=['GET'])
@login_required
def get_history():
    """Get user's search history"""
    limit = request.args.get('limit', 20, type=int)
    history = get_user_history(session['user_id'], limit=limit)
    return jsonify({
        'status': 'success',
        'history': history,
        'count': len(history)
    })


@app.route('/history/<int:history_id>', methods=['GET'])
@login_required
def get_history_detail(history_id):
    """Get a specific history item"""
    item = get_history_item(history_id, session['user_id'])
    if item:
        return jsonify({
            'status': 'success',
            'item': item
        })
    return jsonify({'error': 'History item not found'}), 404


@app.route('/history/<int:history_id>', methods=['DELETE'])
@login_required
def delete_history(history_id):
    """Delete a history item"""
    if delete_history_item(history_id, session['user_id']):
        return jsonify({
            'status': 'success',
            'message': 'History item deleted'
        })
    return jsonify({'error': 'Failed to delete history item'}), 400


@app.route('/history/clear', methods=['DELETE'])
@login_required
def clear_history():
    """Clear all user's history"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM search_history WHERE user_id = %s", (session['user_id'],))
        conn.commit()
        deleted = cursor.rowcount
        cursor.close()
        conn.close()
        return jsonify({
            'status': 'success',
            'message': f'Deleted {deleted} history items'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/load-db', methods=['POST'])
def load_db():
    """Manually load/reload diseases from database"""
    if load_diseases_from_db():
        return jsonify({'status': 'success', 'message': f'Loaded {len(diseases_db)} diseases'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to load diseases'}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'diseases_loaded': len(diseases_db),
        'bm25_index_size': len(tokenized_symptoms) if tokenized_symptoms else 0,
        'model': 'Gemma 2 via Ollama',
        'embedding_model': 'S-PubMedBert-MS-MARCO (Medical Domain)',
        'cross_encoder': 'ms-marco-MiniLM-L-6-v2',
        'method': 'Hybrid retrieval (70% semantic + 30% BM25) with cross-encoder re-ranking'
    })


if __name__ == '__main__':
    print("=" * 60)
    print("Starting Medical RAG Application...")
    print("=" * 60)
    
    if embedding_model is None:
        print("⚠ Warning: Embedding model not loaded. Please check your setup.")
    
    # Initialize database tables
    print("\nInitializing database tables...")
    init_users_table()
    init_history_table()
    
    # Load diseases on startup
    print("\nLoading diseases from database...")
    if load_diseases_from_db():
        print("\n✓ Application ready!")
        print("Make sure Ollama is running: ollama serve")
        print("=" * 60)
        app.run(debug=True, port=5000)
    else:
        print("\n✗ Failed to load diseases. Check your database connection.")
        print("=" * 60)