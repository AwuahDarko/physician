"""
Search history service
"""
import json
from app.database import get_db_connection


def init_history_table():
    """Create search_history table if it doesn't exist"""
    try:
        conn = get_db_connection()
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
        conn = get_db_connection()
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
        conn = get_db_connection()
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
        conn = get_db_connection()
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
        conn = get_db_connection()
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


def clear_user_history(user_id):
    """Clear all history for a user"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM search_history WHERE user_id = %s", (user_id,))
        conn.commit()
        deleted = cursor.rowcount
        cursor.close()
        conn.close()
        return deleted
    except Exception as e:
        print(f"✗ Error clearing history: {e}")
        return 0

