"""
Main application routes
"""
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, session

from app.models.user import login_required, get_current_user
from app.services.retrieval import retrieve_similar_diseases
from app.services.prediction import generate_prediction
from app.services.history import save_search_history

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
@login_required
def index():
    """Render the main page"""
    user = get_current_user()
    return render_template('index.html', user=user)


@main_bp.route('/predict', methods=['POST'])
def predict():
    """API endpoint for disease prediction"""
    try:
        data = request.get_json()
        user_symptoms = data.get('symptoms', '').strip()
        user_geography = data.get('geography', None)
        
        if not user_symptoms:
            return jsonify({'error': 'Please provide symptoms'}), 400
        
        print(f"\n--- New prediction request ---")
        print(f"User symptoms: {user_symptoms}")
        if user_geography:
            print(f"User geography: {user_geography}")
        
        # Step 1: Retrieve similar diseases
        retrieved_diseases = retrieve_similar_diseases(
            user_symptoms, 
            top_k=5, 
            user_geography=user_geography
        )
        
        if not retrieved_diseases:
            return jsonify({
                'error': 'No matching diseases found in database. The system may not have data for these symptoms.'
            }), 404
        
        # Add confidence score to each disease
        for disease in retrieved_diseases:
            similarity = disease.get('similarity_score', 0)
            
            if similarity > 0.8:
                disease['confidence_level'] = 'HIGH'
            elif similarity > 0.6:
                disease['confidence_level'] = 'MEDIUM'
            else:
                disease['confidence_level'] = 'LOW'
            
            disease['confidence_score'] = float(similarity)
            print(f"Disease: {disease['disease']}, Similarity: {similarity}, "
                  f"Confidence: {disease['confidence_level']}")
        
        # Step 2: Generate prediction using LLM
        prediction = generate_prediction(user_symptoms, retrieved_diseases)
        
        # Debug output
        print(f"✓ Sending {len(retrieved_diseases)} diseases in response")
        for disease in retrieved_diseases:
            print(f"  - {disease['disease']}: confidence={disease.get('confidence_level')}, "
                  f"score={disease.get('confidence_score')}")
        
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

