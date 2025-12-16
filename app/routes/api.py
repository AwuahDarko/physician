"""
API routes (history, health, etc.)
"""
from flask import Blueprint, jsonify, request, session

from app.models.user import login_required
from app.services.history import (
    get_user_history, 
    get_history_item, 
    delete_history_item,
    clear_user_history
)
from app.services.retrieval import load_diseases_from_db, get_diseases_count, get_bm25_index_size
from app.config import Config

api_bp = Blueprint('api', __name__)


# ============== History Routes ==============

@api_bp.route('/history', methods=['GET'])
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


@api_bp.route('/history/<int:history_id>', methods=['GET'])
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


@api_bp.route('/history/<int:history_id>', methods=['DELETE'])
@login_required
def delete_history(history_id):
    """Delete a history item"""
    if delete_history_item(history_id, session['user_id']):
        return jsonify({
            'status': 'success',
            'message': 'History item deleted'
        })
    return jsonify({'error': 'Failed to delete history item'}), 400


@api_bp.route('/history/clear', methods=['DELETE'])
@login_required
def clear_history():
    """Clear all user's history"""
    deleted = clear_user_history(session['user_id'])
    return jsonify({
        'status': 'success',
        'message': f'Deleted {deleted} history items'
    })


# ============== System Routes ==============

@api_bp.route('/load-db', methods=['POST'])
def reload_db():
    """Manually load/reload diseases from database"""
    if load_diseases_from_db():
        return jsonify({
            'status': 'success', 
            'message': f'Loaded {get_diseases_count()} diseases'
        })
    else:
        return jsonify({
            'status': 'error', 
            'message': 'Failed to load diseases'
        }), 500


@api_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'diseases_loaded': get_diseases_count(),
        'bm25_index_size': get_bm25_index_size(),
        'model': f'{Config.LLM_MODEL} via Ollama',
        'embedding_model': f'{Config.EMBEDDING_MODEL} (Medical Domain)',
        'cross_encoder': Config.CROSS_ENCODER_MODEL,
        'method': f'Hybrid retrieval ({int(Config.SEMANTIC_WEIGHT*100)}% semantic + '
                  f'{int(Config.BM25_WEIGHT*100)}% BM25) with cross-encoder re-ranking'
    })

