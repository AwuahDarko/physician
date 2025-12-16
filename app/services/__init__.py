"""
Services package
"""
from app.services.embedding import (
    get_embedding_model,
    get_cross_encoder,
    init_models
)
from app.services.retrieval import (
    load_diseases_from_db,
    retrieve_similar_diseases,
    get_diseases_count,
    get_bm25_index_size
)
from app.services.prediction import generate_prediction
from app.services.history import (
    init_history_table,
    save_search_history,
    get_user_history,
    get_history_item,
    delete_history_item,
    clear_user_history
)

__all__ = [
    # Embedding
    'get_embedding_model',
    'get_cross_encoder',
    'init_models',
    # Retrieval
    'load_diseases_from_db',
    'retrieve_similar_diseases',
    'get_diseases_count',
    'get_bm25_index_size',
    # Prediction
    'generate_prediction',
    # History
    'init_history_table',
    'save_search_history',
    'get_user_history',
    'get_history_item',
    'delete_history_item',
    'clear_user_history'
]

