"""
Disease retrieval service
"""
import re
import numpy as np
from rank_bm25 import BM25Okapi

from app.config import Config
from app.database import get_db_connection
from app.services.embedding import get_embedding_model, get_cross_encoder

# Global state for diseases and index
_diseases_db = []
_bm25_index = None
_tokenized_symptoms = []


def tokenize_text(text: str) -> list:
    """Tokenize text for BM25 indexing"""
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def load_diseases_from_db():
    """Load all diseases from MySQL database, create embeddings, and build BM25 index"""
    global _diseases_db, _bm25_index, _tokenized_symptoms
    
    embedding_model = get_embedding_model()
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT id, article_title as disease, symptoms, 
                   exams_and_tests as exam_and_tests, geography, prevalence_score 
            FROM medical_articles_new
        """)
        results = cursor.fetchall()
        
        _diseases_db = []
        _tokenized_symptoms = []
        
        for row in results:
            # Create embedding for symptoms
            if embedding_model:
                symptom_embedding = embedding_model.encode(row['symptoms'])
            else:
                symptom_embedding = None
            
            # Tokenize symptoms for BM25
            tokens = tokenize_text(row['symptoms'])
            _tokenized_symptoms.append(tokens)
            
            _diseases_db.append({
                'id': row['id'],
                'disease': row['disease'],
                'symptoms': row['symptoms'],
                'exam_and_tests': row['exam_and_tests'],
                'embedding': symptom_embedding,
                'geography': row.get('geography'),
                'prevalence_score': row.get('prevalence_score', 0.5)
            })
        
        # Build BM25 index
        if _tokenized_symptoms:
            _bm25_index = BM25Okapi(_tokenized_symptoms)
            print(f"✓ BM25 index built with {len(_tokenized_symptoms)} documents")
        
        cursor.close()
        conn.close()
        print(f"✓ Loaded {len(_diseases_db)} diseases from database")
        return True
        
    except Exception as e:
        print(f"✗ Error loading diseases: {e}")
        return False


def get_diseases_count():
    """Get number of loaded diseases"""
    return len(_diseases_db)


def get_bm25_index_size():
    """Get BM25 index size"""
    return len(_tokenized_symptoms) if _tokenized_symptoms else 0


def retrieve_similar_diseases(user_symptoms: str, top_k=None, user_geography=None):
    """
    Hybrid retrieval using:
    1. Medical semantic embeddings (configurable weight)
    2. BM25 keyword search (configurable weight)
    3. Cross-encoder re-ranking for final results
    """
    embedding_model = get_embedding_model()
    cross_encoder = get_cross_encoder()
    
    if top_k is None:
        top_k = Config.TOP_K_RESULTS
    
    if not embedding_model or not _diseases_db:
        print("✗ Cannot retrieve diseases - embedding model or database not loaded")
        return []
    
    try:
        # Convert user symptoms to embedding
        user_embedding = embedding_model.encode(user_symptoms)
        
        # Get BM25 scores for keyword matching
        user_tokens = tokenize_text(user_symptoms)
        bm25_scores = []
        if _bm25_index:
            bm25_scores = _bm25_index.get_scores(user_tokens)
            # Normalize BM25 scores to 0-1 range
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
            bm25_scores = [s / max_bm25 for s in bm25_scores]
        else:
            bm25_scores = [0] * len(_diseases_db)
        
        # Calculate hybrid similarity scores
        similarities = []
        for i, disease in enumerate(_diseases_db):
            if disease['embedding'] is not None:
                # Semantic similarity (cosine)
                semantic_similarity = np.dot(user_embedding, disease['embedding']) / (
                    np.linalg.norm(user_embedding) * np.linalg.norm(disease['embedding']) + 1e-8
                )
                
                # BM25 keyword score
                bm25_score = bm25_scores[i]
                
                # Hybrid score with configurable weights
                hybrid_score = (
                    Config.SEMANTIC_WEIGHT * semantic_similarity + 
                    Config.BM25_WEIGHT * bm25_score
                )
                
                # Apply geographic weighting
                geographic_boost = 1.0
                if user_geography and disease.get('geography'):
                    if user_geography.lower() in disease['geography'].lower():
                        geographic_boost = 1.3
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
                    'similarity_score': float(weighted_score),
                    'weighted_similarity': float(weighted_score),
                    'geography': disease.get('geography'),
                    'prevalence_score': prevalence
                })
        
        # Sort by weighted similarity and get top candidates for re-ranking
        similarities.sort(key=lambda x: x['weighted_similarity'], reverse=True)
        top_candidates = similarities[:Config.RERANK_CANDIDATES]
        
        print(f"✓ Hybrid retrieval found {len(similarities)} potential matches")
        print(f"  Top {Config.RERANK_CANDIDATES} candidates selected for re-ranking...")
        
        # Cross-encoder re-ranking
        if cross_encoder and len(top_candidates) > 0:
            print("  Applying cross-encoder re-ranking...")
            pairs = [(user_symptoms, candidate['symptoms']) for candidate in top_candidates]
            rerank_scores = cross_encoder.predict(pairs)
            
            for j, candidate in enumerate(top_candidates):
                candidate['rerank_score'] = float(rerank_scores[j])
                candidate['final_score'] = (
                    0.5 * candidate['weighted_similarity'] + 
                    0.5 * candidate['rerank_score']
                )
            
            top_candidates.sort(key=lambda x: x['final_score'], reverse=True)
            print("  ✓ Cross-encoder re-ranking complete")
        else:
            for candidate in top_candidates:
                candidate['rerank_score'] = 0.0
                candidate['final_score'] = candidate['weighted_similarity']
        
        # Get final top_k results
        final_results = top_candidates[:top_k]
        
        print(f"✓ Final {len(final_results)} matches after re-ranking:")
        for i, match in enumerate(final_results, 1):
            print(f"  {i}. {match['disease']} (semantic: {match['semantic_score']:.2%}, "
                  f"bm25: {match['bm25_score']:.2%}, rerank: {match.get('rerank_score', 0):.2f}, "
                  f"final: {match['final_score']:.2%})")
        
        return final_results
    
    except Exception as e:
        print(f"✗ Error retrieving diseases: {e}")
        import traceback
        traceback.print_exc()
        return []

