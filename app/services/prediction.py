"""
LLM Prediction service
"""
import ollama
from app.config import Config


def generate_prediction(user_symptoms: str, retrieved_diseases: list):
    """Use Gemma to generate disease prediction"""
    
    # Calculate overall confidence based on similarity scores
    if retrieved_diseases:
        avg_similarity = sum(d['similarity_score'] for d in retrieved_diseases) / len(retrieved_diseases)
        max_similarity = max(d['similarity_score'] for d in retrieved_diseases)
        
        if max_similarity > 0.8 and avg_similarity > 0.7:
            confidence_level = "HIGH"
            confidence_score = min(0.95, max_similarity * 1.1)
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
    
    # Create prompt
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
        print("Generating prediction with Gemma...")
        response = ollama.generate(
            model=Config.LLM_MODEL,
            prompt=prompt,
            stream=False
        )
        print("✓ Prediction generated successfully")
        return response['response']
    except Exception as e:
        print(f"✗ Error generating prediction: {e}")
        return f"Error generating prediction: {str(e)}"

