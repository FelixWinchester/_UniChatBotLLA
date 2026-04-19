from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import torch
import re
from typing import List, Dict, Tuple

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(EMBEDDING_MODEL).to(device)
client = QdrantClient(host='localhost', port=6333)


def get_keyword_score(text: str, query: str) -> float:
    """Вычисляет буст на основе совпадения ключевых слов."""
    text_lower = text.lower()
    query_words = re.findall(r'\w+', query.lower())
    
    if not query_words:
        return 1.0
    
    matches = sum(1 for word in query_words if word in text_lower)
    return 1.0 + (matches / len(query_words)) * 0.3


def hybrid_search(
    query: str, 
    collection: str = "my_collection", 
    limit: int = 10,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3
) -> List[Dict]:
    """
    Гибридный поиск: комбинирует векторное сходство с бустом по ключевым словам.
    
    Args:
        query: Поисковый запрос
        collection: Название коллекции
        limit: Максимальное количество результатов
        vector_weight: Вес векторного поиска (0-1)
        keyword_weight: Вес ключевых слов (0-1)
    
    Returns:
        Список результатов с комбинированным скором
    """
    with torch.no_grad():
        query_vector = model.encode(query).tolist()
    
    vector_results = client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=limit * 2,
        score_threshold=0.0
    )
    
    if not vector_results:
        return []
    
    scored_results = []
    max_score = max(r.score for r in vector_results)
    
    for r in vector_results:
        payload = r.payload
        content = payload.get("content", "")
        
        keyword_boost = get_keyword_score(content, query)
        vector_score = r.score / max_score if max_score > 0 else r.score
        
        combined_score = (
            vector_weight * vector_score + 
            keyword_weight * keyword_boost
        )
        
        scored_results.append({
            "id": r.id,
            "score": combined_score,
            "vector_score": r.score,
            "keyword_boost": keyword_boost,
            "content": content,
            "metadata": payload.get("metadata", {})
        })
    
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    return scored_results[:limit]


def vector_search_only(
    query: str,
    collection: str = "my_collection",
    limit: int = 5,
    score_threshold: float = 0.2
) -> List[Dict]:
    """Простой векторный поиск (для сравнения)."""
    with torch.no_grad():
        query_vector = model.encode(query).tolist()
    
    results = client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=limit,
        score_threshold=score_threshold
    )
    
    return [{
        "id": r.id,
        "score": r.score,
        "content": r.payload.get("content", ""),
        "metadata": r.payload.get("metadata", {})
    } for r in results]
