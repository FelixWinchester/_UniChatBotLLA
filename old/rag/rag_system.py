from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama
import torch
import os

# Загрузка модели эмбеддингов один раз при импорте
device = "cuda" if torch.cuda.is_available() else "cpu"
sentence_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
client = QdrantClient(host='localhost', port=6333)

def get_rag_answer(query_text: str) -> str:
    if not query_text.strip():
        return "Пустой запрос"

    # Векторизация запроса
    with torch.no_grad():
        query_vector = sentence_model.encode(query_text)

    # Поиск ближайших векторов
    results = client.search(
        collection_name="my_collection",
        query_vector=query_vector.tolist(),
        limit=5,
        score_threshold=0.2
    )

    if not results:
        return "Не найдено релевантной информации."

    # Формирование контекста
    total_score = sum(r.score for r in results)
    context_parts = []
    for i, r in enumerate(results, 1):
        score = r.score
        relative_score = (score / total_score) * 100
        if relative_score >= 10:
            context_parts.append(f"Фрагмент {i}:\n{r.payload['text']}")
    context = "\n\n".join(context_parts)

    # Формирование запроса к LLM
    prompt = f"""Задача: обобщи информацию из предоставленных фрагментов текста.
Используй ТОЛЬКО факты из этих фрагментов.

Контекст:
{context}

Вопрос:
{query_text}"""

    # Запрос к Ollama
    try:
        ollama_response = ollama.chat(
            model="llama3",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return ollama_response['message']['content']
    except Exception as e:
        return f"Ошибка при обращении к модели: {str(e)}"
