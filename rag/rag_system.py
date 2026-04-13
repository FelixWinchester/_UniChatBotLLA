from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama
import torch
import re
from typing import Optional

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

COLLECTIONS = {
    "schedule": "kniit_schedule",
    "faculty": "kniit_faculty",
    "university": "kniit_university",
}

SCHEDULE_KEYWORDS = ["расписание", "пара", "занятие", "аудитория", "преподаватель", "группа"]
FACULTY_KEYWORDS = ["поступить", "направление", "учебный план", "факультет", "книит", "приём", "олимпиада"]
UNIVERSITY_KEYWORDS = ["общежити", "стипендия", "материальн", "военн", "кафедра", "помощь", "сво", "питание", "психолог", "общежил"]

device = "cuda" if torch.cuda.is_available() else "cpu"
sentence_model = SentenceTransformer(EMBEDDING_MODEL).to(device)
client = QdrantClient(host='localhost', port=6333)


def extract_group_number(query: str) -> Optional[str]:
    match = re.search(r"\b(\d{3})\b", query)
    return match.group(1) if match else None


def detect_collections(query: str) -> list[str]:
    query_lower = query.lower()
    collections = []
    
    if extract_group_number(query):
        collections.append("schedule")
    
    for kw in SCHEDULE_KEYWORDS:
        if kw in query_lower:
            collections.append("schedule")
            break
    
    for kw in FACULTY_KEYWORDS:
        if kw in query_lower:
            collections.append("faculty")
            break
    
    for kw in UNIVERSITY_KEYWORDS:
        if kw in query_lower:
            collections.append("university")
            break
    
    if not collections:
        collections = ["schedule", "faculty", "university"]
    
    return list(dict.fromkeys(collections))


def search(query_text: str, collections: Optional[list[str]] = None, limit: int = 5) -> list[dict]:
    """Поиск релевантных документов."""
    if not query_text.strip():
        return []

    with torch.no_grad():
        query_vector = sentence_model.encode(query_text).tolist()

    if collections is None:
        collections = detect_collections(query_text)

    group = extract_group_number(query_text)

    all_results = []
    per_collection = max(5, limit * 2 // len(collections))

    for col in collections:
        if col not in COLLECTIONS:
            continue
        
        results = client.search(
            collection_name=COLLECTIONS[col],
            query_vector=query_vector,
            limit=per_collection,
            score_threshold=0.0,
        )

        for r in results:
            payload = r.payload
            if group and payload.get("group") != group:
                continue
            all_results.append({
                "id": r.id,
                "score": r.score,
                "collection": col,
                "content": payload.get("content", ""),
                "title": payload.get("title", ""),
                "group": payload.get("group", ""),
                "day": payload.get("day", ""),
            })

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:limit]


def get_rag_answer(query_text: str) -> str:
    """Получает ответ от RAG-системы."""
    if not query_text.strip():
        return "Пустой запрос"

    results = search(query_text, limit=5)

    if not results:
        return "Не найдено релевантной информации."

    context_parts = []
    for i, r in enumerate(results, 1):
        content = r.get("content", "")
        if content:
            meta = []
            if r.get("group"):
                meta.append(f"группа {r['group']}")
            if r.get("day"):
                meta.append(r['day'])
            if meta:
                context_parts.append(f"Фрагмент {i} ({', '.join(meta)}):\n{content[:400]}")
            else:
                context_parts.append(f"Фрагмент {i}:\n{content[:400]}")

    context = "\n\n".join(context_parts)

    prompt = f"""Задача: обобщи информацию из предоставленных фрагментов текста.
Используй ТОЛЬКО факты из этих фрагментов.

Контекст:
{context}

Вопрос:
{query_text}"""

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


if __name__ == "__main__":
    print("RAG System - нажми Ctrl+C для выхода")
    print("-" * 40)
    
    while True:
        try:
            query = input("\nВопрос: ").strip()
            if not query:
                continue
            
            print("Поиск...")
            answer = get_rag_answer(query)
            print(f"\nОтвет:\n{answer}")
            
        except KeyboardInterrupt:
            print("\nВыход")
            break
