from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama
import torch
import re
from typing import Optional

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_LIMIT = 5
DEFAULT_SCORE_THRESHOLD = 0.25

COLLECTIONS = {
    "schedule": "kniit_schedule",
    "faculty": "kniit_faculty",
    "university": "kniit_university",
}

SCHEDULE_KEYWORDS = ["расписание", "пара", "занятие", "аудитория", "преподаватель", "группа"]
FACULTY_KEYWORDS = ["поступить", "направление", "учебный план", "факультет", "книит", "приём", "олимпиада"]
UNIVERSITY_KEYWORDS = ["общежитие", "стипендия", "материальн", "военн", "кафедра", "помощь", "сво", "питание", "психолог"]


def extract_group_number(query: str) -> Optional[str]:
    """Извлекает номер группы из запроса."""
    match = re.search(r"\b(\d{3})\b", query)
    return match.group(1) if match else None


def detect_collections(query: str) -> list[str]:
    """Определяет в каких коллекциях искать."""
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


device = "cuda" if torch.cuda.is_available() else "cpu"
sentence_model = SentenceTransformer(EMBEDDING_MODEL).to(device)
client = QdrantClient(host='localhost', port=6333)


def search_in_collection(
    collection_name: str,
    query_vector: list,
    group: Optional[str] = None,
    limit: int = 3,
    score_threshold: float = 0.2,
) -> list[dict]:
    """Поиск в одной коллекции."""
    
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=limit,
        score_threshold=score_threshold if score_threshold > 0 else 0.0,
    )

    search_results = []
    for r in results:
        payload = r.payload
        if group and payload.get("group") != group:
            continue
        search_results.append({
            "id": r.id,
            "score": r.score,
            "collection": collection_name,
            "source": payload.get("source", ""),
            "title": payload.get("title", ""),
            "content": payload.get("content", ""),
            "group": payload.get("group", ""),
            "day": payload.get("day", ""),
            "time": payload.get("time", ""),
            "subject": payload.get("subject", ""),
            "teacher": payload.get("teacher", ""),
            "room": payload.get("room", ""),
            "topic": payload.get("topic", ""),
            "url": payload.get("url", ""),
        })
    
    return search_results


def search(
    query_text: str,
    collections: Optional[list[str]] = None,
    group: Optional[str] = None,
    limit: int = DEFAULT_LIMIT,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> list[dict]:
    """
    Поиск релевантных документов.
    
    Args:
        query_text: текст запроса
        collections: список коллекций для поиска (auto-detect если None)
        group: фильтр по номеру группы
        limit: максимальное количество результатов
        score_threshold: минимальный порог релевантности
    """
    if not query_text.strip():
        return []

    with torch.no_grad():
        query_vector = sentence_model.encode(query_text).tolist()

    if collections is None:
        collections = detect_collections(query_text)
    
    if group is None:
        group = extract_group_number(query_text)

    all_results = []
    
    for col in collections:
        if col not in COLLECTIONS:
            continue
        
        if group and col == "schedule":
            per_collection = 50
            per_threshold = 0.1
        else:
            per_collection = max(2, limit // len(collections))
            per_threshold = score_threshold
        
        results = search_in_collection(
            COLLECTIONS[col],
            query_vector,
            group=group,
            limit=per_collection,
            score_threshold=per_threshold,
        )
        all_results.extend(results)
    
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:limit]


def format_context(results: list[dict]) -> str:
    """Форматирует результаты поиска в текстовый контекст для LLM."""
    if not results:
        return ""

    context_parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        content = r.get("content", "")
        collection = r.get("collection", "")
        
        if collection == "schedule":
            meta = []
            if r.get("group"):
                meta.append(f"группа {r['group']}")
            if r.get("day"):
                meta.append(f"{r['day']}")
            if r.get("time"):
                meta.append(f"{r['time']}")
            if r.get("teacher"):
                meta.append(f"преподаватель {r['teacher']}")
            if r.get("room"):
                meta.append(f"аудитория {r['room']}")
            meta_str = " | ".join(meta) if meta else ""
            context_parts.append(f"[{collection.upper()}] {meta_str}\n{content}")
        else:
            topic = r.get("topic", "info")
            context_parts.append(f"[{collection.upper()} - {topic}] {title}\n{content[:300]}")

    return "\n\n---\n\n".join(context_parts)


def get_rag_answer(
    query_text: str,
    collections: Optional[list[str]] = None,
    group: Optional[str] = None,
) -> str:
    """
    Получает ответ от RAG-системы.
    """
    if not query_text.strip():
        return "Пустой запрос"

    results = search(
        query_text,
        collections=collections,
        group=group,
        limit=5,
        score_threshold=0.2,
    )

    if not results:
        return "Не найдено релевантной информации по вашему запросу."

    context = format_context(results)

    prompt = f"""Ты - помощник студентов и абитуриентов факультета КНиИТ СГУ.
Отвечай на вопросы на основе предоставленного контекста.
Используй ТОЛЬКО факты из контекста. Если информации недостаточно - так и скажи.

Контекст:
{context}

Вопрос:
{query_text}

Ответ:"""

    try:
        ollama_response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": "Ты - полезный помощник. Отвечай кратко и по существу."},
                {"role": "user", "content": prompt}
            ]
        )
        return ollama_response['message']['content']
    except Exception as e:
        return f"Ошибка при обращении к модели: {str(e)}"


if __name__ == "__main__":
    print("RAG System - 3 collections")
    print("=" * 60)
    print("Commands:")
    print("  q <text> - search everywhere")
    print("  s <text> - search in schedule")
    print("  f <text> - search in faculty")
    print("  u <text> - search in university")
    print("  quit - exit")
    print("=" * 60)
    
    while True:
        cmd = input("\n> ").strip()
        if cmd.lower() == 'quit':
            break
        
        if cmd.startswith('q '):
            query = cmd[2:]
            collections = None
        elif cmd.startswith('s '):
            query = cmd[2:]
            collections = ["schedule"]
        elif cmd.startswith('f '):
            query = cmd[2:]
            collections = ["faculty"]
        elif cmd.startswith('u '):
            query = cmd[2:]
            collections = ["university"]
        else:
            query = cmd
            collections = None
        
        if not query:
            continue
        
        print("\nDetected collections:", collections or detect_collections(query))
        print("-" * 40)
        
        results = search(query, collections=collections)
        print(f"Found: {len(results)} results\n")
        
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r['collection']}] {r.get('title', r.get('subject', ''))}")
            print(f"   Score: {r['score']:.2f}")
            if r.get('group'):
                print(f"   Group: {r['group']}")
            print(f"   {r['content'][:100]}...")
            print()
