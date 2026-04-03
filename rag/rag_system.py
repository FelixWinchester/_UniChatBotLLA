from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama
import torch
import re
from typing import Optional

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_LIMIT = 5
DEFAULT_SCORE_THRESHOLD = 0.3

QUERY_REPLACEMENTS = {
    "книит": "книит",
    "книит": "книит",
    "кнit": "книит",
    "сгу": "сгу",
    "сгу": "сгу",
    "sap": "сап",
    "sap/fico": "сап фико",
}

DAY_ALIASES = {
    "пн": "понедельник",
    "пон": "понедельник",
    "вт": "вторник",
    "ср": "среда",
    "чт": "четверг",
    "пт": "пятница",
    "сб": "суббота",
}


def normalize_query(query: str) -> str:
    """Нормализует запрос: исправляет опечатки, приводит к нижнему регистру."""
    query = query.lower().strip()
    
    for wrong, correct in QUERY_REPLACEMENTS.items():
        query = query.replace(wrong, correct)
    
    for abbr, full in DAY_ALIASES.items():
        pattern = r'\b' + abbr + r'\b'
        query = re.sub(pattern, full, query)
    
    query = re.sub(r'\s+', ' ', query)
    
    return query.strip()

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
    query_text: str = "",
    group: Optional[str] = None,
    limit: int = 3,
    score_threshold: float = 0.2,
) -> list[dict]:
    """
    Поиск в одной коллекции (гибридный: векторный + полнотекстовый).
    """
    if query_vector is None:
        with torch.no_grad():
            query_vector = sentence_model.encode(query_text).tolist()
    
    results_vector = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=100,
        score_threshold=0.0,
    )
    
    all_results = []
    for r in results_vector:
        payload = r.payload
        if group and payload.get("group") != group:
            continue
        
        content = payload.get("content", "") + " " + payload.get("title", "")
        if query_text and len(query_text) > 2:
            words = query_text.lower().split()
            matches = sum(1 for w in words if w in content.lower())
            text_boost = 1.0 + (matches / len(words)) * 0.5
        else:
            text_boost = 1.0
        
        boosted_score = r.score * text_boost
        
        if boosted_score < score_threshold:
            continue
            
        all_results.append({
            "id": r.id,
            "score": boosted_score,
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
    
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:limit]


def search(
    query_text: str,
    collections: Optional[list[str]] = None,
    group: Optional[str] = None,
    limit: int = DEFAULT_LIMIT,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> list[dict]:
    """
    Поиск релевантных документов (гибридный).
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
        
        results = search_in_collection(
            COLLECTIONS[col],
            query_vector,
            query_text=query_text,
            group=group,
            limit=limit * 3,
            score_threshold=score_threshold,
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


def format_context(results: list[dict]) -> str:
    """Формирует контекст для LLM."""
    if not results:
        return ""
    
    context_parts = []
    for i, r in enumerate(results, 1):
        content = r.get("content", "")
        if content:
            context_parts.append(f"Фрагмент {i}:\n{content[:500]}")
    
    return "\n\n".join(context_parts)


def get_rag_answer(
    query_text: str,
    collections: Optional[list[str]] = None,
    group: Optional[str] = None,
) -> str:
    """
    Получает ответ от RAG-системы с нормализацией и человечными ответами.
    """
    print(f"[LOG] Получен запрос: {query_text}")
    
    if not query_text.strip():
        print("[LOG] Пустой запрос")
        return "Напишите ваш вопрос"

    normalized_query = normalize_query(query_text)
    print(f"[LOG] Нормализованный запрос: {normalized_query}")

    print(f"[LOG] Поиск в коллекциях: {collections or 'auto'}")
    results = search(
        normalized_query,
        collections=collections,
        group=group,
        limit=5,
        score_threshold=0.25,
    )
    print(f"[LOG] Найдено результатов: {len(results)}")

    if not results:
        print("[LOG] Результатов нет - возвращаю 'не найдено'")
        return "К сожалению, я не нашел информации по вашему запросу. Попробуйте переформулировать вопрос или спросить о другом."

    context = format_context(results)
    print(f"[LOG] Контекст сформирован, длина: {len(context)} символов")
    
    print(f"\n[DB] Найдено: {len(results)}")
    for i, r in enumerate(results[:3], 1):
        print(f"  {i}. [{r['score']:.2f}] {r.get('title', '')[:50]}")
    
    prompt = f"""Задача: обобщи информацию из предоставленных фрагментов текста.
Используй ТОЛЬКО факты из этих фрагментов.

Контекст:
{context}

Вопрос:
{normalized_query}"""
    
    print(f"\n[OLLAMA] Запрос к модели...")
    try:
        ollama_response = ollama.chat(
            model="llama3",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        answer = ollama_response['message']['content']
        print(f"[OLLAMA] Ответ: {answer[:100]}...")
        return answer
    except Exception as e:
        return f"Ошибка: {str(e)}"


if __name__ == "__main__":
    print("RAG System - 3 collections + LLM")
    print("=" * 60)
    print("Commands:")
    print("  <text>       - ask question (gets LLM answer)")
    print("  s <text>     - search in schedule")
    print("  f <text>     - search in faculty")
    print("  u <text>     - search in university")
    print("  quit        - exit")
    print("=" * 60)
    
    while True:
        cmd = input("\n> ").strip()
        if cmd.lower() == 'quit':
            break
        
        if cmd.startswith('s '):
            query = cmd[2:]
            collections = ["schedule"]
            results = search(query, collections=collections)
            print(f"Found: {len(results)} results")
            for i, r in enumerate(results, 1):
                print(f"  {i}. [{r['collection']}] {r.get('title', r.get('subject', ''))[:50]}")
        
        elif cmd.startswith('f '):
            query = cmd[2:]
            collections = ["faculty"]
            results = search(query, collections=collections)
            print(f"Found: {len(results)} results")
            for i, r in enumerate(results, 1):
                print(f"  {i}. [{r['collection']}] {r.get('title', r.get('subject', ''))[:50]}")
        
        elif cmd.startswith('u '):
            query = cmd[2:]
            collections = ["university"]
            results = search(query, collections=collections)
            print(f"Found: {len(results)} results")
            for i, r in enumerate(results, 1):
                print(f"  {i}. [{r['collection']}] {r.get('title', r.get('subject', ''))[:50]}")
        
        elif cmd:
            print("\nДумаю...")
            answer = get_rag_answer(cmd)
            print(f"\n{answer}")
