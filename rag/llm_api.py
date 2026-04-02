from rag.rag_system import get_rag_answer, search, detect_collections
from typing import Optional


async def ask_llm(
    query_text: str,
    collections: Optional[list[str]] = None,
    group: Optional[str] = None,
) -> str:
    """
    Получает ответ от RAG-системы.
    
    Args:
        query_text: вопрос пользователя
        collections: список коллекций ("schedule" | "faculty" | "university")
                      если None - автоопределение
        group: фильтр по номеру группы
    
    Returns:
        сгенерированный ответ
    """
    if collections is None:
        collections = detect_collections(query_text)
    
    return get_rag_answer(
        query_text,
        collections=collections,
        group=group,
    )


async def search_docs(
    query_text: str,
    collections: Optional[list[str]] = None,
    group: Optional[str] = None,
    limit: int = 5,
) -> list[dict]:
    """
    Поиск документов без генерации ответа.
    
    Returns:
        список найденных документов
    """
    if collections is None:
        collections = detect_collections(query_text)
    
    return search(
        query_text,
        collections=collections,
        group=group,
        limit=limit,
    )
