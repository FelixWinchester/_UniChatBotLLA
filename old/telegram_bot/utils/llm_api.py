import sys
import os

# Добавляем путь к RAG модулю
rag_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'rag'))
sys.path.insert(0, rag_path)

from rag_system import get_rag_answer


async def ask_llm(prompt: str) -> str:
    return get_rag_answer(prompt)
