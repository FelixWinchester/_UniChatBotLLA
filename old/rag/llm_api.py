from rag.rag_system import get_rag_answer

async def ask_llm(query_text: str) -> str:
    return get_rag_answer(query_text)
