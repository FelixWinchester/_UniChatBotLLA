from hybrid_search import hybrid_search
import ollama

def get_rag_answer(query_text: str) -> str:
    if not query_text.strip():
        return "Пустой запрос"

    results = hybrid_search(query_text, collection="my_collection", limit=5)

    if not results:
        return "Не найдено релевантной информации."

    context_parts = []
    for i, r in enumerate(results, 1):
        content = r.get("content", "")
        if content:
            context_parts.append(f"Фрагмент {i}:\n{content[:500]}")
    
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
    print("RAG System (Hybrid) - Ctrl+C для выхода")
    print("-" * 40)
    
    while True:
        try:
            query = input("\nВопрос: ").strip()
            if not query:
                continue
            
            results = hybrid_search(query, limit=5)
            print(f"Найдено {len(results)} результатов")
            for i, r in enumerate(results[:3], 1):
                print(f"  [{i}] score={r['score']:.3f} (vector={r['vector_score']:.3f}, kw={r['keyword_boost']:.2f})")
                print(f"      {r['content'][:100]}...")
            
            print("\nОтвет:")
            answer = get_rag_answer(query)
            print(answer)
            
        except KeyboardInterrupt:
            print("\nВыход")
            break
