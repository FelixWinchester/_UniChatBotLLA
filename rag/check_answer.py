from rag.rag_system import get_rag_answer

result = get_rag_answer("материальная помощь")
print(f"Type: {type(result)}")
print(f"Value: {result}")
