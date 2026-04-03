import time
from rag.rag_system import get_rag_answer

print("Testing fast model...")
start = time.time()
answer = get_rag_answer("материальная помощь")
duration = time.time() - start

print(f"\nDuration: {duration:.2f}s")
print(f"Answer length: {len(answer) if answer else 0}")
