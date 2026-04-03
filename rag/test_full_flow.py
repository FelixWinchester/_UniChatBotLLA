import time
from rag.rag_system import get_rag_answer

print("Testing full flow...")
print(f"Start: {time.time()}")

print("Calling get_rag_answer...")
start = time.time()
answer = get_rag_answer("материальная помощь")
duration = time.time() - start

print(f"Duration: {duration:.2f}s")
print(f"Answer type: {type(answer)}")
print(f"Answer length: {len(answer) if answer else 0}")
print(f"Answer: {answer[:200] if answer else 'None'}")
