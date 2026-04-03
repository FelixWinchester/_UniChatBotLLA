import ollama

print("Checking Ollama connection...")

try:
    print("1. Checking models...")
    models = ollama.list()
    print(f"   Models: {models}")
except Exception as e:
    print(f"   Error listing models: {e}")

try:
    print("\n2. Testing chat...")
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": "Привет"}]
    )
    print(f"   Response: {response}")
except Exception as e:
    print(f"   Error: {e}")

print("\n3. Testing search and get_rag_answer...")

from rag.rag_system import get_rag_answer, search

# Test search
print("\nSearch test:")
results = search("материальная помощь", limit=5)
print(f"Results: {len(results)}")
for r in results[:3]:
    print(f"  - {r.get('title', '')[:50]} [{r['score']:.2f}]")

# Test full answer
print("\nFull answer test:")
answer = get_rag_answer("материальная помощь")
print(f"Answer type: {type(answer)}")
print(f"Answer: {answer[:200] if answer else 'None'}")
