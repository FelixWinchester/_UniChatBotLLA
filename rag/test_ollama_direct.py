import ollama
import time

print("Testing Ollama chat...")
print(f"Model: llama3")

# Test 1: Simple request
print("\n1. Simple test (Привет):")
start = time.time()
try:
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": "Привет, как дела?"}],
        options={"timeout": 30}
    )
    print(f"   OK! Duration: {time.time()-start:.2f}s")
    print(f"   Response: {response['message']['content'][:100]}")
except Exception as e:
    print(f"   Error: {e}")

# Test 2: With our prompt
print("\n2. With full prompt:")
context = """[UNIVERSITY - financial_aid] Материальная помощь
В этом году каждый нуждающийся студент может получить адресную социальную поддержку в виде материальной помощи."""

prompt = f"""Контекст (факты для ответа):
{context}

Вопрос пользователя: материальная помощь

Дай ответ на основе фактов из контекста."""

start = time.time()
try:
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": "Ты — дружелюбный помощник."},
            {"role": "user", "content": prompt}
        ],
        options={"timeout": 60}
    )
    print(f"   OK! Duration: {time.time()-start:.2f}s")
    print(f"   Response: {response['message']['content'][:200]}")
except Exception as e:
    print(f"   Error: {e}")

print("\n3. Check Ollama service:")
try:
    ollama.show("llama3")
    print("   Model info: OK")
except Exception as e:
    print(f"   Model info error: {e}")
