from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama
import torch
import os
import sys
from time import time
from datetime import datetime

def print_progress(step, message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {step}: {message}", flush=True)

def display_results(results):
    """Отображает найденные результаты с их релевантностью"""
    total_score = sum(r.score for r in results)
    
    print("\nНайденные векторы:")
    print("=" * 80)
    for i, result in enumerate(results, 1):
        relative_score = (result.score / total_score) * 100
        print(f"Вектор {i} (релевантность: {relative_score:.1f}%):")
        print("-" * 40)
        print(result.payload['text'])
    print("=" * 80)
    print(client.get_collection("my_collection").config.params)

def format_context(results):
    """Форматирует контекст для модели"""
    contexts = []
    total_score = sum(r.score for r in results)
    
    for i, r in enumerate(results, 1):
        text = r.payload['text']
        score = r.score
        relative_score = (score / total_score) * 100
        
        if relative_score >= 10:
            context_entry = f"Фрагмент {i}:\n{text}"
            contexts.append(context_entry)
            
    return "\n\n".join(contexts)

try:
    start_time = time()
    print_progress("START", "Инициализация системы")

    # GPU setup
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_progress("SYSTEM", f"Используется {device.upper()}")
    
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # Подключение к БД
    print_progress("SYSTEM", "Подключение к базе данных...")
    client = QdrantClient(host='localhost', port=6333)
    
    # Загрузка модели для векторизации
    print_progress("SYSTEM", "Загрузка модели эмбеддингов...")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence_model = sentence_model.to(device)

    # Получение запроса
    query_text = input('\nВаш вопрос: ').strip()
    if not query_text:
        print_progress("ERROR", "Пожалуйста, задайте вопрос")
        sys.exit(1)

    # Векторный поиск
    print_progress("PROCESS", "Поиск релевантной информации...")
    with torch.no_grad():
        query_vector = sentence_model.encode(query_text)

    results = client.search(
        collection_name="my_collection",
        query_vector=query_vector.tolist(),
        limit=5,
        score_threshold=0.2
    )

    if not results:
        print_progress("RESULT", "Не найдено релевантной информации")
        sys.exit(0)

    print_progress("PROCESS", f"Найдено {len(results)} релевантных фрагментов")
    display_results(results)

    # Подготовка контекста
    context = format_context(results)

    # Используем Ollama
    print_progress("SYSTEM", "Отправка запроса в Ollama...")

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

        response_text = ollama_response['message']['content']

        if not response_text or len(response_text) < 10:
            print_progress("ERROR", "Не удалось сгенерировать содержательный ответ")
            print("\nИспользуйте информацию из показанных выше фрагментов")
        else:
            print("\nСгенерированный ответ:")
            print("=" * 80)
            print(response_text)
            print("=" * 80)

        total_time = time() - start_time
        print_progress("COMPLETE", f"Обработка заняла {total_time:.1f} секунд")

    except Exception as e:
        print_progress("ERROR", f"Ошибка во время генерации через Ollama: {str(e)}")
        raise

except Exception as e:
    print_progress("ERROR", str(e))
    if hasattr(e, '__traceback__'):
        import traceback
        traceback.print_exc()
