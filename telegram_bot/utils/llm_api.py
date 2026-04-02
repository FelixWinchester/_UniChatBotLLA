from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama
import torch
import os

app = FastAPI()

# Инициализация моделей и клиентов один раз при запуске
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.empty_cache()

client = QdrantClient(host='localhost', port=6333)
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
sentence_model = sentence_model.to(device)

class LLMRequest(BaseModel):
    prompt: str

class LLMResponse(BaseModel):
    response: str

def format_context(results):
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

async def ask_llm(prompt: str) -> Optional[str]:
    try:
        # Векторизация запроса
        with torch.no_grad():
            query_vector = sentence_model.encode(prompt)
        
        # Поиск в Qdrant
        results = client.search(
            collection_name="my_collection",
            query_vector=query_vector.tolist(),
            limit=5,
            score_threshold=0.2
        )
        if not results:
            return "Информация не найдена."
        
        # Формирование контекста
        context = format_context(results)
        
        # Формируем prompt для Ollama
        full_prompt = f"""Задача: обобщи информацию из предоставленных фрагментов текста.
Используй ТОЛЬКО факты из этих фрагментов.

Контекст:
{context}

Вопрос:
{prompt}"""
        
        # Отправка запроса в Ollama через Python API
        ollama_response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )
        response_text = ollama_response['message']['content']
        
        if not response_text or len(response_text) < 10:
            return "Не удалось сгенерировать содержательный ответ."
        return response_text
    except Exception as e:
        return f"Ошибка при обработке запроса: {str(e)}"

@app.post("/ask_llm/", response_model=LLMResponse)
async def ask_llm_endpoint(request: LLMRequest):
    response = await ask_llm(request.prompt)
    if response is None:
        raise HTTPException(status_code=500, detail="Произошла ошибка при обращении к модели.")
    return LLMResponse(response=response)
