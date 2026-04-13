import json
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import torch
from semantic_chunker import semantic_chunk

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "my_collection"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(EMBEDDING_MODEL).to(device)
client = QdrantClient(host='localhost', port=6333)

DIMENSION = model.get_sentence_embedding_dimension()


def create_collection():
    """Создает коллекцию с нужными параметрами."""
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Удалена старая коллекция {COLLECTION_NAME}")
    except:
        pass
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=DIMENSION, distance=Distance.COSINE)
    )
    print(f"Создана коллекция {COLLECTION_NAME} (dim={DIMENSION})")


def index_data(file_path: str, data_type: str):
    """Индексирует данные с семантическим чанкованием."""
    with open(file_path, 'r', encoding='utf-8') as f:
        if data_type == "faculty":
            data = json.load(f)
        else:
            data = json.load(f)
    
    print(f"Чанкование данных ({data_type})...")
    chunks = semantic_chunk(data, data_type)
    print(f"Создано чанков: {len(chunks)}")
    
    print("Векторизация...")
    texts = [c["content"] for c in chunks]
    
    with torch.no_grad():
        vectors = model.encode(texts).tolist()
    
    print("Загрузка в Qdrant...")
    points = [
        PointStruct(
            id=chunk["id"],
            vector=vectors[i],
            payload={
                "content": chunk["content"],
                "metadata": chunk["metadata"]
            }
        )
        for i, chunk in enumerate(chunks)
    ]
    
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Загружено {len(points)} векторов")
    
    return len(chunks)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Индексация данных в Qdrant')
    parser.add_argument('--type', choices=['schedule', 'faculty', 'university', 'all'], 
                        default='all', help='Тип данных для индексации')
    args = parser.parse_args()
    
    create_collection()
    
    total = 0
    
    base_path = os.path.join(os.path.dirname(__file__), '..')
    
    if args.type in ['schedule', 'all']:
        try:
            total += index_data(os.path.join(base_path, 'parse', 'full_knt_schedule.json'), 'schedule')
        except FileNotFoundError:
            print("Не найден parse/full_knt_schedule.json")
    
    if args.type in ['faculty', 'all']:
        try:
            total += index_data(os.path.join(base_path, 'data', 'Data.json'), 'faculty')
        except FileNotFoundError:
            print("Не найден data/Data.json")
    
    if args.type in ['university', 'all']:
        try:
            total += index_data(os.path.join(base_path, 'data', 'university.json'), 'university')
        except FileNotFoundError:
            print("Не найден data/university.json")
    
    print(f"\nИтого: {total} чанков загружено в коллекцию {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
