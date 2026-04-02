from qdrant.client import client
from sentence_transformers import SentenceTransformer
import json

model = SentenceTransformer('all-MiniLM-L6-v2')

def store_vectors(data_path):
    with open(data_path, 'r') as file:
        data = json.load(file)

    vectors = [PointStruct(id=item['id'], vector=model.encode(item['content'])) for item in data]
    client.upsert(collection_name='vectors', points=vectors)

def get_vectors(query, top=5):
    query_vector = model.encode([query])
    results = client.search(collection_name='vectors', query_vector=query_vector, top=top)
    return [result.payload for result in results]
