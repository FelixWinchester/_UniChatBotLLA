"""
Indexer — загрузка данных в Qdrant (3 коллекции)
=================================================

Коллекции:
  - kniit_schedule   — расписание
  - kniit_faculty    — информация факультета
  - kniit_university — информация университета

Использование:
    python indexer.py                        # полная индексация
    python indexer.py --reset               # удалить все и создать заново
    python indexer.py --collection schedule  # только расписание
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_SIZE = 384

PARSE_DIR = Path(__file__).parent.parent / "parse"

SCHEDULE_FILE = PARSE_DIR / "full_knt_schedule.json"
FACULTY_FILE = PARSE_DIR / "netutor_chunks.json"
UNIVERSITY_FILE = PARSE_DIR / "university_chunks.json"

COLLECTIONS = {
    "schedule": "kniit_schedule",
    "faculty": "kniit_faculty",
    "university": "kniit_university",
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()}")


def get_client() -> QdrantClient:
    return QdrantClient(host="localhost", port=6333)


def collection_exists(client: QdrantClient, name: str) -> bool:
    try:
        client.get_collection(name)
        return True
    except Exception:
        return False


def create_collection(client: QdrantClient, name: str) -> None:
    if collection_exists(client, name):
        client.delete_collection(name)
        print(f"    Deleted: {name}")
    
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"    Created: {name}")


def load_json(file_path: Path) -> dict | list:
    if not file_path.exists():
        return {} if "chunks" in str(file_path) else []
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def schedule_to_items(schedule_data: list[dict]) -> list[dict]:
    items = []
    
    for group_data in schedule_data:
        group = group_data.get("group", "unknown")
        
        for day, lessons in group_data.get("days", {}).items():
            if not lessons:
                continue
                
            for lesson in lessons:
                subject = lesson.get("subject", "")
                if not subject or subject == "—":
                    continue
                
                content = f"{subject}"
                if lesson.get("teacher"):
                    content += f", преподаватель {lesson['teacher']}"
                if lesson.get("room"):
                    content += f", аудитория {lesson['room']}"
                if lesson.get("type"):
                    content += f", {lesson['type']}"
                if lesson.get("week_type") and lesson["week_type"] != "Всегда":
                    content += f", {lesson['week_type']}"
                
                items.append({
                    "source": "schedule",
                    "collection": "schedule",
                    "title": f"{subject} - {day}",
                    "content": content,
                    "group": group,
                    "day": day,
                    "time": lesson.get("time", ""),
                    "subject": subject,
                    "teacher": lesson.get("teacher", ""),
                    "room": lesson.get("room", ""),
                    "type": lesson.get("type", ""),
                    "week_type": lesson.get("week_type", ""),
                    "topic": "",
                    "url": "",
                })
    
    return items


FACULTY_TOPIC_CATEGORY = {
    "study_plans": "study",
    "direction_bachelor": "study",
    "direction_master": "study",
    "direction_specialist": "study",
    "admission": "admission",
    "faculty_info": "faculty",
    "student_life": "faculty",
    "alumni": "faculty",
    "staff": "faculty",
    "contacts": "contacts",
    "faculty_general": "general",
}


def faculty_to_items(chunks_data: dict) -> list[dict]:
    items = []
    chunks = chunks_data.get("chunks", [])
    
    for chunk in chunks:
        content = chunk.get("content", "")
        if not content or len(content) < 20:
            continue
        
        title = chunk.get("title", "")
        topic = chunk.get("topic", "faculty_general")
        category = FACULTY_TOPIC_CATEGORY.get(topic, "general")
        url = chunk.get("source_url", "")
        
        items.append({
            "source": "faculty",
            "collection": "faculty",
            "category": category,
            "title": title,
            "content": content,
            "group": "",
            "day": "",
            "time": "",
            "subject": "",
            "teacher": "",
            "room": "",
            "type": "",
            "week_type": "",
            "topic": topic,
            "url": url,
        })
    
    return items


UNIV_TOPIC_CATEGORY = {
    "dormitory": "housing",
    "financial_aid": "financial",
    "scholarship": "financial",
    "military": "military",
    "student_support": "support",
    "food": "food",
    "psychology": "support",
    "military_support": "support",
    "student_life": "life",
    "general": "general",
}


def university_to_items(chunks_data: dict) -> list[dict]:
    items = []
    chunks = chunks_data.get("chunks", [])
    
    for chunk in chunks:
        content = chunk.get("content", "")
        if not content or len(content) < 20:
            continue
        
        title = chunk.get("title", "")
        topic = chunk.get("topic", "general")
        category = UNIV_TOPIC_CATEGORY.get(topic, "general")
        url = chunk.get("source_url", "")
        
        items.append({
            "source": "university",
            "collection": "university",
            "category": category,
            "title": title,
            "content": content,
            "group": "",
            "day": "",
            "time": "",
            "subject": "",
            "teacher": "",
            "room": "",
            "type": "",
            "week_type": "",
            "topic": topic,
            "url": url,
        })
    
    return items


def index_data(
    client: QdrantClient,
    collection_name: str,
    items: list[dict],
    model,
    batch_size: int = 100,
) -> int:
    total = len(items)
    indexed = 0
    
    for i in range(0, total, batch_size):
        batch = items[i:i + batch_size]
        texts = [item["content"] for item in batch]
        
        with torch.no_grad():
            vectors = model.encode(texts).tolist()
        
        points = [
            PointStruct(
                id=i + j,
                vector=vectors[j],
                payload=batch[j],
            )
            for j in range(len(batch))
        ]
        
        client.upsert(
            collection_name=collection_name,
            points=points,
        )
        
        indexed += len(batch)
        print(f"    {indexed}/{total}")
    
    return indexed


def main():
    parser = argparse.ArgumentParser(description="Indexer - 3 collections in Qdrant")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate all collections")
    parser.add_argument("--collection", choices=["schedule", "faculty", "university"],
                        help="Index only one collection")
    args = parser.parse_args()

    print("=" * 60)
    print("  INDEXER - 3 collections in Qdrant")
    print("=" * 60)

    client = get_client()
    
    print("\n[1/2] Loading model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    model = model.to(device)
    print(f"    {EMBEDDING_MODEL}")

    print("\n[2/2] Indexing...")

    collections_to_process = [args.collection] if args.collection else ["schedule", "faculty", "university"]

    for col_key in collections_to_process:
        collection_name = COLLECTIONS[col_key]
        print(f"\n  >> {collection_name}")
        
        if args.reset:
            create_collection(client, collection_name)
        elif not collection_exists(client, collection_name):
            create_collection(client, collection_name)

        if col_key == "schedule":
            data = load_json(SCHEDULE_FILE)
            if isinstance(data, list):
                items = schedule_to_items(data)
            else:
                items = []
        elif col_key == "faculty":
            data = load_json(FACULTY_FILE)
            if isinstance(data, dict):
                items = faculty_to_items(data)
            else:
                items = []
        else:
            data = load_json(UNIVERSITY_FILE)
            if isinstance(data, dict):
                items = university_to_items(data)
            else:
                items = []

        if not items:
            print(f"    No data for {collection_name}")
            continue

        print(f"    Items: {len(items)}")
        index_data(client, collection_name, items, model)

    print("\n" + "=" * 60)
    print("[OK] Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
