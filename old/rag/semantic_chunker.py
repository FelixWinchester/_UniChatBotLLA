import re
from typing import List, Dict, Any


def chunk_schedule(data: List[Dict]) -> List[Dict]:
    """Чанкование данных расписания по парам."""
    chunks = []
    chunk_id = 0
    
    for group_data in data:
        group = group_data.get("group", "unknown")
        days = group_data.get("days", {})
        
        for day_name, lessons in days.items():
            for lesson in lessons:
                if not lesson.get("subject"):
                    continue
                    
                content = f"{day_name}: {lesson['time']} - {lesson['subject']}"
                if lesson.get("type"):
                    content += f" ({lesson['type']})"
                if lesson.get("teacher"):
                    content += f", преп. {lesson['teacher']}"
                if lesson.get("room"):
                    content += f", ауд. {lesson['room']}"
                if lesson.get("week_type") and lesson['week_type'] != "Всегда":
                    content += f", {lesson['week_type']}"
                
                chunks.append({
                    "id": chunk_id,
                    "content": content,
                    "metadata": {
                        "type": "schedule",
                        "group": group,
                        "day": day_name,
                        "time": lesson.get("time", ""),
                        "subject": lesson.get("subject", ""),
                        "teacher": lesson.get("teacher", ""),
                        "room": lesson.get("room", ""),
                        "week_type": lesson.get("week_type", "Всегда")
                    }
                })
                chunk_id += 1
    
    return chunks


def chunk_faculty_data(data: Dict) -> List[Dict]:
    """Семантическое чанкование данных о факультете по секциям."""
    chunks = []
    chunk_id = 0
    
    directions = data.get("directions", [])
    
    for direction in directions:
        name = direction.get("name", "Unknown")
        documents = direction.get("documents", [])
        
        for doc in documents:
            doc_type = doc.get("type", "Document")
            sections = doc.get("sections", [])
            
            for section in sections:
                category = section.get("category", "")
                section_chunks = _chunk_section_recursive(category, section, name)
                
                for content in section_chunks:
                    chunks.append({
                        "id": chunk_id,
                        "content": content,
                        "metadata": {
                            "type": "faculty",
                            "direction": name,
                            "document_type": doc_type,
                            "category": category
                        }
                    })
                    chunk_id += 1
    
    return chunks


def _chunk_section_recursive(category: str, section: Dict, direction: str, depth: int = 0) -> List[str]:
    """Рекурсивно извлекает чанки из секции."""
    chunks = []
    
    if depth > 2:
        return chunks
    
    details = section.get("details", {})
    if details:
        content_parts = []
        for key, value in details.items():
            if value:
                content_parts.append(f"{_normalize_key(key)}: {value}")
        
        if content_parts:
            header = f"{direction} - {category}"
            chunks.append(f"{header}\n" + "\n".join(content_parts))
    
    standards = section.get("standards", [])
    if standards:
        standards_text = "\n".join([f"- {s.get('name', '')} (код: {s.get('code', '')})" for s in standards])
        header = f"{direction} - {category}"
        chunks.append(f"{header}\n{standards_text}")
    
    courses = section.get("courses", [])
    if courses:
        for i in range(0, len(courses), 5):
            course_batch = courses[i:i+5]
            courses_text = "\n".join([
                f"- {c.get('name', '')} ({c.get('type', '')})" 
                for c in course_batch
            ])
            header = f"{direction} - {category}"
            chunks.append(f"{header}\n{courses_text}")
    
    for key in ["details", "standards", "courses"]:
        if key in section:
            continue
        for sub_key, sub_value in section.items():
            if isinstance(sub_value, dict):
                sub_chunks = _chunk_section_recursive(
                    f"{category} - {sub_key}", 
                    sub_value, 
                    direction, 
                    depth + 1
                )
                chunks.extend(sub_chunks)
    
    return chunks


def _normalize_key(key: str) -> str:
    """Нормализует ключи для читаемости."""
    key_map = {
        "code": "Код направления",
        "degree": "Степень",
        "duration_years": "Срок обучения",
        "language": "Язык обучения",
        "department": "Кафедра",
        "profile": "Профиль"
    }
    return key_map.get(key, key.replace("_", " ").capitalize())


def chunk_university_info(data: List[Dict]) -> List[Dict]:
    """Чанкование общей информации об университете."""
    chunks = []
    chunk_id = 0
    
    for item in data:
        content = item.get("content", "")
        if not content:
            continue
            
        title = item.get("title", "")
        
        chunks.append({
            "id": chunk_id,
            "content": f"{title}\n{content}" if title else content,
            "metadata": {
                "type": "university",
                "title": title,
                "category": item.get("category", "")
            }
        })
        chunk_id += 1
    
    return chunks


def semantic_chunk(data: Dict, data_type: str) -> List[Dict]:
    """Главная функция семантического чанкования."""
    if data_type == "schedule":
        return chunk_schedule(data)
    elif data_type == "faculty":
        return chunk_faculty_data(data)
    elif data_type == "university":
        return chunk_university_info(data)
    else:
        raise ValueError(f"Unknown data type: {data_type}")
