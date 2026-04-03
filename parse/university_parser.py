"""
University Parser — парсер информации СГУ с семантическим чанкованием
========================================================================

Собирает информацию общего характера:
- Общежития
- Материальная помощь
- Стипендии
- Военное обучение
- Поддержка студентов

Особенности:
- Семантическое чанкование: разбиение по смыслу, не по размеру
- Каждый чанк = одна логическая секция (до h2/h3)
- Объединение коротких параграфов
- Минимум 2 предложения или списки

Использование:
    python university_parser.py
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

try:
    import requests
    from bs4 import BeautifulSoup, Tag
except ImportError:
    print("Установи зависимости:\n  pip install requests beautifulsoup4")
    sys.exit(1)


BASE_URL = "https://www.sgu.ru"
REQUEST_TIMEOUT = 15
REQUEST_DELAY = 1.0
MAX_PAGES = 50

OUTPUT_FILE = "university_chunks.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NETutor-RAG-Crawler/1.0)",
    "Accept-Language": "ru-RU,ru;q=0.9",
}

TOPIC_RULES = [
    (r"obschezhitiya", "dormitory"),
    (r"materialnaya-pomosch", "financial_aid"),
    (r"stipend", "scholarship"),
    (r"voen", "military"),
    (r"svoy-pomosch|social", "student_support"),
    (r"pitani", "food"),
    (r"psiholog", "psychology"),
    (r"bronnik|voinskiy", "military"),
    (r"svo|uchastnikov", "military_support"),
    (r"studentskaya-zhizn|studencheskaya", "student_life"),
]

JUNK_SELECTORS = [
    "header", "footer", "nav",
    ".menu", ".navigation", ".breadcrumb",
    ".region-primary-menu", ".region-secondary-menu",
    ".block-menu", ".block-superfish",
    ".social", ".copyright",
    "script", "style", "noscript",
]

SEED_URLS = [
    f"{BASE_URL}/struktura/social/studencheskie-obschezhitiya",
    f"{BASE_URL}/struktura/social/v-pomosch-studentu",
    f"{BASE_URL}/struktura/social/v-pomosch-studentu/materialnaya-pomosch",
    f"{BASE_URL}/struktura/social/v-pomosch-studentu/stipendialnoe-obespechenie",
    f"{BASE_URL}/voennoe-obuchenie",
    f"{BASE_URL}/studentu-vsyo-pro-uchyobu/vstat-na-voinskiy-uchyot",
    f"{BASE_URL}/struktura/social/v-pomosch-studentu/mery-podderzhki-uchastnikov-i-chlenov-semey-uchastnikov-svo",
]


def make_id(url: str, suffix: str = "") -> str:
    import hashlib
    return hashlib.md5((url + "|" + suffix).encode()).hexdigest()[:16]


def clean(text: str) -> str:
    text = re.sub(r"[\r\n\t]+", " ", text)
    return re.sub(r" {2,}", " ", text).strip()


def count_sentences(text: str) -> int:
    """Подсчёт предложений (по точке, восклицательному, вопросительному)."""
    sentences = re.split(r"[.!?]+", text)
    return len([s for s in sentences if len(s.strip()) > 3])


def detect_topic(url: str) -> str:
    for pattern, topic in TOPIC_RULES:
        if re.search(pattern, url, re.IGNORECASE):
            return topic
    return "general"


def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc != "www.sgu.ru":
        return False
    path = parsed.path
    if re.search(r"\.(pdf|doc|docx|xls|xlsx|zip|rar|png|jpg|jpeg|gif)$", path, re.I):
        return False
    return True


def normalize_url(href: str, base: str) -> Optional[str]:
    full = urljoin(base, href)
    p = urlparse(full)
    clean_url = p._replace(query="", fragment="").geturl()
    return clean_url if is_valid_url(clean_url) else None


http_session = requests.Session()
http_session.headers.update(HEADERS)


def fetch_html(url: str) -> Optional[str]:
    try:
        r = http_session.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        return r.text
    except requests.RequestException as e:
        print(f"    Error: {e}")
        return None


def strip_junk(soup: BeautifulSoup) -> None:
    for sel in JUNK_SELECTORS:
        for el in soup.select(sel):
            el.decompose()


def get_content_root(soup: BeautifulSoup) -> Optional[Tag]:
    for sel in ["#main-content", "main", "article",
                ".node__content", ".layout-container", ".region-content"]:
        el = soup.select_one(sel)
        if el:
            return el
    return soup.find("body")


def parse_html_table(table: Tag) -> list[dict]:
    rows = table.find_all("tr")
    if not rows:
        return []

    headers = []
    result = []

    for i, row in enumerate(rows):
        cells = row.find_all(["th", "td"])
        texts = [clean(c.get_text()) for c in cells]
        if not any(texts):
            continue
        if i == 0 and row.find("th"):
            headers = texts
            continue
        if headers:
            result.append(dict(zip(headers, texts)))
        else:
            result.append({str(j): v for j, v in enumerate(texts)})

    return result


def table_to_text(rows: list[dict]) -> str:
    lines = [" | ".join(f"{k}: {v}" for k, v in row.items() if v) for row in rows]
    return ". ".join(lines)


def extract_semantic_chunks(
    soup: BeautifulSoup,
    url: str,
    page_title: str,
    topic: str,
) -> list[dict]:
    """
    Семантическое чанкование:
    - Чанк = всё до следующего h2 (или конец страницы)
    - Короткие параграфы (<2 предложений) объединяются с предыдущим
    - h3 создаёт подсекцию внутри чанка
    """
    root = get_content_root(soup)
    if not root:
        return []

    chunks = []
    
    current_section = {
        "title": page_title,
        "paragraphs": [],
        "list_items": [],
        "highlights": [],
        "tables": [],
        "h3_sections": [],
    }
    
    current_h3 = None
    seen = set()

    def flush_section(is_h3: bool = False):
        nonlocal current_section, current_h3
        
        content_parts = []
        
        if current_h3:
            section_data = current_h3
        else:
            section_data = current_section
        
        paragraphs_text = " ".join(section_data["paragraphs"])
        
        if paragraphs_text and count_sentences(paragraphs_text) >= 1:
            content_parts.append(paragraphs_text)
        
        if section_data["list_items"]:
            content_parts.append(" ".join(f"• {item}" for item in section_data["list_items"]))
        
        if section_data["tables"]:
            content_parts.append(table_to_text(section_data["tables"]))
        
        content = clean(" ".join(content_parts))
        
        title = current_h3["title"] if current_h3 else section_data["title"]
        
        if content and len(content) > 20:
            chunk = {
                "id": make_id(url, title),
                "source_url": url,
                "title": title,
                "page_title": page_title,
                "topic": topic,
                "content": content,
            }
            meta = {}
            if section_data["highlights"]:
                meta["highlights"] = list(dict.fromkeys(section_data["highlights"]))
            if section_data["list_items"]:
                meta["list_items"] = list(dict.fromkeys(section_data["list_items"]))
            if section_data["tables"]:
                meta["table_data"] = section_data["tables"]
            if meta:
                chunk["metadata"] = meta
            
            chunks.append(chunk)
        
        if current_h3 and not is_h3:
            current_section["paragraphs"].extend(current_h3["paragraphs"])
            current_section["list_items"].extend(current_h3["list_items"])
            current_h3 = None

    for el in root.find_all(["h1", "h2", "h3", "p", "li", "strong", "b", "blockquote", "table"], recursive=True):
        tag = el.name
        text = clean(el.get_text())
        if not text or len(text) < 3:
            continue
        
        if tag in ("h1", "h2"):
            flush_section()
            current_section = {
                "title": text,
                "paragraphs": [],
                "list_items": [],
                "highlights": [],
                "tables": [],
                "h3_sections": [],
            }
            current_h3 = None
        
        elif tag == "h3":
            if current_h3:
                flush_section(is_h3=True)
            current_h3 = {
                "title": text,
                "paragraphs": [],
                "list_items": [],
                "highlights": [],
                "tables": [],
            }
        
        elif tag == "p":
            target = current_h3 if current_h3 else current_section
            if text not in seen:
                target["paragraphs"].append(text)
                seen.add(text)
        
        elif tag == "li":
            if el.parent and el.parent.name in ("ul", "ol"):
                if not el.find(["ul", "ol"]) and text not in seen:
                    target = current_h3 if current_h3 else current_section
                    target["list_items"].append(text)
                    seen.add(text)
        
        elif tag in ("strong", "b"):
            target = current_h3 if current_h3 else current_section
            if len(text) > 3 and text not in seen:
                target["highlights"].append(text)
        
        elif tag == "blockquote":
            target = current_h3 if current_h3 else current_section
            if text not in seen:
                target["paragraphs"].append(f'"{text}"')
                seen.add(text)
        
        elif tag == "table":
            tdata = parse_html_table(el)
            if tdata:
                target = current_h3 if current_h3 else current_section
                target["tables"].extend(tdata)
                for cell in el.find_all(["td", "th"]):
                    if cell not in root.find_all(["td", "th"]):
                        continue

    flush_section()

    return chunks


def crawl() -> list[dict]:
    all_chunks = []
    visited = set()
    queue = list(dict.fromkeys(SEED_URLS))

    print(f"University Parser — Semantic Chunking")
    print(f"Starting URLs: {len(queue)}")
    print()

    pages_done = 0

    while queue and pages_done < MAX_PAGES:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        pages_done += 1

        print(f"[{pages_done:>3}] {url}")
        html = fetch_html(url)
        if not html:
            time.sleep(REQUEST_DELAY)
            continue

        soup = BeautifulSoup(html, "html.parser")
        strip_junk(soup)

        h1 = soup.find("h1")
        page_title = clean(h1.get_text()) if h1 else url
        topic = detect_topic(url)

        chunks = extract_semantic_chunks(soup, url, page_title, topic)
        print(f"       -> {len(chunks)} chunks | {topic}")
        all_chunks.extend(chunks)

        for a in soup.find_all("a", href=True):
            norm = normalize_url(a["href"], url)
            if norm and norm not in visited and norm not in queue:
                queue.append(norm)

        time.sleep(REQUEST_DELAY)

    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="University Parser — семантическое чанкование")
    args = parser.parse_args()

    print("=" * 60)
    print("  University Parser — Semantic Chunking")
    print("=" * 60)

    chunks = crawl()

    output = {
        "project": "NETutor",
        "source": "sgu.ru — University Info",
        "total_chunks": len(chunks),
        "chunks": chunks,
    }
    
    output_path = Path(__file__).parent / OUTPUT_FILE
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"[OK] Done! Chunks: {len(chunks)} -> {OUTPUT_FILE}")
    print("\nBy topic:")
    for topic, cnt in Counter(c["topic"] for c in chunks).most_common():
        print(f"  {topic:<25} {cnt:>4}")


if __name__ == "__main__":
    main()
