"""
University Parser — парсер информации СГУ для студентов и абитуриентов
===================================================================

Собирает информацию общего характера:
- Общежития
- Материальная помощь
- Стипендии
- Военное обучение
- Поддержка студентов

Использование:
    python university_parser.py
"""

import argparse
import json
import os
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
MIN_CHUNK_LENGTH = 30

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
    return "; ".join(lines)


def extract_chunks(
    soup: BeautifulSoup,
    url: str,
    page_title: str,
    topic: str,
) -> list[dict]:
    root = get_content_root(soup)
    if not root:
        return []

    chunks = []
    section_title = page_title
    parts = []
    list_items = []
    highlights = []
    tables = []
    seen = set()
    processed = set()

    def flush():
        nonlocal parts, list_items, highlights, tables
        content = clean(" ".join(parts))
        if len(content) < MIN_CHUNK_LENGTH:
            parts, list_items, highlights, tables = [], [], [], []
            return

        chunk = {
            "id": make_id(url, section_title),
            "source_url": url,
            "title": section_title,
            "page_title": page_title,
            "topic": topic,
            "content": content,
        }
        meta = {}
        if list_items:
            meta["list_items"] = list(dict.fromkeys(list_items))
        if highlights:
            meta["highlights"] = list(dict.fromkeys(highlights))
        if tables:
            meta["table_data"] = tables
        if meta:
            chunk["metadata"] = meta

        chunks.append(chunk)
        parts, list_items, highlights, tables = [], [], [], []

    for el in root.find_all(
        ["h1", "h2", "h3", "p", "li", "strong", "b",
         "td", "th", "blockquote", "table"],
        recursive=True,
    ):
        eid = id(el)
        if eid in processed:
            continue
        processed.add(eid)

        tag = el.name
        text = clean(el.get_text())
        if not text:
            continue

        if tag in ("h1", "h2", "h3"):
            flush()
            section_title = text

        elif tag == "p":
            if text not in seen:
                parts.append(text)
                seen.add(text)

        elif tag == "li":
            if el.parent and el.parent.name in ("ul", "ol"):
                if not el.find(["ul", "ol"]) and text not in seen:
                    list_items.append(text)
                    parts.append(f"• {text}")
                    seen.add(text)

        elif tag in ("strong", "b"):
            if len(text) > 3 and text not in highlights:
                highlights.append(text)

        elif tag == "table":
            tdata = parse_html_table(el)
            if tdata:
                tables.extend(tdata)
                ttext = table_to_text(tdata)
                if ttext and ttext not in seen:
                    parts.append(ttext)
                    seen.add(ttext)
            for cell in el.find_all(["td", "th"]):
                processed.add(id(cell))

        elif tag == "blockquote":
            if text not in seen:
                parts.append(f'"{text}"')
                seen.add(text)

    flush()
    return chunks


def crawl() -> list[dict]:
    all_chunks = []
    visited = set()
    queue = list(dict.fromkeys(SEED_URLS))

    print(f"University Parser — Crawling SGU")
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

        chunks = extract_chunks(soup, url, page_title, topic)
        print(f"       -> {len(chunks)} chunks | {topic}")
        all_chunks.extend(chunks)

        for a in soup.find_all("a", href=True):
            norm = normalize_url(a["href"], url)
            if norm and norm not in visited and norm not in queue:
                queue.append(norm)

        time.sleep(REQUEST_DELAY)

    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="University Parser — парсер информации СГУ")
    args = parser.parse_args()

    print("=" * 60)
    print("  University Parser — Информация СГУ для студентов")
    print("=" * 60)

    chunks = crawl()

    output = {
        "project": "NETutor",
        "source": "sgu.ru — Университетская информация",
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
