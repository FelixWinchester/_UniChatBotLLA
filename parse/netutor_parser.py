"""
NETutor — Единый парсер данных факультета КНиИТ СГУ
====================================================
Всё в одном файле: краулер сайта + Vision для учебных планов
+ повторная попытка для неудачных картинок.

Что парсится:
  • Весь раздел /struktura/computersciences/ (HTML-страницы)
  • Учебные планы (картинки) — через Claude Vision API
  • Алгоритм поступления, направления, студенческая жизнь и др.

Структура чанка (единица для RAG):
  {
    "id":         "MD5(url + заголовок раздела)",
    "source_url": "https://www.sgu.ru/...",
    "title":      "Заголовок раздела",
    "page_title": "H1 всей страницы",
    "topic":      "study_plans | admission | direction_bachelor | ...",
    "content":    "Текст — именно это идёт в эмбеддинг",
    "metadata":   { список_пунктов, таблицы, structured_curriculum, ... }
  }

Установка:
    pip install requests beautifulsoup4 openai python-dotenv

Переменная окружения (для Vision):
    Windows:    set OPENROUTER_API_KEY=sk-or-...
    Linux/Mac:  export OPENROUTER_API_KEY=sk-or-...
    Или файл:   .env  →  OPENROUTER_API_KEY=sk-or-...

    Бесплатный ключ: https://openrouter.ai → Sign Up → Keys

Использование:
    # Полный запуск (краулер + Vision)
    python netutor_parser.py

    # Только повторить неудачные картинки
    python netutor_parser.py --retry

Выходные файлы:
    netutor_chunks.json      — все чанки для RAG
    netutor_failed.json      — картинки, которые не удалось распознать
                               (только если есть неудачи)
"""

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from typing import Optional
from urllib.parse import urljoin, urlparse

# ── Обязательные зависимости ──────────────────────────────────────────────────
try:
    import requests
    from bs4 import BeautifulSoup, Tag
except ImportError:
    print("Установи зависимости:\n  pip install requests beautifulsoup4")
    sys.exit(1)

# Vision использует чистый requests — никакие SDK не нужны
OPENAI_AVAILABLE = True  # всегда True, SDK не требуется

# ── python-dotenv (опционально) ───────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================================
# КОНФИГ
# =============================================================================

BASE_URL     = "https://www.sgu.ru"
CRAWL_PREFIX = "/struktura/computersciences"   # весь раздел факультета

SEED_URLS = [
    f"{BASE_URL}{CRAWL_PREFIX}/zhizn-fakulteta/uchebnye-plany-napravleniy-podgotovki-fakulteta-kniit",
    f"{BASE_URL}{CRAWL_PREFIX}/postupit-k-nam",
    f"{BASE_URL}{CRAWL_PREFIX}/postupit-k-nam/algoritm-tvoego-postupleniya-na-byudzhet-ili-kak-prosto",
    f"{BASE_URL}{CRAWL_PREFIX}",
]

# ── OpenRouter API ключ ──────────────────────────────────────────────────────
OPENROUTER_API_KEY = "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"
os.environ.setdefault("OPENROUTER_API_KEY", OPENROUTER_API_KEY)

REQUEST_DELAY    = 1.2    # сек между запросами к сайту (вежливость)
VISION_DELAY     = 2.0    # сек между вызовами Claude Vision
MAX_PAGES        = 200    # защита от бесконечного обхода
MIN_CHUNK_LENGTH = 40     # минимум символов для сохранения чанка

# openrouter/free — специальный роутер OpenRouter, который автоматически
# выбирает доступную бесплатную модель с поддержкой vision.
# Не нужно следить за конкретными моделями — роутер делает это сам.
VISION_MODEL = "openrouter/healer-alpha"

OUTPUT_CHUNKS = "netutor_chunks.json"
OUTPUT_FAILED = "netutor_failed.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NETutor-RAG-Crawler/1.0)",
    "Accept-Language": "ru-RU,ru;q=0.9",
}

# Маппинг URL → топик (порядок важен: более специфичные — первыми)
TOPIC_RULES: list[tuple[str, str]] = [
    (r"\d{2}-03-\d{2}",         "direction_bachelor"),
    (r"\d{2}-04-\d{2}",         "direction_master"),
    (r"\d{2}-05-\d{2}",         "direction_specialist"),
    ("uchebnye-plany",          "study_plans"),
    ("algoritm",                "admission"),
    ("o-rabote-priemnoy",       "admission"),
    ("privetstvennoe-slovo",    "faculty_info"),
    ("informaciya-o-fakultete", "faculty_info"),
    ("gde-rabotaet",            "alumni"),
    ("studencheskaya-zhizn",    "student_life"),
    ("prepodavateli",           "staff"),
    ("kontakty",                "contacts"),
    ("postupit-k-nam",          "admission"),
]

# Блоки, которые нужно вырезать из страниц (меню, навигация, футер)
JUNK_SELECTORS = [
    "header", "footer", "nav",
    ".menu", ".navigation", ".breadcrumb",
    ".region-primary-menu", ".region-secondary-menu",
    ".block-menu", ".block-superfish",
    ".social", ".copyright",
    "script", "style", "noscript",
    ".mini-slider",
]

# Промпт для Claude Vision — универсальный, не привязан к конкретной структуре таблицы
VISION_PROMPT = """Перед тобой учебный план университетского направления подготовки.
Внимательно изучи изображение и извлеки всю информацию.

Верни ответ СТРОГО в формате JSON (без markdown-блоков, без пояснений):
{
  "direction": "название направления если видно на картинке",
  "degree": "бакалавриат | специалитет | магистратура",
  "duration_years": 4,
  "curriculum": [
    {
      "year": 1,
      "semester": 1,
      "subjects": [
        {
          "name": "Название предмета",
          "hours_total": 144,
          "hours_lecture": 36,
          "hours_practice": 36,
          "hours_lab": 0,
          "hours_self_study": 72,
          "credits": 4,
          "control": "экзамен | зачёт | зачёт с оценкой | курсовая"
        }
      ]
    }
  ],
  "practices": ["Учебная практика — 2 семестр", "..."],
  "thesis": "описание ВКР если есть",
  "notes": "любая другая важная информация с картинки"
}

Правила:
- Если поле не видно — ставь null
- Если таблица устроена иначе — адаптируй структуру, но сохрани все данные
- Не теряй ни одного предмета и ни одного семестра
- Отвечай ТОЛЬКО JSON, никакого текста вокруг"""


# =============================================================================
# УТИЛИТЫ
# =============================================================================

def make_id(url: str, suffix: str = "") -> str:
    """Уникальный 16-символьный ID из URL + заголовка."""
    return hashlib.md5((url + "|" + suffix).encode()).hexdigest()[:16]


def clean(text: str) -> str:
    """Нормализует пробелы и переносы строк."""
    text = re.sub(r"[\r\n\t]+", " ", text)
    return re.sub(r" {2,}", " ", text).strip()


def detect_topic(url: str) -> str:
    """Определяет топик чанка по URL."""
    for pattern, topic in TOPIC_RULES:
        if re.search(pattern, url):
            return topic
    return "faculty_general"


def is_faculty_url(url: str) -> bool:
    """Проверяет, что ссылка ведёт внутрь раздела факультета."""
    parsed = urlparse(url)
    return (
        parsed.netloc == "www.sgu.ru"
        and parsed.path.startswith(CRAWL_PREFIX)
        and not re.search(
            r"\.(pdf|doc|docx|xls|xlsx|zip|rar|png|jpg|jpeg|gif)$",
            parsed.path, re.I,
        )
    )


def normalize_url(href: str, base: str) -> Optional[str]:
    """Приводит ссылку к абсолютному виду, убирает якорь и query.
    Фильтрует задвоенные пути /struktura/.../struktura/...
    которые возникают из-за относительных ссылок на страницах СГУ.
    """
    full = urljoin(base, href)
    p = urlparse(full)
    path = p.path
    # Если путь содержит CRAWL_PREFIX дважды — берём второе вхождение
    if path.count(CRAWL_PREFIX) > 1:
        second = path.index(CRAWL_PREFIX, len(CRAWL_PREFIX))
        path = path[second:]
    clean_url = p._replace(path=path, query="", fragment="").geturl()
    return clean_url if is_faculty_url(clean_url) else None


# =============================================================================
# СЕТЬ
# =============================================================================

http_session = requests.Session()
http_session.headers.update(HEADERS)


def fetch_html(url: str) -> Optional[str]:
    """Загружает HTML страницы. Возвращает None при ошибке."""
    try:
        r = http_session.get(url, timeout=15)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        return r.text
    except requests.RequestException as e:
        print(f"    ✗ {e}")
        return None


def fetch_image_b64(url: str) -> Optional[tuple[str, str]]:
    """Скачивает изображение, возвращает (base64_string, media_type)."""
    try:
        r = http_session.get(url, timeout=20)
        r.raise_for_status()
        media_type = r.headers.get("Content-Type", "image/png").split(";")[0].strip()
        # Claude поддерживает только эти типы
        if media_type not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
            media_type = "image/png"
        b64 = base64.standard_b64encode(r.content).decode("utf-8")
        return b64, media_type
    except requests.RequestException as e:
        print(f"    ✗ Изображение {url}: {e}")
        return None


# =============================================================================
# OPENROUTER VISION (бесплатный tier, чистый requests — без SDK)
# =============================================================================

OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"


def parse_image_with_vision(img_url: str, direction_title: str) -> dict:
    """
    Отправляет картинку учебного плана в OpenRouter Vision.
    Использует openrouter/healer-alpha — роутер сам выбирает доступную
    бесплатную модель с поддержкой vision. Чистый requests, никаких SDK.
    Возвращает распознанный JSON или {"error": "..."} при неудаче.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return {"error": "OPENROUTER_API_KEY не задан"}

    img_data = fetch_image_b64(img_url)
    if not img_data:
        return {"error": f"Не удалось скачать картинку: {img_url}"}

    b64_str, media_type = img_data

    payload = {
        "model": VISION_MODEL,
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{b64_str}"},
                    },
                    {
                        "type": "text",
                        "text": VISION_PROMPT,
                    },
                ],
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/netutor",
        "X-Title": "NETutor",
    }

    try:
        resp = requests.post(OPENROUTER_BASE, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        raw = data["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except requests.HTTPError:
        return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except json.JSONDecodeError as e:
        raw_preview = raw[:200] if "raw" in dir() else "нет ответа"
        return {"error": f"JSON parse: {e}", "raw": raw_preview}
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# КОНВЕРТАЦИЯ УЧЕБНОГО ПЛАНА В ТЕКСТ
# =============================================================================

def curriculum_to_text(data: dict, direction_title: str) -> str:
    """
    Превращает структурированный учебный план в читаемый текст для эмбеддинга.
    Работает с любой структурой которую вернула Vision-модель.
    """
    parts = [f"Учебный план направления {direction_title}."]

    if data.get("degree"):
        parts.append(f"Уровень подготовки: {data['degree']}.")
    if data.get("duration_years"):
        parts.append(f"Срок обучения: {data['duration_years']} лет.")

    for block in data.get("curriculum", []):
        year     = block.get("year", "?")
        sem      = block.get("semester", "?")
        subjects = block.get("subjects", [])
        if subjects:
            names = [
                s.get("name", str(s)) if isinstance(s, dict) else str(s)
                for s in subjects
            ]
            parts.append(f"{year} курс, {sem} семестр: {', '.join(names)}.")

    if data.get("practices"):
        parts.append("Практики: " + "; ".join(data["practices"]) + ".")

    if data.get("thesis"):
        parts.append(f"ВКР: {data['thesis']}.")

    if data.get("notes"):
        parts.append(data["notes"])

    return " ".join(parts)


# =============================================================================
# HTML → ЧАНКИ
# =============================================================================

def strip_junk(soup: BeautifulSoup) -> None:
    """Удаляет меню, навигацию, футер и прочий мусор."""
    for sel in JUNK_SELECTORS:
        for el in soup.select(sel):
            el.decompose()


def get_content_root(soup: BeautifulSoup) -> Optional[Tag]:
    """Находит основной контентный блок страницы."""
    for sel in ["#main-content", "main", "article",
                ".node__content", ".layout-container", ".region-content"]:
        el = soup.select_one(sel)
        if el:
            return el
    return soup.find("body")


def parse_html_table(table: Tag) -> list[dict]:
    """Парсит HTML-таблицу в список словарей {заголовок: значение}."""
    rows = table.find_all("tr")
    if not rows:
        return []

    headers: list[str] = []
    result:  list[dict] = []

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
    """Превращает таблицу в строку для эмбеддинга."""
    lines = [
        " | ".join(f"{k}: {v}" for k, v in row.items() if v)
        for row in rows
    ]
    return "; ".join(lines)


def extract_chunks(
    soup: BeautifulSoup,
    url: str,
    page_title: str,
    topic: str,
) -> list[dict]:
    """
    Основной парсер HTML-страниц.
    Разбивает контент на чанки по заголовкам h2/h3.
    Каждый чанк — один смысловой раздел страницы.
    """
    root = get_content_root(soup)
    if not root:
        return []

    chunks: list[dict] = []

    # Состояние текущего чанка
    section_title = page_title
    parts:      list[str] = []
    list_items: list[str] = []
    highlights: list[str] = []
    tables:     list[dict] = []
    seen:       set[str]  = set()   # дедупликация внутри страницы
    processed:  set[int]  = set()   # уже обработанные узлы

    def flush():
        """Сохраняет накопленный чанк и сбрасывает буферы."""
        nonlocal parts, list_items, highlights, tables
        content = clean(" ".join(parts))
        if len(content) < MIN_CHUNK_LENGTH:
            parts, list_items, highlights, tables = [], [], [], []
            return

        chunk: dict = {
            "id":         make_id(url, section_title),
            "source_url": url,
            "title":      section_title,
            "page_title": page_title,
            "topic":      topic,
            "content":    content,
        }
        meta: dict = {}
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

        tag  = el.name
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
            # Только листовые li (без вложенных списков)
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
            # Помечаем все ячейки таблицы как обработанные
            for cell in el.find_all(["td", "th"]):
                processed.add(id(cell))

        elif tag == "blockquote":
            if text not in seen:
                parts.append(f'"{text}"')
                seen.add(text)

    flush()
    return chunks


# =============================================================================
# ПАРСЕР УЧЕБНЫХ ПЛАНОВ (картинки → Claude Vision)
# =============================================================================

def parse_study_plans(soup: BeautifulSoup, url: str) -> tuple[list[dict], list[dict]]:
    """
    Парсит страницу с учебными планами.
    Для каждого направления:
      - Находит картинку
      - Отправляет в Claude Vision
      - При успехе: чанк с полным текстом предметов по семестрам
      - При ошибке: чанк только со ссылкой на картинку

    Возвращает (chunks, failed_list).
    failed_list — список направлений для которых Vision не сработал.
    """
    chunks:      list[dict] = []
    failed_list: list[dict] = []

    root = get_content_root(soup)
    if not root:
        return chunks, failed_list

    api_key   = os.environ.get("OPENROUTER_API_KEY", "")
    vision_on = bool(api_key) and OPENAI_AVAILABLE

    if not vision_on:
        reasons = []
        if not OPENAI_AVAILABLE:
            reasons.append("pip install openai")
        if not api_key:
            reasons.append("задай OPENROUTER_API_KEY")
        print(f"    [!] Vision отключён ({', '.join(reasons)})")
        print(f"        Учебные планы сохранятся только со ссылками на картинки.")

    for h2 in root.find_all("h2"):
        direction_title = clean(h2.get_text())
        if not re.match(r"\d{2}\.\d{2}\.\d{2}", direction_title):
            continue

        code = direction_title.split()[0]
        name = " ".join(direction_title.split()[1:])

        # Ссылки на картинки
        img_full    = None
        img_preview = None
        next_a = h2.find_next("a")
        if next_a:
            href = next_a.get("href", "")
            if "styles" in href:
                img_full = urljoin(BASE_URL, href)
            img_tag = next_a.find("img")
            if img_tag:
                img_preview = urljoin(BASE_URL, img_tag.get("src", ""))

        # Vision
        vision_data: dict = {}
        vision_ok = False

        if vision_on and img_full:
            print(f"    Vision → {direction_title} ...", end=" ", flush=True)
            vision_data = parse_image_with_vision(img_full, direction_title)

            if vision_data.get("error"):
                print(f"✗ {vision_data['error']}")
                failed_list.append({
                    "direction_title": direction_title,
                    "direction_code":  code,
                    "image_url":       img_full,
                    "source_url":      url,
                    "error":           vision_data["error"],
                })
                vision_data = {}
            else:
                sems = len(vision_data.get("curriculum", []))
                print(f"✓ ({sems} сем.)")
                vision_ok = True

            time.sleep(VISION_DELAY)

        # content: полный текст если Vision сработал, иначе ссылка
        if vision_ok:
            content = curriculum_to_text(vision_data, direction_title)
        else:
            content = (
                f"Учебный план направления {direction_title} — {name}. "
                f"Код направления: {code}. "
                f"Учебный план доступен в виде изображения: "
                f"{img_full or 'ссылка недоступна'}."
            )

        chunks.append({
            "id":         make_id(url, direction_title),
            "source_url": url,
            "title":      f"Учебный план: {direction_title}",
            "page_title": "Учебные планы направлений КНиИТ",
            "topic":      "study_plans",
            "content":    content,
            "metadata": {
                "direction_code":        code,
                "direction_name":        name,
                "image_full_url":        img_full,
                "image_preview_url":     img_preview,
                "vision_extracted":      vision_ok,
                "curriculum_structured": vision_data if vision_ok else None,
            },
        })

    return chunks, failed_list



# =============================================================================
# ПАРСЕР ХАБ-СТРАНИЦ (навигационные карточки без текста)
# =============================================================================

# Страницы-хабы — содержат только карточки-ссылки, дни открытых дверей,
# списки направлений. Обычный extract_chunks возвращает 0 чанков.
HUB_URLS = {
    f"{BASE_URL}{CRAWL_PREFIX}/postupit-k-nam",
}


def is_hub_page(url: str, soup: BeautifulSoup) -> bool:
    """Определяет является ли страница хабом (нет текста, только карточки)."""
    # Явно известные хабы
    if url.rstrip("/") in {u.rstrip("/") for u in HUB_URLS}:
        return True
    return False


def parse_hub_page(soup: BeautifulSoup, url: str, page_title: str, topic: str) -> list[dict]:
    """
    Парсит страницы-хабы: собирает список ссылок-карточек,
    таблицы (дни открытых дверей), списки направлений.
    Формирует один сводный чанк со всей навигационной информацией.
    """
    root = get_content_root(soup)
    if not root:
        return []

    parts:      list[str] = []
    list_items: list[str] = []
    tables:     list[dict] = []
    seen:       set[str]  = set()

    # Карточки — ссылки внутри факультета с текстом
    for a in root.find_all("a", href=True):
        href = a.get("href", "")
        text = clean(a.get_text())
        if (
            CRAWL_PREFIX in href
            and len(text) > 3
            and text not in seen
            # Исключаем сами себя и пункты глобального меню
            and href.rstrip("/") != urlparse(url).path.rstrip("/")
        ):
            parts.append(f"• {text}")
            list_items.append(text)
            seen.add(text)

    # Таблицы (дни открытых дверей и т.п.)
    for table in root.find_all("table"):
        tdata = parse_html_table(table)
        if tdata:
            tables.extend(tdata)
            ttext = table_to_text(tdata)
            if ttext and ttext not in seen:
                parts.append(ttext)
                seen.add(ttext)

    # Аспирантура и прочие текстовые блоки
    for tag in root.find_all(["p", "li"]):
        text = clean(tag.get_text())
        if len(text) > 10 and text not in seen:
            parts.append(text)
            seen.add(text)

    content = clean(" ".join(parts))
    if len(content) < MIN_CHUNK_LENGTH:
        return []

    chunk: dict = {
        "id":         make_id(url, page_title),
        "source_url": url,
        "title":      page_title,
        "page_title": page_title,
        "topic":      topic,
        "content":    content,
    }
    meta: dict = {}
    if list_items:
        meta["nav_links"] = list(dict.fromkeys(list_items))
    if tables:
        meta["table_data"] = tables
    if meta:
        chunk["metadata"] = meta

    return [chunk]


# =============================================================================
# КРАУЛЕР
# =============================================================================

def crawl() -> tuple[list[dict], list[dict]]:
    """
    Обходит весь раздел /struktura/computersciences/.
    Возвращает (все_чанки, неудачные_картинки).
    """
    all_chunks:  list[dict] = []
    all_failed:  list[dict] = []
    visited:     set[str]   = set()
    queue:       list[str]  = list(dict.fromkeys(SEED_URLS))

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    print(f"Префикс : {CRAWL_PREFIX}")
    print(f"Vision  : {'✓ ' + VISION_MODEL if api_key and OPENAI_AVAILABLE else '✗ отключён'}")
    if not api_key:
        print("          → задай OPENROUTER_API_KEY (бесплатно: openrouter.ai)")
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

        h1         = soup.find("h1")
        page_title = clean(h1.get_text()) if h1 else url
        topic      = detect_topic(url)

        if "uchebnye-plany" in url:
            chunks, failed = parse_study_plans(soup, url)
            all_failed.extend(failed)
        elif is_hub_page(url, soup):
            chunks = parse_hub_page(soup, url, page_title, topic)
        else:
            chunks = extract_chunks(soup, url, page_title, topic)

        print(f"       → {len(chunks)} чанков | {topic}")
        all_chunks.extend(chunks)

        # Собираем новые ссылки
        for a in soup.find_all("a", href=True):
            norm = normalize_url(a["href"], url)
            if norm and norm not in visited and norm not in queue:
                queue.append(norm)

        time.sleep(REQUEST_DELAY)

    if pages_done >= MAX_PAGES:
        print(f"\n[!] Достигнут лимит {MAX_PAGES} страниц")

    return all_chunks, all_failed


# =============================================================================
# ПОВТОРНАЯ ОБРАБОТКА НЕУДАЧНЫХ КАРТИНОК
# =============================================================================

def retry_failed() -> None:
    """
    Читает netutor_failed.json, повторно вызывает Vision для каждой картинки
    и обновляет чанки в netutor_chunks.json.
    """
    print("=" * 60)
    print("  NETutor — Повторная обработка неудачных учебных планов")
    print("=" * 60)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("✗ OPENROUTER_API_KEY не задан")
        sys.exit(1)

    if not os.path.exists(OUTPUT_FAILED):
        print(f"✓ {OUTPUT_FAILED} не найден — нечего повторять")
        return

    with open(OUTPUT_FAILED, encoding="utf-8") as f:
        failed_data = json.load(f)

    items = failed_data.get("items", [])
    if not items:
        print("✓ Нет неудачных картинок")
        return

    print(f"Найдено: {len(items)} картинок\n")

    # Загружаем существующие чанки
    if not os.path.exists(OUTPUT_CHUNKS):
        print(f"✗ {OUTPUT_CHUNKS} не найден — сначала запусти полный краулер")
        sys.exit(1)

    with open(OUTPUT_CHUNKS, encoding="utf-8") as f:
        chunks_data = json.load(f)

    chunks: list[dict] = chunks_data["chunks"]
    by_id = {c["id"]: i for i, c in enumerate(chunks)}

    still_failed = []
    success_count = 0

    for item in items:
        direction_title = item["direction_title"]
        img_url         = item["image_url"]
        source_url      = item["source_url"]
        code            = item["direction_code"]

        print(f"[→] {direction_title}")
        vision_data = parse_image_with_vision(img_url, direction_title)

        if vision_data.get("error"):
            print(f"    ✗ {vision_data['error']}")
            still_failed.append({**item, "error": vision_data["error"]})
            time.sleep(VISION_DELAY)
            continue

        sems = len(vision_data.get("curriculum", []))
        print(f"    ✓ ({sems} сем.)")

        content  = curriculum_to_text(vision_data, direction_title)
        chunk_id = make_id(source_url, direction_title)

        if chunk_id in by_id:
            idx = by_id[chunk_id]
            chunks[idx]["content"] = content
            chunks[idx]["metadata"]["vision_extracted"]      = True
            chunks[idx]["metadata"]["curriculum_structured"] = vision_data
        else:
            name = " ".join(direction_title.split()[1:])
            new_chunk = {
                "id":         chunk_id,
                "source_url": source_url,
                "title":      f"Учебный план: {direction_title}",
                "page_title": "Учебные планы направлений КНиИТ",
                "topic":      "study_plans",
                "content":    content,
                "metadata": {
                    "direction_code":        code,
                    "direction_name":        name,
                    "image_full_url":        img_url,
                    "image_preview_url":     None,
                    "vision_extracted":      True,
                    "curriculum_structured": vision_data,
                },
            }
            chunks.append(new_chunk)
            by_id[chunk_id] = len(chunks) - 1

        success_count += 1
        time.sleep(VISION_DELAY)

    # Сохраняем обновлённые чанки
    chunks_data["chunks"]       = chunks
    chunks_data["total_chunks"] = len(chunks)
    with open(OUTPUT_CHUNKS, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    # Обновляем / удаляем failed
    if still_failed:
        with open(OUTPUT_FAILED, "w", encoding="utf-8") as f:
            json.dump({"total": len(still_failed), "items": still_failed},
                      f, ensure_ascii=False, indent=2)
    else:
        os.remove(OUTPUT_FAILED)

    print(f"\n{'=' * 60}")
    print(f"✅ Готово! Успешно: {success_count} / Осталось: {len(still_failed)}")
    if not still_failed:
        print("   Все учебные планы распознаны!")


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NETutor — парсер данных факультета КНиИТ СГУ"
    )
    parser.add_argument(
        "--retry",
        action="store_true",
        help=f"Повторить Vision для неудачных картинок из {OUTPUT_FAILED}",
    )
    args = parser.parse_args()

    if args.retry:
        retry_failed()
        return

    # ── Полный краулер ────────────────────────────────────────────────────────
    print("=" * 60)
    print("  NETutor — RAG-краулер КНиИТ СГУ")
    print("=" * 60)

    chunks, failed = crawl()

    # Сохраняем чанки
    output = {
        "project":      "NETutor",
        "source":       "sgu.ru — Факультет КНиИТ",
        "crawl_prefix": CRAWL_PREFIX,
        "total_chunks": len(chunks),
        "chunks":       chunks,
    }
    with open(OUTPUT_CHUNKS, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Сохраняем неудачные картинки (если есть)
    if failed:
        with open(OUTPUT_FAILED, "w", encoding="utf-8") as f:
            json.dump({"total": len(failed), "items": failed},
                      f, ensure_ascii=False, indent=2)

    # ── Итог ──────────────────────────────────────────────────────────────────
    vision_ok    = sum(1 for c in chunks
                       if c.get("topic") == "study_plans"
                       and c.get("metadata", {}).get("vision_extracted"))
    vision_total = sum(1 for c in chunks if c.get("topic") == "study_plans")

    print(f"\n{'=' * 60}")
    print(f"✅ Готово!")
    print(f"   Чанков : {len(chunks)}  →  {OUTPUT_CHUNKS}")
    if vision_total:
        print(f"   Vision : ✓ {vision_ok} / {vision_total} учебных планов", end="")
        if failed:
            print(f"  (✗ {len(failed)} не удалось  →  {OUTPUT_FAILED})")
            print(f"   Повтори позже: python netutor_parser.py --retry")
        else:
            print()
    print("\nПо топикам:")
    for topic, cnt in Counter(c["topic"] for c in chunks).most_common():
        print(f"  {topic:<30} {cnt:>4}")


if __name__ == "__main__":
    main()