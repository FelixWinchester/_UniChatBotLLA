import requests
from bs4 import BeautifulSoup
import json
import re
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

BASE_URL = "https://www.sgu.ru"
REQUEST_TIMEOUT = 10
REQUEST_DELAY = 0.5
MAX_RETRIES = 3

DAYS_RU = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота"]

# Регулярные выражения для парсинга "грязных" строк
_WEEK_PREFIX_RE = re.compile(
    r"^(З|Ч|Числитель|Знаменатель|з\.|ч\.)\s+",
    re.IGNORECASE | re.UNICODE,
)

_WEEK_PREFIX_MAP = {
    "з": "Знаменатель",
    "з.": "Знаменатель",
    "знаменатель": "Знаменатель",
    "ч": "Числитель",
    "ч.": "Числитель",
    "числитель": "Числитель",
}

_ROOM_PATTERNS = [
    r"\d+\s*корпус\s*,\s*\d+\s*комн(?:ата)?",
    r"(?:ауд(?:итория)?\.?\s*)\d+[а-яА-Я]?",
    r"(?:к(?:орп)?\.?\s*)\d+\s*[,/]\s*\d+",
    r"Спортивный комплекс[^,;]*",
    r"онлайн",
    r"\b\d{3,4}[а-яА-Я]?\b",
]
_ROOM_RE = re.compile(r"(?:" + "|".join(_ROOM_PATTERNS) + r")", re.IGNORECASE | re.UNICODE)

# ─────────────────────────── сетевой слой ────────────────────────────────────

def fetch(url: str) -> requests.Response | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            resp.encoding = "utf-8"
            return resp
        except requests.RequestException as e:
            log.warning("Попытка %d/%d — ошибка %s", attempt, MAX_RETRIES, url)
            if attempt < MAX_RETRIES:
                time.sleep(REQUEST_DELAY * attempt)
    return None

# ─────────────────────────── вспомогательные функции ──────────────────────────

def _extract_week_prefix(text: str) -> tuple[str, str]:
    """Извлекает З/Ч/Числитель/Знаменатель из начала строки."""
    m = _WEEK_PREFIX_RE.match(text)
    if m:
        key = m.group(1).lower().rstrip('.')
        week_type = _WEEK_PREFIX_MAP.get(key, "Всегда")
        return week_type, text[m.end():].strip()
    return "Всегда", text

def _is_phys_ed(text: str) -> bool:
    return "Элективные дисциплины по физической культуре" in text or "физкультур" in text.lower()

def _normalize_day_header(raw: str) -> str:
    _DAY_ALIASES = {
        "пн": "Понедельник", "вт": "Вторник", "ср": "Среда",
        "чт": "Четверг", "пт": "Пятница", "сб": "Суббота"
    }
    key = raw.strip().rstrip(".").lower()[:2]
    return _DAY_ALIASES.get(key, raw)

# ──────────────────────────── парсинг ────────────────────────────────────────

def parse_raw_lesson(raw_text: str, time_text: str, week_type: str = "Всегда") -> dict | None:
    """Универсальный парсер строки предмета."""
    # 1. Пробуем вытащить префикс из самого начала сырой строки (если он там есть)
    if week_type == "Всегда":
        week_type, raw_text = _extract_week_prefix(raw_text)
    else:
        _, raw_text = _extract_week_prefix(raw_text)

    # 2. Ищем аудиторию
    room_match = _ROOM_RE.search(raw_text)
    room_text = room_match.group(0).strip() if room_match else ""
    prefix = raw_text[: room_match.start()].strip() if room_match else raw_text.strip()

    # 3. Ищем тип занятия
    type_match = re.match(
        r"^(ЛЕКЦИЯ|ПРАКТИКА|СЕМИНАР|КОНСУЛЬТАЦИЯ|ЛАБОРАТОРНАЯ(?:\s*\d+\s*под)?)\s*",
        prefix, re.IGNORECASE
    )
    lesson_type = type_match.group(1).strip().upper() if type_match else ""
    rest = prefix[type_match.end():].strip() if type_match else prefix

    # 4. Ищем преподавателя
    teacher_match = re.search(
        r"([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.?)+(?:\s*и\s*др\.)?)",
        rest, re.UNICODE
    )
    
    if teacher_match:
        teacher_text = teacher_match.group(1).strip()
        subject_text = rest[: teacher_match.start()].strip()
    else:
        teacher_text = ""
        subject_text = rest.strip()

    # 5. Чистка от висящих запятых, точек и корпусов в конце
    subject_text = re.sub(r"[,.\s\-]+$", "", subject_text)
    
    # 6. Убираем подгруппы из названия, если они там остались ("1 под")
    subject_text = re.sub(r"^\d+\s*под\s+", "", subject_text, flags=re.IGNORECASE)

    # 7. ГЛАВНЫЙ ФИКС: Ищем З/Ч непосредственно в очищенном названии предмета!
    m_week = _WEEK_PREFIX_RE.match(subject_text)
    if m_week:
        prefix_val = m_week.group(1).lower().rstrip('.')
        if week_type == "Всегда":
            week_type = _WEEK_PREFIX_MAP.get(prefix_val, "Всегда")
        subject_text = subject_text[m_week.end():].strip()

    if not subject_text:
        return None

    return {
        "time": time_text,
        "subject": subject_text,
        "type": lesson_type,
        "teacher": teacher_text,
        "room": room_text,
        "week_type": week_type,
    }

def _parse_physical_education(full_text: str, time_text: str) -> dict:
    teacher_match = re.search(r"([А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.?)", full_text)
    teacher_name = teacher_match.group(1) if teacher_match else ""
    return {
        "time": time_text,
        "subject": "Элективные дисциплины по физической культуре",
        "type": "ПРАКТИКА",
        "teacher": teacher_name,
        "room": "Спортивный комплекс",
        "week_type": "Всегда",
    }

def parse_cell(cell, time_text: str, day_name: str) -> list[dict]:
    lessons = []
    wrappers = cell.find_all("div", class_="l-qs-wrapper")
    
    if wrappers:
        for item in wrappers:
            classes = item.get("class", [])
            w_type = "Числитель" if "nom" in classes else "Знаменатель" if "denom" in classes else "Всегда"
            
            subj_div = item.find("div", class_="l-dn")
            if not subj_div: continue
            
            raw_content = item.get_text(separator=" ", strip=True)
            lesson = parse_raw_lesson(raw_content, time_text, w_type)
            if lesson:
                if _is_phys_ed(lesson["subject"]):
                    lesson = _parse_physical_education(raw_content, time_text)
                lessons.append(lesson)
    else:
        raw_text = cell.get_text(separator=" ", strip=True)
        if raw_text and raw_text != "—":
            # Обработка лабораторных
            chunks = re.split(r"(ЛАБОРАТОРНАЯ\s*\d+\s*под)", raw_text, flags=re.IGNORECASE)
            if len(chunks) > 1:
                for i in range(1, len(chunks), 2):
                    combined = chunks[i] + " " + chunks[i+1]
                    res = parse_raw_lesson(combined, time_text)
                    if res: lessons.append(res)
            else:
                res = parse_raw_lesson(raw_text, time_text)
                if res:
                    if _is_phys_ed(res["subject"]):
                        res = _parse_physical_education(raw_text, time_text)
                    lessons.append(res)
    return lessons

# ─────────────────────── логика обхода страниц ──────────────────────────────

def get_all_group_links() -> list[str]:
    resp = fetch(f"{BASE_URL}/schedule/knt")
    if not resp: return []
    soup = BeautifulSoup(resp.content, "html.parser")
    links = soup.find_all("a", href=re.compile(r"/schedule/knt/(do|zo)/\d+"))
    return [a["href"] for a in links]

def parse_group_schedule(group_href: str) -> dict | None:
    resp = fetch(f"{BASE_URL}{group_href}")
    if not resp: return None
    soup = BeautifulSoup(resp.text, "html.parser")
    
    table = soup.find("table", id="schedule")
    if not table: return None

    header_row = table.find("tr")
    day_columns = [""] + [_normalize_day_header(th.get_text()) for th in header_row.find_all(["th", "td"])][1:]

    group_num = group_href.split("/")[-1]
    schedule_data = {"group": group_num, "days": {day: [] for day in DAYS_RU}}

    for row in table.find_all("tr")[1:]:
        cells = row.find_all(["td", "th"])
        if not cells: continue
        time_text = cells[0].get_text(strip=True).replace(" ", "")
        if not re.match(r"\d{2}:\d{2}", time_text): continue

        for col_idx, cell in enumerate(cells[1:], start=1):
            if col_idx >= len(day_columns): break
            day_name = day_columns[col_idx]
            if day_name in schedule_data["days"]:
                lessons = parse_cell(cell, time_text, day_name)
                schedule_data["days"][day_name].extend(lessons)

    return schedule_data

def run_parser():
    group_links = get_all_group_links()
    full_result = []
    for i, link in enumerate(group_links, start=1):
        log.info(f"Обработка {i}/{len(group_links)}: {link}")
        data = parse_group_schedule(link)
        if data: full_result.append(data)
        time.sleep(REQUEST_DELAY)
    
    with open("full_knt_schedule.json", "w", encoding="utf-8") as f:
        json.dump(full_result, f, ensure_ascii=False, indent=2)
    log.info("Парсинг окончен.")

if __name__ == "__main__":
    run_parser()