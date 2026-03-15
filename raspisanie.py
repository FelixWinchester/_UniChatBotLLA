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
REQUEST_TIMEOUT = 10       # секунд
REQUEST_DELAY = 0.5        # пауза между запросами
MAX_RETRIES = 3            # попыток при сетевой ошибке

DAYS_RU = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота"]

# ─────────────────────────── сетевой слой ────────────────────────────────────

def fetch(url: str) -> requests.Response | None:
    """GET-запрос с retry и timeout. Возвращает None при неудаче."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            resp.encoding = "utf-8"
            return resp
        except requests.RequestException as e:
            log.warning("Попытка %d/%d — ошибка %s: %s", attempt, MAX_RETRIES, url, e)
            if attempt < MAX_RETRIES:
                time.sleep(REQUEST_DELAY * attempt)
    log.error("Не удалось загрузить: %s", url)
    return None

# ─────────────────────────── сбор ссылок ─────────────────────────────────────

def get_all_group_links() -> list[str]:
    """Возвращает список href на все группы КНиИТ."""
    resp = fetch(f"{BASE_URL}/schedule/knt")
    if resp is None:
        return []
    soup = BeautifulSoup(resp.content, "html.parser")
    links = soup.find_all("a", href=re.compile(r"/schedule/knt/(do|zo)/\d+"))
    hrefs = [a["href"] for a in links]
    log.info("Найдено групп: %d", len(hrefs))
    return hrefs

# ──────────────────────── парсинг заголовков таблицы ─────────────────────────

def parse_day_columns(table) -> list[str]:
    """
    Читает заголовки столбцов из таблицы динамически.
    Возвращает список вида ['', 'Понедельник', 'Вторник', ...].
    Если заголовки не найдены — падает обратно на жёстко заданный список.
    """
    header_row = table.find("tr")
    if not header_row:
        return [""] + DAYS_RU

    headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]
    if len(headers) < 2:
        return [""] + DAYS_RU

    # Нормализуем: первый столбец — время, остальные — дни
    normalized = []
    for i, h in enumerate(headers):
        if i == 0:
            normalized.append("")   # столбец времени
            continue
        # Сайт может вернуть "Пн", "Пн." и т.д. — приводим к полным названиям
        day = _normalize_day_header(h)
        normalized.append(day)

    log.debug("Заголовки столбцов: %s", normalized)
    return normalized

_DAY_ALIASES = {
    "пн": "Понедельник", "пон": "Понедельник", "понедельник": "Понедельник",
    "вт": "Вторник",  "вто": "Вторник",  "вторник": "Вторник",
    "ср": "Среда",    "сре": "Среда",    "среда": "Среда",
    "чт": "Четверг",  "чет": "Четверг",  "четверг": "Четверг",
    "пт": "Пятница",  "пят": "Пятница",  "пятница": "Пятница",
    "сб": "Суббота",  "суб": "Суббота",  "суббота": "Суббота",
}

def _normalize_day_header(raw: str) -> str:
    key = raw.strip().rstrip(".").lower()
    return _DAY_ALIASES.get(key, raw)   # неизвестный заголовок оставляем как есть

# ──────────────────────────── парсинг ячейки ─────────────────────────────────

def parse_cell(cell, time_text: str, day_name: str) -> list[dict]:
    """Парсит одну ячейку расписания и возвращает список пар."""
    lessons = []

    wrappers = cell.find_all("div", class_="l-qs-wrapper")
    if wrappers:
        for item in wrappers:
            lesson = _parse_wrapper(item, time_text)
            if lesson:
                lessons.append(lesson)
    else:
        raw_text = cell.get_text(separator=" ", strip=True)
        if not raw_text or raw_text == "—":
            return []
        raw_text = re.sub(r"\s+", " ", raw_text).strip()
        lessons = _parse_raw_text(raw_text, time_text)

    return lessons

def _parse_wrapper(item, time_text: str) -> dict | None:
    """Структурированный блок l-qs-wrapper → пара."""
    classes = item.get("class", [])
    if "nom" in classes:
        week_info = "Числитель"
    elif "denom" in classes:
        week_info = "Знаменатель"
    else:
        week_info = "Всегда"

    subj    = item.find("div", class_="l-dn")
    typ     = item.find("div", class_="l-pr")
    teacher = item.find("div", class_="l-tn")
    room    = item.find("div", class_="l-p")

    subject_text = subj.get_text(strip=True)    if subj    else ""
    type_text    = typ.get_text(strip=True)     if typ     else ""
    teacher_text = teacher.get_text(strip=True) if teacher else ""
    room_text    = room.get_text(strip=True)    if room    else ""

    if not subject_text or subject_text == "—":
        return None

    # CSS-класс (nom/denom) приоритетнее текстового префикса.
    # Если класса нет — пробуем извлечь тип недели из самого текста предмета.
    if week_info == "Всегда":
        week_info, subject_text = _extract_week_prefix(subject_text)
    else:
        # Класс есть — просто убираем префикс из текста, если он там есть
        _, subject_text = _extract_week_prefix(subject_text)

    lesson = {
        "time":      time_text,
        "subject":   subject_text,
        "type":      type_text,
        "teacher":   teacher_text,
        "room":      room_text,
        "week_type": week_info,
    }

    if _is_phys_ed(subject_text):
        lesson = _parse_physical_education(subject_text, time_text)

    return lesson

_WEEK_PREFIX_RE = re.compile(
    r"^(З|Ч|Числитель|Знаменатель)\s+",
    re.IGNORECASE | re.UNICODE,
)
_WEEK_PREFIX_MAP = {
    "з": "Знаменатель",
    "знаменатель": "Знаменатель",
    "ч": "Числитель",
    "числитель": "Числитель",
}

def _extract_week_prefix(text: str) -> tuple[str, str]:
    """
    Если строка начинается с 'З ', 'Ч ', 'Числитель ' или 'Знаменатель ' —
    возвращает (week_type, текст_без_префикса).
    Иначе возвращает ('Всегда', исходный_текст).
    """
    m = _WEEK_PREFIX_RE.match(text)
    if m:
        key = m.group(1).lower()
        week_type = _WEEK_PREFIX_MAP.get(key, "Всегда")
        return week_type, text[m.end():].strip()
    return "Всегда", text


def _parse_raw_text(raw_text: str, time_text: str) -> list[dict]:
    """Fallback-парсинг сырого текста ячейки."""
    # Разбиваем сдвоенные лабораторные
    lab_parts = re.split(r"(ЛАБОРАТОРНАЯ\s*\d+)", raw_text)
    if len(lab_parts) > 2 and "ЛАБОРАТОРНАЯ" in raw_text.upper():
        chunks = []
        for j in range(0, len(lab_parts), 2):
            if j == 0 and lab_parts[0].strip():
                chunks.append(lab_parts[0].strip())
            elif j > 0:
                chunks.append((lab_parts[j - 1] + " " + lab_parts[j]).strip())
    else:
        chunks = [raw_text]

    lessons = []
    for chunk in chunks:
        if not chunk:
            continue
        week_type, chunk_clean = _extract_week_prefix(chunk)
        lesson = parse_raw_lesson(chunk_clean, time_text, week_type)
        if lesson:
            if _is_phys_ed(lesson["subject"]):
                lesson = _parse_physical_education(lesson["subject"], time_text)
                lesson["week_type"] = week_type
            lessons.append(lesson)
    return lessons

# ───────────────────── вспомогательные парсеры ───────────────────────────────

def _is_phys_ed(text: str) -> bool:
    return "Элективные дисциплины по физической культуре" in text or "физкультур" in text.lower()

def _parse_physical_education(full_text: str, time_text: str) -> dict:
    """Специальный парсер для физкультуры."""
    teacher_match = re.search(r"([А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.?)", full_text)
    teacher_name = teacher_match.group(1) if teacher_match else ""

    room_text = ""
    if "Спортивный комплекс" in full_text:
        m = re.search(r"(Спортивный комплекс[^.]+(?:\.\s*[^.]+)*)", full_text)
        if m:
            room_text = m.group(1).strip()
    else:
        base = full_text.replace(teacher_name, "").strip() if teacher_name else full_text
        m = re.search(r"[–\-]\s*(.+)$", base)
        if m:
            room_text = m.group(1).strip()

    return {
        "time":      time_text,
        "subject":   "Элективные дисциплины по физической культуре",
        "type":      "ПРАКТИКА",
        "teacher":   teacher_name,
        "room":      room_text,
        "week_type": "Всегда",
    }

# Расширенный regex для аудиторий:
# теперь принимает: "3 корпус, 301 комната", "ауд. 301", "301", "Спортивный комплекс", "онлайн" и т.д.
_ROOM_PATTERNS = [
    r"\d+\s*корпус\s*,\s*\d+\s*комн(?:ата)?",   # стандартный формат СГУ
    r"(?:ауд(?:итория)?\.?\s*)\d+[а-яА-Я]?",     # ауд. 301
    r"(?:к(?:орп)?\.?\s*)\d+\s*[,/]\s*\d+",      # к.3/301
    r"Спортивный комплекс[^,;]*",                  # спорткомплекс
    r"онлайн",                                     # дистант
    r"\b\d{3,4}[а-яА-Я]?\b",                      # просто номер аудитории
]
_ROOM_RE = re.compile(
    r"(?:" + "|".join(_ROOM_PATTERNS) + r")",
    re.IGNORECASE | re.UNICODE,
)

def parse_raw_lesson(raw_text: str, time_text: str, week_type: str = "Всегда") -> dict | None:
    """Парсит строку с парой. Поддерживает разные форматы аудиторий."""
    # Пробуем найти аудиторию
    room_match = _ROOM_RE.search(raw_text)
    room_text = room_match.group(0).strip() if room_match else ""

    # Всё до аудитории — тип + предмет + преподаватель
    prefix = raw_text[: room_match.start()].strip() if room_match else raw_text.strip()

    # Отделяем тип занятия
    type_match = re.match(
        r"^(ЛЕКЦИЯ|ПРАКТИКА|СЕМИНАР|КОНСУЛЬТАЦИЯ|ЛАБОРАТОРНАЯ(?:\s*\d+\s*под)?)\s*",
        prefix,
        re.IGNORECASE,
    )
    lesson_type = type_match.group(1).strip().upper() if type_match else ""
    rest = prefix[type_match.end():].strip() if type_match else prefix

    # Последний токен в формате "Фамилия И.О." — преподаватель
    teacher_match = re.search(
        r"([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.?)+(?:\s*и\s*др\.)?)\s*$",
        rest,
        re.UNICODE,
    )
    teacher_text = teacher_match.group(1).strip() if teacher_match else ""
    subject_text = rest[: teacher_match.start()].strip() if teacher_match else rest.strip()

    if not subject_text:
        # Совсем не распознали — сохраняем как есть
        return {
            "time":      time_text,
            "subject":   raw_text,
            "type":      "",
            "teacher":   "",
            "room":      "",
            "week_type": week_type,
        }

    return {
        "time":      time_text,
        "subject":   subject_text,
        "type":      lesson_type,
        "teacher":   teacher_text,
        "room":      room_text,
        "week_type": week_type,
    }

# ─────────────────────── парсинг расписания группы ───────────────────────────

def parse_group_schedule(group_href: str) -> dict | None:
    """Загружает и парсит страницу группы."""
    resp = fetch(f"{BASE_URL}{group_href}")
    if resp is None:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    group_num = group_href.split("/")[-1]

    table = soup.find("table", id="schedule")
    if not table:
        log.warning("Таблица расписания не найдена: %s", group_href)
        return None

    day_columns = parse_day_columns(table)

    schedule_data: dict = {
        "group": group_num,
        "days":  {day: [] for day in DAYS_RU},
    }

    for row in table.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if not cells:
            continue

        time_text = cells[0].get_text(strip=True)
        if not re.match(r"\d{2}:\d{2}", time_text):
            continue  # заголовочная строка

        for col_idx in range(1, len(cells)):
            if col_idx >= len(day_columns):
                break
            day_name = day_columns[col_idx]
            if day_name not in schedule_data["days"]:
                continue

            lessons = parse_cell(cells[col_idx], time_text, day_name)
            schedule_data["days"][day_name].extend(lessons)

    return schedule_data

# ──────────────────────────── главная функция ────────────────────────────────

def run_parser(output_file: str = "full_knt_schedule.json") -> None:
    log.info("Начинаю сбор ссылок на группы...")
    group_links = get_all_group_links()
    if not group_links:
        log.error("Список групп пуст. Завершаю.")
        return

    full_result = []
    failed = []

    for i, link in enumerate(group_links, start=1):
        log.info("Обработка %d/%d: %s", i, len(group_links), link)
        data = parse_group_schedule(link)
        if data:
            full_result.append(data)
        else:
            failed.append(link)
        time.sleep(REQUEST_DELAY)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_result, f, ensure_ascii=False, indent=2)

    log.info("Готово! Сохранено групп: %d. Файл: %s", len(full_result), output_file)
    if failed:
        log.warning("Не удалось обработать (%d): %s", len(failed), failed)


if __name__ == "__main__":
    run_parser()