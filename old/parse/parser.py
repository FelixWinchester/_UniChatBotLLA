import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime

BASE_URL = "https://www.sgu.ru"

def get_all_group_links():
    """Собирает все актуальные ссылки на группы КНиИТ."""
    url = f"{BASE_URL}/schedule/knt"
    res = requests.get(url)
    soup = BeautifulSoup(res.content, 'html.parser')
    
    # Ищем все ссылки, которые ведут на конкретные группы (do - дневное, zo - заочное)
    links = soup.find_all('a', href=re.compile(r'/schedule/knt/(do|zo)/\d+'))
    return [link['href'] for link in links]

def parse_group_schedule(group_href):
    """Парсит страницу группы и возвращает объект с парами по дням."""
    url = f"{BASE_URL}{group_href}"
    res = requests.get(url)
    res.encoding = 'utf-8'
    soup = BeautifulSoup(res.text, 'html.parser')
   
    # Определяем номер группы из URL или заголовка
    group_num = group_href.split('/')[-1]
   
    schedule_data = {
        "group": group_num,
        "days": {
            "Понедельник": [],
            "Вторник": [],
            "Среда": [],
            "Четверг": [],
            "Пятница": [],
            "Суббота": []
        }
    }
    table = soup.find('table', id='schedule')
    if not table:
        return None
    # Заголовки дней недели (чтобы сопоставить индекс столбца с названием дня)
    # Обычно: Время, Пн, Вт, Ср, Чт, Пт, Сб
    day_columns = ["", "Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота"]
    rows = table.find_all('tr')
    for row in rows:
        cells = row.find_all(['td', 'th'])
        if not cells: continue
       
        # Первая ячейка — это всегда время (например, "08:20-09:50")
        time_text = cells[0].get_text(strip=True)
        if not re.match(r'\d{2}:\d{2}', time_text):
            continue # Пропускаем заголовочную строку таблицы
        # Проходим по дням недели (столбцы 1-6)
        for i in range(1, len(cells)):
            if i >= len(day_columns): break
           
            day_name = day_columns[i]
            cell = cells[i]
           
            # В одной ячейке может быть несколько "плашек" (подгруппы или ч/з)
            # СГУ использует классы l-dn (название), l-pr (тип), l-tn (препод), l-p (аудитория)
            items = cell.find_all('div', class_='l-qs-wrapper')
            if items:
                # Структурированный случай (как раньше)
                for item in items:
                    week_info = "Всегда"
                    if "nom" in item.get('class', []): week_info = "Числитель"
                    elif "denom" in item.get('class', []): week_info = "Знаменатель"
                    subj = item.find('div', class_='l-dn')
                    typ = item.find('div', class_='l-pr')
                    teacher = item.find('div', class_='l-tn')
                    room = item.find('div', class_='l-p')
                    
                    subject_text = subj.get_text(strip=True) if subj else ""
                    type_text = typ.get_text(strip=True) if typ else ""
                    
                    lesson = {
                        "time": time_text,
                        "subject": subject_text,
                        "type": type_text,
                        "teacher": teacher.get_text(strip=True) if teacher else "",
                        "room": room.get_text(strip=True) if room else "",
                        "week_type": week_info
                    }
                    
                    # Специальная обработка для физкультуры (оставляем, так как не связано с проблемой)
                    if "Элективные дисциплины по физической культуре" in subject_text or "физкультур" in subject_text.lower():
                        # Извлекаем преподавателя
                        teacher_match = re.search(r'([А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.?)', subject_text)
                        if teacher_match:
                            lesson["teacher"] = teacher_match.group(1)
                        
                        # Извлекаем аудиторию (только место проведения)
                        room_text = ""
                        
                        # Ищем "Спортивный комплекс" или место после тире
                        if "Спортивный комплекс" in subject_text:
                            room_match = re.search(r'(Спортивный комплекс[^\.]+(?:\.\s*[^\.]+)*)', subject_text)
                            if room_match:
                                room_text = room_match.group(1).strip()
                        else:
                            # Ищем текст после тире
                            room_match = re.search(r'[–-]\s*(.+)$', subject_text)
                            if room_match:
                                room_text = room_match.group(1).strip()
                        
                        if room_text:
                            lesson["room"] = room_text
                        
                        lesson["subject"] = "Элективные дисциплины по физической культуре"
                        lesson["type"] = "ПРАКТИКА"
                    
                    if lesson["subject"] and lesson["subject"] != "—":
                        schedule_data["days"][day_name].append(lesson)
            else:
                # Fallback: сырой текст без структуры — парсим regex
                raw_text = cell.get_text(separator=" ", strip=True)
                if not raw_text or raw_text == "—":
                    continue
                raw_text = re.sub(r'\s+', ' ', raw_text).strip()  # Нормализуем пробелы
                
                # Проверяем на сдвоенные (ключевые слова вроде "ЛАБОРАТОРНАЯ 2" в середине)
                lab_parts = re.split(r'(ЛАБОРАТОРНАЯ\s*\d+)', raw_text)
                if len(lab_parts) > 2 and "ЛАБОРАТОРНАЯ" in raw_text.upper():  # Сдвоенная
                    for j in range(0, len(lab_parts), 2):
                        if j == 0 and lab_parts[0].strip():
                            part = lab_parts[0].strip()
                        else:
                            part_type = lab_parts[j-1].strip() if j > 0 else ""
                            part = (part_type + " " + lab_parts[j].strip()).strip()
                        if not part: continue
                        lesson = parse_raw_lesson(part, time_text)
                        if lesson:
                            # Специальная обработка для физкультуры в сыром тексте
                            if lesson["subject"] and ("Элективные дисциплины" in lesson["subject"] or "физкультур" in lesson["subject"].lower()):
                                lesson = parse_physical_education(lesson, time_text)
                            
                            schedule_data["days"][day_name].append(lesson)
                else:
                    # Одиночная
                    lesson = parse_raw_lesson(raw_text, time_text)
                    if lesson:
                        # Специальная обработка для физкультуры в сыром тексте
                        if lesson["subject"] and ("Элективные дисциплины" in lesson["subject"] or "физкультур" in lesson["subject"].lower()):
                            lesson = parse_physical_education(lesson, time_text)
                        
                        schedule_data["days"][day_name].append(lesson)
    return schedule_data

def parse_physical_education(lesson, time_text):
    """Специальный парсер для физкультуры."""
    full_text = lesson["subject"]
    
    # Извлекаем преподавателя
    teacher_match = re.search(r'([А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.?)', full_text)
    teacher_name = teacher_match.group(1) if teacher_match else ""
    
    # Извлекаем аудиторию (только место проведения)
    room_text = ""
    
    # Ищем "Спортивный комплекс"
    if "Спортивный комплекс" in full_text:
        room_match = re.search(r'(Спортивный комплекс[^\.]+(?:\.\s*[^\.]+)*)', full_text)
        if room_match:
            room_text = room_match.group(1).strip()
    else:
        # Если нет "Спортивный комплекс", ищем текст после тире
        if teacher_name:
            # Убираем преподавателя из текста
            text_without_teacher = full_text.replace(teacher_name, "").strip()
            room_match = re.search(r'[–-]\s*(.+)$', text_without_teacher)
            if room_match:
                room_text = room_match.group(1).strip()
        else:
            room_match = re.search(r'[–-]\s*(.+)$', full_text)
            if room_match:
                room_text = room_match.group(1).strip()
    
    return {
        "time": time_text,
        "subject": "Элективные дисциплины по физической культуре",
        "type": "ПРАКТИКА",
        "teacher": teacher_name,
        "room": room_text,
        "week_type": "Всегда"
    }

def parse_raw_lesson(raw_text, time_text):
    """Вспомогательная функция для парсинга сырой строки."""
    # Обновленный regex: захватываем "ЛАБОРАТОРНАЯ \d+ под" полностью в type
    m = re.match(
        r'(?P<type>(?:ЛЕКЦИЯ|ПРАКТИКА|СЕМИНАР|КОНСУЛЬТАЦИЯ|ЛАБОРАТОРНАЯ(?:\s*\d+\s*под)?)?)\s*'
        r'(?P<subject>.+?)\s+'
        r'(?P<teacher>[А-ЯЁ][а-яё]+(?:\s*[А-ЯЁ]\.\s*[А-ЯЁ]\.?)+(?:\s*и\s*др\.)?)\s+'
        r'(?P<room>\d+\s*корпус\s*,\s*\d+\s*комната.*)$',
        raw_text,
        re.UNICODE | re.DOTALL
    )
    if m:
        return {
            "time": time_text,
            "subject": m.group('subject').strip(),
            "type": m.group('type').strip() if m.group('type') else "",
            "teacher": m.group('teacher').strip(),
            "room": m.group('room').strip(),
            "week_type": "Всегда"  # По умолчанию, если не указано
        }
    else:
        # Если не подошло — fallback в subject
        return {
            "time": time_text,
            "subject": raw_text,
            "type": "",
            "teacher": "",
            "room": "",
            "week_type": "Всегда"
        }


def run_parser():
    print("Начинаю сбор ссылок на группы...")
    group_links = get_all_group_links()
    print(f"Найдено групп: {len(group_links)}")
    
    full_result = []
    
    for i, link in enumerate(group_links):
        print(f" Обработка {i+1}/{len(group_links)}: {link}")
        data = parse_group_schedule(link)
        if data:
            full_result.append(data)
            
    # Сохраняем все в один большой JSON
    with open("full_knt_schedule.json", "w", encoding="utf-8") as f:
        json.dump(full_result, f, ensure_ascii=False, indent=2)
    
    print("\nГотово! Все данные в файле full_knt_schedule.json")

if __name__ == "__main__":
    run_parser()