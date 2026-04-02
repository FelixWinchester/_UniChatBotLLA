import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logger(name: str = __name__, max_bytes: int = 1073741824, backup_count: int = 2):
    """Настройка логгера с ротацией файлов"""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Ротирующий файловый обработчик (1 ГБ макс, 3 бэкап-файла)
    file_handler = RotatingFileHandler(
        filename=logs_dir / 'bot.log',
        maxBytes=max_bytes,          # 1 GB = 1073741824 bytes
        backupCount=backup_count,     # Сколько бэкап-файлов хранить
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    # Консольный обработчик
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Очистка старых обработчиков
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Инициализация глобального логгера (1 ГБ на файл, максимум 3 файла)
logger = setup_logger(max_bytes=1073741824, backup_count=3)