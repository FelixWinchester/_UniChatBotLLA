import asyncio
import logging

from aiogram import Bot, Dispatcher

from config import TOKEN
from app.users.handlers import router
from app.admin.handlers import router_adm

from logger import logger

bot = Bot(token=TOKEN)
dp = Dispatcher()

async def main():
    logger.info("Запуск основного процесса...")
    dp.include_router(router)
    dp.include_router(router_adm)
    logger.info("Бот запущен и готов к использованию.")
    

    try:
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"Произошла ошибка во время работы бота: {e}")
    finally:
        logger.info("Поллинг остановлен.")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Завершение работы программы.")
    except Exception as e:
        logger.error(f"Произошла ошибка: {e}")
    finally:
        logger.info("Запуск завершен.")