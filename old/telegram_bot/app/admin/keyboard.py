from aiogram.types import ReplyKeyboardMarkup, KeyboardButton,InlineKeyboardMarkup, InlineKeyboardButton

admin_keyboard = InlineKeyboardMarkup(
    keyboard=[
        [InlineKeyboardButton(text="📊 Статистика"), InlineKeyboardButton(text="⚙️ Настройки")],
        [InlineKeyboardButton(text="🔙 Назад")]
    ],
    resize_keyboard=True
)