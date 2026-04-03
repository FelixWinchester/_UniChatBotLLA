from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.types import Message

import app.users.keyboard as adm_kb

from config import ADMINS


router_adm = Router()

@router_adm.message(F.text == "/admin")
async def admin_panel(message: Message):
    user_id = message.from_user.id

    if user_id in ADMINS:
        await message.answer("🔐 Добро пожаловать в админ-панель!", reply_markup=adm_kb.admin_keyboard)
    else:
        await message.answer("🚫 У вас нет доступа к этой команде.")

