from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils.keyboard import ReplyKeyboardBuilder, InlineKeyboardBuilder

main_keyboard = ReplyKeyboardMarkup(keyboard=[
    [KeyboardButton(text='🕵️‍♂️ Обо мне')]
], 
    resize_keyboard = True)

start_inline_keyboard = InlineKeyboardMarkup(inline_keyboard= [
    [InlineKeyboardButton(text = 'Сайт СГУ', url= "https://www.sgu.ru")]
])