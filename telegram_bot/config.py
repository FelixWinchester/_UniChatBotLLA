import os
from dotenv import load_dotenv

load_dotenv()


TOKEN = os.getenv('TOKEN')
LLM_SERVER_URL = os.getenv('LLM_SERVER_URL')
ADMINS = os.getenv('ADMINS')