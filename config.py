import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4.1-mini")
VECTOR_DB_DIR: str = os.getenv("VECTOR_DB_DIR", "./chroma_db")
TOP_K_CANDIDATES: int = int(os.getenv("TOP_K_CANDIDATES", "20"))
TOP_K_RETURN: int = int(os.getenv("TOP_K_RETURN", "5"))
