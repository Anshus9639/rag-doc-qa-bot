import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

UPLOAD_DIR = BASE_DIR / os.getenv("UPLOAD_DIR", "data/uploads")
INDEX_DIR = BASE_DIR / os.getenv("INDEX_DIR", "data/index")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))


def ensure_dirs() -> None:
    """Create data folders if they don't exist."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)