"""
Core Configuration
Loads all settings from environment variables with sensible defaults.
"""
from pathlib import Path
from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
class Settings(BaseSettings):
    # ===== API KEYS =====
    GROQ_API_KEY: str | None = None

    # ===== APP CONFIG =====
    APP_NAME: str = "Mineral Exploration Intelligence System"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ===== LLM =====
    MAX_TOKENS: int = 4096

    # ===== EMBEDDINGS =====
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # ===== VECTOR DB =====
    CHROMA_PERSIST_DIR: str = str(_PROJECT_ROOT / "data" / "chroma_db")
    COLLECTION_GEOLOGICAL: str = "geological_reports"
    COLLECTION_MINERAL: str = "mineral_datasets"
    COLLECTION_HYPERSPECTRAL: str = "hyperspectral_data"

    # ===== RAG =====
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.1

    # ===== FILES =====
    UPLOAD_DIR: str = "./data/uploads"
    MAX_UPLOAD_SIZE_MB: int = 50

    # ===== LOGGING =====
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"

    # ===== CORS =====
    ALLOWED_ORIGINS: list[str] = []

    class Config:
        env_file = ".env"
        extra = "ignore"   # prevents crashes on unknown env vars

    
    def __getattr__(self, name: str):
        upper_name = name.upper()
        if upper_name in self.__dict__:
            return self.__dict__[upper_name]
        raise AttributeError(f"{name} not found in Settings")


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()