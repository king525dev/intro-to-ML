# app/config.py

from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):

    # Agent
    AGENT_NAME: str = "GlasgowBaby"

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3.1:8b"
    
    # Embeddings
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    
    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "glasgow_baby"
    VECTOR_SIZE: int = 384   # embedding dimension for bge-small-en
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.6
    
    # File storage for uploaded docs
    UPLOAD_DIR: str = "data/uploads"
    
    class Config:
        env_file = ".env"

settings = Settings()
# Ensure upload dir exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)