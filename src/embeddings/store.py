from chromadb import Client
from chromadb.config import Settings
from src.utils.logging import setup_logging

logger = setup_logging("vector_store")

def get_chroma_client():
    return Client(
        Settings(
            anonymized_telemetry=False
        )
    )
