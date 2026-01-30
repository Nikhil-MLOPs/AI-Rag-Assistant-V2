from pathlib import Path
import yaml
from pydantic import BaseModel

class IngestionConfig(BaseModel):
    raw_dir: str
    processed_dir: str
    skip_start_pages: int
    skip_end_after: int

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_ingestion_config(path: str) -> IngestionConfig:
    return IngestionConfig(**load_yaml(path))