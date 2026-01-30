import json
from pathlib import Path
import yaml
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.logging import setup_logging

logger = setup_logging("Embeddings")


# ======================================================
# CONFIG
# ======================================================

def load_embedding_config() -> dict:
    with open("configs/embeddings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ======================================================
# CORE EMBEDDING LOGIC
# ======================================================

def embed_chunks():
    cfg = load_embedding_config()

    model = SentenceTransformer(cfg["model_name"])

    chunks_file = Path("data/processed/chunks/chunks.jsonl")
    out_dir = Path("data/embeddings")
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = []
    metadatas = []

    # ---------- Load chunks ----------
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["text"])
            metadatas.append(record["metadata"])

    logger.info(f"Loaded {len(texts)} chunks for embedding")

    # ---------- Embed ----------
    embeddings = model.encode(
        texts,
        batch_size=cfg["batch_size"],
        show_progress_bar=True,
        normalize_embeddings=True,  # recommended for cosine similarity
    )

    # ---------- Save ----------
    emb_path = out_dir / "embeddings.npy"
    meta_path = out_dir / "metadata.json"

    np.save(emb_path, embeddings)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved embeddings to {emb_path}")
    logger.info(f"Saved metadata to {meta_path}")
    logger.info("Embedding pipeline completed successfully")


# ======================================================
# ENTRYPOINT
# ======================================================

if __name__ == "__main__":
    embed_chunks()
