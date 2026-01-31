import json
from pathlib import Path
from functools import lru_cache

import yaml
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from src.utils.logging import setup_logging

logger = setup_logging("hybrid_retrieval")

# -------------------------------------------------
# PATHS & CONSTANTS
# -------------------------------------------------

CHUNKS_FILE = Path("data/processed/chunks/chunks.jsonl")
EMBEDDINGS_FILE = Path("data/embeddings/embeddings.npy")
EMBED_CFG_FILE = Path("configs/embeddings.yaml")

VECTOR_TOP_K = 50
FINAL_TOP_K = 5

TOPIC_BOOST = 2.5
DEFINITION_BOOST = 1.5


# -------------------------------------------------
# LOADERS (CACHED)
# -------------------------------------------------

@lru_cache(maxsize=1)
def _load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@lru_cache(maxsize=1)
def _load_embeddings():
    return np.load(EMBEDDINGS_FILE)


@lru_cache(maxsize=1)
def _load_embedder():
    with open(EMBED_CFG_FILE, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model_name"]
    logger.info(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


# -------------------------------------------------
# VECTOR → BM25 HYBRID RETRIEVER
# -------------------------------------------------

def hybrid_retrieve(query: str, _unused=None):
    """
    Vector-first retrieval with BM25 reranking,
    topic awareness, and section prioritization.
    """

    chunks = _load_chunks()
    embeddings = _load_embeddings()
    embedder = _load_embedder()

    query_lower = query.lower()

    # ---------------------------
    # 1️⃣ VECTOR SEARCH
    # ---------------------------
    query_emb = embedder.encode(
        query,
        normalize_embeddings=True,
    )

    vector_scores = np.dot(embeddings, query_emb)
    top_vector_idx = np.argsort(vector_scores)[-VECTOR_TOP_K:][::-1]

    candidate_chunks = [chunks[i] for i in top_vector_idx]

    # ---------------------------
    # 2️⃣ BM25 RERANK
    # ---------------------------
    bm25_corpus = [
        f"{c['metadata'].get('topic','')} "
        f"{c['metadata'].get('section','')} "
        f"{c['text']}".lower().split()
        for c in candidate_chunks
    ]

    bm25 = BM25Okapi(bm25_corpus)
    bm25_scores = bm25.get_scores(query_lower.split())

    # ---------------------------
    # 3️⃣ INTENT-AWARE BOOSTING
    # ---------------------------
    boosted_scores = []

    for score, chunk in zip(bm25_scores, candidate_chunks):
        topic = chunk["metadata"].get("topic", "").lower()
        section = chunk["metadata"].get("section", "").lower()

        # Strong boost for exact topic match
        if topic and (topic in query_lower or query_lower in topic):
            score *= TOPIC_BOOST

        # Definition section priority
        if section == "definition":
            score *= DEFINITION_BOOST

        boosted_scores.append(score)

    # ---------------------------
    # 4️⃣ FINAL SELECTION
    # ---------------------------
    reranked_idx = np.argsort(boosted_scores)[::-1][:FINAL_TOP_K]
    results = [candidate_chunks[i] for i in reranked_idx]

    # ---------------------------
    # 🔍 LOG FINAL RESULTS
    # ---------------------------
    for r in results:
        logger.info(
            f"Retrieved -> {r['metadata'].get('topic')} | {r['metadata'].get('section')}"
        )

    return results
