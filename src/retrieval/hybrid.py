import json
from pathlib import Path
from functools import lru_cache

import yaml
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from langsmith import traceable

from src.utils.logging import setup_logging

logger = setup_logging("hybrid_retrieval")

# -------------------------------------------------
# PATHS & CONSTANTS
# -------------------------------------------------

CHUNKS_FILE = Path("data/processed/chunks/chunks.jsonl")
EMBEDDINGS_FILE = Path("data/embeddings/embeddings.npy")
EMBED_CFG_FILE = Path("configs/embeddings.yaml")

VECTOR_TOP_K = 50

TOPIC_BOOST = 2.5
DEFINITION_BOOST = 1.5

# -------------------------------------------------
# LOADERS (CACHED — SAFE)
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
# 🔥 QUERY EMBEDDING CACHE (MAJOR LATENCY WIN)
# -------------------------------------------------

@lru_cache(maxsize=256)
def _embed_query(text: str):
    embedder = _load_embedder()
    return embedder.encode(text, normalize_embeddings=True)


# -------------------------------------------------
# 🔥 BM25 CACHE (SAFE, ADDITIVE)
# -------------------------------------------------

@lru_cache(maxsize=8)
def _build_bm25(corpus_tuple):
    """
    Cache BM25 index for repeated candidate sets
    """
    return BM25Okapi([list(doc) for doc in corpus_tuple])


# -------------------------------------------------
# VECTOR → BM25 HYBRID RETRIEVER
# -------------------------------------------------
@traceable(name="Hybrid Retrieval")
def hybrid_retrieve(query: str, chunks=None, top_k: int = 5):
    """
    Vector-first retrieval with BM25 reranking,
    topic awareness, and section prioritization.
    """

    final_k = top_k

    chunks = _load_chunks()
    embeddings = _load_embeddings()

    query_lower = query.lower()

    # ---------------------------
    # 1️⃣ VECTOR SEARCH
    # ---------------------------

    query_emb = _embed_query(query)

    vector_scores = np.dot(embeddings, query_emb)
    top_vector_idx = np.argsort(vector_scores)[-VECTOR_TOP_K:][::-1]

    candidate_chunks = [chunks[i] for i in top_vector_idx]

    # ---------------------------
    # 🚀 EARLY EXIT (DOMINANT TOPIC)
    # ---------------------------

    top_topics = [
        c["metadata"].get("topic") for c in candidate_chunks[:5]
    ]

    if top_topics and len(set(top_topics)) == 1:
        results = candidate_chunks[:final_k]

        for r in results:
            logger.info(
                f"Retrieved -> {r['metadata'].get('topic')} | "
                f"{r['metadata'].get('section')}"
            )
        return results

    # ---------------------------
    # 2️⃣ BM25 RERANK
    # ---------------------------

    bm25_corpus = [
        (
            f"{c['metadata'].get('topic','')} "
            f"{c['metadata'].get('section','')} "
            f"{c['text']}"
        ).lower().split()
        for c in candidate_chunks
    ]

    corpus_tuple = tuple(tuple(doc) for doc in bm25_corpus)
    bm25 = _build_bm25(corpus_tuple)

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

    reranked_idx = np.argsort(boosted_scores)[::-1][:final_k]
    results = [candidate_chunks[i] for i in reranked_idx]

    # ---------------------------
    # 🔍 LOG FINAL RESULTS
    # ---------------------------

    for r in results:
        logger.info(
            f"Retrieved -> {r['metadata'].get('topic')} | "
            f"{r['metadata'].get('section')}"
        )

    return results
