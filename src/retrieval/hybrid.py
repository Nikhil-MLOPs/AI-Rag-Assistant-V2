from functools import lru_cache
from src.retrieval.bm25 import build_bm25

@lru_cache(maxsize=128)
def _cached_bm25_scores(query: str, texts: tuple[str]):
    bm25 = build_bm25([{"text": t, "metadata": {}} for t in texts])
    return bm25.get_scores(query.split())


def hybrid_retrieve(query: str, docs: tuple):
    """
    docs: tuple of {"text": str, "metadata": dict}
    """
    texts = tuple(doc["text"] for doc in docs)

    scores = _cached_bm25_scores(query, texts)

    ranked = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for _, doc in ranked[:5]]
