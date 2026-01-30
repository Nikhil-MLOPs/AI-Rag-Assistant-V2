from rank_bm25 import BM25Okapi

def build_bm25(docs: list[dict]):
    """
    docs: [{"text": str, "metadata": dict}, ...]
    """
    tokenized = [doc["text"].split() for doc in docs]
    return BM25Okapi(tokenized)