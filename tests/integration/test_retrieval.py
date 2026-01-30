from src.retrieval.hybrid import hybrid_retrieve

def test_hybrid_retrieval():
    docs = (
        {"text": "diabetes mellitus is a chronic disease", "metadata": {"topic": "diabetes"}},
        {"text": "hypertension affects blood pressure", "metadata": {"topic": "hypertension"}},
        {"text": "insulin regulates blood sugar levels", "metadata": {"topic": "diabetes"}},
    )

    results = hybrid_retrieve("diabetes", docs)

    assert isinstance(results, list)
    assert len(results) > 0
    assert all("text" in doc for doc in results)