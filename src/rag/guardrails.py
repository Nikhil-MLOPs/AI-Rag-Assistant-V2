def validate_answer(answer: str, sources: list[dict], cfg: dict) -> str:
    # If retrieval returned nothing, block
    if not sources:
        return "I am not confident based on the provided documents."

    # If the answer is extremely short, likely hallucinated
    if len(answer.strip()) < 50:
        return "I am not confident based on the provided documents."

    return answer