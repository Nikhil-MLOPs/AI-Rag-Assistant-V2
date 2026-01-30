def validate_answer(answer: str, sources: list[str], cfg: dict) -> str:
    if cfg["guardrails"]["require_citations"] and not sources:
        return "I am not confident based on the provided documents."

    if cfg["guardrails"]["refuse_if_no_sources"] and "[SOURCE" not in answer:
        return "I am not confident based on the provided documents."

    return answer
