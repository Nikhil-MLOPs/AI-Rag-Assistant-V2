from typing import Dict, Any

# HELPERS

def _safe_lower(text: str | None) -> str:
    return text.lower() if isinstance(text, str) else ""


def _contains_refusal(answer: str | None) -> bool:
    if not answer:
        return True

    markers = [
        "cannot help",
        "can't help",
        "not able to",
        "i am not a doctor",
        "consult a medical professional",
        "cannot provide",
        "sorry",
    ]

    ans = answer.lower()
    return any(m in ans for m in markers)


# MAIN METRIC FUNCTION

def compute_metrics(
    question: str,
    answer: str | None,
    golden: Dict[str, Any],
) -> Dict[str, float]:
    """
     HARD GUARANTEE:
    - never raises
    - always returns floats
    """

    try:
        answer_text = _safe_lower(answer)

        expected_facts = golden.get("expected_facts", [])
        expected_sources = golden.get("expected_sources", [])

        # REFUSAL PENALTY
        refusal_penalty = 1.0 if _contains_refusal(answer) else 0.0

        # RECALL
        if not expected_facts:
            recall = 1.0
        else:
            matched_facts = [
                f for f in expected_facts
                if _safe_lower(f) in answer_text
            ]
            recall = len(matched_facts) / max(len(expected_facts), 1)

        recall = min(max(recall, 0.0), 1.0)

        # -------------------------
        # PRECISION (IMPROVED)
        # -------------------------
        if not expected_facts:
            precision = 1.0
        else:
            detected_hits = sum(
                1 for f in expected_facts
                if _safe_lower(f) in answer_text
            )
            precision = detected_hits / max(len(expected_facts), 1)

        precision = min(max(precision, 0.0), 1.0)

        # -------------------------
        # FAITHFULNESS
        # -------------------------
        has_sources_section = "sources:" in answer_text

        if not expected_sources:
            faithfulness = 1.0
        else:
            pdf_hits = sum(
                1 for s in expected_sources
                if s.get("pdf", "").lower() in answer_text
            )
            faithfulness = 1.0 if has_sources_section and pdf_hits > 0 else 0.0

        faithfulness = min(max(faithfulness, 0.0), 1.0)

        # -------------------------
        # ACCURACY (RAG-SAFE, CONTINUOUS)
        # -------------------------
        accuracy = (0.6 * recall) + (0.4 * faithfulness)

        if refusal_penalty == 1.0:
            accuracy *= 0.2

        accuracy = min(max(accuracy, 0.0), 1.0)

        # -------------------------
        # FINAL RETURN
        # -------------------------
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "faithfulness": float(faithfulness),
            "refusal_penalty": float(refusal_penalty),
        }

    except Exception:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "faithfulness": 0.0,
            "refusal_penalty": 1.0,
        }
