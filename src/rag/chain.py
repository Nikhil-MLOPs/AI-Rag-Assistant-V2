import os
import json
import yaml
import time
from pathlib import Path
from typing import List, Dict, Tuple

from dotenv import load_dotenv
load_dotenv()

from src.utils.logging import setup_logging
logger = setup_logging("Rag-Chain")

from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.memory import ConversationBufferWindowMemory

from src.rag.prompts import MEDICAL_RAG_PROMPT
from src.rag.guardrails import validate_answer
from src.retrieval.hybrid import hybrid_retrieve
from src.utils.runtime_config import load_runtime_config

# ======================================================
# ENV
# ======================================================

def is_test_env() -> bool:
    return os.getenv("CI") == "true" or os.getenv("PYTEST_CURRENT_TEST") is not None

# ======================================================
# CONFIG
# ======================================================

RAG_CFG = load_runtime_config()

# ======================================================
# LLM + MEMORY
# ======================================================

LLM = None
MEMORY = None

if not is_test_env():
    LLM = OllamaLLM(
        model=RAG_CFG["llm"]["model"],
        temperature=RAG_CFG["llm"]["temperature"],
        streaming=True,
    )

    MEMORY = ConversationBufferWindowMemory(
        k=RAG_CFG["memory"]["window_size"],
        return_messages=True,
    )

# ======================================================
# 🔒 AUTHORITATIVE TOPIC STATE
# ======================================================

TOPIC_STATE = {
    "current": None
}

# ======================================================
# DATA
# ======================================================

def load_chunks() -> Tuple[Dict, ...]:
    path = Path("data/processed/chunks/chunks.jsonl")
    if not path.exists():
        return tuple()

    with open(path, "r", encoding="utf-8") as f:
        return tuple(json.loads(line) for line in f)

# ======================================================
# FOLLOW-UP DETECTION
# ======================================================

def is_followup(question: str) -> bool:
    q = question.lower().strip()

    if " it " in f" {q} " or " its " in f" {q} ":
        return True

    if q.startswith((
        "how ",
        "why ",
        "can ",
        "does ",
        "is ",
        "are ",
    )):
        return True

    return False

# ======================================================
# 🔑 GENERIC TOPIC MATCHING (SCALABLE)
# ======================================================

def select_best_topic_from_question(
    question: str,
    candidate_topics: List[str]
) -> str | None:
    """
    Selects the topic that best matches the user's question.
    Scales to thousands of diseases.
    """
    q = question.lower()
    q_tokens = set(q.split())

    scored = []

    for topic in set(candidate_topics):
        t = topic.lower()
        t_tokens = set(t.split())
        score = 0

        if t == q:
            score += 10

        if t in q:
            score += 8

        if q in t:
            score += 5

        score += len(q_tokens & t_tokens)

        if score > 0:
            scored.append((score, topic))

    if not scored:
        return None

    return sorted(scored, reverse=True)[0][1]

# ======================================================
# 🔑 EXPLICIT TOPIC EXTRACTION (ROBUST)
# ======================================================

def extract_explicit_topic(question: str, docs: List[Dict]) -> str | None:
    topics = [
        d.get("metadata", {}).get("topic")
        for d in docs
        if d.get("metadata", {}).get("topic")
    ]
    return select_best_topic_from_question(question, topics)

# ======================================================
# 🔥 EXPLICIT NEW TOPIC OVERRIDE
# ======================================================

def introduces_new_topic(question: str, chunks: List[Dict]) -> bool:
    q = question.lower()
    for c in chunks:
        topic = c.get("metadata", {}).get("topic")
        if topic and topic.lower() in q:
            return True
    return False

# ======================================================
# FALLBACK TOPIC INFERENCE (RETRIEVAL-BASED)
# ======================================================

def infer_topic_from_retrieval(docs: List[Dict], question: str) -> str | None:
    topics = [
        d.get("metadata", {}).get("topic")
        for d in docs
        if d.get("metadata", {}).get("topic")
    ]
    return select_best_topic_from_question(question, topics)

# ======================================================
# CONTEXT
# ======================================================

def format_context(docs: List[Dict]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        meta = d["metadata"]
        blocks.append(
            f"[{i}] {d['text']}\nSOURCE: {meta.get('pdf')} | page {meta.get('page')}"
        )
    return "\n\n".join(blocks[:RAG_CFG["retrieval"]["max_context_chunks"]])

def append_sources(answer: str, docs: List[Dict]) -> str:
    sources = [
        f"[{i}] {d['metadata'].get('pdf')} | page {d['metadata'].get('page')}"
        for i, d in enumerate(docs, 1)
    ]
    return answer.strip() + "\n\nSources:\n" + "\n".join(sources)

# ======================================================
# 🔗 FINAL RAG CHAIN (WITH TIMING + STABLE TOPICS)
# ======================================================

def rag_chain(question: str):
    total_start = time.perf_counter()
    logger.info(f"Received question: {question}")

    chunks = load_chunks()
    explicit_new_topic = introduces_new_topic(question, chunks)
    followup = is_followup(question)

    # --------------------------------------------------
    # 🔍 RETRIEVAL (TIMED)
    # --------------------------------------------------

    retrieval_start = time.perf_counter()

    if explicit_new_topic:
        logger.info("[NEW_TOPIC][EXPLICIT_OVERRIDE]")
        retrieved_docs = hybrid_retrieve(
            question,
            chunks,
            top_k=RAG_CFG["retrieval"]["top_k"],
        )

        topic = extract_explicit_topic(question, retrieved_docs)
        if topic:
            TOPIC_STATE["current"] = topic
            logger.info(f"[TOPIC_LOCKED][EXPLICIT] {topic}")

    elif followup and TOPIC_STATE["current"]:
        topic = TOPIC_STATE["current"]
        logger.info(f"[FOLLOWUP] Using locked topic: {topic}")

        topic_chunks = tuple(
            c for c in chunks
            if c.get("metadata", {}).get("topic", "").lower() == topic.lower()
        )

        retrieved_docs = hybrid_retrieve(
            question,
            chunks,
            top_k=RAG_CFG["retrieval"]["top_k"],
        )

    else:
        logger.info("[NEW_TOPIC] Global retrieval")
        retrieved_docs = hybrid_retrieve(
            question,
            chunks,
            top_k=RAG_CFG["retrieval"]["top_k"],
        )

        topic = infer_topic_from_retrieval(retrieved_docs, question)
        if topic:
            TOPIC_STATE["current"] = topic
            logger.info(f"[TOPIC_LOCKED][INFERRED] {topic}")

    retrieval_time = time.perf_counter() - retrieval_start
    logger.info(f"[RAG] Retrieval completed in {retrieval_time:.3f}s")

    if is_test_env():
        return

    # --------------------------------------------------
    # 🧠 LLM INFERENCE (TIMED)
    # --------------------------------------------------

    inference_start = time.perf_counter()
    context = format_context(retrieved_docs)

    chain = (
        {
            "context": lambda _: context,
            "question": RunnablePassthrough(),
            "history": RunnableLambda(
                lambda _: "\n".join(
                    str(m) for m in MEMORY.load_memory_variables({}).get("history", [])
                )
            ),
        }
        | MEDICAL_RAG_PROMPT
        | LLM
        | StrOutputParser()
    )

    full_answer = ""
    for token in chain.stream(question):
        full_answer += token
        yield token

    inference_time = time.perf_counter() - inference_start
    logger.info(f"[RAG] LLM inference completed in {inference_time:.3f}s")

    # --------------------------------------------------
    # 🛡 VALIDATION + SOURCES
    # --------------------------------------------------

    final_answer = validate_answer(
        answer=full_answer,
        sources=retrieved_docs,
        cfg=RAG_CFG,
    )

    final_answer = append_sources(final_answer, retrieved_docs)
    yield "\n\n" + final_answer[len(full_answer):]

    total_time = time.perf_counter() - total_start
    logger.info(f"[RAG] Total request time: {total_time:.3f}s")

    if MEMORY:
        MEMORY.save_context(
            {"input": question},
            {"output": final_answer},
        )
