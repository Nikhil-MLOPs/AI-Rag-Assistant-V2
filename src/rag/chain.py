import os
import json
import time
import yaml
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv()

# -------------------------------------------------
# 🔐 LangSmith tracing (SAFE DEFAULTS)
# -------------------------------------------------
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "AI-Rag-Assistant-V2")

from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.memory import ConversationBufferWindowMemory

from src.rag.prompts import MEDICAL_RAG_PROMPT
from src.rag.guardrails import validate_answer
from src.retrieval.hybrid import hybrid_retrieve
from src.utils.logging import setup_logging

logger = setup_logging("Rag-Chain")

# ======================================================
# ENV DETECTION (CI / TEST SAFE)
# ======================================================

def _is_test_env() -> bool:
    return (
        os.getenv("CI") == "true"
        or os.getenv("PYTEST_CURRENT_TEST") is not None
    )

# ======================================================
# CONFIG
# ======================================================

with open("configs/rag.yaml", "r", encoding="utf-8") as f:
    RAG_CFG = yaml.safe_load(f)

logger.info("RAG config loaded")

# ======================================================
# LLM + MEMORY (PROD ONLY)
# ======================================================

LLM = None
MEMORY = None

if not _is_test_env():
    logger.info("Initializing Ollama LLM (production mode)")
    LLM = OllamaLLM(
        model=RAG_CFG["llm"]["model"],
        temperature=RAG_CFG["llm"]["temperature"],
        streaming=RAG_CFG["llm"]["stream"],
    )

    MEMORY = ConversationBufferWindowMemory(
        k=RAG_CFG["memory"]["window_size"],
        return_messages=True,
    )
else:
    logger.warning("Test/CI environment detected — LLM disabled")

# ======================================================
# DATA LOADING
# ======================================================

def load_chunks() -> tuple[Dict, ...]:
    chunks_file = Path("data/processed/chunks/chunks.jsonl")

    if not chunks_file.exists():
        logger.warning("chunks.jsonl not found — using empty corpus")
        return tuple()

    chunks = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    logger.info(f"Loaded {len(chunks)} chunks")
    return tuple(chunks)

# ======================================================
# HELPERS
# ======================================================

def _load_history(_: dict) -> str:
    if MEMORY is None:
        return ""
    history = MEMORY.load_memory_variables({}).get("history", [])
    return "\n".join(str(h) for h in history)


def _resolve_followup_question(question: str) -> str:
    """
    Resolve short / ambiguous follow-up questions using conversation memory
    BEFORE retrieval.
    """
    if MEMORY is None:
        return question

    history = MEMORY.load_memory_variables({}).get("history", [])
    if not history:
        return question

    last_user_questions = [
        h.content for h in history
        if getattr(h, "type", "") == "human"
    ]

    if not last_user_questions:
        return question

    last_topic = last_user_questions[-1]

    # Heuristic: very short / vague question
    if len(question.split()) <= 4:
        resolved = f"{question} in the context of {last_topic}"
        logger.info(f"Resolved follow-up query: {resolved}")
        return resolved

    return question


def _format_context(docs: List[Dict]) -> str:
    """
    Numbered, source-aware context.
    This is CRITICAL for good answers.
    """
    formatted = []

    for i, d in enumerate(docs, 1):
        meta = d["metadata"]
        source = f"{meta.get('pdf')} | page {meta.get('page')}"
        formatted.append(
            f"[{i}] {d['text']}\nSOURCE: {source}"
        )

    max_k = RAG_CFG["retrieval"]["max_context_chunks"]
    return "\n\n".join(formatted[:max_k])


def _append_sources(answer: str, docs: List[Dict]) -> str:
    sources = []
    for i, d in enumerate(docs, 1):
        meta = d["metadata"]
        sources.append(
            f"[{i}] {meta.get('pdf')} | page {meta.get('page')}"
        )

    return answer.strip() + "\n\nSources:\n" + "\n".join(sources)

# ======================================================
# 🔗 RAG CHAIN (FINAL)
# ======================================================

def rag_chain(question: str):
    """
    Streaming RAG chain.
    - Structured context
    - Real citations
    - Conversational retrieval
    - CI safe
    """

    logger.info(f"Received question: {question}")

    # -------------------------------
    # 🔎 Retrieval (WITH CONTEXT RESOLUTION)
    # -------------------------------
    start = time.perf_counter()

    chunks = load_chunks()
    resolved_question = _resolve_followup_question(question)
    retrieved_docs = hybrid_retrieve(resolved_question, chunks)

    retrieval_time = time.perf_counter() - start
    logger.info(
        f"Retrieved {len(retrieved_docs)} docs in {retrieval_time:.3f}s"
    )

    # -------------------------------
    # 🚫 CI SHORT-CIRCUIT
    # -------------------------------
    if _is_test_env():
        logger.warning("Skipping LLM execution in CI/test environment")
        return

    # -------------------------------
    # 🧠 Context + Chain
    # -------------------------------
    context = _format_context(retrieved_docs)

    chain = (
        {
            "context": lambda _: context,
            "question": RunnablePassthrough(),
            "history": RunnableLambda(_load_history),
        }
        | MEDICAL_RAG_PROMPT
        | LLM
        | StrOutputParser()
    )

    # -------------------------------
    # 🔄 Streaming
    # -------------------------------
    full_answer = ""

    for token in chain.stream(question):
        full_answer += token
        yield token

    # -------------------------------
    # 🛡 Guardrails
    # -------------------------------
    final_answer = validate_answer(
        answer=full_answer,
        sources=retrieved_docs,
        cfg=RAG_CFG,
    )

    final_answer = _append_sources(final_answer, retrieved_docs)

    # Stream only the appended part
    if final_answer.startswith(full_answer):
        yield final_answer[len(full_answer):]
    else:
        yield "\n\n" + final_answer

    if MEMORY is not None:
        MEMORY.save_context(
            {"input": question},
            {"output": final_answer},
        )
