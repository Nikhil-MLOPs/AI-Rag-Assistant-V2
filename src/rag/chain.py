import os
import json
import time
import yaml
from pathlib import Path

# -------------------------------------------------
# 🔐 LangSmith tracing — HARD ENFORCEMENT
# -------------------------------------------------
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "AI-Rag-Assistant-V2"

from langchain_ollama import OllamaLLM
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.memory import ConversationBufferWindowMemory

from src.rag.prompts import MEDICAL_RAG_PROMPT
from src.rag.guardrails import validate_answer
from src.retrieval.hybrid import hybrid_retrieve
from src.utils.logging import setup_logging

logger = setup_logging("Rag-Chain")

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

with open("configs/rag.yaml", "r", encoding="utf-8") as f:
    RAG_CFG = yaml.safe_load(f)

logger.info("RAG config loaded")

# -------------------------------------------------
# LLM + MEMORY (WARM START)
# -------------------------------------------------

logger.info("Initializing Ollama LLM (warm start)")

LLM = OllamaLLM(
    model=RAG_CFG["llm"]["model"],
    temperature=RAG_CFG["llm"]["temperature"],
    streaming=RAG_CFG["llm"]["stream"],
)

MEMORY = ConversationBufferWindowMemory(
    k=RAG_CFG["memory"]["window_size"],
    return_messages=True,
)

# -------------------------------------------------
# DATA LOADING (LAZY + CI SAFE)
# -------------------------------------------------

def load_chunks():
    """
    Loads cleaned chunks from JSONL.
    Returns empty tuple if file is missing (CI-safe).
    """
    chunks_file = Path("data/processed/chunks/chunks.jsonl")

    if not chunks_file.exists():
        logger.warning("chunks.jsonl not found — using empty corpus")
        return tuple()

    chunks = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    logger.info(f"Loaded {len(chunks)} chunks into memory")
    return tuple(chunks)


# -------------------------------------------------
# HELPERS
# -------------------------------------------------

def _load_history(_: dict) -> str:
    """Conversation history (NOT a knowledge source)."""
    history = MEMORY.load_memory_variables({}).get("history", [])
    return "\n".join(str(h) for h in history)


def _format_context(docs: list[dict]) -> str:
    texts = [doc["text"] for doc in docs]
    return "\n\n".join(
        texts[: RAG_CFG["retrieval"]["max_context_chunks"]]
    )


# -------------------------------------------------
# 🔗 RAG CHAIN (streaming + memory + guardrails)
# -------------------------------------------------

def rag_chain(question: str):
    """
    Streaming, warm-start, memory-aware RAG chain.
    """

    logger.info(f"Received question: {question}")

    # -------------------------------
    # 🔎 Retrieval
    # -------------------------------
    retrieval_start = time.perf_counter()

    chunks = load_chunks()
    retrieved_docs = hybrid_retrieve(question, chunks)

    retrieval_time = time.perf_counter() - retrieval_start
    logger.info(
        f"Retrieved {len(retrieved_docs)} docs "
        f"in {retrieval_time:.3f}s"
    )

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
    sources = [doc["metadata"] for doc in retrieved_docs]

    final_answer = validate_answer(
        answer=full_answer,
        sources=sources,
        cfg=RAG_CFG,
    )

    if final_answer != full_answer:
        logger.warning("Guardrails triggered — overriding response")

    MEMORY.save_context(
        {"input": question},
        {"output": final_answer},
    )
