import json
import time
import yaml
from pathlib import Path

from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.memory import ConversationBufferWindowMemory

from src.rag.prompts import MEDICAL_RAG_PROMPT
from src.rag.guardrails import validate_answer
from src.retrieval.hybrid import hybrid_retrieve
from src.utils.logging import setup_logging

logger = setup_logging("rag_chain")


# ======================================================
# CONFIG
# ======================================================

with open("configs/rag.yaml", "r", encoding="utf-8") as f:
    RAG_CFG = yaml.safe_load(f)

logger.info("RAG config loaded successfully")


# ======================================================
# LLM (WARM START)
# ======================================================

logger.info("Initializing Ollama LLM")

LLM = OllamaLLM(
    model=RAG_CFG["llm"]["model"],
    temperature=RAG_CFG["llm"]["temperature"],
    streaming=RAG_CFG["llm"]["stream"],
)

MEMORY = ConversationBufferWindowMemory(
    k=RAG_CFG["memory"]["window_size"],
    return_messages=True,
)

logger.info("LLM and memory initialized")


# ======================================================
# DATA LOADING (LAZY, CI-SAFE)
# ======================================================

def load_chunks():
    """
    Loads cleaned chunks from JSONL.
    Safe for CI: returns empty tuple if file is missing.
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


# ======================================================
# HELPERS
# ======================================================

def format_context(docs: list[dict]) -> str:
    """
    Extracts text from retrieved docs and formats context.
    """
    texts = [doc["text"] for doc in docs]
    max_chunks = RAG_CFG["retrieval"]["max_context_chunks"]
    return "\n\n".join(texts[:max_chunks])


# ======================================================
# RAG CHAIN
# ======================================================

def rag_chain(question: str):
    """
    Streaming RAG pipeline with hybrid retrieval + guardrails.
    """
    logger.info(f"Received question: {question}")

    start = time.perf_counter()

    chunks = load_chunks()
    logger.info(f"Corpus size for retrieval: {len(chunks)}")

    retrieved_docs = hybrid_retrieve(question, chunks)
    retrieval_time = time.perf_counter() - start

    logger.info(
        f"Retrieved {len(retrieved_docs)} documents "
        f"in {retrieval_time:.3f}s"
    )

    context = format_context(retrieved_docs)

    chain = (
        {
            "context": lambda _: context,
            "question": RunnablePassthrough(),
        }
        | MEDICAL_RAG_PROMPT
        | LLM
        | StrOutputParser()
    )

    logger.info("Starting LLM streaming response")

    answer = ""
    for token in chain.stream(question):
        answer += token
        yield token

    logger.info("LLM streaming completed")

    # -------------------------------
    # GUARDRAILS
    # -------------------------------

    sources = [doc["metadata"] for doc in retrieved_docs]
    final_answer = validate_answer(answer, sources, RAG_CFG)

    if final_answer != answer:
        logger.warning("Guardrails triggered — overriding response")
        yield final_answer
    else:
        logger.info("Response passed guardrails successfully")
