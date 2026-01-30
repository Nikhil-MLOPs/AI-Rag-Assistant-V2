import json
import yaml
import time
from pathlib import Path
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.memory import ConversationBufferWindowMemory

from src.rag.prompts import MEDICAL_RAG_PROMPT
from src.rag.guardrails import validate_answer
from src.retrieval.hybrid import hybrid_retrieve
from src.utils.logging import setup_logging

logger = setup_logging("Rag-Chain")

# -------------------------------
# CONFIG
# -------------------------------

with open("configs/rag.yaml", "r", encoding="utf-8") as f:
    RAG_CFG = yaml.safe_load(f)

# -------------------------------
# LOAD CHUNKS (JSONL, NOT TXT)
# -------------------------------

logger.info("Loading chunks into memory")

CHUNKS = []
chunks_file = Path("data/processed/chunks/chunks.jsonl")

with open(chunks_file, "r", encoding="utf-8") as f:
    for line in f:
        CHUNKS.append(json.loads(line))

CHUNKS = tuple(CHUNKS)  # safe to cache on text later
logger.info(f"Loaded {len(CHUNKS)} chunks")

# -------------------------------
# LLM (WARM START)
# -------------------------------

LLM = OllamaLLM(
    model=RAG_CFG["llm"]["model"],
    temperature=RAG_CFG["llm"]["temperature"],
    streaming=RAG_CFG["llm"]["stream"],
)

MEMORY = ConversationBufferWindowMemory(
    k=RAG_CFG["memory"]["window_size"],
    return_messages=True,
)

# -------------------------------
# HELPERS
# -------------------------------

def format_context(docs: list[dict]) -> str:
    texts = [doc["text"] for doc in docs]
    return "\n\n".join(texts[:RAG_CFG["retrieval"]["max_context_chunks"]])

# -------------------------------
# RAG CHAIN
# -------------------------------

def rag_chain(question: str):
    start = time.perf_counter()

    retrieved_docs = hybrid_retrieve(question, CHUNKS)
    retrieval_time = time.perf_counter() - start
    logger.info(f"Retrieval time: {retrieval_time:.3f}s")

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

    answer = ""
    for token in chain.stream(question):
        answer += token
        yield token

    # -------------------------------
    # GUARDRAILS
    # -------------------------------

    sources = [doc["metadata"] for doc in retrieved_docs]
    final_answer = validate_answer(answer, sources, RAG_CFG)

    if final_answer != answer:
        yield final_answer
