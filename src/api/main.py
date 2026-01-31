from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
import time
import asyncio

from src.rag.chain import rag_chain
from src.utils.logging import setup_logging

logger = setup_logging("API")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI lifespan starting — RAG already warm")
    yield
    logger.info("FastAPI lifespan shutting down")

app = FastAPI(
    title="AI-RAG-Assistant",
    lifespan=lifespan
)

@app.get("/health", response_class=JSONResponse)
async def health():
    return {"status": "ok"}

@app.post("/query")
async def query(payload: dict):
    question = payload.get("question", "").strip()
    if not question:
        return JSONResponse(
            status_code=400,
            content={"error": "Question cannot be empty"}
        )

    start_time = time.perf_counter()

    async def stream():
        for token in rag_chain(question):
            yield token
            await asyncio.sleep(0)

        total_time = time.perf_counter() - start_time
        logger.info(f"Total response time: {total_time:.3f}s")

    return StreamingResponse(
        stream(),
        media_type="text/plain"
    )
