import json
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document

from src.utils.logging import setup_logging
from src.utils.config import load_ingestion_config

logger = setup_logging("Ingestion")

FOOTER_KEYWORDS = [
    "g a l e e n c y c l o p e d i a",
]


def clean_footer(text: str) -> str:
    lines = []
    for line in text.splitlines():
        lower = line.lower()
        if any(k in lower for k in FOOTER_KEYWORDS):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def ingest():
    cfg = load_ingestion_config("configs/ingestion.yaml")

    raw_dir = Path(cfg.raw_dir)
    out_dir = Path(cfg.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "pages.jsonl"

    total_pages = 0

    with open(out_file, "w", encoding="utf-8") as f_out:
        for pdf_path in raw_dir.glob("*.pdf"):
            logger.info(f"Ingesting {pdf_path.name}")
            doc = fitz.open(pdf_path)

            for page_index in range(len(doc)):
                if page_index < cfg.skip_start_pages:
                    continue
                if page_index > cfg.skip_end_after:
                    break

                page = doc.load_page(page_index)
                text = page.get_text("text")
                text = clean_footer(text)

                if not text.strip():
                    continue

                record = {
                    "text": text,
                    "metadata": {
                        "pdf": pdf_path.name,
                        "page": page_index + 1,
                    },
                }

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_pages += 1

    logger.info(f"Total pages ingested: {total_pages}")
    logger.info(f"Wrote pages to {out_file}")


if __name__ == "__main__":
    ingest()
