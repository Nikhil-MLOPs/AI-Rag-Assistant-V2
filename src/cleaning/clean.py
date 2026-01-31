import json
import yaml
from pathlib import Path
from typing import List, Dict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.logging import setup_logging

logger = setup_logging("Cleaning")

# ======================================================
# STRUCTURAL CONSTANTS
# ======================================================

SECTION_HEADERS = {
    "definition",
    "description",
    "purpose",
    "preparation",
    "causes and symptoms",
    "causes",
    "symptoms",
    "diagnosis",
    "treatment",
    "alternative treatment",
    "alternative treatments",
    "prevention",
    "prognosis",
    "risks",
    "aftercare",
    "normal results",
    "abnormal results",
    "precautions",
    "cost",
    "results",
    "key terms",
}

CONTROL_CHARS = ["\u0002"]

# ======================================================
# CONFIG
# ======================================================

def load_cleaning_config() -> dict:
    with open("configs/cleaning.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ======================================================
# LINE UTILITIES
# ======================================================

def clean_line(line: str) -> str:
    for ch in CONTROL_CHARS:
        line = line.replace(ch, "")
    return line.strip()


def is_noise_line(line: str) -> bool:
    return not line or line.isdigit() or len(line) <= 2


def is_cross_reference(line: str) -> bool:
    return " see " in line.lower()


def is_section_header(line: str) -> bool:
    return line.lower() in SECTION_HEADERS


def is_author_line(line: str) -> bool:
    words = line.split()
    if 2 <= len(words) <= 4:
        return all(w[0].isupper() for w in words)
    return False


# ======================================================
# 🔵 ADDED: ALPHABET INDEX HEADERS (A–Z)
# ======================================================

def is_alphabet_header(line: str) -> bool:
    return len(line) == 1 and line.isalpha() and line.isupper()


# ======================================================
# 🔵 ADDED: CROSS-REFERENCE BLOCK (WITH TERMINATION)
# ======================================================

def is_cross_reference_block(lines: List[str], idx: int) -> bool:
    """
    Cross-reference blocks end IMMEDIATELY
    when a real topic + Definition starts.
    """

    # 🔵 ADDITION: if this line starts a topic, it's NOT cross-ref
    if idx + 1 < len(lines) and lines[idx + 1].lower() == "definition":
        return False

    if is_cross_reference(lines[idx]):
        return True

    if idx > 0 and is_cross_reference(lines[idx - 1]):
        return True

    return False


# ======================================================
# DE-HYPHENATION LOGIC
# ======================================================

def merge_hyphenated_lines(lines: List[str]) -> List[str]:
    merged: List[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if (
            line.endswith("-")
            and i + 1 < len(lines)
            and lines[i + 1]
            and lines[i + 1][0].islower()
        ):
            merged.append(line[:-1] + lines[i + 1])
            i += 2
        else:
            merged.append(line)
            i += 1

    return merged


# ======================================================
# STRICT TOPIC DETECTION (EXTENDED, NOT CHANGED)
# ======================================================

def detect_topic(lines: List[str], idx: int) -> tuple[str | None, int]:
    def is_valid_topic_line(line: str) -> bool:
        return not (
            is_cross_reference(line)
            or is_author_line(line)
            or is_section_header(line)
            or ";" in line
        )

    # 🔵 ADDITION: allow topic even after cross-ref if Definition follows
    if is_cross_reference_block(lines, idx):
        return None, 0

    line = lines[idx]

    # ---------- Single-line topic ----------
    if (
        idx + 1 < len(lines)
        and lines[idx + 1].lower() == "definition"
        and is_valid_topic_line(line)
    ):
        return line.strip(), 1

    # ---------- Two-line topic ----------
    if (
        idx + 2 < len(lines)
        and lines[idx + 2].lower() == "definition"
        and is_valid_topic_line(line)
        and is_valid_topic_line(lines[idx + 1])
    ):
        return f"{line} {lines[idx + 1]}".strip(), 2

    return None, 0


# ======================================================
# CORE CLEANING + HIERARCHICAL CHUNKING
# ======================================================

def clean_and_chunk(pages: List[Document]) -> List[Document]:
    cfg = load_cleaning_config()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
    )

    chunks: List[Document] = []
    current_topic = None

    for page in pages:
        raw_lines = [
            clean_line(l)
            for l in page.page_content.splitlines()
            if clean_line(l)
        ]

        raw_lines = merge_hyphenated_lines(raw_lines)

        i = 0
        current_section = None
        section_buffers: Dict[str, List[str]] = {}
        inside_resources = False

        while i < len(raw_lines):
            line = raw_lines[i]

            if is_noise_line(line):
                i += 1
                continue

            # 🔵 Alphabet headers (A, B, C…)
            if is_alphabet_header(line):
                i += 1
                continue

            # ---------- Resources ----------
            if line.lower() == "resources":
                inside_resources = True
                i += 1
                continue

            if inside_resources:
                topic, consumed = detect_topic(raw_lines, i)
                if topic:
                    inside_resources = False
                    current_topic = topic
                    current_section = "definition"
                    section_buffers = {"definition": []}
                    i += consumed + 1
                else:
                    i += 1
                continue

            # ---------- Topic ----------
            topic, consumed = detect_topic(raw_lines, i)
            if topic:
                current_topic = topic
                current_section = "definition"
                section_buffers = {"definition": []}
                i += consumed + 1
                continue

            # ---------- Section ----------
            if is_section_header(line):
                current_section = line.lower()
                section_buffers.setdefault(current_section, [])
                i += 1
                continue

            # ---------- Cross refs ----------
            if current_section is None and is_cross_reference_block(raw_lines, i):
                i += 1
                continue

            # ---------- Content ----------
            if current_section:
                section_buffers[current_section].append(line)

            i += 1

        # ---------- Emit chunks ----------
        for section, lines in section_buffers.items():
            if not lines:
                continue

            text = " ".join(lines).replace("  ", " ").strip()

            for chunk in splitter.split_text(text):
                chunks.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "topic": current_topic,
                            "section": section,
                            "pdf": page.metadata["pdf"],
                            "page": page.metadata["page"],
                        },
                    )
                )

    logger.info(f"Generated {len(chunks)} hierarchical chunks")
    return chunks


# ======================================================
# DISK IO
# ======================================================

if __name__ == "__main__":
    pages_file = Path("data/processed/pages/pages.jsonl")
    out_dir = Path("data/processed/chunks")
    out_dir.mkdir(parents=True, exist_ok=True)

    pages: List[Document] = []

    with open(pages_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            pages.append(
                Document(
                    page_content=record["text"],
                    metadata=record["metadata"],
                )
            )

    logger.info(f"Loaded {len(pages)} pages")

    chunks = clean_and_chunk(pages)

    out_file = out_dir / "chunks.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(
                json.dumps(
                    {
                        "text": chunk.page_content,
                        "metadata": chunk.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    logger.info(f"Wrote {len(chunks)} chunks to {out_file}")
