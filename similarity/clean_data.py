"""
QA Dataset Pipeline: Clean → Embed → ChromaDB
Supports: RAG queries and Similarity search
Usage: python pipeline.py --input your_dataset.md
"""

import re
import json
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 1. DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class QAEntry:
    id: int
    category: str
    question: str
    answer: str
    combined: str          # question + answer, used for embedding
    token_count: int       # rough word count


# ─────────────────────────────────────────────
# 2. PARSER
# ─────────────────────────────────────────────

CATEGORY_PATTERN = re.compile(r"^#{1,3}\s+(.+?)\s*\(\d+[-–]\d+\)", re.MULTILINE)
ENTRY_PATTERN    = re.compile(
    r"\*{0,2}(\d+)\.\s+Q:\s*\*{0,2}(.+?)\*{0,2}\s*\nA:\s*(.+?)(?=\n\*{0,2}\d+\.\s+Q:|\n#{1,3}|\Z)",
    re.DOTALL,
)

def parse_markdown(text: str) -> list[QAEntry]:
    """Parse the Markdown QA file into structured QAEntry objects."""
    # Build a map: entry_id → category by scanning headers
    cat_map: dict[int, str] = {}
    current_cat = "Unknown"
    for line in text.splitlines():
        m = CATEGORY_PATTERN.match(line)
        if m:
            current_cat = m.group(1).strip()
            # Extract the range, e.g. (1-10)
            rng = re.search(r"\((\d+)[-–](\d+)\)", line)
            if rng:
                for i in range(int(rng.group(1)), int(rng.group(2)) + 1):
                    cat_map[i] = current_cat

    entries: list[QAEntry] = []
    for m in ENTRY_PATTERN.finditer(text):
        eid      = int(m.group(1))
        question = m.group(2).strip()
        answer   = m.group(3).strip()
        category = cat_map.get(eid, "Unknown")
        combined = f"Question: {question} Answer: {answer}"
        entries.append(QAEntry(
            id          = eid,
            category    = category,
            question    = question,
            answer      = answer,
            combined    = combined,
            token_count = len(combined.split()),
        ))

    log.info(f"Parsed {len(entries)} entries across {len(set(e.category for e in entries))} categories.")
    return entries


# ─────────────────────────────────────────────
# 3. CLEANER  (sklearn-based)
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalize a single text string."""
    text = re.sub(r"\*{1,2}",   "",   text)   # remove markdown bold/italic
    text = re.sub(r"`{1,3}",    "",   text)   # remove code ticks
    text = re.sub(r"\[.*?\]",   "",   text)   # remove markdown links labels
    text = re.sub(r"\(https?://\S+\)", "", text)  # remove URLs
    text = re.sub(r"\s+",       " ",  text)   # collapse whitespace
    return text.strip()


def clean_entries(entries: list[QAEntry]) -> list[QAEntry]:
    """Clean all text fields and remove near-duplicates via TF-IDF cosine similarity."""
    # Clean text fields
    for e in entries:
        e.question   = clean_text(e.question)
        e.answer     = clean_text(e.answer)
        e.combined   = f"Question: {e.question} Answer: {e.answer}"
        e.token_count = len(e.combined.split())

    # Deduplication: flag entries whose cosine similarity > 0.92
    log.info("Running TF-IDF deduplication…")
    corpus = [e.combined for e in entries]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20_000, sublinear_tf=True)
    tfidf = normalize(vectorizer.fit_transform(corpus))

    # Only check consecutive windows for speed on large datasets
    duplicates: set[int] = set()
    chunk = 500
    for i in range(0, len(entries), chunk):
        block = tfidf[i : i + chunk]
        scores = (block @ block.T).toarray()
        for r in range(scores.shape[0]):
            for c in range(r + 1, scores.shape[1]):
                if scores[r, c] > 0.92:
                    idx = i + c
                    if idx not in duplicates:
                        log.warning(f"Near-duplicate found: entry {entries[i+r].id} ↔ {entries[idx].id} (sim={scores[r,c]:.3f})")
                        duplicates.add(idx)

    cleaned = [e for j, e in enumerate(entries) if j not in duplicates]
    log.info(f"After dedup: {len(cleaned)} entries ({len(duplicates)} removed).")
    return cleaned


# ─────────────────────────────────────────────
# 4. EXPORT  (JSON + CSV)
# ─────────────────────────────────────────────

def export_json(entries: list[QAEntry], out_dir: Path) -> Path:
    """Export grouped JSON: { category: [entries] }"""
    grouped: dict[str, list] = {}
    for e in entries:
        grouped.setdefault(e.category, []).append(asdict(e))

    path = out_dir / "dataset_clean.json"
    path.write_text(json.dumps(grouped, indent=2, ensure_ascii=False))
    log.info(f"JSON saved → {path}")
    return path


def export_csv(entries: list[QAEntry], out_dir: Path) -> Path:
    """Export flat CSV for inspection / backup."""
    import csv
    path = out_dir / "dataset_clean.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id","category","question","answer","token_count"])
        writer.writeheader()
        for e in entries:
            writer.writerow({
                "id": e.id, "category": e.category,
                "question": e.question, "answer": e.answer,
                "token_count": e.token_count,
            })
    log.info(f"CSV saved → {path}")
    return path

