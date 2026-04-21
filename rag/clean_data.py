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





# ─────────────────────────────────────────────
# 5. EMBEDDINGS
# ─────────────────────────────────────────────

def embed_entries(entries: list[QAEntry], model_name: str = "all-MiniLM-L6-v2"):
    """Generate sentence embeddings. Returns (entries, embeddings np.ndarray)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit("Run: pip install sentence-transformers")

    log.info(f"Loading embedding model '{model_name}'…")
    model = SentenceTransformer(model_name)

    corpus = [e.combined for e in entries]
    log.info(f"Embedding {len(corpus)} entries…")
    embeddings = model.encode(corpus, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    log.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings


# ─────────────────────────────────────────────
# 6. CHROMADB INGEST
# ─────────────────────────────────────────────

def ingest_chromadb(entries: list[QAEntry], embeddings, db_path: str = "./chroma_db"):
    """
    Creates two types of ChromaDB collections:
      - 'qa_all'              → unified collection (for global RAG + similarity)
      - 'qa_<category_slug>'  → one per theme (for scoped RAG)
    """
    try:
        import chromadb
    except ImportError:
        raise SystemExit("Run: pip install chromadb")

    client = chromadb.PersistentClient(path=db_path)
    log.info(f"ChromaDB path: {db_path}")

    def slug(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

    def batch_upsert(collection, ids, embeddings_list, documents, metadatas, batch=500):
        for i in range(0, len(ids), batch):
            collection.upsert(
                ids        = ids[i:i+batch],
                embeddings = embeddings_list[i:i+batch],
                documents  = documents[i:i+batch],
                metadatas  = metadatas[i:i+batch],
            )

    ids        = [str(e.id) for e in entries]
    documents  = [e.combined for e in entries]
    metadatas  = [{"category": e.category, "question": e.question,
                   "answer": e.answer, "token_count": e.token_count} for e in entries]
    emb_list   = embeddings.tolist()

    # ── Unified collection ──────────────────────────────
    col_all = client.get_or_create_collection(
        name     = "qa_all",
        metadata = {"hnsw:space": "cosine", "description": "All QA entries unified"},
    )
    batch_upsert(col_all, ids, emb_list, documents, metadatas)
    log.info(f"'qa_all' collection: {col_all.count()} documents.")

    # ── Per-category collections ────────────────────────
    categories = list({e.category for e in entries})
    for cat in categories:
        cat_entries = [(e, emb) for e, emb in zip(entries, emb_list) if e.category == cat]
        col = client.get_or_create_collection(
            name     = f"qa_{slug(cat)}",
            metadata = {"hnsw:space": "cosine", "category": cat},
        )
        batch_upsert(
            col,
            ids        = [str(e.id) for e, _ in cat_entries],
            embeddings_list = [emb for _, emb in cat_entries],
            documents  = [e.combined for e, _ in cat_entries],
            metadatas  = [{"category": e.category, "question": e.question,
                           "answer": e.answer, "token_count": e.token_count}
                          for e, _ in cat_entries],
        )
        log.info(f"  '{col.name}': {col.count()} docs")

    log.info("ChromaDB ingest complete.")
    return client


# ─────────────────────────────────────────────
# 7. QUERY HELPERS  (RAG + Similarity)
# ─────────────────────────────────────────────

class QARetriever:
    """
    Two query modes:
      - similarity_search(query)  → returns top-k most similar QA pairs
      - rag_query(query)          → returns context string ready for an LLM prompt
    """

    def __init__(self, db_path: str = "./chroma_db", model_name: str = "all-MiniLM-L6-v2",
                 collection: str = "qa_all"):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.model  = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=db_path)
        self.col    = self.client.get_collection(collection)
        log.info(f"Retriever ready on '{collection}' ({self.col.count()} docs).")

    def _embed_query(self, query: str):
        return self.model.encode([query], normalize_embeddings=True).tolist()

    def similarity_search(self, query: str, k: int = 5, category: Optional[str] = None) -> list[dict]:
        """Return top-k QA pairs most similar to the query."""
        where = {"category": category} if category else None
        results = self.col.query(
            query_embeddings = self._embed_query(query),
            n_results        = k,
            where            = where,
            include          = ["metadatas", "distances"],
        )
        hits = []
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            hits.append({
                "score"   : round(1 - dist, 4),   # cosine similarity
                "category": meta["category"],
                "question": meta["question"],
                "answer"  : meta["answer"],
            })
        return hits

    def rag_query(self, query: str, k: int = 5, category: Optional[str] = None) -> str:
        """
        Return a formatted context string ready to inject into an LLM prompt.
        Plug this into: f'Context:\\n{context}\\n\\nUser question: {query}'
        """
        hits = self.similarity_search(query, k=k, category=category)
        lines = [f"[{i+1}] (category: {h['category']}, score: {h['score']})\n"
                 f"  Q: {h['question']}\n  A: {h['answer']}"
                 for i, h in enumerate(hits)]
        return "\n\n".join(lines)


# ─────────────────────────────────────────────
# 8. CLI ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QA Pipeline: clean → embed → ChromaDB")
    parser.add_argument("--input",      required=True,              help="Path to your .md dataset file")
    parser.add_argument("--out_dir",    default="./output",         help="Directory for JSON/CSV exports")
    parser.add_argument("--db_path",    default="./chroma_db",      help="ChromaDB persistence directory")
    parser.add_argument("--model",      default="all-MiniLM-L6-v2", help="Sentence-transformer model name")
    parser.add_argument("--skip_embed", action="store_true",        help="Stop after cleaning/exporting (no embedding)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Parse ──────────────────────────────────────────
    raw_text = Path(args.input).read_text(encoding="utf-8")
    entries  = parse_markdown(raw_text)

    # ── Clean ──────────────────────────────────────────
    entries = clean_entries(entries)

    # ── Export ─────────────────────────────────────────
    export_json(entries, out_dir)
    export_csv(entries,  out_dir)

    if args.skip_embed:
        log.info("--skip_embed set. Stopping before embedding.")
        return

    # ── Embed ──────────────────────────────────────────
    embeddings = embed_entries(entries, model_name=args.model)

    # ── ChromaDB ───────────────────────────────────────
    ingest_chromadb(entries, embeddings, db_path=args.db_path)

    log.info("✅ Pipeline complete.")
    log.info(f"   JSON/CSV  → {out_dir}/")
    log.info(f"   ChromaDB  → {args.db_path}/")
    log.info("")
    log.info("Example usage after pipeline:")
    log.info("  from pipeline import QARetriever")
    log.info("  r = QARetriever()")
    log.info("  print(r.similarity_search('What is a black hole?', k=3))")
    log.info("  print(r.rag_query('How does machine learning work?'))")


if __name__ == "__main__":
    main()