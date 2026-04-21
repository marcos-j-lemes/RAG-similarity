"""
examples.py — How to use QARetriever after running pipeline.py
 
Run the pipeline first:
    python pipeline.py --input your_dataset.md
 
Then run these examples:
    python examples.py
"""
 
from pipeline import QARetriever
 
# ─────────────────────────────────────────────
# SIMILARITY SEARCH
# Find QA pairs most semantically similar to a free-text query.
# Useful for: "show me questions like this one", duplicate detection,
#             topic clustering, recommendation.
# ─────────────────────────────────────────────
 
def demo_similarity():
    print("=" * 60)
    print("SIMILARITY SEARCH")
    print("=" * 60)
 
    retriever = QARetriever(collection="qa_all")
 
    queries = [
        "How does the universe expand?",
        "What makes humans conscious?",
        "How does encryption keep data safe?",
    ]
 
    for query in queries:
        print(f"\nQuery: \"{query}\"")
        print("-" * 40)
        results = retriever.similarity_search(query, k=3)
        for r in results:
            print(f"  [{r['score']:.3f}] ({r['category']})  {r['question']}")
            print(f"         → {r['answer'][:120]}…")
 
 
# ─────────────────────────────────────────────
# CATEGORY-SCOPED SIMILARITY
# Search only within a specific thematic group.
# ─────────────────────────────────────────────
 
def demo_scoped_similarity():
    print("\n" + "=" * 60)
    print("SCOPED SIMILARITY  (within a single category)")
    print("=" * 60)
 
    # Use the per-category collection directly
    retriever = QARetriever(collection="qa_space_astronomy")  # adjust slug as needed
 
    query = "Why do stars die?"
    print(f"\nQuery: \"{query}\"  [Space & Astronomy only]")
    print("-" * 40)
    for r in retriever.similarity_search(query, k=3):
        print(f"  [{r['score']:.3f}]  {r['question']}")
        print(f"         → {r['answer'][:120]}…")

        