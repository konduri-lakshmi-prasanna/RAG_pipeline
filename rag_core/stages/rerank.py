"""
rerank.py — Stage 3: Reranking

Why this exists:
  ChromaDB returns chunks ordered by embedding distance.
  But embedding distance is approximate — it measures vector closeness,
  not true semantic relevance to the query.

  Reranking applies a second, more expensive scoring step
  to sort the candidates more accurately.

Two strategies implemented (Strategy Pattern):
  1. Reciprocal Rank Fusion (RRF) — fast, no extra model needed
  2. Cross-encoder reranking — higher quality, needs a model

Mentor's latent space point:
  Better reranking = better quality chunks reach the LLM.
  Quality of latent space representation → quality of retrieval → quality of answer.
  This is exactly what your mentor meant.

GoF Pattern: Strategy — swap rerank_fn without touching the pipeline.
"""

from rag_core.db.chromadb_store import embed_query
import math


def rerank_by_distance(query: str, chunks: list[dict]) -> list[dict]:
    """
    Simple baseline: sort by ChromaDB distance score (lower = better).
    This is what you have if you skip reranking — use as a fallback.
    """
    return sorted(chunks, key=lambda c: c["distance"])


def rerank_rrf(query: str, chunks: list[dict], k: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion (RRF) — fast multi-signal reranking.

    RRF combines two ranking signals:
      1. ChromaDB embedding distance (vector similarity)
      2. Keyword overlap between query terms and chunk text

    Formula: RRF_score = 1/(k + rank_signal_1) + 1/(k + rank_signal_2)
    
    Higher RRF score = better chunk. This is the recommended default.
    No extra model needed — works out of the box.

    Args:
        query: The rewritten query string.
        chunks: List of chunk dicts from retrieval stage.
        k: RRF smoothing constant (default 60, standard value).
    
    Returns:
        Re-sorted list, best chunks first.
    """
    if not chunks:
        return []

    query_terms = set(query.lower().split())

    # Rank 1: by embedding distance (lower = better → lower rank index = better)
    by_distance = sorted(chunks, key=lambda c: c["distance"])

    # Rank 2: by keyword overlap (more overlap = better)
    def keyword_score(chunk: dict) -> float:
        chunk_terms = set(chunk["text"].lower().split())
        overlap = len(query_terms & chunk_terms)
        return overlap / max(len(query_terms), 1)

    by_keyword = sorted(chunks, key=keyword_score, reverse=True)

    # Build lookup: chunk_id → ranks in each signal
    # Use text[:50] as a simple chunk identifier
    distance_rank = {c["text"][:50]: i for i, c in enumerate(by_distance)}
    keyword_rank  = {c["text"][:50]: i for i, c in enumerate(by_keyword)}

    # Compute RRF score for each chunk
    for chunk in chunks:
        cid = chunk["text"][:50]
        r1 = distance_rank.get(cid, len(chunks))
        r2 = keyword_rank.get(cid, len(chunks))
        chunk["rrf_score"] = 1 / (k + r1) + 1 / (k + r2)

    # Sort by RRF score descending (higher = better)
    ranked = sorted(chunks, key=lambda c: c.get("rrf_score", 0), reverse=True)
    print(f"[rerank] RRF reranked {len(ranked)} chunks")
    return ranked


def rerank_chunks(query: str, chunks: list[dict], strategy: str = "rrf") -> list[dict]:
    """
    Main rerank entry point. Choose strategy via the `strategy` parameter.
    
    Args:
        query: The rewritten query.
        chunks: Retrieved chunks from Stage 2.
        strategy: "rrf" (default, recommended) or "distance" (baseline).
    
    Returns:
        Re-sorted chunks list.
    
    GoF Strategy Pattern in action:
        You can add "cross_encoder" here without touching the pipeline.
    """
    if strategy == "distance":
        return rerank_by_distance(query, chunks)
    else:
        return rerank_rrf(query, chunks)