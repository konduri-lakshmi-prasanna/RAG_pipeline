"""
refine.py — Stage 4: Context Refinement

Why this exists:
  After reranking, we still have many chunks.
  Not all of them should go to the LLM because:
    1. LLMs have token limits — too much context causes truncation
    2. Irrelevant chunks confuse the LLM ("noise")
    3. Duplicate chunks waste tokens

  Refine filters to the best, non-redundant chunks.

Low coupling / high cohesion:
  This module has ONE job: clean the chunk list.
  It does not know about retrieval, the LLM, or the DB.
  This is high cohesion.
  It receives a list and returns a list — no side effects.
  This is loose coupling.
"""


def remove_duplicates(chunks: list[dict], threshold: float = 0.9) -> list[dict]:
    """
    Removes near-duplicate chunks by text overlap ratio.
    
    Two chunks are considered duplicates if their text overlap
    (Jaccard similarity) exceeds the threshold.

    Args:
        chunks: Sorted chunk list (best first).
        threshold: Similarity above which a chunk is considered duplicate.
    
    Returns:
        De-duplicated list, preserving original order.
    """
    seen = []
    result = []

    for chunk in chunks:
        chunk_words = set(chunk["text"].lower().split())
        is_duplicate = False

        for existing_words in seen:
            if not chunk_words or not existing_words:
                continue
            intersection = chunk_words & existing_words
            union = chunk_words | existing_words
            jaccard = len(intersection) / len(union)
            if jaccard >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            seen.append(chunk_words)
            result.append(chunk)

    print(f"[refine] Duplicates removed: {len(chunks)} → {len(result)} chunks")
    return result


def filter_low_relevance(chunks: list[dict], max_distance: float = 1.2) -> list[dict]:
    """
    Removes chunks with very high distance scores (i.e., low relevance).
    
    ChromaDB cosine distance: 0 = identical, 2 = opposite.
    Chunks above max_distance are almost certainly irrelevant.

    Args:
        chunks: Chunk list.
        max_distance: Discard chunks with distance above this value.
    
    Returns:
        Filtered list.
    """
    filtered = [c for c in chunks if c.get("distance", 0) <= max_distance]
    if len(filtered) < len(chunks):
        print(f"[refine] Low-relevance removed: {len(chunks)} → {len(filtered)} chunks")
    return filtered


def select_top_k(chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Keeps only the top-k chunks for the LLM context window.
    
    Even after dedup and filtering, we cap the total to avoid
    exceeding the LLM's context limit.

    Args:
        chunks: Sorted, filtered chunk list.
        top_k: Maximum number of chunks to send to the LLM.
    """
    return chunks[:top_k]


def refine_chunks(chunks: list[dict], top_k: int = 5, max_distance: float = 1.2) -> list[str]:
    """
    Full refinement pipeline: filter → deduplicate → top-k → extract text.
    
    Args:
        chunks: Reranked chunk list from Stage 3.
        top_k: How many chunks to keep for the LLM.
        max_distance: Max allowed distance for relevance.
    
    Returns:
        List of text strings ready to be joined as LLM context.
    """
    chunks = filter_low_relevance(chunks, max_distance)
    chunks = remove_duplicates(chunks)
    chunks = select_top_k(chunks, top_k)
    texts = [c["text"] for c in chunks]
    print(f"[refine] Final context: {len(texts)} chunks selected for LLM")
    return texts