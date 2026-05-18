"""
retrieval.py — Stage 2: Vector Database Retrieval

Why this exists:
  Searches ChromaDB for document chunks that are semantically
  similar to the query. Returns raw candidates for reranking.

Low coupling design:
  This module ONLY knows about chromadb_store.
  It does NOT know about the LLM, the query rewriter, or the generator.
  Changing ChromaDB to Pinecone = only change this file.

Latent space (your mentor's point):
  embed_query() converts the text to a vector.
  ChromaDB finds vectors closest in cosine distance.
  "Closest" means semantically similar — this IS the latent space search.

GoF Pattern: Strategy (the retrieval algorithm is swappable)
"""

from rag_core.db.chromadb_store import get_collection, embed_query


def retrieve_chunks(
    query: str,
    collection_name: str,
    k: int = 10,
    where: dict = None
) -> list[dict]:
    """
    Searches the ChromaDB collection for the top-k most relevant chunks.

    Args:
        query: The rewritten query string.
        collection_name: Which ChromaDB collection to search.
        k: Number of results to return.
        where: Optional metadata filter (e.g., {"type": "resume"}).

    Returns:
        List of dicts: [{
            "text": "chunk text",
            "metadata": {...},
            "distance": 0.12   # lower = more similar
        }]
    """
    collection = get_collection(collection_name)

    if collection.count() == 0:
        print(f"[retrieval] Warning: collection '{collection_name}' is empty")
        return []

    # Embed the query into the latent space
    query_embedding = embed_query(query)

    # Limit k to whats actually in the collection
    actual_k = min(k, collection.count())

    # Build query kwargs
    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": actual_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    try:
        results = collection.query(**query_kwargs)
    except Exception as e:
        print(f"[retrieval] Error during query: {e}")
        return []

    # Flatten ChromaDB response format into clean dicts
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "metadata": meta,
            "distance": dist,
        })

    print(f"[retrieval] Retrieved {len(chunks)} chunks from '{collection_name}'")
    return chunks