"""
insert.py — Stage 6: Document Insertion

Why this exists:
  Before the RAG pipeline can retrieve anything, documents must
  be chunked, embedded, and stored in ChromaDB.

  For careerbot: insert job listings, career guides, alumni profiles.
  For resumeanalyser: insert the student's resume PDF.

  This is the "data ingestion" stage — it runs before any chat,
  not inside the run() query flow.

High cohesion:
  This file ONLY handles document insertion.
  It knows about chunking, embedding, and the DB.
  It does NOT know about the LLM or query answering.
"""

import hashlib
from rag_core.db.chromadb_store import get_collection, embed_texts


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Splits text into overlapping word-based chunks.

    Why overlap?
      A sentence at a chunk boundary appears in BOTH adjacent chunks.
      This prevents loss of context at boundaries.
      Better chunks = better retrieval = better latent space representation.

    Args:
        text: Raw document text.
        chunk_size: Words per chunk.
        overlap: Words shared between adjacent chunks.
    
    Returns:
        List of text chunk strings.
    """
    words = text.split()
    chunks = []
    step = max(chunk_size - overlap, 1)

    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def compute_hash(text: str) -> str:
    """SHA-256 hash of text — used as a unique chunk ID."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def insert_document(
    text: str,
    collection_name: str,
    metadata: dict = None,
    chunk_size: int = 500,
    overlap: int = 100,
    doc_id_prefix: str = "doc",
) -> dict:
    """
    Full insertion pipeline: chunk → embed → store in ChromaDB.

    Args:
        text: Raw document text (from PDF, TXT, etc.).
        collection_name: Which ChromaDB collection to store in.
        metadata: Key-value metadata attached to every chunk.
                  e.g., {"type": "resume", "student_id": "123"}
        chunk_size: Words per chunk.
        overlap: Overlap between consecutive chunks.
        doc_id_prefix: Prefix for generated chunk IDs.

    Returns:
        Summary dict: {"chunks_inserted": N, "collection": name}

    Example usage in careerbot:
        insert_document(
            text=resume_text,
            collection_name="careerbot_db",
            metadata={"type": "alumni_resume", "company": "Google"},
        )
    """
    if not text.strip():
        print("[insert] Warning: empty text, nothing inserted")
        return {"chunks_inserted": 0, "collection": collection_name}

    # Step 1: Chunk
    chunks = chunk_text(text, chunk_size, overlap)
    if not chunks:
        return {"chunks_inserted": 0, "collection": collection_name}

    # Step 2: Embed
    embeddings = embed_texts(chunks)

    # Step 3: Prepare IDs and metadata
    ids = [f"{doc_id_prefix}_{compute_hash(c)}_{i}" for i, c in enumerate(chunks)]
    metadatas = [metadata or {} for _ in chunks]

    # Step 4: Store in ChromaDB (upsert = safe to re-run, no duplicates)
    collection = get_collection(collection_name)
    collection.upsert(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )

    print(f"[insert] Stored {len(chunks)} chunks in '{collection_name}'")
    return {"chunks_inserted": len(chunks), "collection": collection_name}