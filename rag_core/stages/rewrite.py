"""
rewrite.py — Stage 1: Query Rewriting

Why this exists:
  Users type vague queries like "jobs for me" or "analyse resume".
  The LLM rewrites these into precise, searchable queries.
  Better query = better retrieval = better final answer.

GoF Pattern: Strategy (partial)
  The rewrite strategy can be swapped — LLM-based vs rule-based.

Latent space connection (what your mentor mentioned):
  The rewritten query gets embedded into the latent space.
  A precise query lands in a more specific region of latent space.
  That gives ChromaDB a better vector to search with.
  So rewriting DIRECTLY improves retrieval quality.
"""

from rag_core.llm.factory import get_llm
def rewrite_query(query: str, context_hint: str = "") -> str:
    """
    Rewrites a raw user query into a precise, searchable form.

    Args:
        query: The raw user input.
        context_hint: Optional domain context (e.g., "career guidance", "resume analysis").

    Returns:
        An improved query string.

    Example:
        Input:  "jobs for me"
        Output: "software engineering job opportunities for a student with Python and machine learning skills"
    """
    llm = get_llm()

    prompt = f"""You are a query optimization expert for a RAG (Retrieval-Augmented Generation) system.

Your job: Rewrite the user's query to be more specific, detailed, and searchable.

Rules:
- Keep the original intent exactly
- Add relevant domain-specific terms
- Expand abbreviations and vague pronouns
- Make it 1-3 sentences maximum
- Return ONLY the rewritten query, no explanation

Domain context: {context_hint if context_hint else "general career and resume assistance"}

Original query: {query}

Rewritten query:"""

    try:
        response = llm.invoke(prompt)
        rewritten = response.content.strip() if hasattr(response, "content") else str(response).strip()
        # Fallback: if LLM returns something too short or empty, use original
        if len(rewritten) < 10:
            return query
        return rewritten
    except Exception as e:
        print(f"[rewrite] Warning: LLM rewrite failed ({e}), using original query")
        return query