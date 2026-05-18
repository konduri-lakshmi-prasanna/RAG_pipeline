"""
generate.py — Stage 5: Answer Generation

Why this exists:
  Takes the refined context + original question and calls the LLM.
  This is where RAG produces its grounded answer.

  The key: the LLM is CONSTRAINED to the provided context.
  It cannot hallucinate information that isn't in the retrieved chunks.
  This is the core value of RAG over plain LLM.

Reliability of LLM (your mentor's point):
  We add explicit instructions: "only use the context provided".
  We also add a fallback if the LLM returns nothing.
  This is what your mentor meant by "reliability of LLM".

GoF Pattern: Template Method (generate is one step in the fixed run() order)
"""

from rag_core.llm.factory import get_llm


def generate_answer(
    query: str,
    context: str,
    system_prompt: str = None,
    temperature: float = 0.3,
) -> str:
    """
    Calls the LLM with context and returns a grounded answer.

    Args:
        query: The original (or rewritten) user query.
        context: The refined, joined chunks from Stage 4.
        system_prompt: Optional domain-specific instructions.
                       careerbot passes its own, resumeanalyser passes its own.
                       This is where low coupling happens — each project
                       customizes generation without changing this file.
        temperature: LLM creativity (0 = deterministic, 1 = creative).

    Returns:
        The LLM's final answer string.
    """
    llm = get_llm()

    default_system = """You are a helpful AI assistant. 
Answer the user's question ONLY using the information provided in the context below.
If the context does not contain enough information to answer, say so clearly.
Do not hallucinate or make up information not present in the context.
Be specific, concise, and structured in your response."""

    system = system_prompt or default_system

    full_prompt = f"""{system}

--- CONTEXT (retrieved from knowledge base) ---
{context if context.strip() else "No relevant context was found in the knowledge base."}
--- END CONTEXT ---

User Question: {query}

Answer:"""

    try:
        response = llm.invoke(full_prompt)
        answer = response.content.strip() if hasattr(response, "content") else str(response).strip()

        if not answer:
            return "I could not generate a response. Please try rephrasing your question."

        return answer

    except Exception as e:
        print(f"[generate] LLM call failed: {e}")
        return f"An error occurred while generating the response. Please try again. (Error: {type(e).__name__})"