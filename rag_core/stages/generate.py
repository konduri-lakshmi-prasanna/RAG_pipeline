"""
generate.py — Stage 5: Answer Generation

Why this exists:
  Takes the refined context + original question and calls the LLM.
  This is where RAG produces its grounded answer.

  The key: the LLM is CONSTRAINED to the provided context.
  It should not invent information that is not present in the retrieved chunks.

Reliability of LLM:
  We explicitly instruct the model to use the retrieved context.
  We also provide a fallback if the LLM returns nothing or fails.

GoF Pattern: Template Method
"""

from rag_core.llm.factory import get_llm
from langchain_core.messages import SystemMessage, HumanMessage


def generate_answer(
    query: str,
    context: str,
    system_prompt: str = None,
    temperature: float = 0.3,
) -> str:
    """
    Calls the LLM with context and returns a grounded answer.

    Args:
        query: The original or rewritten user query.
        context: The refined, joined chunks from Stage 4.
        system_prompt: Optional domain-specific instructions.
        temperature: LLM creativity.

    Returns:
        The LLM's final answer string.
    """

    llm = get_llm()

    # Default instructions used when the project does not
    # provide its own system prompt.
    default_system = """You are a helpful AI assistant.

Answer the user's question using the information provided
in the retrieved context.

IMPORTANT RULES:
1. Use the retrieved context as the primary source of information.
2. If the context contains information about the user's education,
   skills, experience, projects, or achievements, use that information
   to personalize the answer.
3. Do not say that information is unavailable when it is actually
   present in the context.
4. Do not invent qualifications, experience, skills, companies,
   achievements, or other facts that are not present in the context.
5. You may reason about information present in the context.
6. If the context genuinely does not contain enough information,
   clearly say what information is missing.
7. Be specific, concise, and structured."""

    system = system_prompt or default_system

    # Make sure context is never None or empty.
    context_text = context.strip() if context else ""

    if not context_text:
        context_text = "No relevant context was found in the knowledge base."

    # Send system instructions separately from the retrieved context.
    messages = [
        SystemMessage(content=system),

        HumanMessage(
            content=f"""The following information was retrieved from
the user's uploaded documents.

--- BEGIN RETRIEVED CONTEXT ---

{context_text}

--- END RETRIEVED CONTEXT ---

User Question:
{query}

Answer the user's question using the retrieved context.

If the question is about the user personally, such as their career,
skills, education, experience, projects, or achievements, use the
corresponding information from the retrieved context.

Do not claim that the user's information is missing if it is present
in the retrieved context."""
        ),
    ]

    try:
        # Call the LLM.
        response = llm.invoke(messages)

        # Extract the response text.
        if hasattr(response, "content"):
            answer = response.content.strip()
        else:
            answer = str(response).strip()

        # Fallback if the LLM returns an empty response.
        if not answer:
            return (
                "I could not generate a response. "
                "Please try rephrasing your question."
            )

        return answer

    except Exception as e:
        print(f"[generate] LLM call failed: {e}")

        return (
            "An error occurred while generating the response. "
            f"Please try again. (Error: {type(e).__name__})"
        )