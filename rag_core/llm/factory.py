from langchain_groq import ChatGroq
import os

# Singleton pattern — only one LLM instance created and reused
_llm_instance = None


def get_llm(model_name: str = "llama3-8b-8192"):
    """
    Factory + Singleton Pattern.
    Creates LLM only once and reuses it.
    
    If tomorrow you want to switch from Groq to OpenAI,
    you only change this file — not careerbot or resumeanalyser.
    That is the Open/Closed Principle.
    """
    global _llm_instance
    if _llm_instance is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. "
                "Please add it to your .env file."
            )
        _llm_instance = ChatGroq(
            model=model_name,
            groq_api_key=api_key,
            temperature=0.3
        )
    return _llm_instance