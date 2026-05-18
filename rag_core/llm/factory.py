import os

_llm_instance = None

def get_llm(model_name: str = None):
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        _llm_instance = ChatGroq(
            model=model_name or "llama3-8b-8192",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3
        )
    elif os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        _llm_instance = ChatGoogleGenerativeAI(
            model=model_name or "gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3
        )
    else:
        raise ValueError(
            "No LLM API key found. Set GROQ_API_KEY or GOOGLE_API_KEY in your .env file."
        )

    return _llm_instance