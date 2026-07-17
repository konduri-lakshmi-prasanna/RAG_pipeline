import os

_llm_instance = None

def get_llm(model_name: str = None):
    global _llm_instance
    
    # 1. Early return if already initialized (from main)
    if _llm_instance is not None:
        return _llm_instance

    # 2. Try Groq first
    if os.getenv("GROQ_API_KEY"):
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError("langchain-groq is required to use the shared Groq LLM.")
            
        _llm_instance = ChatGroq(
            model=model_name or "llama-3.3-70b-versatile",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3
        )
        
    # 3. Fallback to Google Gemini (from main)
    elif os.getenv("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("langchain-google-genai is required to use the Google LLM.")
            
        _llm_instance = ChatGoogleGenerativeAI(
            model=model_name or "gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3
        )
        
    # 4. Error if no keys found
    else:
        raise ValueError(
            "No LLM API key found. Set GROQ_API_KEY or GOOGLE_API_KEY in your .env file."
        )

    return _llm_instance
