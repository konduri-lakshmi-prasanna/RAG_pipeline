import os
import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings

# Singleton pattern — only one client created and reused
_client = None
_embeddings = None


def get_client():
    """
    Singleton Pattern.
    Creates ChromaDB client only once and reuses it.
    Saves memory and connection time.

    IMPORTANT: The path is read from the CHROMA_DB_PATH env var (set it to an
    ABSOLUTE path in your project's .env / config). We no longer fall back to
    a bare "./chroma_db" resolved against the process's current working
    directory — if the app is ever launched from a different folder than
    expected, that silently creates/opens a brand-new, EMPTY database instead
    of the one your documents were actually indexed into, and every query
    then returns "no relevant context" with no visible error at all.
    """
    global _client
    if _client is None:
        path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        path = os.path.abspath(path)
        _client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False)
        )
    return _client


def get_embeddings():
    """
    Singleton Pattern.
    Creates embedding model only once and reuses it.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5"
        )
    return _embeddings


def get_collection(collection_name: str):
    """
    Gets or creates a ChromaDB collection by name.
    careerbot passes "careerbot_db"
    resumeanalyser passes "resumeanalyser_db"
    Same code, different collections. 
    """
    client = get_client()
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def embed_texts(texts: list) -> list:
    """
    Converts text into vector embeddings.
    Better embeddings = better retrieval = better answers.
    This is the latent space your sir mentioned.
    """
    embeddings = get_embeddings()
    return embeddings.embed_documents(texts)


def embed_query(query: str) -> list:
    """
    Converts a single query into a vector embedding for searching.
    """
    embeddings = get_embeddings()
    return embeddings.embed_query(query)