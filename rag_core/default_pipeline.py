"""
default_pipeline.py — Concrete implementation of the RAG pipeline.

GoF Pattern: Facade
  Hides all 6 stages behind one simple pipeline.run("question") call.
  Projects do not need to know about rewrite, rerank, refine etc.

GoF Pattern: Template Method (inherited from IRagPipeline)
  The run() order is fixed in the interface.
  This class provides the actual implementations.

How careerbot uses it:
    from rag_core.default_pipeline import DefaultRagPipeline
    pipeline = DefaultRagPipeline(collection_name="careerbot_db")
    answer = pipeline.run("What jobs suit my resume?")

How resumeanalyser uses it:
    pipeline = DefaultRagPipeline(collection_name="resumeanalyser_db")
    answer = pipeline.run("Analyse my resume for Google SDE role")

SOLID — Open/Closed:
  Projects extend this class and override only what they need.
  This class itself never changes when a project adds custom behavior.
"""

from rag_core.interface import IRagPipeline
from rag_core.stages.rewrite import rewrite_query
from rag_core.stages.retrieval import retrieve_chunks
from rag_core.stages.rerank import rerank_chunks
from rag_core.stages.refine import refine_chunks
from rag_core.stages.generate import generate_answer
from rag_core.stages.insert import insert_document


class DefaultRagPipeline(IRagPipeline):
    """
    Default implementation of the 6-stage RAG pipeline.
    Projects extend this and pass their collection name.
    """

    def __init__(
        self,
        collection_name: str,
        top_k: int = 5,
        rerank_strategy: str = "rrf",
        context_hint: str = "",
        system_prompt: str = None,
    ):
        """
        Args:
            collection_name: ChromaDB collection this pipeline uses.
                             careerbot = "careerbot_db"
                             resumeanalyser = "resumeanalyser_db"
            top_k: Max chunks to send to the LLM.
            rerank_strategy: "rrf" (recommended) or "distance".
            context_hint: Domain hint for query rewriting.
            system_prompt: Custom LLM system instructions.
                           Each project passes its own — this is the
                           abstraction boundary your mentor described.
        """
        self.collection_name = collection_name
        self.top_k = top_k
        self.rerank_strategy = rerank_strategy
        self.context_hint = context_hint
        self.system_prompt = system_prompt

    # ── Stage Implementations ──────────────────────────────────────

    def rewrite(self, query: str) -> str:
        """Stage 1: Improve the query before searching."""
        return rewrite_query(query, context_hint=self.context_hint)

    def retrieve(self, query: str) -> list:
        """Stage 2: Search ChromaDB for relevant chunks."""
        return retrieve_chunks(query, self.collection_name, k=self.top_k * 3)

    def rerank(self, query: str, chunks: list) -> list:
        """Stage 3: Sort chunks by relevance using RRF."""
        return rerank_chunks(query, chunks, strategy=self.rerank_strategy)

    def refine(self, chunks: list) -> list:
        """Stage 4: Remove duplicates and low-relevance chunks."""
        return refine_chunks(chunks, top_k=self.top_k)

    def generate(self, query: str, context: str) -> str:
        """Stage 5: Call LLM and get final answer."""
        return generate_answer(query, context, system_prompt=self.system_prompt)

    def insert(self, text: str, metadata: dict = None) -> dict:
        """Stage 6: Chunk, embed, and store documents."""
        return insert_document(text, self.collection_name, metadata=metadata)