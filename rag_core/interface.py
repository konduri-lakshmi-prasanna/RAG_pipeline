from abc import ABC, abstractmethod
from typing import List


class IRagPipeline(ABC):
    """
    Interface (contract) for the RAG pipeline.
    All 6 stages must be implemented by any class that uses this.
    This is the Open/Closed Principle — this file never changes.
    """

    @abstractmethod
    def rewrite(self, query: str) -> str:
        """Stage 1: Improve the user's question"""
        pass

    @abstractmethod
    def retrieve(self, query: str) -> List[str]:
        """Stage 2: Search documents from vector DB"""
        pass

    @abstractmethod
    def rerank(self, query: str, docs: List[str]) -> List[str]:
        """Stage 3: Sort results by relevance"""
        pass

    @abstractmethod
    def refine(self, docs: List[str]) -> List[str]:
        """Stage 4: Remove useless parts, keep best context"""
        pass

    @abstractmethod
    def generate(self, query: str, context: str) -> str:
        """Stage 5: Send to LLM and get answer"""
        pass

    @abstractmethod
    def insert(self, documents: List[str]) -> None:
        """Stage 6: Save new documents into the database"""
        pass

    def run(self, query: str) -> str:
        """
        Template Method Pattern.
        Runs all 6 stages in fixed order.
        Never change this method.
        """
        clean_query = self.rewrite(query)
        docs        = self.retrieve(clean_query)
        ranked      = self.rerank(clean_query, docs)
        refined     = self.refine(ranked)
        context     = "\n\n".join(refined)
        return self.generate(clean_query, context)