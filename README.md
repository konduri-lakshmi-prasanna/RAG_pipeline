# RAG-CORE — Shared RAG Pipeline

## What is this repo?
This is a shared library used by two projects:
- **Repo 1** — careerbot (my friend's project)
- **Repo 2** — resumeanalyser (my project)

Instead of both projects having their own separate RAG pipeline,
we extracted the common logic into this one shared repo.
Both projects import from here.

---

## Why did we create this?
Our sir asked us to:
- Remove duplicate code from both projects
- Follow SOLID principles (especially Open/Closed principle)
- Follow GoF Design Patterns
- Make the system loosely coupled and highly cohesive
- Have one shared repo that both projects depend on

---

## Folder Structure and Purpose of Each File

rag_core/
interface.py          → The contract/rulebook (IRagPipeline abstract class)
default_pipeline.py   → The actual implementation of all 6 stages
stages/
rewrite.py          → Stage 1: Improves the user's question before searching
retrieval.py        → Stage 2: Searches ChromaDB for relevant documents
rerank.py           → Stage 3: Sorts retrieved docs by relevance
refine.py           → Stage 4: Removes useless content, keeps best context
generate.py         → Stage 5: Sends context + question to LLM, gets answer
insert.py           → Stage 6: Saves new documents into the database
db/
chromadb_store.py   → Shared ChromaDB connection used by all stages
llm/
factory.py          → Creates the LLM object (Groq, OpenAI etc.)
requirements.txt        → All Python packages needed

---

## The 6 RAG Pipeline Stages

| Stage | File | What it does |
|-------|------|--------------|
| 1 | rewrite.py   | Cleans and improves the user's query |
| 2 | retrieval.py | Searches vector database for relevant chunks |
| 3 | rerank.py    | Sorts chunks, most relevant first |
| 4 | refine.py    | Removes low quality chunks |
| 5 | generate.py  | Calls the LLM and gets the final answer |
| 6 | insert.py    | Stores new documents into ChromaDB |

---

## Design Patterns Used (GoF)

| Pattern | Where | Why |
|---------|-------|-----|
| Template Method | interface.py → run() | Fixes the order of all 6 stages |
| Strategy | retrieval.py | Swap retriever without changing pipeline |
| Factory | llm/factory.py | Create LLM without exposing logic |
| Singleton | db/chromadb_store.py | One shared DB connection |
| Facade | default_pipeline.py | Hides all 6 stages behind one run() call |
| Observer | stages/ | Log events after each stage completes |
| Decorator | stages/ | Add caching on top of any stage |

---

## SOLID Principles Applied

| Principle | How we applied it |
|-----------|-------------------|
| S — Single Responsibility | Each stage file does exactly one job |
| O — Open/Closed | IRagPipeline never changes, we only extend it |
| L — Liskov Substitution | Any pipeline implementation can replace another |
| I — Interface Segregation | Interface has only what is needed, nothing extra |
| D — Dependency Inversion | Both repos depend on IRagPipeline, not the concrete class |

---

## How careerbot uses this

```python
from rag_core.default_pipeline import DefaultRagPipeline

pipeline = DefaultRagPipeline(collection_name="careerbot")
answer = pipeline.run("What jobs suit my resume?")
print(answer)
```

## How resumeanalyser uses this

```python
from rag_core.default_pipeline import DefaultRagPipeline

pipeline = DefaultRagPipeline(collection_name="resumeanalyser")
answer = pipeline.run("Analyse my resume for Google")
print(answer)
```

---

## Key Rule
If something changes **inside** rag-core (example: we improve the reranker),
neither careerbot nor resumeanalyser needs to change their code.
They only depend on the **interface**, not the implementation.
This is the **Open/Closed Principle**.

---

## Team
- Person 1 (your name) — resumeanalyser + rag-core contributor
- Person 2 (friend's name) — careerbot + rag-core contributor