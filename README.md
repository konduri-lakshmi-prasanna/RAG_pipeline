---

## Purpose of Each File

### `rag_core/interface.py`
The most important file in this repo. Defines `IRagPipeline` — an abstract
class that acts as a contract. It declares all 6 pipeline stages as abstract
methods. Neither careerbot nor resumeanalyser can skip any stage.
Also contains the `run()` method which executes all 6 stages in fixed order
— this is the **Template Method** design pattern.

### `rag_core/default_pipeline.py`
The concrete implementation of `IRagPipeline`. This is what careerbot and
resumeanalyser actually import and use. It calls each stage file in order.
Acts as a **Facade** — hides all internal complexity behind one simple
`pipeline.run("your question")` call.

### `rag_core/stages/rewrite.py` — Stage 1
Takes the user's raw question and improves it before searching.
Example: "jobs for me" becomes "software engineering jobs for
a student with Python and ML skills".

### `rag_core/stages/retrieval.py` — Stage 2
Searches ChromaDB vector database for the most relevant document
chunks based on the rewritten query. Returns a list of matching text chunks.

### `rag_core/stages/rerank.py` — Stage 3
Takes the retrieved chunks and sorts them by how relevant they are
to the query. Most relevant chunks come first. Uses the **Strategy**
pattern — the ranking algorithm can be swapped without changing the pipeline.

### `rag_core/stages/refine.py` — Stage 4
Removes low quality or duplicate chunks from the list.
Keeps only the best context to send to the LLM.
This improves answer quality and reduces token usage.

### `rag_core/stages/generate.py` — Stage 5
Takes the refined context and the user's question, sends them to
the LLM (Groq/Llama3), and returns the final answer.

### `rag_core/stages/insert.py` — Stage 6
When new documents are uploaded (resumes, interview experiences etc.),
this stage chunks them, creates embeddings, and stores them in ChromaDB
for future retrieval.

### `rag_core/db/chromadb_store.py`
Single shared connection to ChromaDB. Uses the **Singleton** pattern —
only one instance is created and reused across all stages.
Both careerbot and resumeanalyser use this same database layer.

### `rag_core/llm/factory.py`
Creates LLM instances using the **Factory** pattern. If we want to
switch from Groq to OpenAI, we change it only here — not in both projects.

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
| Template Method | `interface.py` → `run()` | Fixes the order of all 6 stages |
| Strategy | `retrieval.py`, `rerank.py` | Swap algorithms without changing pipeline |
| Factory | `llm/factory.py` | Create LLM instances without exposing logic |
| Singleton | `db/chromadb_store.py` | One shared DB connection across all stages |
| Facade | `default_pipeline.py` | Hides all 6 stages behind one `run()` call |
| Observer | `stages/` | Log events after each stage completes |
| Decorator | `stages/` | Add caching on top of any stage |

---

## SOLID Principles Applied

| Principle | How we applied it |
|-----------|-------------------|
| S — Single Responsibility | Each stage file does exactly one job |
| O — Open/Closed | `IRagPipeline` never changes, we only extend it |
| L — Liskov Substitution | Any pipeline implementation can replace another |
| I — Interface Segregation | Interface has only what is needed, nothing extra |
| D — Dependency Inversion | Both repos depend on `IRagPipeline`, not the concrete class |

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
If something changes inside rag-core (example: we improve the reranker),
neither careerbot nor resumeanalyser needs to change their code.
They only depend on the interface, not the implementation.
This is the **Open/Closed Principle**.

---

## Team
- Kommireddy Mounika Iswarya — resumeanalyser + rag-core contributor
- Konduri Lakshmi Prasanna — careerbot + rag-core contributor