# Build Instructions: Local RAG System for IPCC Climate Reports

## Goal

Build a clean, minimal, well-documented local RAG application. This is a **learning
project** to demonstrate solid RAG fundamentals — prioritize clarity and correctness
over features. No over-engineering, no speculative abstractions.

**What it does:** Lets a user ask climate-science questions and get answers grounded
*only* in IPCC Assessment Report text, with section-level citations. This is a
structured retrieval system over scientific reports — **not** a generic "chat with PDF"
tool.

---

## Target Environment

- **Hardware:** Apple Silicon (M-series Mac). Ollama runs natively via Metal — pick
  model sizes that run comfortably on unified memory. An 8B-class generation model is
  the safe default.
- **Sources:** IPCC AR6 only. The user will place 3 PDFs in `data/`:
  - AR6 Synthesis Report — Longer Report
  - AR6 Synthesis Report — Summary for Policymakers
  - AR6 WG1 — Summary for Policymakers

---

## Stack (all must be used)

| Component   | Choice | Notes |
|-------------|--------|-------|
| Orchestration | **LangChain** | document loaders, structure-aware splitter, retrieval logic |
| Inference   | **Ollama** | embeddings + generation, runs locally |
| Embeddings  | `nomic-embed-text` | via Ollama |
| Generation  | configurable in `config.py` | default to a current Llama 3.x or Qwen3 instruct model, 8B class |
| Vector store | **ChromaDB** | persistent, on-disk |
| Reranker    | **Hugging Face** cross-encoder | `bge-reranker` class model; CPU is fine at this scale |

For the generation model: put the name in `config.py` as a variable and add a README
note telling the user to check the Ollama library / run `ollama list` and swap in the
best model their Mac's memory supports.

---

## Core Requirements

1. **Ingestion pipeline** (`ingest.py`)
   - Load IPCC report PDFs from `data/`.
   - Extract text while preserving document-structure metadata: report name,
     chapter/section heading, page number.
   - Chunk with a structure-aware splitter.
   - Store chunks + metadata in persistent ChromaDB.

2. **Retrieval** (`rag.py`)
   - Two-stage: vector similarity search → HF cross-encoder rerank → top-k passages.
   - Initial retrieval count, final `k`, chunk size, and overlap all configurable.

3. **Grounded answering**
   - Prompt strictly instructs the LLM to answer *only* from retrieved context.
   - When context is insufficient, respond exactly:
     `"I cannot answer this from the provided IPCC reports."`
   - Never use outside knowledge.

4. **Citations**
   - Every answer lists the source passages used: report name, section heading,
     page number.

5. **Interface** (`app.py`)
   - Minimal Streamlit app: one input box, the answer, and an expandable "Sources"
     section showing each retrieved chunk with its metadata.

6. Sanity-check script (eval.py): ~5 hardcoded questions (3 answerable, 2 unanswerable). Prints each question, the answer, and whether unanswerable ones correctly returned the refusal string. Not a formal eval framework — just a smoke test.

---

## Project Structure

```
.
├── ingest.py          # load PDFs → chunk → embed → store in ChromaDB
├── rag.py             # retrieval (vector search + rerank) + answer chain
├── app.py             # minimal Streamlit UI
├── config.py          # ALL tunables: model names, chunk size/overlap, retrieval counts, paths
├── requirements.txt
├── README.md
├── .gitignore         # must ignore data/ and chroma_db/
├── data/              # user drops IPCC PDFs here (gitignored)
└── chroma_db/         # persistent vector store (gitignored)
```

---

## Quality Bar

- Keep functions small and readable. Comment the non-obvious parts.
- Use **only** the libraries listed above.
- `config.py` is the single source of truth for every tunable value.
- `README.md` must include:
  - What the project is.
  - The architecture flow: ingestion → embed → store → retrieve → rerank → generate.
  - Setup steps: install Ollama, `ollama pull` the two models, add IPCC PDFs to
    `data/`, run `ingest.py`, run `app.py`.
  - A short **"Design decisions"** section explaining the structure-aware chunking
    and the two-stage retrieval choice.

---

## Explicitly Out of Scope

Do **not** add any of the following:

- Authentication
- Docker
- Conversation memory / multi-turn chat history
- Arxiv or any non-IPCC source
- Any database beyond ChromaDB
- Any agent framework

If a feature isn't in the Core Requirements above, don't add it. Keep it simple.

---

## Build Order (suggested)

1. `config.py` — define all constants first.
2. `ingest.py` — get PDFs loading with clean metadata. This is the most important
   and trickiest part (IPCC PDFs have multi-column layouts, figures, footnotes).
   Verify section headings are extracted cleanly before moving on.
3. `rag.py` — vector retrieval, then add the reranker, then the grounded answer chain.
4. `app.py` — wire the Streamlit UI to `rag.py`.
5. `README.md` — write last, once the pieces are real.