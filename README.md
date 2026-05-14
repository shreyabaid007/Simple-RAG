# IPCC AR6 RAG

A small, local Retrieval-Augmented Generation system over the IPCC Sixth Assessment
Report (AR6). Ask climate-science questions and get answers grounded *only* in the
AR6 text, with section-level citations.

This is a learning project — the goal is to demonstrate solid RAG fundamentals
end-to-end on commodity hardware (Apple Silicon), not to ship a product.

## Architecture

```
PDFs ──► load + section-tag ──► chunk ──► embed (Ollama) ──► ChromaDB
                                                                │
                          question ──► vector search (top-K) ◄──┘
                                            │
                                            ▼
                              cross-encoder rerank (top-k)
                                            │
                                            ▼
                                grounded LLM answer + citations
```

Two-stage retrieval — a fast vector search pulls a wide candidate set, then a
cross-encoder reranker re-scores each candidate against the question to keep only
the most relevant passages. This is more accurate than vector search alone (which
optimizes for semantic similarity, not answer relevance) without paying the
cross-encoder cost on the whole corpus.

## Stack

- **LangChain** — document loaders, splitter, retrieval glue
- **Ollama** — local inference for both embeddings and generation
- **ChromaDB** — persistent on-disk vector store
- **Hugging Face cross-encoder** (`bge-reranker-base`) — reranker, runs on CPU
- **Streamlit** — UI

## Setup

### 1. Install Ollama

Download from [ollama.com](https://ollama.com) and install. On Apple Silicon it
runs natively via Metal.

### 2. Pull the models

```bash
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

The generation model is configurable in [`config.py`](config.py) via `GEN_MODEL`.
The default (`llama3.1:8b`) is a safe pick for any M-series Mac with ≥16 GB of
unified memory. To upgrade, check `ollama list` and the [Ollama
library](https://ollama.com/library) and swap in the strongest instruct model your
machine can comfortably run — Qwen3, Llama 3.x, or similar 8B-class (or larger)
instruct models all work as drop-in replacements.

### 3. Add the IPCC PDFs

Drop these three files into `data/`:

- `IPCC_AR6_SYR_LongerReport.pdf`
- `IPCC_AR6_SYR_SPM.pdf`
- `IPCC_AR6_WGI_SPM.pdf`

(All available from [ipcc.ch](https://www.ipcc.ch/assessment-report/ar6/).)

### 4. Install Python dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 5. Ingest the PDFs

```bash
python ingest.py
```

This loads each PDF, tags chunks with `(report, section, page)` metadata, embeds
them with `nomic-embed-text`, and persists everything to `chroma_db/`. Run once;
re-run only if you change chunking parameters or swap embedding models.

### 6. Run the app

```bash
streamlit run app.py
```

Or run the smoke test:

```bash
python eval.py
```

## Files

| File | Purpose |
|---|---|
| [`config.py`](config.py) | Single source of truth for every tunable (paths, model names, chunk size, retrieval counts) |
| [`ingest.py`](ingest.py) | PDF → tagged chunks → ChromaDB |
| [`rag.py`](rag.py) | Two-stage retrieval + grounded answer chain |
| [`app.py`](app.py) | Streamlit UI |
| [`eval.py`](eval.py) | ~5-question smoke test (answerable + unanswerable) |

## Design decisions

### Structure-aware chunking

IPCC reports are heavily structured — content is organized under labeled sections
like `A.1 Observed Warming and Its Causes` or `SPM.B.2`. A naive
fixed-window chunker throws this structure away. Instead, `ingest.py` walks each
page in document order, detects section headings via a regex over the
common AR6 numbering patterns, and carries the "current section" forward across
pages. Every chunk emitted into ChromaDB is tagged with the section it lives
under and the page it came from. This metadata is what makes citations useful:
users (and you, while debugging) can trace any answer back to a specific section
in a specific report.

Within each section, text is split with a standard recursive character splitter
sized to fit comfortably inside the embedding model's context.

### Two-stage retrieval

Dense vector search is great at "find me passages that look semantically similar
to this query" but only weakly correlated with "find me passages that *answer*
this query". Cross-encoders score (query, passage) jointly and are much better
at relevance — but they're too slow to run over the whole corpus. The pipeline
combines them: vector search produces a wide candidate set (`INITIAL_K = 20`),
the cross-encoder reranks those, and only the top `FINAL_K = 5` reach the LLM.
This noticeably improves answer quality on questions where the most
*lexically* similar passage isn't the most *relevant* one.

### Grounded answering

The prompt in [`rag.py`](rag.py) gives the model only the retrieved passages and
forbids outside knowledge. When the context is insufficient, the model is
required to reply with the exact refusal string defined in `config.py`. This
matters because IPCC RAG is a high-stakes domain — confabulated "climate facts"
are worse than no answer.
