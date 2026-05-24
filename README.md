# Climate Intelligence RAG

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain.com)
[![Ollama](https://img.shields.io/badge/Ollama-local_inference-000?style=flat-square)](https://ollama.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector_store-orange?style=flat-square)](https://www.trychroma.com/)

A local Retrieval-Augmented Generation system over the **IPCC Sixth Assessment Report (AR6)**. Ask climate-science questions and get answers grounded *only* in the AR6 text, with section-level citations.

Runs entirely offline — no API keys, no cloud calls, no data leaves your machine.

---

## Why two-stage retrieval?

Vector search alone optimizes for semantic similarity, not answer relevance — a passage about "carbon pricing mechanisms" might be semantically close to a question about "carbon taxes" without actually answering it. This system uses a **two-stage approach**:

1. **Fast vector search** (ChromaDB + nomic-embed-text) pulls a wide top-K candidate set
2. **Cross-encoder reranking** (bge-reranker-base) re-scores each candidate against the question, keeping only the most relevant passages

The cross-encoder sees the full question-passage pair and is far more accurate than cosine similarity alone, but too expensive to run on the whole corpus. The vector search stage narrows the field cheaply; the reranker provides precision.

## Architecture

![Architecture](./architecture.png)

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

## Stack

| Component | Technology |
|---|---|
| Orchestration | LangChain — document loaders, splitter, retrieval |
| Embeddings | Ollama (`nomic-embed-text`) — runs locally via Metal on Apple Silicon |
| Generation | Ollama (`llama3.1:8b`) — local inference |
| Vector store | ChromaDB — persistent on-disk |
| Reranker | Hugging Face cross-encoder (`bge-reranker-base`) — CPU inference |
| UI | Streamlit |

## Setup

### 1. Install Ollama

Download from [ollama.com](https://ollama.com) and install. On Apple Silicon it runs natively via Metal.

### 2. Pull the models

```bash
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

### 3. Install Python dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Ingest the PDFs

```bash
python ingest.py
```

This loads each PDF, tags chunks with `(report, section, page)` metadata, embeds them with `nomic-embed-text`, and persists everything to `chroma_db/`. Run once; re-run only if you change chunking parameters or swap embedding models.

### 5. Run the app

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
| `config.py` | Single source of truth for every tunable (paths, model names, chunk size, retrieval counts) |
| `ingest.py` | PDF → section-tagged chunks → ChromaDB |
| `rag.py` | Two-stage retrieval + grounded answer chain |
| `app.py` | Streamlit UI |
| `eval.py` | ~5-question smoke test (answerable + unanswerable) |

## License

MIT
