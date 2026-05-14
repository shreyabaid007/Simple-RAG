"""Single source of truth for every tunable value in the RAG pipeline."""

from pathlib import Path

# --- Paths ---------------------------------------------------------------

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / "chroma_db"

# IPCC AR6 PDFs the user is expected to drop into data/.
# Filename → human-readable report name used in citations.
PDF_FILES: dict[str, str] = {
    "IPCC_AR6_SYR_LongerReport.pdf": "AR6 Synthesis Report — Longer Report",
    "IPCC_AR6_SYR_SPM.pdf": "AR6 Synthesis Report — Summary for Policymakers",
    "IPCC_AR6_WGI_SPM.pdf": "AR6 WG1 — Summary for Policymakers",
}

# --- Models --------------------------------------------------------------

# Ollama embedding model. Pull with: `ollama pull nomic-embed-text`.
EMBED_MODEL = "nomic-embed-text"

# Ollama generation model. Pick the strongest instruct model your Mac's
# unified memory can run comfortably. Check `ollama list` and the Ollama
# library; swap this string to upgrade. 8B-class is the safe default on
# Apple Silicon.
GEN_MODEL = "llama3.1:8b"

# HF cross-encoder for reranking. CPU is fine at this scale.
RERANKER_MODEL = "BAAI/bge-reranker-base"

# --- Chunking ------------------------------------------------------------

# Character-based, applied after structure-aware section tagging.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Retrieval -----------------------------------------------------------

# Pull this many candidates from the vector store...
INITIAL_K = 20
# ...then keep this many after cross-encoder reranking.
FINAL_K = 5

# --- Vector store --------------------------------------------------------

COLLECTION_NAME = "ipcc_ar6"

# --- Generation ----------------------------------------------------------

# Deterministic answers for a grounded QA system.
GEN_TEMPERATURE = 0.0

# Exact string the model must return when retrieved context is insufficient.
REFUSAL_STRING = "I cannot answer this from the provided IPCC reports."
