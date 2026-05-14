"""Load IPCC PDFs → tag with section/page metadata → chunk → embed → persist to ChromaDB.

Run once after dropping the AR6 PDFs into data/:
    python ingest.py
"""

import re

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBED_MODEL,
    PDF_FILES,
)

# Matches IPCC AR6 section headings:
#   A.   A.1   A.1.2   SPM.A   SPM.A.1   Section 3.2
# Requires an uppercase title word after the number to avoid false positives
# from sentences that happen to start with a section reference.
SECTION_RE = re.compile(
    r"^\s*("
    r"SPM\.[A-Z](?:\.\d+)?"
    r"|[A-Z]\.\d+(?:\.\d+)?"
    r"|[A-Z]\."
    r"|Section\s+\d+(?:\.\d+)?"
    r")\s+[A-Z][A-Za-z]"
)


def is_section_heading(line: str) -> bool:
    stripped = line.strip()
    # Headings are short. Long lines that match the pattern are paragraphs
    # citing a section, not headings themselves.
    if not (4 <= len(stripped) <= 200):
        return False
    return bool(SECTION_RE.match(stripped))


def split_page_by_sections(
    page_text: str, carried_section: str
) -> list[tuple[str, str]]:
    """Split a page at section-heading boundaries.

    Returns (section, text) pairs. Text appearing before the page's first
    heading inherits the section carried over from the previous page.
    """
    segments: list[tuple[str, str]] = []
    current = carried_section
    buf: list[str] = []
    for line in page_text.split("\n"):
        if is_section_heading(line):
            if buf:
                segments.append((current, "\n".join(buf)))
                buf = []
            current = line.strip()
        else:
            buf.append(line)
    if buf:
        segments.append((current, "\n".join(buf)))
    return segments


def load_documents() -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_chunks: list[Document] = []

    for filename, report_name in PDF_FILES.items():
        pdf_path = DATA_DIR / filename
        if not pdf_path.exists():
            raise FileNotFoundError(
                f"Missing IPCC PDF at {pdf_path}. "
                "Drop the 3 AR6 PDFs into data/ before running ingest.py."
            )

        print(f"Loading {filename}...")
        pages = PyMuPDFLoader(str(pdf_path)).load()

        carried_section = "Front matter"
        for page in pages:
            # PyMuPDF page numbers are 0-indexed in metadata.
            page_num = page.metadata.get("page", 0) + 1
            segments = split_page_by_sections(page.page_content, carried_section)
            if segments:
                carried_section = segments[-1][0]
            for section, text in segments:
                if not text.strip():
                    continue
                for sub in splitter.split_text(text):
                    all_chunks.append(
                        Document(
                            page_content=sub,
                            metadata={
                                "report": report_name,
                                "section": section,
                                "page": page_num,
                                "source": filename,
                            },
                        )
                    )

    print(f"Produced {len(all_chunks)} chunks across {len(PDF_FILES)} reports.")
    return all_chunks


def build_vector_store(chunks: list[Document]) -> Chroma:
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    print(f"Embedding {len(chunks)} chunks with {EMBED_MODEL} via Ollama...")
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"Persisted vector store to {CHROMA_DIR}.")
    return store


if __name__ == "__main__":
    build_vector_store(load_documents())
