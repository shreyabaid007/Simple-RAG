"""Two-stage retrieval (vector search → cross-encoder rerank) and grounded answer chain."""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from sentence_transformers import CrossEncoder

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    FINAL_K,
    GEN_MODEL,
    GEN_TEMPERATURE,
    INITIAL_K,
    REFUSAL_STRING,
    RERANKER_MODEL,
)

PROMPT_TEMPLATE = """You are an assistant that answers questions strictly from IPCC Assessment Report excerpts.

Rules:
- Answer ONLY using information present in the CONTEXT below.
- Do NOT use outside knowledge, even if you are confident it is correct.
- If the CONTEXT does not contain enough information to answer the question, reply with this EXACT sentence and nothing else:
{refusal}
- Be concise. Quote or paraphrase the context faithfully.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


class RAGPipeline:
    def __init__(self) -> None:
        self.embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        self.store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(CHROMA_DIR),
        )
        self.reranker = CrossEncoder(RERANKER_MODEL)
        self.llm = OllamaLLM(model=GEN_MODEL, temperature=GEN_TEMPERATURE)
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def retrieve(self, question: str) -> list[Document]:
        candidates = self.store.similarity_search(question, k=INITIAL_K)
        if not candidates:
            return []
        pairs = [(question, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:FINAL_K]]

    @staticmethod
    def _format_context(docs: list[Document]) -> str:
        blocks = []
        for i, doc in enumerate(docs, 1):
            m = doc.metadata
            header = (
                f"[{i}] {m.get('report', '?')} — {m.get('section', '?')} "
                f"(p. {m.get('page', '?')})"
            )
            blocks.append(f"{header}\n{doc.page_content}")
        return "\n\n".join(blocks)

    def answer(self, question: str) -> tuple[str, list[Document]]:
        docs = self.retrieve(question)
        if not docs:
            return REFUSAL_STRING, []
        prompt_text = self.prompt.format(
            refusal=REFUSAL_STRING,
            context=self._format_context(docs),
            question=question,
        )
        response = self.llm.invoke(prompt_text)
        return response.strip(), docs
