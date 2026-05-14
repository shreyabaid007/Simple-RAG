"""Minimal Streamlit UI for the IPCC AR6 RAG system.

Run with:
    streamlit run app.py
"""

import streamlit as st

from rag import RAGPipeline


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


st.set_page_config(page_title="IPCC AR6 RAG", layout="wide")
st.title("Ask the IPCC AR6 reports")
st.caption(
    "Answers are grounded only in the AR6 Synthesis Report (Longer Report + SPM) "
    "and the WG1 Summary for Policymakers."
)

question = st.text_input(
    "Question",
    placeholder="e.g. What has caused observed warming since 1850?",
)

if question:
    pipeline = get_pipeline()
    with st.spinner("Retrieving and answering..."):
        answer, sources = pipeline.answer(question)

    st.markdown("### Answer")
    st.write(answer)

    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for i, doc in enumerate(sources, 1):
            m = doc.metadata
            st.markdown(
                f"**[{i}] {m.get('report', '?')}** — *{m.get('section', '?')}* "
                f"· page {m.get('page', '?')}"
            )
            st.write(doc.page_content)
            st.markdown("---")
