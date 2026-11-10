import os, re
import streamlit as st

st.set_page_config(page_title="MS-ADS Assistant", page_icon="🎓", layout="centered")
# Copy deployment secrets into env so rag.py can read them (Streamlit Cloud / HF Spaces both work)
try:
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "LLM_MODEL"):
        if k in st.secrets and not os.getenv(k): os.environ[k] = st.secrets[k]
except Exception:
    pass
from rag import Bot

@st.cache_resource(show_spinner="Loading knowledge base...")
def get_bot(): return Bot()

SAMPLES = ["What is tuition cost for the program?", "What scholarships are available?",
           "What are the TOEFL and IELTS minimum scores?", "What are the deadlines for the in-person program?",
           "Where can I mail my official transcripts?", "Is the program STEM/OPT eligible?",
           "How many courses do I need to graduate?", "What are the core courses?",
           "Can I set up an advising appointment?", "How do I apply to the MBA/MS program?"]

st.markdown("<h2 style='color:#800000;margin-bottom:0'>UChicago MS in Applied Data Science</h2>"
            "<div style='color:#555;margin-bottom:12px'>Ask about admissions, tuition, curriculum, deadlines, and more. "
            "Answers are generated from the official program website.</div>", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Try a question")
    for q in SAMPLES:
        if st.button(q, use_container_width=True): st.session_state["pending"] = q
    st.divider()
    if st.button("Clear conversation", use_container_width=True): st.session_state.messages = []; st.rerun()
    st.caption("Built with a hybrid RAG pipeline: BGE embeddings + FAISS + BM25, cross-encoder reranking, grounded LLM answers with citations. "
               "Do not enter personal data; details like SSNs, emails and phone numbers are redacted. "
               "Always confirm deadlines and costs on the official website.")

st.session_state.setdefault("messages", [])

def render_sources(srcs):
    seen = {}
    for s in srcs:
        seen.setdefault(s["url"], s)
    with st.expander(f"Sources ({len(seen)})"):
        for i, s in enumerate(srcs, 1):
            st.markdown(f"**[{i}] [{s['title']} › {s['heading'][:70]}]({s['url']})**")
            st.caption(s["text"].split("\n", 1)[-1][:280] + "...")

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🎓" if m["role"] == "assistant" else None):
        st.markdown(m["content"])
        if m.get("sources"): render_sources(m["sources"])

q = st.chat_input("Ask a question about the MS in Applied Data Science...") or st.session_state.pop("pending", None)
if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"): st.markdown(q)
    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Searching the program website..."):
            hist = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
            res = get_bot().ask(q, hist)
        st.markdown(res["answer"])
        if res["sources"]: render_sources(res["sources"])
    st.session_state.messages.append({"role": "assistant", "content": res["answer"], "sources": res["sources"]})
