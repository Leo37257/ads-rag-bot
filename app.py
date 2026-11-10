import base64, os
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="MS-ADS Assistant", page_icon="🎓", layout="centered")
# Copy deployment secrets into env so rag.py can read them (Streamlit Cloud / HF Spaces both work)
try:
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "LLM_MODEL"):
        if k in st.secrets and not os.getenv(k): os.environ[k] = st.secrets[k]
except Exception:
    pass
from rag import Bot

MAROON = "#800000"

@st.cache_resource(show_spinner="Loading knowledge base...")
def get_bot(): return Bot()

def logo_tag():
    p = Path(__file__).parent / "assets" / "dsi_logo.svg"
    try:
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f'<img src="data:image/svg+xml;base64,{b64}" alt="UChicago Data Science Institute" style="height:42px">'
    except Exception:
        return ""

st.markdown(f"""
<style>
  .block-container {{ padding-top: 3.5rem; max-width: 820px; }}
  #MainMenu, footer {{ visibility: hidden; }}
  .hero {{ background: linear-gradient(135deg, {MAROON} 0%, #5c0000 100%); border-radius: 16px; padding: 22px 26px; color: #fff; margin-bottom: 14px; }}
  .hero .logo {{ background: #fff; display: inline-block; padding: 8px 14px; border-radius: 10px; margin-bottom: 14px; }}
  .hero h1 {{ color: #fff; font-size: 1.7rem; margin: 0 0 4px 0; padding: 0; line-height: 1.25; }}
  .hero p {{ color: #f3dede; margin: 0; font-size: 0.98rem; }}
  [data-testid="stSidebar"] .stButton button {{ border: 1px solid #e6d9d9; border-radius: 10px; text-align: left; font-size: 0.9rem; }}
  [data-testid="stSidebar"] .stButton button:hover {{ border-color: {MAROON}; color: {MAROON}; }}
  [data-testid="stChatMessage"] {{ border-radius: 14px; }}
  .fine {{ color:#777; font-size:0.78rem; text-align:center; margin-top:10px; }}
</style>
<div class="hero">
  <div class="logo">{logo_tag()}</div>
  <h1>MS in Applied Data Science Assistant</h1>
  <p>Ask about admissions, tuition, curriculum, deadlines and more. Answers come from the official program website, with sources.</p>
</div>
""", unsafe_allow_html=True)

SAMPLES = ["What is tuition cost for the program?", "What scholarships are available?",
           "What are the TOEFL and IELTS minimum scores?", "What are the deadlines for the in-person program?",
           "Where can I mail my official transcripts?", "Is the program STEM/OPT eligible?",
           "How many courses do I need to graduate?", "What are the core courses?",
           "Can I set up an advising appointment?", "How do I apply to the MBA/MS program?"]

with st.sidebar:
    st.subheader("Try a question")
    for q in SAMPLES:
        if st.button(q, use_container_width=True): st.session_state["pending"] = q
    st.divider()
    if st.button("Clear conversation", use_container_width=True): st.session_state.messages = []; st.rerun()
    st.caption("Hybrid RAG: BGE embeddings + FAISS + BM25, cross-encoder reranking, grounded LLM answers with citations. "
               "Do not enter personal data; SSNs, emails and phone numbers are redacted. "
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

if not st.session_state.messages:
    st.info("Try a question from the sidebar, or type your own below. For example: *What are the deadlines for the in-person program?*")

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

st.markdown('<div class="fine">Unofficial student course project. Not affiliated with or endorsed by the University of Chicago. '
            'Logo shown for identification of the source program.</div>', unsafe_allow_html=True)
