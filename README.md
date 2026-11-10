# Ask-ADS: RAG assistant for UChicago MS in Applied Data Science

Pipeline: `scrape.py` -> `rag.py index` (BGE-small + FAISS + BM25) -> `app.py` (Streamlit) ; `eval.py [--llm]` scores retrieval and answers.

## Run locally
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY (+ optional OPENAI_BASE_URL, LLM_MODEL)
    streamlit run app.py

## Deploy (public URL) - Streamlit Community Cloud
1. Push this folder (including `data/`) to a GitHub repo.
2. share.streamlit.io -> New app -> select repo, main file `app.py`.
3. App settings -> Secrets: `ANTHROPIC_API_KEY = "..."` (use a key with a spend cap).
Hugging Face Spaces (Streamlit SDK) works the same way, with the key as a Space secret.

Refresh data: `python scrape.py && python rag.py index`, then commit `data/`.
