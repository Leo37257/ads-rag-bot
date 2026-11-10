"""Retrieval ablation: dense only, BM25 only, hybrid (RRF), hybrid + cross-encoder rerank.
Uses the same 15 questions and gold keys as eval.py. No LLM or API key needed."""
import json, numpy as np
from rag import Retriever, QUERY_PREFIX, tok
from eval import GOLD, EXTRA, has

ret = Retriever()
QS = GOLD + EXTRA
K = 6

def dense(q):
    v = np.array(list(ret.emb.embed([QUERY_PREFIX + q])), dtype="float32"); v /= np.linalg.norm(v, axis=1, keepdims=True)
    _, ids = ret.index.search(v, K); return [ret.chunks[i] for i in ids[0]]

def bm25(q):
    return [ret.chunks[i] for i in np.argsort(-ret.bm25.get_scores(tok(q)))[:K]]

SYSTEMS = {
    "Dense only (BGE + FAISS)": dense,
    "BM25 only": bm25,
    "Hybrid (RRF)": lambda q: ret.search(q, k=K, rerank=False),
    "Hybrid + rerank (ours)": lambda q: ret.search(q, k=K, rerank=True),
}

def score(fn):
    h1 = h3 = h6 = rr_sum = 0
    for q, gold in QS:
        hits = fn(q)
        def ok(k): blob = " ".join(h["text"] for h in hits[:k]); return all(has(blob, g) for g in gold)
        h1 += ok(1); h3 += ok(3); h6 += ok(6)
        rr = next((i + 1 for i, h in enumerate(hits) if any(has(h["text"], g) for g in gold)), 0)
        rr_sum += 1 / rr if rr else 0
    n = len(QS)
    return {"hit@1": h1 / n, "hit@3": h3 / n, "hit@6": h6 / n, "mrr": rr_sum / n}

if __name__ == "__main__":
    res = {name: score(fn) for name, fn in SYSTEMS.items()}
    print(f"{'System':28s} hit@1  hit@3  hit@6   MRR")
    for n, m in res.items():
        print(f"{n:28s} {m['hit@1']:.2f}   {m['hit@3']:.2f}   {m['hit@6']:.2f}   {m['mrr']:.2f}")
    json.dump(res, open("data/ablation_results.json", "w"), indent=1)
