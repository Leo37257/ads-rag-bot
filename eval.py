"""Evaluate retrieval (hit@k, MRR) and answer quality (gold-keyword recall) on the course's sample set."""
import json, sys, re
from rag import Bot, llm_complete

# each: question -> list of gold groups; a group is satisfied if ANY of its strings appears (case-insens.)
GOLD = [
 ("What is tuition cost for the program?", [["$6,830"]]),
 ("What scholarships are available for the program?", [["Data Science Institute Scholarship"], ["Alumni Scholarship"]]),
 ("What are the minimum scores for the TOEFL and IELTS English Language Requirement?", [["TOEFL"], ["IELTS, 7"]]),
 ("Is there an application fee waiver?", [["fee waiver", "FeeWaiver"]]),
 ("What are the deadlines for the in-person program?", [["Priority Application Deadline"], ["International Application Deadline"], ["Final Application Deadline"]]),
 ("How long will it take for me to receive a decision on my application?", [["1 to 2 months"], ["rolling basis"]]),
 ("Can I set up an advising appointment with the enrollment management team?", [["Schedule an Appointment"], ["apply-psd.uchicago.edu/portal"]]),
 ("Where can I mail my official transcripts?", [["455 N Cityfront Plaza"], ["Attention: MS in Applied Data Science Admissions"]]),
 ("Does the Master's in Applied Data Science Online program provide visa sponsorship?", [["In-Person"], ["visa"]]),
 ("How do I apply to the MBA/MS program?", [["Booth"]]),
 ("Is the MS in Applied Data Science program STEM/OPT eligible?", [["STEM"]]),
 ("How many courses must you complete to earn UChicago's Master's in Applied Data Science?", [["12"], ["6 core", "six core", "Core"]]),
]
EXTRA = [  # held-out style questions in the spirit of the project PDF
 ("What are the core courses in the MS in Applied Data Science program?", [["Machine Learning"], ["Statistical"]]),
 ("Can you provide information about the capstone project?", [["Capstone"]]),
 ("What are the admission requirements?", [["bachelor"]]),
]
def has(t, grp): return any(g.lower() in t.lower() for g in grp)

def run(use_llm=True, ks=(1,3,6)):
    bot = Bot(); rows = []
    for q, gold in GOLD + EXTRA:
        hits = bot.ret.search(q, k=max(ks))
        def ok(k):  # all gold groups covered by the union of top-k chunks
            blob = " ".join(h["text"] for h in hits[:k]); return all(has(blob, g) for g in gold)
        rr = next((i+1 for i,h in enumerate(hits) if any(has(h["text"], g) for g in gold)), 0)
        row = {"q": q, **{f"hit@{k}": ok(k) for k in ks}, "mrr": 1/rr if rr else 0}
        if use_llm:
            r = bot.ask(q)
            if r["provider"] != "extractive":
                row["answer"] = r["answer"]; row["ans_recall"] = sum(has(r["answer"], g) for g in gold)/len(gold)
        rows.append(row)
    n = len(rows)
    summ = {f"hit@{k}": sum(r[f"hit@{k}"] for r in rows)/n for k in ks}; summ["mrr"] = sum(r["mrr"] for r in rows)/n
    ar = [r["ans_recall"] for r in rows if "ans_recall" in r]
    if ar: summ["answer_keyword_recall"] = sum(ar)/len(ar)
    return rows, summ

if __name__ == "__main__":
    import os
    if "--llm" in sys.argv and not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        sys.exit("No OPENAI_API_KEY / ANTHROPIC_API_KEY found in this terminal. Run:\n  OPENAI_API_KEY=sk-... python3 eval.py --llm")
    rows, summ = run(use_llm="--llm" in sys.argv)
    for r in rows: print(("OK  " if r["hit@6"] else "MISS"), f"h1={int(r['hit@1'])} h3={int(r['hit@3'])}", r["q"])
    print(json.dumps(summ, indent=1)); json.dump({"rows": rows, "summary": summ}, open("data/eval_results.json", "w"), indent=1)
