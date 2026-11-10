"""Crawl the MS in Applied Data Science site and write structured pages to data/pages.jsonl.

Each record: {url, title, sections: [{heading, text}]}.  Headings are kept so the chunker
can prepend the section path to every chunk (big win for retrieval precision).
"""
import json, re, time, sys
from collections import deque
from urllib.parse import urljoin, urldefrag, urlparse
import requests
from bs4 import BeautifulSoup

SEEDS = [
    "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/",
    "https://datascience.uchicago.edu/education/masters-programs/",
    "https://datascience.uchicago.edu/how-to-apply/",
    "https://datascience.uchicago.edu/capstone-projects/",
]
# Extra pages the evaluation set points to (fee waiver, scheduling, Booth joint degree).
EXTRA = [
    "https://physicalsciences.uchicago.edu/academics/admissions/application-requirements/",
    "https://www.chicagobooth.edu/admissions/apply",
]
ALLOW = re.compile(r"^/(education/masters-programs|how-to-apply|capstone-projects|education/?$|about/contact|about/leadership-staff|dsi-communications)")
DENY = re.compile(r"(/news/|/events/|/insights/|/newsletter|/research/|/outreach/|/people/)")
MAX_PAGES = 80
HEAD = {"User-Agent": "Mozilla/5.0 (course-project RAG crawler)"}


def norm(u):
    u = urldefrag(u.strip())[0]
    return u.rstrip("/") + "/" if not re.search(r"\.\w{2,4}$", u) else u


def extract(html, url):
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "footer", "header", "form", "noscript", "svg", "iframe"]):
        t.decompose()
    main = soup.find("main") or soup.find("article") or soup.body
    title = (soup.title.string if soup.title else url).split("|")[0].strip()
    sections, cur, buf = [], title, []

    def flush():
        txt = re.sub(r"\n{3,}", "\n\n", "\n".join(buf)).strip()
        if len(txt) > 40:
            sections.append({"heading": cur, "text": txt})

    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "table", "button", "summary", "dt", "dd", "div", "a"]):
        name = el.name
        if name == "a":
            if "accordion-title" not in (el.get("class") or []):
                continue
            flush(); buf = []
            t = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).replace("Letter Grade", "").replace("Pass/Fail", "").strip()
            grp = el.find_parent(class_="course-group")
            qtr = el.find_parent(class_="quarter")
            tags = []
            if grp is not None and grp.find(class_="course-group__indicator"):
                tags.append(grp.find(class_="course-group__indicator").get_text(" ", strip=True))
            if qtr is not None and qtr.find(class_="quarter__title"):
                tags.append(qtr.find(class_="quarter__title").get_text(" ", strip=True))
            cur = t + (f" ({'; '.join(tags)})" if tags else "")
            buf.append(f"Topic: {cur}")
        elif name in ("h1", "h2", "h3", "h4"):
            flush(); buf = []
            cur = el.get_text(" ", strip=True)
        elif name == "table":
            rows = []
            for tr in el.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
                if any(cells):
                    rows.append(" | ".join(cells))
            buf.append("\n".join(rows))
        elif name == "li":
            if el.find_parent("table") is None and el.find(["p", "li"]) is None:
                buf.append("- " + el.get_text(" ", strip=True))
        elif name == "p":
            if el.find_parent("table") is None:
                t = el.get_text(" ", strip=True)
                # keep link targets: they are often the answer (scheduling, fee waiver, apply)
                for a in el.find_all("a", href=True):
                    h = urljoin(url, a["href"])
                    if h.startswith("http") and a.get_text(strip=True):
                        t += f" [{a.get_text(strip=True)}]({h})"
                buf.append(t)
        elif name == "div":
            # leaf-ish divs (e.g. mailing addresses): no block children, not nested in captured blocks
            if el.find(["p", "li", "div", "table", "h1", "h2", "h3", "h4"]) is None and el.find_parent(["p", "li", "table", "button", "summary", "dt", "dd"]) is None:
                t = el.get_text(" ", strip=True)
                if t:
                    buf.append(t)
        else:
            t = el.get_text(" ", strip=True)
            if t:
                buf.append(t)
    flush()
    links = [urljoin(url, a["href"]) for a in main.find_all("a", href=True)]
    return {"url": url, "title": title, "sections": sections}, links


def crawl():
    seen, q, out = set(), deque(norm(s) for s in SEEDS), []
    for e in EXTRA:
        q.append(e)
    while q and len(out) < MAX_PAGES:
        u = q.popleft()
        if u in seen:
            continue
        seen.add(u)
        try:
            r = requests.get(u, headers=HEAD, timeout=25)
            if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
                continue
        except Exception as e:
            print("ERR", u, e, file=sys.stderr); continue
        rec, links = extract(r.text, u)
        if rec["sections"]:
            out.append(rec)
            print(f"[{len(out):02d}] {u}  ({len(rec['sections'])} sections)")
        if u in EXTRA:
            continue  # do not crawl outward from external pages
        for l in links:
            l = norm(l)
            p = urlparse(l)
            if p.netloc == "datascience.uchicago.edu" and ALLOW.match(p.path) and not DENY.search(p.path) and l not in seen:
                q.append(l)
        time.sleep(0.4)
    with open("data/pages.jsonl", "w") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("pages:", len(out))


if __name__ == "__main__":
    crawl()
