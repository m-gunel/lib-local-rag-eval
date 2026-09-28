"""Независимый пул кандидатов для разметки (методика: кандидаты отбирает не
индексатор, а отдельный BM25).

Текст документов — чанки, которые выдают парсеры самого индексатора
(work/indexer_view/*.chunks.json, см. indexer_view.py), поэтому асессор видит
ровно то, что может найти поиск. Ранжирование — rank_bm25 со стеммером
Snowball, отдельная от lance реализация.

    .venv/bin/python src/pool.py noanswer   — пул top-10 для запросов сценария 6
    (как библиотека) passages(qid_text, doc_id, k) — лучшие фрагменты документа
"""

import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import snowballstemmer
from rank_bm25 import BM25Okapi

EVAL = Path(__file__).resolve().parent.parent
DATA = EVAL / "data"
VIEW = EVAL / "work" / "indexer_view_corpus"
_stem = snowballstemmer.stemmer("russian")
_en = snowballstemmer.stemmer("english")
TOKEN = re.compile(r"[0-9a-zа-яё]+", re.I)


def tokens(text: str) -> list[str]:
    out = []
    for t in TOKEN.findall(text.lower().replace("ё", "е")):
        out.append(_stem.stemWord(t) if re.search("[а-я]", t) else _en.stemWord(t))
    return out


def view_file(doc_id: str) -> Path:
    return VIEW / (doc_id.replace("/", "__") + ".chunks.json")


@lru_cache(maxsize=1)
def corpus_index():
    manifest = [json.loads(l) for l in open(DATA / "manifest.jsonl", encoding="utf-8")]
    chunk_doc, chunk_text = [], []
    for m in manifest:
        f = view_file(m["doc_id"])
        if not f.exists():
            continue
        for c in json.load(open(f, encoding="utf-8"))["chunks"]:
            chunk_doc.append(m["doc_id"])
            chunk_text.append(c)
    bm = BM25Okapi([tokens(t) for t in chunk_text])
    return bm, chunk_doc, chunk_text


def top_docs(query: str, k: int = 10) -> list[tuple[str, float]]:
    bm, chunk_doc, _ = corpus_index()
    scores = bm.get_scores(tokens(query))
    best: dict[str, float] = {}
    for i, s in enumerate(scores):
        d = chunk_doc[i]
        if s > best.get(d, 0.0):
            best[d] = float(s)
    return sorted(best.items(), key=lambda x: -x[1])[:k]


@lru_cache(maxsize=64)
def doc_index(doc_id: str):
    f = view_file(doc_id)
    chunks = json.load(open(f, encoding="utf-8"))["chunks"] if f.exists() else []
    return chunks, (BM25Okapi([tokens(c) for c in chunks]) if chunks else None)


def passages(query: str, doc_id: str, k: int = 3, extra: list[str] | None = None, limit: int = 1500) -> list[str]:
    """Лучшие по BM25 фрагменты документа для запроса (+ фрагменты, которые
    вернул индексатор, если переданы) — это то, что видит асессор."""
    chunks, bm = doc_index(doc_id)
    out = []
    for e in extra or []:
        if e and e[:limit] not in out:
            out.append(e[:limit])
    if chunks:
        scores = bm.get_scores(tokens(query))
        order = sorted(range(len(chunks)), key=lambda i: -scores[i])
        for i in order:
            if len(out) >= k + len(extra or []):
                break
            if chunks[i][:limit] not in out:
                out.append(chunks[i][:limit])
    return out


def cmd_noanswer() -> None:
    qs = [json.loads(l) for l in open(DATA / "queries.jsonl", encoding="utf-8")]
    out_dir = EVAL / "work" / "pool"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {json.loads(l)["doc_id"]: json.loads(l) for l in open(DATA / "manifest.jsonl", encoding="utf-8")}
    items = []
    for q in qs:
        if q["scenario"] != 6:
            continue
        for d, s in top_docs(q["query"], 10):
            items.append({"qid": q["qid"], "query": q["query"], "doc_id": d, "title": manifest[d]["title"],
                          "bm25": round(s, 2), "passages": passages(q["query"], d, 3)})
    (out_dir / "noanswer_pool.json").write_text(json.dumps(items, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"пар на оценку: {len(items)}")


if __name__ == "__main__":
    if sys.argv[1:] == ["noanswer"]:
        cmd_noanswer()
