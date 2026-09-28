"""Чанки индексатора для всех файлов корпуса (кэш по sha1 содержимого).

Запуск интерпретатором окружения демона (нужны парсеры проекта):
    daemon-venv/bin/python src/view_corpus.py precache <файлы...>   — заранее, по исходникам
    daemon-venv/bin/python src/view_corpus.py corpus                 — по data/manifest.jsonl

Результат: work/indexer_view_cache/<sha1>.chunks.json и
work/indexer_view_corpus/<doc_id с / → __>.chunks.json (для pool.py и асессоров).
"""

import hashlib
import json
import os
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL / "src"))
os.environ.setdefault("INDEXER_VIEW_HOME", str(EVAL / "work" / "indexer_view_home"))

from indexer_view import chunks_of  # noqa: E402  (импортирует парсеры проекта)

CACHE = EVAL / "work" / "indexer_view_cache"
OUT = EVAL / "work" / "indexer_view_corpus"


def sha1(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cached_chunks(p: Path) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    c = CACHE / f"{sha1(p)}.chunks.json"
    if c.exists():
        return json.load(open(c, encoding="utf-8"))
    try:
        chunks, err = chunks_of(p), None
    except Exception as e:
        chunks, err = [], f"{type(e).__name__}: {e}"
    rec = {"file": str(p), "error": err, "chunks": chunks}
    c.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return rec


def main() -> None:
    mode = sys.argv[1]
    if mode == "precache":
        for f in sys.argv[2:]:
            rec = cached_chunks(Path(f).resolve())
            print(json.dumps({"file": f, "chunks": len(rec["chunks"]), "error": rec["error"]}, ensure_ascii=False), flush=True)
    elif mode == "corpus":
        OUT.mkdir(parents=True, exist_ok=True)
        summary = []
        for l in open(EVAL / "data" / "manifest.jsonl", encoding="utf-8"):
            m = json.loads(l)
            rec = cached_chunks(Path(m["path"]))
            (OUT / (m["doc_id"].replace("/", "__") + ".chunks.json")).write_text(
                json.dumps({"doc_id": m["doc_id"], **rec}, ensure_ascii=False), encoding="utf-8")
            summary.append({"doc_id": m["doc_id"], "chunks": len(rec["chunks"]), "chars": sum(map(len, rec["chunks"])), "error": rec["error"]})
        (EVAL / "work" / "indexer_view_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=0), encoding="utf-8")
        empty = [s for s in summary if not s["chunks"]]
        print(f"документов: {len(summary)}; без чанков: {len(empty)}")
        for s in empty:
            print("  нет чанков:", s["doc_id"], s["error"])


if __name__ == "__main__":
    main()
