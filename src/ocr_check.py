"""КТ-OCR1: качество и скорость OCR (Tesseract внутри PyMuPDF) на этом компьютере.

Режим sim — цифровые страницы МСФО превращаются в «скан» (картинка JPEG без текстового слоя,
100 и 200 dpi), распознаются и сверяются с текстовым слоем: доля сумм от 4 цифр (без годов,
разряды склеены) и слов от 5 букв из текстового слоя, найденных в распознанном тексте.
Режим answers — эталонные страницы 31 исключённого запроса (три настоящих скана): доля сумм
из эталонных ответов, найденных в распознанном тексте страницы; потолок той же проверки на
цифровом тексте — 0,77 по суммам (замер исследования OCR).

  daemon-venv/bin/python src/ocr_check.py sim --pages 20 --scan-dpi 100,200 --ocr-dpi 200,300
  daemon-venv/bin/python src/ocr_check.py answers --ocr-dpi 300
  LIB_LOCAL_RAG=<проект> daemon-venv/bin/python src/ocr_check.py chunks   # КТ-OCR2

Режим chunks — три скана целиком парсером проекта (с OCR, как при индексации): сколько
чанков, сколько секунд, и есть ли суммы эталонных ответов в чанках эталонных страниц.

Распознавание — как в воркере проекта: страница → RGB pixmap → pdfocr_tobytes → get_text.
Языковые данные tessdata_fast rus+eng — work/ocr/tessdata (с GitHub, tag 4.1.0).
"""

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import pymupdf

EVAL = Path(__file__).resolve().parent.parent
AI360 = EVAL / "raw/ai360/data"
TESSDATA = EVAL / "work/ocr/tessdata"
OUT = EVAL / "work/ocr"
SIM_DOCS = ["alfa/2023/annual.pdf", "gpb/2024/annual.pdf"]
MAX_PIXELS = 20_000_000  # ~A3 при 300 dpi: больше не рисуем

_SUM = re.compile(r"(?<![\d.,])\d{1,3}(?:[  ,.]\d{3})+(?![\d])|(?<![\d.,])\d{4,}(?![\d])")
_WORD = re.compile(r"[а-яёa-z]{5,}")


def sums(text: str) -> list[int]:
    out = []
    for m in _SUM.finditer(text):
        n = int(re.sub(r"\D", "", m.group()))
        if 1900 <= n <= 2099 and len(m.group()) == 4:  # год
            continue
        if n >= 1000:
            out.append(n)
    return out


def words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower().replace("ё", "е")))


def ocr_page(page: pymupdf.Page, dpi: int, language: str = "rus+eng") -> tuple[str, float]:
    t = time.perf_counter()
    zoom = dpi / 72
    px = page.rect.width * zoom * page.rect.height * zoom
    if px > MAX_PIXELS:
        dpi = int(dpi * (MAX_PIXELS / px) ** 0.5)
    # Только RGB: по серому pixmap pdfocr молча отдаёт пустой текст (PyMuPDF 1.28.2).
    pix = page.get_pixmap(dpi=dpi)
    data = pix.pdfocr_tobytes(compress=False, language=language, tessdata=str(TESSDATA))
    with pymupdf.open("pdf", data) as d:
        text = d[0].get_text()
    return text, time.perf_counter() - t


def fake_scan(page: pymupdf.Page, dpi: int) -> pymupdf.Document:
    """Страница → PDF из одной JPEG-картинки той же страницы (текстового слоя нет)."""
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    jpeg = pix.tobytes("jpeg", jpg_quality=75)
    doc = pymupdf.open()
    p = doc.new_page(width=page.rect.width, height=page.rect.height)
    p.insert_image(p.rect, stream=jpeg)
    return doc


def pick_pages(doc: pymupdf.Document, n: int) -> list[int]:
    dense = [i for i, p in enumerate(doc) if len(sums(p.get_text())) >= 5]
    if len(dense) <= n:
        return dense
    step = len(dense) / n
    return [dense[int(k * step)] for k in range(n)]


def share(found: int, total: int) -> str:
    return f"{found}/{total} = {found / total:.3f}" if total else "—"


def cmd_sim(a) -> None:
    rows = []
    for rel in SIM_DOCS:
        src = pymupdf.open(AI360 / "raw" / rel)
        for i in pick_pages(src, a.pages):
            ref = src[i].get_text()
            ref_sums, ref_words = sums(ref), words(ref)
            for sd in a.scan_dpi:
                scan = fake_scan(src[i], sd)
                assert not scan[0].get_text().strip()
                for od in a.ocr_dpi:
                    text, sec = ocr_page(scan[0], od)
                    got_sums, got_words = set(sums(text)), words(text)
                    rows.append({"doc": rel, "page": i + 1, "scan_dpi": sd, "ocr_dpi": od, "sec": round(sec, 2),
                                 "sums": len(ref_sums), "sums_found": sum(1 for s in ref_sums if s in got_sums),
                                 "words": len(ref_words), "words_found": len(ref_words & got_words)})
                    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sim.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print("\n| Скан, dpi | OCR, dpi | Страниц | Суммы | Слова от 5 букв | с/стр (медиана) | с/стр (макс) |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for sd in a.scan_dpi:
        for od in a.ocr_dpi:
            rs = [r for r in rows if r["scan_dpi"] == sd and r["ocr_dpi"] == od]
            print(f"| {sd} | {od} | {len(rs)} | {share(sum(r['sums_found'] for r in rs), sum(r['sums'] for r in rs))} | "
                  f"{share(sum(r['words_found'] for r in rs), sum(r['words'] for r in rs))} | "
                  f"{statistics.median(r['sec'] for r in rs):.2f} | {max(r['sec'] for r in rs):.2f} |")


def excluded_gold() -> list[dict]:
    """31 исключённый запрос по сканам → документ, страницы, эталонный ответ."""
    ex = [json.loads(x) for x in open(EVAL / "data/excluded.jsonl", encoding="utf-8")]
    scans = {e["query"].strip() for e in ex if "скан" in e["reason"]}
    out = []
    for line in open(AI360 / "dataset.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["question"].strip() in scans:
            for ev in r["gold_evidence"]:
                bank, year, kind = ev["doc_id"].split("_")
                out.append({"q": r["question_id"], "file": f"{bank}/{year}/{kind}.pdf", "pages": ev["pages"],
                            "answer": r["gold_answer"]})
    return out


def cmd_answers(a) -> None:
    gold = excluded_gold()
    print(f"запросов {len({g['q'] for g in gold})}, страниц {len({(g['file'], p) for g in gold for p in g['pages']})}")
    for od in a.ocr_dpi:
        cache: dict[tuple[str, int], str] = {}
        secs = []
        for g in gold:
            for p in g["pages"]:
                if (g["file"], p) not in cache:
                    with pymupdf.open(AI360 / "raw" / g["file"]) as d:
                        text, sec = ocr_page(d[p - 1], od)
                    cache[(g["file"], p)] = text
                    secs.append(sec)
        q_sums = q_all = n_sums = n_found = 0
        for q in sorted({g["q"] for g in gold}):
            gs = [g for g in gold if g["q"] == q]
            want = sums(gs[0]["answer"])
            if not want:
                continue
            got = set()
            for g in gs:
                for p in g["pages"]:
                    got |= set(sums(cache[(g["file"], p)]))
            q_sums += 1
            n_sums += len(want)
            n_found += sum(1 for s in want if s in got)
            q_all += all(s in got for s in want)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"answers_{od}.json").write_text(json.dumps({f"{f}#{p}": t for (f, p), t in cache.items()},
                                                           ensure_ascii=False), encoding="utf-8")
        print(f"OCR {od} dpi: страниц {len(cache)}, с/стр медиана {statistics.median(secs):.2f}, макс {max(secs):.2f}; "
              f"суммы ответов {share(n_found, n_sums)}; запросов со всеми суммами {share(q_all, q_sums)}")


def cmd_chunks(a) -> None:
    import os
    import sys

    os.environ.setdefault("INDEXER_VIEW_HOME", str(EVAL / "work/indexer_view_home"))
    sys.path.insert(0, str(EVAL / "src"))
    import indexer_view  # noqa: F401  парсеры проекта (LIB_LOCAL_RAG) и set_config()
    from src.parsers import file_chunks

    gold = excluded_gold()
    by_file: dict[str, list[dict]] = {}
    for f in sorted({g["file"] for g in gold}):
        t = time.perf_counter()
        chunks = list(file_chunks(AI360 / "raw" / f))
        by_file[f] = chunks
        with pymupdf.open(AI360 / "raw" / f) as d:
            pages = d.page_count
        print(f"{f}: страниц {pages}, чанков {len(chunks)}, с OCR {sum(c.get('kind') == 'ocr' for c in chunks)}, "
              f"{time.perf_counter() - t:.0f} с ({(time.perf_counter() - t) / pages:.2f} с/стр)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chunks.json").write_text(json.dumps({f: [c["text"] for c in ch] for f, ch in by_file.items()},
                                                ensure_ascii=False), encoding="utf-8")
    q_sums = q_doc = q_page = n_sums = n_doc = n_page = 0
    for q in sorted({g["q"] for g in gold}):
        gs = [g for g in gold if g["q"] == q]
        want = sums(gs[0]["answer"])
        if not want:
            continue
        in_doc, in_page = set(), set()
        for g in gs:
            for c in by_file[g["file"]]:
                got = set(sums(c["text"]))
                in_doc |= got
                if any(c.get("page_start", 0) <= p <= c.get("page_end", -1) for p in g["pages"]):
                    in_page |= got
        q_sums += 1
        n_sums += len(want)
        n_doc += sum(1 for s in want if s in in_doc)
        n_page += sum(1 for s in want if s in in_page)
        q_doc += all(s in in_doc for s in want)
        q_page += all(s in in_page for s in want)
    print(f"суммы ответов в чанках эталонных страниц: {share(n_page, n_sums)}; где-либо в документе: "
          f"{share(n_doc, n_sums)}; запросов со всеми суммами на эталонных страницах: {share(q_page, q_sums)}, "
          f"в документе: {share(q_doc, q_sums)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim")
    s.add_argument("--pages", type=int, default=20)
    s.add_argument("--scan-dpi", type=lambda v: [int(x) for x in v.split(",")], default=[100, 200])
    s.add_argument("--ocr-dpi", type=lambda v: [int(x) for x in v.split(",")], default=[300])
    b = sub.add_parser("answers")
    b.add_argument("--ocr-dpi", type=lambda v: [int(x) for x in v.split(",")], default=[300])
    sub.add_parser("chunks")
    a = ap.parse_args()
    {"sim": cmd_sim, "answers": cmd_answers, "chunks": cmd_chunks}[a.cmd](a)


if __name__ == "__main__":
    main()
