"""Сборка корпуса и эталона для замера ранжирования.

corpus/<набор>/<файл>        — файлы, которые индексирует демон
data/manifest.jsonl          — doc_id (путь относительно corpus), набор, формат, название
data/queries.jsonl           — qid, query, dataset, scenario (1–6), base (кластер), флаги
data/qrels.tsv               — qid, doc_id, оценка 0–3, источник оценки
data/excluded.jsonl          — запросы, не вошедшие в замер, с причиной

Правила оценок (методика, раздел «Перевод и разметка»):
- ЦБ: акт, к которому задан вопрос, = 3; другие акты, на которые ссылается ответ, = 2;
- ai360, ZX Bank, ObliQA, FAQ ЦБ, xlsx: эталонный документ набора = 3;
- варианты сценариев 2–4 наследуют разметку базового вопроса.
Запросы без ответа (сценарий 6) получают пустую разметку; их проверяет пул.
"""

import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render import render  # noqa: E402

EVAL = Path(__file__).resolve().parent.parent
RAW = EVAL / "raw"
WORK = EVAL / "work"
CORPUS = EVAL / "corpus"
DATA = EVAL / "data"

# Кейсы «нет ответа» (сценарий 6) отложены по решению пользователя (2026-09-27):
# исключённые ради них акты и отчёты МКБ вернулись в корпус.
EXCLUDED_ACTS: set[str] = set()
AI360_SCANS = {"alfa_2024_annual", "gpb_2023_annual", "gpb_2025_annual"}  # PDF без текста
AI360_NOANSWER_BANK = None
# Дубли страниц FAQ под другим адресом (совпадение текста ~100%).
EXTRA_PAGE_DUPLICATES = {"faq_bank_s", "faq_credit_h", "faq_nps"}
BANKS = {
    "alfa": "Альфа-Банк (Беларусь)", "domrf": "ДОМ.РФ", "gpb": "Газпромбанк", "mkb": "МКБ",
    "rshb": "Россельхозбанк", "sber": "Сбербанк", "sovkom": "Совкомбанк", "tbank": "Т-Банк", "vtb": "ВТБ",
}


def safe_name(title: str, limit: int = 90) -> str:
    t = re.sub(r"[\\/:*?\"<>|\n\r\t]+", " ", title)
    t = re.sub(r"\s+", " ", t).strip().strip(".").lstrip("~")
    return t[:limit].rstrip() or "документ"


class Builder:
    def __init__(self) -> None:
        self.manifest: dict[str, dict] = {}
        self.queries: list[dict] = []
        self.qrels: dict[str, dict[str, tuple[int, str]]] = defaultdict(dict)
        self.excluded: list[dict] = []

    # ---- документы -------------------------------------------------------
    def add_doc(self, dataset: str, src: Path | None, name: str, *, render_from: Path | None = None,
                fmt: str | None = None, title: str | None = None, key: str | None = None) -> str:
        folder = CORPUS / dataset
        fmt = fmt or (src.suffix[1:] if src else "txt")
        dest = folder / f"{safe_name(name)}.{fmt}"
        n = 2
        while dest.exists() and str(dest.relative_to(CORPUS)) in self.manifest:
            dest = folder / f"{safe_name(name)} ({n}).{fmt}"
            n += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        if render_from is not None:
            render(render_from, fmt, dest)
        else:
            shutil.copy2(src, dest)
        doc_id = str(dest.relative_to(CORPUS))
        self.manifest[doc_id] = {"doc_id": doc_id, "path": str(dest), "dataset": dataset, "format": fmt,
                                 "title": title or name, "key": key or doc_id}
        return doc_id

    def add_query(self, qid: str, query: str, dataset: str, scenario: int, base: str | None = None,
                  gold: dict[str, int] | None = None, src: str = "dataset", **extra) -> None:
        self.queries.append({"qid": qid, "query": query.strip(), "dataset": dataset, "scenario": scenario,
                             "base": base or qid, **extra})
        for d, g in (gold or {}).items():
            self.qrels[qid][d] = (g, src)

    def exclude(self, qid: str, dataset: str, reason: str, query: str = "") -> None:
        self.excluded.append({"qid": qid, "dataset": dataset, "reason": reason, "query": query})

    # ---- наборы ----------------------------------------------------------
    def cbr(self) -> dict[str, str]:
        src = json.load(open(RAW / "cbr_acts" / "sources.json", encoding="utf-8"))
        src = src if isinstance(src, list) else list(src.values())
        extra = RAW / "cbr_acts" / "sources_extra.json"
        if extra.exists():
            ex = json.load(open(extra, encoding="utf-8"))
            src = src + (ex if isinstance(ex, list) else list(ex.values()))
        titles_file = RAW / "cbr_acts" / "titles.json"
        titles = json.load(open(titles_file, encoding="utf-8")) if titles_file.exists() else {}
        for s_ in src:
            if s_.get("title") and s_["act"] not in titles:
                titles[s_["act"]] = s_["title"]
        act_doc = {}
        for s in src:
            act = s["act"]
            if act in EXCLUDED_ACTS or s.get("status") != "ok":
                continue
            f = RAW / "cbr_acts" / Path(s["file"]).name
            act_doc[act] = self.add_doc("cbr_acts", f, f"Акт Банка России {act}", title=titles.get(act, act), key=act)
        rows = [json.loads(l) for l in open(RAW / "cbr_dbrnfaq" / "questions.jsonl", encoding="utf-8")]
        for r in rows:
            sec = r["section"].strip()
            refs = set(r["ref_acts"]) - {sec}
            gold = {}
            if sec in act_doc:
                gold[act_doc[sec]] = 3
            for a in refs:
                if a in act_doc:
                    gold.setdefault(act_doc[a], 2)
            explicit = sec.split("-")[0] in r["question"]
            if sec in EXCLUDED_ACTS and not gold:
                self.add_query(r["qid"], r["question"], "cbr", 6, explicit_act=explicit, source_act=sec)
            elif gold:
                self.add_query(r["qid"], r["question"], "cbr", 1, gold=gold, explicit_act=explicit, source_act=sec)
            else:
                self.exclude(r["qid"], "cbr", "эталонных актов нет в корпусе", r["question"])
        return act_doc

    def ai360(self) -> dict[str, str]:
        doc = {}
        for pdf in sorted((RAW / "ai360" / "data" / "raw").glob("*/*/*.pdf")):
            bank, year, kind = pdf.parts[-3], pdf.parts[-2], pdf.stem
            did = f"{bank}_{year}_{kind}"
            if did in AI360_SCANS or (AI360_NOANSWER_BANK and bank == AI360_NOANSWER_BANK):
                continue
            period = "годовая" if kind == "annual" else "за 6 месяцев"
            name = f"{BANKS[bank]} МСФО {year} {period}"
            doc[did] = self.add_doc("ai360", pdf, name, title=name, key=did)
        for l in open(RAW / "ai360" / "data" / "dataset.jsonl", encoding="utf-8"):
            q = json.loads(l)
            ids = {e["doc_id"] for e in q["gold_evidence"]}
            qid = "ai_" + q["question_id"][2:]
            gold = {doc[d]: 3 for d in ids if d in doc}
            if AI360_NOANSWER_BANK and all(d.startswith(AI360_NOANSWER_BANK) for d in ids):
                self.add_query(qid, q["question"], "ai360", 6)
            elif gold:
                self.add_query(qid, q["question"], "ai360", 5, gold=gold, split=q["split"])
            else:
                self.exclude(qid, "ai360", "эталонный PDF — скан без текстового слоя", q["question"])
        return doc

    def faq(self, paraphrases: dict[str, str]) -> None:
        pages = {}
        pairs = [json.loads(l) for l in open(RAW / "cbr_faq" / "pairs.jsonl", encoding="utf-8")]
        slugs = sorted({p["page"] for p in pairs})
        for i, s in enumerate(slugs):
            txt = RAW / "cbr_faq" / "pages" / f"{s}.txt"
            if not txt.exists():
                continue
            title = txt.read_text(encoding="utf-8").splitlines()[0].strip() or s
            fmt = "docx" if i % 2 == 0 else "txt"
            if fmt == "docx":
                md = WORK / "faq_md" / f"{s}.md"
                md.parent.mkdir(parents=True, exist_ok=True)
                body = txt.read_text(encoding="utf-8").split("\n", 1)[1]
                md.write_text(f"# {title}\n\n{body}", encoding="utf-8")
                pages[s] = self.add_doc("cbr_faq", None, f"ЦБ — {title}", render_from=md, fmt="docx", title=title, key=s)
            else:
                pages[s] = self.add_doc("cbr_faq", txt, f"ЦБ — {title}", title=title, key=s)
        by_fid = {p["fid"]: p for p in pairs}
        for fid, text in paraphrases.items():
            p = by_fid[fid]
            if p["page"] in pages:
                self.add_query(fid, text, "cbr_faq", 1, gold={pages[p["page"]]: 3}, original=p["question"])

    def zx_obliqa(self) -> dict[str, str]:
        units = [json.loads(l) for l in open(WORK / "translate" / "units.jsonl", encoding="utf-8")]
        unit_doc = {}
        for u in units:
            parts = [WORK / "translate" / "parts" / f"{p}.ru.md" for p in u["parts"]]
            if not all(p.exists() for p in parts):
                raise FileNotFoundError(f"нет перевода для {u['unit_id']}")
            ru = "\n\n".join(p.read_text(encoding="utf-8").strip() for p in parts) + "\n"
            md = WORK / "translate" / "docs" / f"{u['unit_id']}.ru.md"
            md.write_text(ru, encoding="utf-8")
            first = next((l.lstrip("#").strip() for l in ru.splitlines() if l.strip()), u["unit_id"])
            first = re.sub(r"^\[[^\]]+\]\s*", "", first)
            if u["dataset"] == "zx":
                unit_doc[u["unit_id"]] = self.add_doc("zx_bank", None, first, render_from=md,
                                                      fmt=u["corpus_format"], title=first, key=u["unit_id"])
            else:
                code = u["unit_id"].split("_")[1].upper()
                name = f"ADGM {code} — {first}"
                unit_doc[u["unit_id"]] = self.add_doc("adgm", md, name, fmt="txt", title=name, key=u["unit_id"])
        for f, ds in (("queries_zx", "zx"), ("queries_obliqa", "obliqa")):
            ru = {}
            for qb in sorted((WORK / "translate" / "qbatches").glob("qb*.ru.json")):
                for it in json.load(open(qb, encoding="utf-8")):
                    ru[it["qid"]] = it["q_ru"]
            for l in open(WORK / "translate" / f"{f}.jsonl", encoding="utf-8"):
                q = json.loads(l)
                if q["qid"] not in ru:
                    self.exclude(q["qid"], ds, "нет перевода запроса", q["query_en"])
                    continue
                gold = {unit_doc[u]: 3 for u in q["gold_units"] if u in unit_doc}
                self.add_query(q["qid"], ru[q["qid"]], ds, 1, gold=gold, query_en=q["query_en"])
        return unit_doc

    def xlsx(self, items: list[dict]) -> None:
        files = sorted((RAW / "xlsx").glob("obs_*.xlsx"))
        doc = {f.name: self.add_doc("cbr_stat", f, f"Статистические показатели банковского сектора {f.stem}",
                                    title=f.stem, key=f.name) for f in files}
        chunks = {f.name: json.load(open(WORK / "indexer_view" / f"{f.name}.chunks.json", encoding="utf-8"))["chunks"]
                  for f in files}
        rows = json.load(open(WORK / "querygen" / "xlsx_rows.json", encoding="utf-8"))
        for it in items:
            r = rows[it["idx"]]
            period = it["period"][:10]
            # значение на эту дату в исходной строке
            src_chunk = r["chunk"]
            m = re.search(re.escape(period) + r"[^:\n]*:\s*([-\d.eE]+)", src_chunk)
            if not m:
                self.exclude(f"xl{it['idx']:03d}", "xlsx", "период запроса не найден в строке", it["query"])
                continue
            value = m.group(1)
            gold = {}
            for name, ch in chunks.items():
                for c in ch:
                    if r["label"] in c and re.search(re.escape(period) + r"[^:\n]*:\s*" + re.escape(value), c):
                        gold[doc[name]] = 3
                        break
            if not gold:
                self.exclude(f"xl{it['idx']:03d}", "xlsx", "значение не найдено в чанках", it["query"])
                continue
            sc = 5 if len(gold) < len(files) else 1
            self.add_query(f"xl{it['idx']:03d}", it["query"], "xlsx", sc, gold=gold, period=period,
                           row_label=r["label"], source_file=r["file"])

    def cbr_extra(self) -> None:
        """Разъяснения ЦБ, дополнительные разделы FAQ, обзоры банковского сектора."""
        pages = sorted((RAW / "cbr_extra" / "pages").glob("*.txt"))
        for i, txt in enumerate(pages):
            if txt.stem in EXTRA_PAGE_DUPLICATES:
                continue
            title = txt.read_text(encoding="utf-8").splitlines()[0].strip() or txt.stem
            if i % 2 == 0:
                md = WORK / "faq_md" / f"{txt.stem}.md"
                md.parent.mkdir(parents=True, exist_ok=True)
                md.write_text(f"# {title}\n\n" + txt.read_text(encoding="utf-8").split("\n", 1)[1], encoding="utf-8")
                self.add_doc("cbr_explan", None, f"ЦБ разъяснения — {title}", render_from=md, fmt="docx", title=title, key=txt.stem)
            else:
                self.add_doc("cbr_explan", txt, f"ЦБ разъяснения — {title}", title=title, key=txt.stem)
        for pdf in sorted((RAW / "cbr_extra" / "analytics").glob("*.pdf")):
            yy, mm = pdf.stem.split("_")[-2:]
            name = f"О развитии банковского сектора РФ — 20{yy}-{mm}"
            self.add_doc("cbr_analytics", pdf, name, title=name, key=pdf.stem)

    def variants(self, items: list[dict]) -> None:
        base = {q["qid"]: q for q in self.queries}
        for it in items:
            b = it["base_id"]
            bq = base.get(b) or (base.get("ai_" + b[2:]) if b.startswith("q_") else None)
            if bq is None:
                continue
            gold = {d: g for d, (g, _) in self.qrels[bq["qid"]].items()}
            for sc, field in ((2, "keywords"), (3, "designation"), (4, "paraphrase")):
                text = it.get(field)
                if text:
                    self.add_query(f"{bq['qid']}_s{sc}", text, bq["dataset"], sc, base=bq["qid"], gold=gold, src="inherited")

    def noanswer(self, items: list[dict]) -> None:
        for i, it in enumerate(items, 1):
            self.add_query(f"na{i:03d}", it["query"], "noanswer", 6, topic=it.get("topic"))

    # ---- запись ----------------------------------------------------------
    def write(self) -> None:
        DATA.mkdir(parents=True, exist_ok=True)
        with (DATA / "manifest.jsonl").open("w", encoding="utf-8") as f:
            for m in self.manifest.values():
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        with (DATA / "queries.jsonl").open("w", encoding="utf-8") as f:
            for q in self.queries:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        with (DATA / "qrels.tsv").open("w", encoding="utf-8") as f:
            f.write("qid\tdoc_id\tgrade\tsource\n")
            for q, docs in self.qrels.items():
                for d, (g, s) in sorted(docs.items()):
                    f.write(f"{q}\t{d}\t{g}\t{s}\n")
        with (DATA / "excluded.jsonl").open("w", encoding="utf-8") as f:
            for e in self.excluded:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print("документов:", Counter(m["dataset"] for m in self.manifest.values()))
        print("форматы:", Counter(m["format"] for m in self.manifest.values()))
        print("запросов:", Counter((q["dataset"], q["scenario"]) for q in self.queries))
        print("исключено:", Counter((e["dataset"], e["reason"]) for e in self.excluded))


def main() -> None:
    qg = WORK / "querygen" / "final.json"
    gen = json.load(open(qg, encoding="utf-8")) if qg.exists() else {}
    if CORPUS.exists():
        shutil.rmtree(CORPUS)
    b = Builder()
    b.cbr()
    b.ai360()
    b.faq({it["fid"]: it["query"] for k in ("faq1", "faq2") for it in gen.get(k, [])})
    b.zx_obliqa()
    b.cbr_extra()
    b.xlsx(gen.get("xlsx", []))
    b.variants(gen.get("var_cbr", []) + gen.get("var_ai360", []) + gen.get("var_zx", []) + gen.get("var_obliqa", []))
    # Собственные вопросы «без ответа» отложены вместе со сценарием 6
    # (лежат в work/querygen/final.json, ключ noanswer).
    b.write()


if __name__ == "__main__":
    main()
