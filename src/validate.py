"""Проверки эталона перед прогонами (методика: этап 2).

1. Каждая пара разметки ссылается на документ корпуса.
2. У каждого эталонного документа есть чанки в представлении индексатора.
3. ObliQA: номер эталонного пункта есть в тексте эталонной главы.
4. Сценарий 3: точное обозначение из запроса встречается в тексте хотя бы
   одного эталонного документа; иначе вариант исключается (методика).
Результат: печать сводки; с --apply — варианты сценария 3 без обозначения
в тексте переносятся в data/excluded.jsonl.
"""

import json
import re
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parent.parent
DATA = EVAL / "data"
VIEW = EVAL / "work" / "indexer_view_corpus"

HYPHENS = "‐‑‒–—−\x19"


def norm(s: str) -> str:
    s = s.lower().replace("ё", "е")
    for h in HYPHENS:
        s = s.replace(h, "-")
    return re.sub(r"\s+", " ", s)


def tok(s: str) -> str:
    """Как токенизатор индексатора: всё, кроме букв и цифр, — разделитель."""
    return " " + " ".join(re.findall(r"[0-9a-zа-я]+", norm(s))) + " "


def doc_text(doc_id: str, cache={}) -> str:
    if doc_id not in cache:
        f = VIEW / (doc_id.replace("/", "__") + ".chunks.json")
        cache[doc_id] = tok("\n".join(json.load(open(f, encoding="utf-8"))["chunks"])) if f.exists() else ""
    return cache[doc_id]


# Обозначение в запросе: номер акта, форма, норматив, пункт, стандарт, латинская аббревиатура.
DESIGNATION = re.compile(
    r"\b\d{1,5}\s*-\s*(?:п|у|и|мр)\b|\b0409\d{3}\b|\bн\d+(?:\.\d+)?\b|\b\d+(?:\.\d+){1,5}\b"
    r"|\b(?:мсфо|ifrs|ias)\s*\(?\d+\)?|\b[a-z][a-z0-9]{1,6}\b(?:\s*\d+)?",
    re.I,
)
STOP_LATIN = {"zx", "bank", "ltd", "asia", "adgm"}


def designations(query: str) -> list[str]:
    out = []
    for m in DESIGNATION.finditer(norm(query)):
        t = m.group(0).strip()
        if t in STOP_LATIN:
            continue
        out.append(t)
    return out


def main() -> None:
    apply = "--apply" in sys.argv
    manifest = {json.loads(l)["doc_id"] for l in open(DATA / "manifest.jsonl", encoding="utf-8")}
    queries = [json.loads(l) for l in open(DATA / "queries.jsonl", encoding="utf-8")]
    qrels: dict[str, dict[str, int]] = {}
    for l in open(DATA / "qrels.tsv", encoding="utf-8").read().splitlines()[1:]:
        q, d, g, _ = l.split("\t")
        qrels.setdefault(q, {})[d] = int(g)

    missing_docs = [(q, d) for q, ds in qrels.items() for d in ds if d not in manifest]
    empty_gold = sorted({d for ds in qrels.values() for d in ds if d in manifest and not doc_text(d)})
    print(f"пар разметки: {sum(map(len, qrels.values()))}; ссылок на отсутствующие документы: {len(missing_docs)}; "
          f"эталонных документов без текста: {len(empty_gold)}")

    # ObliQA: номера пунктов
    tr = {json.loads(l)["qid"]: json.loads(l) for l in open(EVAL / "work" / "translate" / "queries_obliqa.jsonl", encoding="utf-8")}
    bad_pass = 0
    total_pass = 0
    for q in queries:
        if q["dataset"] != "obliqa" or q["scenario"] != 1:
            continue
        for f in tr[q["qid"]]["facts_en"]:
            total_pass += 1
            pid = "[" + f["passage_id"].rstrip(".") + "]"
            if not any(tok(pid) in doc_text(d) for d in qrels.get(q["qid"], {})):
                bad_pass += 1
    print(f"ObliQA: эталонных пунктов {total_pass}, не найдено в тексте эталонной главы: {bad_pass}")

    # Сценарий 3
    drop = []
    for q in queries:
        if q["scenario"] != 3:
            continue
        ds = designations(q["query"])
        gold = [d for d, g in qrels.get(q["qid"], {}).items() if g >= 2]
        found = [t for t in ds if any(tok(t) in doc_text(d) for d in gold)]
        if not found:
            drop.append((q["qid"], q["query"], ds))
    print(f"сценарий 3: вариантов {sum(q['scenario'] == 3 for q in queries)}, обозначение не найдено в эталоне: {len(drop)}")
    for x in drop:
        print("  ", x)
    # Сценарий 6: если пул нашёл документ с оценкой ≥ 2, запрос не «без ответа».
    judged = EVAL / "work" / "judge" / "noanswer" / "final.tsv"
    found_answer = set()
    if judged.exists():
        for l in judged.read_text(encoding="utf-8").splitlines():
            qid, _, g, _ = l.split("\t")
            if int(g) >= 2:
                found_answer.add(qid)
    na_drop = [(q["qid"], q["query"], "в корпусе нашёлся документ с ответом (пул)") for q in queries
               if q["scenario"] == 6 and q["qid"] in found_answer]
    print(f"сценарий 6: запросов {sum(q['scenario'] == 6 for q in queries)}, нашёлся ответ: {len(na_drop)}")
    drop = [(qid, text, "обозначение не встречается в тексте эталона") for qid, text, _ in drop] + na_drop
    if apply and drop:
        ids = {x[0] for x in drop}
        kept = [q for q in queries if q["qid"] not in ids]
        with (DATA / "queries.jsonl").open("w", encoding="utf-8") as f:
            for q in kept:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        with (DATA / "excluded.jsonl").open("a", encoding="utf-8") as f:
            for qid, text, reason in drop:
                f.write(json.dumps({"qid": qid, "dataset": "validate", "reason": reason,
                                    "query": text}, ensure_ascii=False) + "\n")
        lines = open(DATA / "qrels.tsv", encoding="utf-8").read().splitlines()
        (DATA / "qrels.tsv").write_text("\n".join([lines[0]] + [l for l in lines[1:] if l.split("\t")[0] not in ids]) + "\n", encoding="utf-8")
        print(f"исключено запросов: {len(ids)}")


if __name__ == "__main__":
    main()
