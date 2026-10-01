"""Проверка дублей файлов: одинаковые файлы в разных папках не должны занимать несколько мест выдачи.

Мини-корпус work/dup_corpus/files: четыре файла (pdf, docx, pptx, txt) лежат в трёх папках —
оригинал, копия под тем же именем в архиве и переименованная копия «копия <имя>»; контрольный pdf —
без копий. Запросы — те, у которых один из этих файлов единственный главный эталонный документ.

  .venv/bin/python src/dup_check.py build
  .venv/bin/python src/harness.py run --run-id DUP0 --corpus work/dup_corpus/files \\
      --queries work/dup_corpus/queries.jsonl --modes hybrid
  .venv/bin/python src/dup_check.py report --run DUP0

Отчёт: сколько мест выдачи заняли копии уже показанного файла (совпадение по sha256 содержимого)
и в скольких запросах так случилось; поле `duplicates` результата, если демон его отдаёт.
"""

import argparse
import collections
import hashlib
import json
import shutil
from pathlib import Path

EVAL = Path(__file__).resolve().parent.parent
DUP = EVAL / "work" / "dup_corpus"
COPIED = [
    "zx_bank/ZX Bank Ltd. — награды и признание.pdf",
    "zx_bank/Сеть отделений ZX Bank в Индауре.docx",
    "zx_bank/ZX Bank — кредиты для бизнеса.pptx",
    "cbr_faq/ЦБ — Маркетплейс.txt",
]
CONTROL = ["zx_bank/Сеть отделений ZX Bank в Лудхиане.pdf"]


def jl(p: Path) -> list[dict]:
    return [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]


def cmd_build(a) -> None:
    man = {m["doc_id"]: m for m in jl(EVAL / "data/manifest.jsonl")}
    files = DUP / "files"
    if files.exists():
        shutil.rmtree(files)
    for doc in COPIED + CONTROL:
        src = Path(man[doc]["path"])
        targets = [files / "Отдел" / src.name]
        if doc in COPIED:
            targets += [files / "Архив" / "2025" / src.name, files / "Почта" / "вложения" / f"копия {src.name}"]
        for t in targets:
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, t)
    gold = collections.defaultdict(dict)
    for line in open(EVAL / "data/qrels.tsv", encoding="utf-8").read().splitlines()[1:]:
        q, d, g, _ = line.split("\t")
        gold[q][d] = int(g)
    def primary(qid):
        return [d for d, g in gold[qid].items() if g == max(gold[qid].values())]

    wanted = set(COPIED + CONTROL)
    queries = [q for q in jl(EVAL / "data/queries.jsonl")
               if len(p := primary(q["qid"])) == 1 and p[0] in wanted]
    (DUP / "queries.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in queries),
                                       encoding="utf-8")
    print(f"файлов {sum(1 for p in files.rglob('*') if p.is_file())} в {files}; запросов {len(queries)}")


def sha256(p: str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cmd_report(a) -> None:
    resp = jl(EVAL / "runs" / a.run / "responses_hybrid.jsonl")
    c = collections.Counter()
    for r in resp:
        seen = set()
        extra = 0
        for item in r.get("results", []):
            h = sha256(item["absolute_path"])
            c["results"] += 1
            c["listed_duplicates"] += len(item.get("duplicates") or [])
            extra += h in seen
            seen.add(h)
        c["copy_slots"] += extra
        c["queries_with_copies"] += extra > 0
    print(f"прогон {a.run}: запросов {len(resp)}, мест в выдаче {c['results']}; заняты копиями уже показанного "
          f"файла {c['copy_slots']} ({c['queries_with_copies']} запросов); путей в поле duplicates "
          f"{c['listed_duplicates']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    r = sub.add_parser("report")
    r.add_argument("--run", required=True)
    a = ap.parse_args()
    {"build": cmd_build, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
