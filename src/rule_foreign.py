"""Оценка 0 по правилу для «очевидно чужих» пар из выдачи прогона.

Документ из набора другой тематики, чем запрос (например, ZX Bank на вопрос
к ЦБ), получает 0 без асессора. Проверка на 3877 уже оценённых чужих парах:
правильными (≥ 2) оказались 25 (0,6%). Пары пишутся в work/judge/judged.tsv
с раундом rule_foreign, чтобы их можно было отличить от оценок асессоров.

    .venv/bin/python src/rule_foreign.py --run R6 --modes text,vector
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report import load_qrels, load_run  # noqa: E402

EVAL = Path(__file__).resolve().parent.parent
CBR = {"cbr_acts", "cbr_explan", "cbr_faq", "cbr_analytics", "cbr_stat"}
OWN = {"cbr": CBR, "cbr_faq": CBR, "xlsx": CBR, "ai360": {"ai360"}, "zx": {"zx_bank"}, "obliqa": {"adgm"}}
ROUND = "rule_foreign"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--modes", default="text,vector")
    a = ap.parse_args()
    queries = {json.loads(l)["qid"]: json.loads(l) for l in open(EVAL / "data" / "queries.jsonl", encoding="utf-8")}
    qrels = load_qrels()
    new = set()
    for run in a.run:
        for mode in a.modes.split(","):
            ranked, _ = load_run(EVAL / "runs" / run, mode)
            for qid, docs in ranked.items():
                if qid not in queries:
                    continue
                own = OWN[queries[qid]["dataset"]]
                for d in docs[:10]:
                    if d not in qrels.get(qid, {}) and d.split("/")[0] not in own:
                        new.add((qid, d))
    judged = EVAL / "work" / "judge" / "judged.tsv"
    with judged.open("a", encoding="utf-8") as f:
        for qid, d in sorted(new):
            f.write(f"{qid}\t{d}\t0\t{ROUND}\n")
    print(f"оценено 0 по правилу: {len(new)} пар")


if __name__ == "__main__":
    main()
