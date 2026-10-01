"""Леммы для варианта FTS_V2 (лемматизация вместо стемминга Snowball).

    .venv/bin/python src/fts_lemma.py

Читает work/fts/chunks.jsonl (выгрузка src/fts_emulate.py --dump-chunks) и
data/queries.jsonl, пишет work/fts/lemma_chunks.jsonl (тот же порядок чанков) и
work/fts/lemma_queries.jsonl. Правила:
- токены — [^\\W_]+, как base_tokenizer=simple у Lance;
- слово из одной кириллицы → нормальная форма pymorphy3, затем ё→е;
- слово из одной латиницы → Snowball English;
- остальное (числа, смешанные токены) — в нижнем регистре как есть.
Леммы склеиваются пробелом: индекс по ним строится с base_tokenizer=whitespace.
"""

import json
import re
import sys
import time
from functools import cache
from pathlib import Path

import pymorphy3
import snowballstemmer

EVAL = Path(__file__).resolve().parent.parent
WORK = EVAL / "work" / "fts"
TOKEN = re.compile(r"[^\W_]+")
CYR = re.compile(r"[а-яё]+")
LAT = re.compile(r"[a-z]+")

morph = pymorphy3.MorphAnalyzer()
english = snowballstemmer.stemmer("english")


@cache
def lemma(token: str) -> str:
    t = token.lower()
    if CYR.fullmatch(t):
        return morph.parse(t)[0].normal_form.replace("ё", "е")
    if LAT.fullmatch(t):
        return english.stemWord(t)
    return t


def lemmatize(text: str) -> str:
    return " ".join(lemma(t) for t in TOKEN.findall(text))


def checks() -> bool:
    cases = [
        (["Счёт", "СЧЕТА", "счета", "счет"], True),
        (["кредит", "кредита", "кредиты", "кредитов"], True),
        (["депозит", "депозиты"], True),
        (["капитал", "капиталы"], True),
        (["актив", "акт"], False),
        (["reports", "report"], True),
    ]
    ok = True
    for words, same in cases:
        got = [lemmatize(w) for w in words]
        passed = (len(set(got)) == 1) == same
        ok &= passed
        print(f"{'OK ' if passed else 'НЕТ'} {' / '.join(words)} → {' / '.join(got)}"
              f" ({'должны совпасть' if same else 'должны различаться'})")
    return ok


def main() -> None:
    if not checks():
        sys.exit("контрольные примеры не прошли")
    t = time.perf_counter()
    n = 0
    with (WORK / "chunks.jsonl").open(encoding="utf-8") as src, \
            (WORK / "lemma_chunks.jsonl").open("w", encoding="utf-8") as out:
        for line in src:
            r = json.loads(line)
            out.write(json.dumps({"path": r["path"], "idx": r["idx"], "text_fts": lemmatize(r["text"])},
                                 ensure_ascii=False) + "\n")
            n += 1
    with (EVAL / "data" / "queries.jsonl").open(encoding="utf-8") as src, \
            (WORK / "lemma_queries.jsonl").open("w", encoding="utf-8") as out:
        for line in src:
            q = json.loads(line)
            out.write(json.dumps({"qid": q["qid"], "lemma": lemmatize(q["query"])}, ensure_ascii=False) + "\n")
    print(f"чанков {n}, разных слов {lemma.cache_info().currsize}, {time.perf_counter() - t:.0f} с → {WORK}")


if __name__ == "__main__":
    main()
