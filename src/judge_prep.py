"""Подготовка пакетов для асессоров (оценка 0–3 пары «запрос — документ»).

Асессор видит: информационную потребность (для вариантов сценариев 2–4 — полный
базовый вопрос, чтобы короткий запрос оценивался по тому же смыслу), название
документа и до 5 фрагментов: до двух из тех, что вернул индексатор (если пара из
выдачи), и три лучших по независимому BM25. Позицию в выдаче и систему он не видит.

    .venv/bin/python src/judge_prep.py noanswer              — пул запросов сценария 6
    .venv/bin/python src/judge_prep.py postrun --run R1      — неоценённые документы из выдачи
Пакеты: work/judge/<раунд>/a_XXX.json (первичная оценка) и b_XXX.json
(независимая вторая оценка ~20% пар в другой разбивке).
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pool import passages  # noqa: E402

EVAL = Path(__file__).resolve().parent.parent
DATA = EVAL / "data"
JUDGE = EVAL / "work" / "judge"
BATCH = 30
DOUBLE_SHARE = 0.2
SEED = 20260925


def load_queries() -> dict[str, dict]:
    return {json.loads(l)["qid"]: json.loads(l) for l in open(DATA / "queries.jsonl", encoding="utf-8")}


def load_qrels() -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    lines = open(DATA / "qrels.tsv", encoding="utf-8").read().splitlines()[1:]
    for l in lines:
        q, d, g, _ = l.split("\t")
        out.setdefault(q, {})[d] = int(g)
    return out


def need_text(q: dict, queries: dict[str, dict]) -> str:
    """Информационная потребность: для вариантов — базовый вопрос."""
    if q["scenario"] in (2, 3, 4) and q["base"] in queries:
        return queries[q["base"]]["query"]
    return q["query"]


def write_batches(round_name: str, pairs: list[dict], batch: int = BATCH) -> None:
    out = JUDGE / round_name
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.json"):
        if old.name.startswith(("a_", "b_")):
            old.unlink()
    rnd = random.Random(SEED)
    for i, p in enumerate(pairs):
        p["pid"] = f"{round_name}-{i:05d}"
    for n, i in enumerate(range(0, len(pairs), batch)):
        (out / f"a_{n:03d}.json").write_text(json.dumps(pairs[i:i + batch], ensure_ascii=False, indent=0), encoding="utf-8")
    double = rnd.sample(pairs, max(1, int(len(pairs) * DOUBLE_SHARE))) if pairs else []
    rnd.shuffle(double)
    for n, i in enumerate(range(0, len(double), batch)):
        (out / f"b_{n:03d}.json").write_text(json.dumps(double[i:i + batch], ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{round_name}: пар {len(pairs)}, пакетов a={-(-len(pairs) // batch)}, двойная оценка {len(double)}")


def manifest() -> dict[str, dict]:
    return {json.loads(l)["doc_id"]: json.loads(l) for l in open(DATA / "manifest.jsonl", encoding="utf-8")}


def cmd_noanswer() -> None:
    queries = load_queries()
    pool = json.load(open(EVAL / "work" / "pool" / "noanswer_pool.json", encoding="utf-8"))
    pairs = [{"qid": p["qid"], "need": need_text(queries[p["qid"]], queries), "doc_id": p["doc_id"],
              "title": p["title"], "passages": p["passages"]} for p in pool]
    write_batches("noanswer", pairs)


def cmd_postrun(runs: list[str], modes: list[str], batch: int) -> None:
    queries = load_queries()
    qrels = load_qrels()
    man = manifest()
    judged = {(q, d) for q, ds in qrels.items() for d in ds}
    judged_file = JUDGE / "judged.tsv"  # уже оценённые в прошлых раундах
    if judged_file.exists():
        for l in open(judged_file, encoding="utf-8"):
            q, d, *_ = l.rstrip("\n").split("\t")
            judged.add((q, d))
    # Уже оценённые пары — эталон наборов и work/judge/judged.tsv (туда
    # judge_merge.py merge сводит оценки всех раундов). Пары прерванных
    # раундов без оценки сюда не попадают и будут отданы асессорам снова.
    todo: dict[tuple[str, str], list[str]] = {}
    # Полностью дооценивается выдача продукта (hybrid); для диагностических
    # режимов vector/text метрики считаются по «сжатым» спискам (неоценённые
    # документы убираются), см. report.py.
    files = [EVAL / "runs" / run / f"responses_{m}.jsonl" for run in runs for m in modes]
    for f in files:
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            for item in r.get("results", [])[:10]:
                d = item["absolute_path"].split("/corpus/", 1)[-1]
                if (r["qid"], d) in judged or d not in man:
                    continue
                chunks = [c["content"] for c in item.get("chunks", [])][:2]
                todo.setdefault((r["qid"], d), [])
                for c in chunks:
                    if c not in todo[(r["qid"], d)]:
                        todo[(r["qid"], d)].append(c)
    pairs = []
    # По документу подряд — кэш BM25 документа в pool.doc_index попадает.
    for (q, d), sys_chunks in sorted(todo.items(), key=lambda x: (x[0][1], x[0][0])):
        need = need_text(queries[q], queries)
        pairs.append({"qid": q, "need": need, "doc_id": d, "title": man[d]["title"],
                      "passages": passages(need, d, 3, extra=sys_chunks[:2])})
    # Пакеты асессорам — вперемешку (иначе в пакете один документ подряд).
    random.Random(SEED).shuffle(pairs)
    write_batches("postrun_" + "_".join(runs), pairs, batch)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["noanswer", "postrun"])
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--modes", default="hybrid")
    ap.add_argument("--batch", type=int, default=BATCH)
    a = ap.parse_args()
    if a.mode == "noanswer":
        cmd_noanswer()
    else:
        cmd_postrun(a.run, a.modes.split(","), a.batch)


if __name__ == "__main__":
    main()
