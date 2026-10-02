"""Отчёт по прогону: метрики ранжирования по сценариям 1–5 (порог «нет ответа» отложен).

    .venv/bin/python src/report.py --run R1 [--out reports/R1.md]

Разметка = data/qrels.tsv (эталон наборов) + work/judge/judged.tsv (оценки
асессоров, в т.ч. дооценка документов из выдачи). В метрики ранжирования
входят запросы, у которых есть документ с оценкой ≥ 2, реально попавший в
индекс (есть чанки в таблице прогона); остальные — отдельной строкой.
"""

import argparse
import gzip
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import METRICS, bootstrap_ci, per_query, relevant  # noqa: E402

EVAL = Path(__file__).resolve().parent.parent
DATA = EVAL / "data"
SCENARIOS = {1: "Обычный вопрос", 2: "Ключевые слова", 3: "Точные обозначения", 4: "Перефраз",
             5: "Выбор среди похожих документов", 6: "Нет ответа в корпусе"}


def corpus_rel(path: str) -> str:
    """Путь документа относительно corpus/ — прогон мог идти на другой машине."""
    return path.split("/corpus/", 1)[-1]


def load_qrels() -> dict[str, dict[str, int]]:
    q: dict[str, dict[str, int]] = defaultdict(dict)
    for l in open(DATA / "qrels.tsv", encoding="utf-8").read().splitlines()[1:]:
        qid, d, g, _ = l.split("\t")
        q[qid][d] = int(g)
    judged = EVAL / "work" / "judge" / "judged.tsv"
    if judged.exists():
        for l in judged.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            qid, d, g, _ = l.split("\t")
            q[qid].setdefault(d, int(g))  # эталон набора приоритетнее
    return q


def load_run(run_dir: Path, mode: str) -> tuple[dict[str, list[str]], dict[str, dict]]:
    ranked, raw = {}, {}
    f = run_dir / f"responses_{mode}.jsonl"
    gz = f.with_name(f.name + ".gz")  # в репозитории ответы R6 лежат сжатыми
    if not f.exists() and not gz.exists():
        return ranked, raw
    for l in (open(f, encoding="utf-8") if f.exists() else gzip.open(gz, "rt", encoding="utf-8")):
        r = json.loads(l)
        raw[r["qid"]] = r
        ranked[r["qid"]] = [corpus_rel(x["absolute_path"]) for x in r.get("results", [])]
    return ranked, raw


def fmt(point: float, lo: float, hi: float) -> str:
    return f"{point:.3f} [{lo:.3f}–{hi:.3f}]"


def table(title: str, groups: dict[str, list[str]], pq: dict, clusters: dict) -> list[str]:
    names = list(METRICS)
    out = [f"### {title}", "", "| Срез | Запросов | " + " | ".join(names) + " |", "|---|---:|" + "---:|" * len(names)]
    for g, qids in groups.items():
        if not qids:
            continue
        sub = {q: pq[q] for q in qids}
        cells = [fmt(*bootstrap_ci(sub, m, clusters)) for m in names]
        out.append(f"| {g} | {len(qids)} | " + " | ".join(cells) + " |")
    return out + [""]


MODES = {"hybrid": "Гибрид (продукт)", "text": "Только BM25 (полнотекстовый)", "vector": "Только вектор"}


def mode_comparison(runs, per_mode, unjudged_by, qrels, queries, answerable, clusters) -> list[str]:
    """Три типа поиска рядом: одни и те же запросы и разметка, разный query_type."""
    mean = lambda mode, m, qids: statistics.mean(per_mode[mode][q][m] for q in qids) if qids else 0.0
    out = ["## Сравнение типов поиска", "",
           "Одни и те же запросы и разметка, разный `query_type`. Неоценённый документ считается "
           "неправильным: где неоценённых много, nDCG, Hit@10 и Recall — нижняя граница.", "",
           "| Тип поиска | Неоценённых в топ-10 | " + " | ".join(METRICS) + " |",
           "|---|---:|" + "---:|" * len(METRICS)]
    for mode in runs:
        out.append(f"| {MODES[mode]} | {unjudged_by[mode]} | "
                   + " | ".join(f"{mean(mode, m, answerable):.3f}" for m in METRICS) + " |")
    modes = list(runs)
    out += ["", "### По сценариям: nDCG@10 и Hit@1", "",
            "| Сценарий | Запросов | " + " | ".join(f"nDCG@10 {MODES[m].split(' (')[0]}" for m in modes) + " | "
            + " | ".join(f"Hit@1 {MODES[m].split(' (')[0]}" for m in modes) + " |",
            "|---|---:|" + "---:|" * (2 * len(modes))]
    for sc in (1, 2, 3, 4, 5):
        qids = [q for q in answerable if queries[q]["scenario"] == sc]
        out.append(f"| {sc}. {SCENARIOS[sc]} | {len(qids)} | "
                   + " | ".join(f"{mean(m, 'nDCG@10', qids):.3f}" for m in modes) + " | "
                   + " | ".join(f"{mean(m, 'Hit@1', qids):.3f}" for m in modes) + " |")
    others = [m for m in modes if m != "hybrid"]
    if "hybrid" in per_mode and others:
        out += ["", "### Разница с гибридом [95% ДИ]", "",
                "| Метрика | " + " | ".join(f"{MODES[m]} − гибрид" for m in others) + " |",
                "|---|" + "---:|" * len(others)]
        for met in METRICS:
            cells = []
            for m in others:
                diff = {q: {met: per_mode[m][q][met] - per_mode["hybrid"][q][met]} for q in answerable}
                cells.append(fmt(*bootstrap_ci(diff, met, clusters)))
            out.append(f"| {met} | " + " | ".join(cells) + " |")
        out.append("")
        out.append("Интервал целиком выше нуля — режим лучше гибрида, ниже нуля — хуже, пересекает ноль — "
                   "разница не доказана.")
    if all(m in runs for m in MODES):
        h, t, v = (runs[m][0] for m in ("hybrid", "text", "vector"))
        dis = [q for q in answerable if h.get(q) and t.get(q) and v.get(q) and t[q][0] != v[q][0]]
        took_v = sum(1 for q in dis if h[q][0] == v[q][0])
        took_t = sum(1 for q in dis if h[q][0] == t[q][0])
        ok_t = sum(1 for q in dis if qrels.get(q, {}).get(t[q][0], 0) >= 2)
        ok_v = sum(1 for q in dis if qrels.get(q, {}).get(v[q][0], 0) >= 2)
        out += ["", f"Первые документы BM25 и вектора разные в {len(dis)} запросах из {len(answerable)}. "
                f"Гибрид поставил первым документ вектора в {took_v}, документ BM25 — в {took_t}. "
                f"Правильным при этом был первый документ BM25 в {ok_t} запросах, вектора — в {ok_v}."]
    return out + [""]


FTS_KEYS = ("base_tokenizer", "language", "stem", "lower_case", "max_token_length", "remove_stop_words",
            "ascii_folding", "with_position")


def fts_params(state: dict) -> str | None:
    """Параметры FTS прогона: блок emulation (src/fts_emulate.py) или details
    индекса из lance_probe."""
    if em := state.get("emulation"):
        idx = em["index"]
        d = idx["details"]
        return (f"эмуляция {em['variant']} на чанках {em['source']}, колонка {', '.join(idx['columns'])}; "
                + ", ".join(f"{k}={d.get(k)}" for k in FTS_KEYS))
    for i in state.get("probe", {}).get("indices", []):
        if i.get("type") == "FTS" and i.get("details"):
            return ", ".join(f"{k}={i['details'].get(k)}" for k in FTS_KEYS)
    return None


def compare_block(run, other, mode, ranked, o_ranked, qrels, queries, answerable, clusters,
                  slice_name, slice_qids) -> list[str]:
    """Парное сравнение одного режима: разницы метрик с кластерным бутстрэпом
    по срезам, сколько запросов стало лучше и хуже, совпавшие топ-10 и
    неоценённые документы у обоих прогонов."""
    pq, opq = per_query(ranked, qrels, answerable), per_query(o_ranked, qrels, answerable)
    diff = {q: {m: pq[q][m] - opq[q][m] for m in METRICS} for q in answerable}
    groups = {"Все": answerable}
    if slice_name:
        groups[f"Срез: {slice_name}"] = [q for q in answerable if q in slice_qids]
        groups["Вне среза"] = [q for q in answerable if q not in slice_qids]
    for sc in (1, 2, 3, 4, 5):
        groups[f"{sc}. {SCENARIOS[sc]}"] = [q for q in answerable if queries[q]["scenario"] == sc]

    def unjudged(rk, qids):
        return sum(1 for q in qids for d in rk.get(q, [])[:10] if d not in qrels.get(q, {}))

    out = [f"## Сравнение с прогоном {other} ({mode}): {run} − {other}", "",
           "Разница метрик [95% ДИ]; «лучше / хуже» — запросы, где nDCG@10 вырос или упал; "
           "неоценённые документы в топ-10 считаются неправильными.", "",
           "| Срез | Запросов | " + " | ".join(METRICS) + f" | Лучше / хуже | Совпал топ-10 | "
           f"Неоценённых {run} / {other} |",
           "|---|---:|" + "---:|" * (len(METRICS) + 3)]
    for g, qids in groups.items():
        if not qids:
            continue
        sub = {q: diff[q] for q in qids}
        better = sum(1 for q in qids if diff[q]["nDCG@10"] > 1e-9)
        worse = sum(1 for q in qids if diff[q]["nDCG@10"] < -1e-9)
        same = sum(1 for q in qids if ranked.get(q, [])[:10] == o_ranked.get(q, [])[:10])
        out.append(f"| {g} | {len(qids)} | " + " | ".join(fmt(*bootstrap_ci(sub, m, clusters)) for m in METRICS)
                   + f" | {better} / {worse} | {same} | {unjudged(ranked, qids)} / {unjudged(o_ranked, qids)} |")
    out += ["", "Интервал целиком выше нуля — прогон лучше, ниже нуля — хуже, пересекает ноль — разница не доказана.",
            ""]
    order = sorted(answerable, key=lambda q: diff[q]["nDCG@10"])
    for title, picked in (("Больше всего выиграли", order[::-1][:5]), ("Больше всего проиграли", order[:5])):
        picked = [q for q in picked if abs(diff[q]["nDCG@10"]) > 1e-9]
        if not picked:
            continue
        out += [f"{title} (nDCG@10 {run} / {other}):", ""]
        for q in picked:
            text = queries[q]["query"].replace("|", "/").replace("\n", " ")
            out.append(f"- `{q}` ({queries[q]['scenario']}) {pq[q]['nDCG@10']:.2f} / {opq[q]['nDCG@10']:.2f} — "
                       f"{text[:110]}{'…' if len(text) > 110 else ''}")
        out.append("")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out")
    ap.add_argument("--compare", action="append", default=[], help="прогон для парного сравнения")
    ap.add_argument("--slice", help='JSON {"name": ..., "qids": [...]}: в сравнении — отдельные строки '
                                    'по срезу и вне его')
    a = ap.parse_args()
    run_dir = EVAL / "runs" / a.run
    queries = {json.loads(l)["qid"]: json.loads(l) for l in open(DATA / "queries.jsonl", encoding="utf-8")}
    qrels = load_qrels()
    state = json.load(open(run_dir / "index_state.json", encoding="utf-8"))
    indexed = {corpus_rel(p) for p, n in state["probe_paths"].get("paths", {}).items() if n > 0}

    answerable, not_indexed = [], []
    for qid, q in queries.items():
        rel = relevant(qrels.get(qid, {}))
        if not rel or q["scenario"] == 6:
            continue
        (answerable if rel & indexed else not_indexed).append(qid)
    clusters = {q: queries[q]["base"] for q in queries}

    lines = [f"# Отчёт прогона {a.run}", "",
             f"Версии: lancedb {state['probe'].get('lancedb')}, pylance {state['probe'].get('pylance')}; "
             f"строк в индексе {state['probe'].get('rows')}; файлов корпуса {state['corpus_files']}, "
             f"проиндексировано файлов {len(indexed)}.", "",
             f"Запросов в метриках ранжирования: {len(answerable)}; "
             f"эталон не попал в индекс (не входят в метрики): {len(not_indexed)}.", ""]
    if fts := fts_params(state):
        lines += [f"FTS: {fts}.", ""]
    # Во всех разделах одно правило: неоценённый документ считается неправильным,
    # поэтому при многих неоценённых значения — нижняя граница.
    runs = {mode: load_run(run_dir, mode) for mode in MODES}
    runs = {mode: r for mode, r in runs.items() if r[0]}
    per_mode = {mode: per_query(r[0], qrels, answerable) for mode, r in runs.items()}
    unjudged_by = {mode: sum(1 for q in answerable for d in r[0].get(q, [])[:10] if d not in qrels.get(q, {}))
                   for mode, r in runs.items()}
    if len(runs) > 1:
        lines += mode_comparison(runs, per_mode, unjudged_by, qrels, queries, answerable, clusters)
    for mode, (ranked, raw) in runs.items():
        pq = per_mode[mode]
        unjudged = unjudged_by[mode]
        by_sc = {"Итого": answerable}
        for sc in (1, 2, 3, 4, 5):
            by_sc[f"{sc}. {SCENARIOS[sc]}"] = [q for q in answerable if queries[q]["scenario"] == sc]
        title = {"hybrid": "Продукт (hybrid, limit=10)", "vector": "Только вектор (диагностика)",
                 "text": "Только BM25, полнотекстовый (диагностика)"}[mode]
        lines += [f"## {title}", "", f"Неоценённых документов в топ-10 по запросам с ответом: {unjudged}.", ""]
        lines += table("По сценариям", by_sc, pq, clusters)
        if mode == "hybrid":
            by_ds = defaultdict(list)
            for q in answerable:
                by_ds[queries[q]["dataset"]].append(q)
            lines += table("По наборам (справочно)", dict(sorted(by_ds.items())), pq, clusters)
            errs = Counter(r["http"] for r in raw.values())
            fallback = sum(1 for r in raw.values() if r.get("results") and (r["results"][0]["score"] or 0) < 0)
            short = [len(r.get("results") or []) for r in raw.values()]
            lines += [f"HTTP-коды: {dict(errs)}; тихий фолбэк hybrid→vector (score<0): {fallback}; "
                      f"документов в ответе: медиана {statistics.median(short) if short else 0}.", ""]
    slice_name, slice_qids = None, set()
    if a.slice:
        sl = json.load(open(a.slice, encoding="utf-8"))
        slice_name, slice_qids = sl.get("name", Path(a.slice).stem), set(sl["qids"])
    for other in a.compare:
        o_state = json.load(open(EVAL / "runs" / other / "index_state.json", encoding="utf-8"))
        if o_fts := fts_params(o_state):
            lines += [f"FTS прогона {other}: {o_fts}.", ""]
        for mode in runs:
            o_ranked, _ = load_run(EVAL / "runs" / other, mode)
            if o_ranked:
                lines += compare_block(a.run, other, mode, runs[mode][0], o_ranked, qrels, queries, answerable,
                                       clusters, slice_name, slice_qids)
    text = "\n".join(lines) + "\n"
    out = Path(a.out) if a.out else EVAL / "reports" / f"{a.run}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
