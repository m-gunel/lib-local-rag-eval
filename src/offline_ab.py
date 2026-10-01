"""Офлайн-A/B «что кладём в индекс»: гибридный поиск точным перебором, без демона и без IVF-PQ.

Минуты вместо ~33 минут полного прогона; решает, стоит ли правка полного прогона.
- строки индекса: чанки R7 (work/indexer_view_corpus — ровно строки runs/R7, сверяет `seed`)
  или чанки текущих парсеров проекта (кэш parse_checks, тот же ключ версии парсера);
- поверх строк — преобразования: префикс «Документ: <имя файла без расширения>» сверху чанка
  (`--prefix add`) или снятие строк-префиксов конвейера (`--prefix strip`); эмбеддинг текста
  в нижнем регистре (`--lower`, и для документов, и для запросов; BM25 и хранимый текст те же);
- векторы — моделью проекта, кэш work/offline_ab/vec/<отпечаток эмбеддера> по (задача, текст);
  затравка — векторы R7 (`seed`), поэтому база не эмбеддится заново;
- поиск — как Indexer.search: hybrid, cosine, RRF по умолчанию, limit 10, FTS ru_stem по text;
  документы — по максимуму оценки чанка, как SearchResponse.from_dataframes. Векторного индекса
  в таблице нет — перебор точный, шум переобучения IVF-PQ сравнению не мешает;
- запросы — как есть и ЗАГЛАВНЫМИ (регистр не должен влиять на выдачу).

Запуск интерпретатором демона (модель и парсеры проекта):
  PY="env PYTHONDONTWRITEBYTECODE=1 daemon-venv/bin/python"
  $PY src/offline_ab.py seed                          # один раз: затравка кэша векторами R7 и сверки
  $PY src/offline_ab.py run B0 --rows view            # база: чанки R7
  LIB_LOCAL_RAG=<проект> $PY src/offline_ab.py run P0 --rows live
  $PY src/offline_ab.py run B0pref --rows view --prefix add
  $PY src/offline_ab.py run P2s --rows live --section  # «Раздел: …» второй строкой чанка
  $PY src/offline_ab.py report B0 P0 B0pref           # первая — база
Выдачи — work/offline_ab/runs/<имя>.json; таблицы — work/offline_ab/db (пересоздаются).
"""

import argparse
import collections
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
import parse_checks as pc  # noqa: E402  манифест, источники чанков, срезы и таблица сравнения

OUT = pc.WORK / "offline_ab"
R7_TABLE = pc.EVAL / "runs/R7/home/.files-search/lance/files.lance"
DOC_PREFIX = "Документ: "
BATCH = 8  # как embed_model_cfg.batch_size проекта
FTS_KW = dict(use_tantivy=False, tokenizer_name="ru_stem")  # как LanceWrapper.create_fts


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Embedder:
    """Векторы моделью проекта с кэшем по (отпечаток эмбеддера, задача, текст)."""

    def __init__(self, src: "pc.LiveSource"):
        self.ov = src.cfg.ov_model
        proj = src.iv.PROJECT
        mp = Path(src.cfg.embed_model_cfg.model_path)
        mp = mp if mp.is_absolute() else proj / mp
        h = hashlib.sha1()
        # отпечаток: файлы модели и код эмбеддера (там будет нижний регистр перед эмбеддингом)
        for f in [*sorted(p for p in mp.rglob("*") if p.is_file()), proj / "src/utils/openvino_encoder.py"]:
            h.update(f.name.encode())
            h.update(f.read_bytes())
        self.fp = h.hexdigest()[:16]
        self.dir = OUT / "vec" / self.fp
        self.store: dict[str, np.ndarray] = {}
        if (self.dir / "keys.npy").exists():
            keys, vecs = np.load(self.dir / "keys.npy"), np.load(self.dir / "vecs.npy")
            self.store = dict(zip(keys.tolist(), vecs))
        self.dirty = 0

    @staticmethod
    def key(text: str, task: str) -> str:
        # hex, а не байты: numpy-строки "S" теряют нулевые байты в конце дайджеста
        return hashlib.sha1(f"{task}\0{text}".encode()).hexdigest()

    def save(self):
        if not self.dirty:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self.store), dtype="U40")
        np.save(self.dir / "vecs.tmp.npy", np.stack(list(self.store.values())).astype(np.float32))
        np.save(self.dir / "keys.tmp.npy", keys)
        (self.dir / "vecs.tmp.npy").replace(self.dir / "vecs.npy")
        (self.dir / "keys.tmp.npy").replace(self.dir / "keys.npy")
        self.dirty = 0

    def put(self, texts, vecs, task):
        for t, v in zip(texts, vecs):
            self.store[self.key(t, task)] = np.asarray(v, np.float32)
        self.dirty += len(texts)

    def get(self, texts: list[str], task: str) -> np.ndarray:
        keys = [self.key(t, task) for t in texts]
        miss = list({k: t for k, t in zip(keys, texts) if k not in self.store}.items())
        if miss:
            log(f"эмбеддинг {len(miss)} новых текстов ({task}), в кэше {len(self.store)}")
            t0 = time.time()
            for i in range(0, len(miss), BATCH):
                part = miss[i:i + BATCH]
                vecs = self.ov.encode([t for _, t in part], task=task)
                for (k, _), v in zip(part, vecs):
                    self.store[k] = np.asarray(v, np.float32)
                self.dirty += len(part)
                if (i // BATCH) % 500 == 499:
                    self.save()
                    log(f"  {i + BATCH}/{len(miss)} за {time.time() - t0:.0f} c")
            self.save()
            log(f"эмбеддинг готов за {time.time() - t0:.0f} c")
        return np.stack([self.store[k] for k in keys])


# ---------- строки индекса ----------

def rows_of(a) -> list[tuple[str, str]]:
    """(doc_id, text) по всем документам манифеста, в порядке манифеста и чанков."""
    src = pc.LiveSource() if a.rows == "live" else pc.ViewSource()
    rows = []
    for m in pc.MANIFEST:
        chunks, _, _, _ = src.chunks(m)
        stem = Path(m["path"]).stem
        for c in chunks:
            t = c["text"]
            if a.prefix == "strip":
                t = pc.own_text(t)
            elif a.prefix == "add" and not t.startswith(DOC_PREFIX):
                t = f"{DOC_PREFIX}{stem}\n{t}"
            if a.section and c.get("section"):  # «Раздел: …» второй строкой, после «Документ: …»
                head, _, rest = t.partition("\n") if t.startswith(DOC_PREFIX) else ("", "", t)
                t = (head + "\n" if head else "") + f"Раздел: {c['section']}\n" + rest
            rows.append((m["doc_id"], t))
    return rows


def build_table(name: str, rows, vecs):
    import lancedb
    import pyarrow as pa

    db = lancedb.connect(str(OUT / "db"))
    data = pa.table({"path": [d for d, _ in rows], "text": [t for _, t in rows],
                     "vector": pa.FixedSizeListArray.from_arrays(pa.array(vecs.ravel()), vecs.shape[1])})
    t = db.create_table(name, data, mode="overwrite")
    t.create_fts_index("text", replace=True, **FTS_KW)
    return t


def search(table, text_query: str, vec) -> tuple[list[str], bool]:
    """Как Indexer.search (гибрид; при ошибке — вектор) + порядок документов SearchResponse."""
    try:
        res = (table.search(query_type="hybrid", vector_column_name="vector").distance_type("cosine")
               .vector(vec).text(text_query).limit(10).to_arrow())
        scores, fallback = res.column("_relevance_score").to_pylist(), False
    except Exception:
        res = (table.search(vec, query_type="vector", vector_column_name="vector").distance_type("cosine")
               .limit(10).to_arrow())
        scores, fallback = [-d for d in res.column("_distance").to_pylist()], True
    best: dict[str, float] = {}
    for p, s in zip(res.column("path").to_pylist(), scores):
        best[p] = max(best.get(p, float("-inf")), s)
    return [p for p, _ in sorted(best.items(), key=lambda x: -x[1])], fallback


# ---------- команды ----------

def cmd_seed(a):
    import lance

    src = pc.LiveSource()
    emb = Embedder(src)
    tb = lance.dataset(str(R7_TABLE)).to_table(columns=["path", "text", "vector"])  # только чтение
    paths = [p.split("/corpus/", 1)[-1] for p in tb.column("path").to_pylist()]
    texts = tb.column("text").to_pylist()
    vecs = np.asarray(tb.column("vector").combine_chunks().flatten().to_numpy(zero_copy_only=False),
                      np.float32).reshape(len(texts), -1)
    log(f"строк R7: {len(texts)}, документов {len(set(paths))}; отпечаток эмбеддера {emb.fp}")
    sample = random.Random(0).sample(range(len(texts)), 64)
    fresh = np.vstack([emb.ov.encode([texts[i] for i in sample[j:j + BATCH]], task="document")
                       for j in range(0, len(sample), BATCH)])
    cos = (fresh * vecs[sample]).sum(1) / (np.linalg.norm(fresh, axis=1) * np.linalg.norm(vecs[sample], axis=1))
    log(f"модель проекта против векторов R7 на 64 случайных чанках: косинус min {cos.min():.6f}, "
        f"медиана {np.median(cos):.6f}")
    if cos.min() < 0.9999:
        raise SystemExit("векторы R7 получены другой моделью — затравка не годится, кэш не тронут")
    emb.put(texts, vecs, "document")
    emb.save()
    r7 = collections.Counter(zip(paths, texts))
    view = collections.Counter((m["doc_id"], c["text"]) for m in pc.MANIFEST for c in pc.ViewSource().chunks(m)[0])
    log(f"чанков view {sum(view.values())}, строк R7 {sum(r7.values())}, совпадают {sum((r7 & view).values())}; "
        f"в кэше {len(emb.store)} векторов")


def cmd_run(a):
    t0 = time.time()
    src = pc.LiveSource()  # модель проекта (и парсеры для --rows live)
    emb = Embedder(src)
    rows = rows_of(a)
    emb_in = [t.lower() for _, t in rows] if a.lower else [t for _, t in rows]
    vecs = emb.get(emb_in, "document")
    table = build_table(a.name, rows, vecs)
    log(f"{a.name}: строк {len(rows)}, документов {len({d for d, _ in rows})}, таблица за {time.time() - t0:.0f} c")
    queries = pc.jl(pc.DATA / "queries.jsonl")
    out = {"args": vars(a), "project": str(src.iv.PROJECT), "embedder": emb.fp, "parser_keys": src.keys,
           "rows": len(rows), "doc_ids": sorted({d for d, _ in rows})}
    fallbacks = 0
    for case, f in (("normal", str), ("upper", str.upper)):
        qtext = [f(q["query"]) for q in queries]
        qvec = emb.get([t.lower() for t in qtext] if a.lower else qtext, "query")
        res = {}
        for q, t, v in zip(queries, qtext, qvec):
            res[q["qid"]], fb = search(table, t, v)
            fallbacks += fb
        out[case] = res
    out["fallbacks"], out["seconds"] = fallbacks, round(time.time() - t0)
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    (OUT / "runs" / f"{a.name}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    log(f"{a.name}: {len(queries)} запросов ×2 (как есть, ЗАГЛАВНЫМИ), откатов на вектор {fallbacks}, "
        f"всего {out['seconds']} c")


def cmd_report(a):
    from metrics import per_query, relevant
    from report import load_qrels

    queries = {q["qid"]: q for q in pc.jl(pc.DATA / "queries.jsonl")}
    qrels = load_qrels()
    runs = {n: json.loads((OUT / "runs" / f"{n}.json").read_text(encoding="utf-8")) for n in a.names}
    ans_by = {n: {q for q in queries if queries[q]["scenario"] != 6 and relevant(qrels.get(q, {})) & set(r["doc_ids"])}
              for n, r in runs.items()}
    ans = set.intersection(*ans_by.values())
    base_name = a.names[0]
    base = runs[base_name]
    print("Запросов с эталоном в индексе: " + ", ".join(f"{n} {len(s)}" for n, s in ans_by.items())
          + f"; в сравнении — пересечение, {len(ans)}")
    print("\n| Вариант | Строк | Документов | Откатов на вектор | Неоценённых в топ-10 | Топ-10 как у базы |")
    print("|---|---:|---:|---:|---:|---:|")
    for n, r in runs.items():
        unj = sum(1 for q in ans for d in r["normal"].get(q, [])[:10] if d not in qrels.get(q, {}))
        same = sum(r["normal"][q] == base["normal"][q] for q in queries)
        print(f"| {n} | {r['rows']} | {len(r['doc_ids'])} | {r['fallbacks']} | {unj} | {same}/{len(queries)} |")
    print("\nРегистр: запросы как есть и ЗАГЛАВНЫМИ (Итого)\n")
    print("| Вариант | nDCG@10 как есть | nDCG@10 ЗАГЛАВНЫМИ | Hit@1 как есть | Hit@1 ЗАГЛАВНЫМИ | топ-10 совпал |")
    print("|---|---:|---:|---:|---:|---:|")
    for n, r in runs.items():
        pn, pu = per_query(r["normal"], qrels, sorted(ans)), per_query(r["upper"], qrels, sorted(ans))
        mean = lambda pq, m: statistics.mean(pq[q][m] for q in ans)  # noqa: E731
        same = sum(r["normal"][q] == r["upper"][q] for q in queries)
        print(f"| {n} | {mean(pn, 'nDCG@10'):.3f} | {mean(pu, 'nDCG@10'):.3f} | {mean(pn, 'Hit@1'):.3f} | "
              f"{mean(pu, 'Hit@1'):.3f} | {same}/{len(queries)} |")
    for n in a.names[1:]:
        print(f"\n### {n} против {base_name}, запросы как есть\n")
        print("\n".join(pc.compare_lines(runs[n]["normal"], base["normal"], ans, n, base_name, a.metrics.split(","))))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seed")
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("--rows", choices=("view", "live"), default="view")
    r.add_argument("--prefix", choices=("keep", "add", "strip"), default="keep")
    r.add_argument("--lower", action="store_true", help="эмбеддинг текста документов и запросов в нижнем регистре")
    r.add_argument("--section", action="store_true", help="строка «Раздел: …» в тексте чанка (live: поле section)")
    p = sub.add_parser("report")
    p.add_argument("names", nargs="+")
    p.add_argument("--metrics", default="nDCG@10,Hit@1,Hit@10")
    a = ap.parse_args()
    {"seed": cmd_seed, "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
