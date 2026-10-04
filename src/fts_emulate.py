"""Эмуляция поиска демона на чанках готового прогона: варианты полнотекстового индекса.

    daemon-venv/bin/python src/fts_emulate.py --dump-chunks work/fts/chunks.jsonl
    daemon-venv/bin/python src/fts_emulate.py --long-tokens-out work/fts/long_token_qids.json
    daemon-venv/bin/python src/fts_emulate.py --variants V0,V1,V2 [--check-against R7]
    daemon-venv/bin/python src/fts_emulate.py --source P3 --work work/fts/P3 --run-prefix FTS_P3_ \\
        --modes text,hybrid --variants V1,V2 --check-against P3

Прогон-источник (--source, по умолчанию R7) только читается: его чанки копируются
в таблицу memory://, на ней строится FTS варианта, запросы идут так же, как в
демоне (Index.search: text — search(q, query_type="fts"), hybrid — vector+text с
RRF; limit 10, при копиях файла — добор с запасом), выдача собирается кодом проекта
(src/server/model.py, SearchResponse.from_dataframes). Результат —
runs/<--run-prefix><вариант>/ в формате harness.py: его читают report.py,
rule_foreign.py и judge_prep.py.

hybrid — точным перебором векторов (векторного индекса в таблице нет): варианты
сравниваются без шума переобучения IVF-PQ, но с выдачей демона совпадают не
полностью. Векторы чанков — из таблицы источника, векторы запросов — из кэша
src/offline_ab.py (тот же эмбеддер проекта).

V2 (лемматизация) берёт леммы из <--work>/lemma_{chunks,queries}.jsonl — их пишет
src/fts_lemma.py в .venv стенда (pymorphy3 в daemon-venv не ставим: там версии
из uv.lock проекта).
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import lance
import lancedb
import numpy as np
import pyarrow as pa
from lancedb.index import FTS

EVAL = Path(__file__).resolve().parent.parent
PROJECT = Path(os.environ.get("LIB_LOCAL_RAG", "/Users/gunel30/Downloads/pip_local_rag_0110"))
DATA = EVAL / "data"
LIMIT = 10  # продуктовый limit, как в harness.py

# Пресет ru_stem (lancedb infer_tokenizer_configs) — то, что строил демон до 1.0.8.
RU_STEM = dict(base_tokenizer="simple", language="Russian", stem=True, lower_case=True,
               max_token_length=40, remove_stop_words=False, ascii_folding=False, with_position=False)
VARIANTS = {
    "V0": {"column": "text", "preset": "ru_stem", "fts": RU_STEM},
    "V1": {"column": "text", "fts": {**RU_STEM, "max_token_length": 100}},
    # Леммы уже в нижнем регистре и с ё→е; стемминг выключен — его заменила лемматизация.
    "V2": {"column": "text_fts", "fts": {**RU_STEM, "base_tokenizer": "whitespace", "stem": False,
                                         "max_token_length": 100}},
}


def load_model():
    """src/server/model.py по пути: импорт пакета src.server тянет src.config,
    а тот при импорте создаёт ~/.files-search. Сам model.py импортирует только
    лёгкий src.common — для него корень проекта в sys.path."""
    sys.path.insert(0, str(PROJECT))
    spec = importlib.util.spec_from_file_location("lrag_server_model", PROJECT / "src" / "server" / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dup_fetch_factor() -> int:
    """Запас выборки при копиях файла (Index.search, с 1.0.8); у старого кода — 1."""
    m = re.search(r"^DUP_FETCH_FACTOR\s*=\s*(\d+)", (PROJECT / "src/indexer/indexer.py").read_text(), re.M)
    return int(m.group(1)) if m else 1


def has_copies(frame) -> bool:
    """Как _has_copies в src/indexer/indexer.py: разные пути одного sha256."""
    if frame is None or "meta" not in frame.columns:
        return False
    first: dict[str, str] = {}
    for meta, path in zip(frame["meta"].to_list(), frame["path"].to_list()):
        sha = (meta or {}).get("sha256")
        if sha and first.setdefault(sha, path) != path:
            return True
    return False


def source_table(run: str):
    return lancedb.connect(str(EVAL / "runs" / run / "home" / ".files-search" / "lance")).open_table("files")


def source_chunks(run: str, with_vectors: bool) -> pa.Table:
    t = source_table(run).to_arrow()
    return t if with_vectors else t.drop_columns(["vector"])


def source_fts(run: str) -> dict:
    for i in source_table(run).list_indices():
        if i.index_type == "FTS":
            return i.index_details or {}
    return {}


def query_vectors(queries: list[dict]) -> dict[str, np.ndarray]:
    """Векторы запросов из кэша src/offline_ab.py: ключ — отпечаток эмбеддера проекта
    (файлы модели и openvino_encoder.py) и sha1(«query\\0текст»)."""
    mp = PROJECT / "models" / "rubert-tiny2_002"
    h = hashlib.sha1()
    for f in [*sorted(p for p in mp.rglob("*") if p.is_file()), PROJECT / "src/utils/openvino_encoder.py"]:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    d = EVAL / "work" / "offline_ab" / "vec" / h.hexdigest()[:16]
    if not (d / "keys.npy").exists():
        raise SystemExit(f"нет кэша векторов {d}: сначала src/offline_ab.py run (он эмбеддит запросы)")
    store = dict(zip(np.load(d / "keys.npy").tolist(), np.load(d / "vecs.npy")))
    out = {}
    for q in queries:
        k = hashlib.sha1(f"query\0{q['query']}".encode()).hexdigest()
        if k not in store:
            raise SystemExit(f"в кэше {d} нет вектора запроса {q['qid']}: обновите кэш src/offline_ab.py run")
        out[q["qid"]] = store[k]
    return out


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def dump_chunks(chunks: pa.Table, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for path, idx, text in zip(*(chunks.column(c).to_pylist() for c in ("path", "idx", "text"))):
            f.write(json.dumps({"path": path, "idx": idx, "text": text}, ensure_ascii=False) + "\n")
    print(f"чанков выгружено: {chunks.num_rows} → {out}")


def long_token_slice(queries: list[dict], out: Path) -> None:
    """Запросы, у которых токен выпадает из-за max_token_length=40 (байт UTF-8)."""
    params = {k: v for k, v in RU_STEM.items() if k != "with_position"}
    qids = []
    for q in queries:
        cut = lance.tokenize(q["query"], **params)
        full = lance.tokenize(q["query"], **{**params, "max_token_length": None})
        if [t.text for t in cut] != [t.text for t in full]:
            qids.append(q["qid"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"name": "токен ≥40 байт UTF-8 в запросе", "qids": qids},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"запросов с длинным токеном: {len(qids)} из {len(queries)} "
          f"({100 * len(qids) / len(queries):.1f}%) → {out}")


def lemma_inputs(chunks: pa.Table, work: Path) -> tuple[pa.Table, dict[str, str]]:
    rows = read_jsonl(work / "lemma_chunks.jsonl")
    keys = list(zip(chunks.column("path").to_pylist(), chunks.column("idx").to_pylist()))
    if [(r["path"], r["idx"]) for r in rows] != keys:
        raise SystemExit("lemma_chunks.jsonl не совпадает с чанками источника: перевыгрузите "
                         "--dump-chunks и перезапустите src/fts_lemma.py")
    chunks = chunks.append_column("text_fts", pa.array([r["text_fts"] for r in rows], pa.string()))
    lq = {r["qid"]: r["lemma"] for r in read_jsonl(work / "lemma_queries.jsonl")}
    return chunks, lq


def build(chunks: pa.Table, name: str, spec: dict) -> tuple[object, dict, float]:
    db = lancedb.connect("memory://")
    table = db.create_table(f"files_{name}", data=chunks)
    t = time.perf_counter()
    if spec.get("preset"):
        table.create_fts_index(spec["column"], use_tantivy=False, tokenizer_name=spec["preset"], replace=True)
    else:
        table.create_index(spec["column"], config=FTS(**spec["fts"]), replace=True)
    seconds = round(time.perf_counter() - t, 2)
    [idx] = [i for i in table.list_indices() if i.index_type == "FTS"]
    details = idx.index_details or {}
    wrong = {k: (details.get(k), v) for k, v in spec["fts"].items() if details.get(k) != v}
    if wrong:
        raise SystemExit(f"{name}: параметры индекса не те, что заданы (в индексе, задано): {wrong}")
    return table, {"name": idx.name, "columns": list(idx.columns), "details": details}, seconds


def run_queries(table, model, queries: list[dict], mode: str, column: str, lemmas: dict[str, str] | None,
                qvecs: dict[str, np.ndarray] | None, out: Path) -> None:
    factor = dup_fetch_factor()
    collect = model.SearchResponse.from_dataframes
    kw = {"limit": LIMIT} if "limit" in inspect.signature(collect).parameters else {}

    def search(text: str, qid: str, limit: int):
        if mode == "text":
            q = table.search(text, query_type="fts", fts_columns=column)
        else:  # как Index.search: hybrid, cosine, RRF по умолчанию
            q = (table.search(query_type="hybrid", vector_column_name="vector", fts_columns=column)
                 .distance_type("cosine").vector(qvecs[qid]).text(text))
        return q.limit(limit).to_polars()

    with out.open("w", encoding="utf-8") as f:
        for q in queries:
            text = lemmas[q["qid"]] if lemmas is not None else q["query"]
            t = time.perf_counter()
            rec = {"qid": q["qid"], "mode": mode}
            try:
                df = search(text, q["qid"], LIMIT)
                if factor > 1 and has_copies(df):
                    df = search(text, q["qid"], LIMIT * factor)
                results = collect([df], **kw).model_dump(mode="json")["results"]
                rec.update(http=200, ms=round((time.perf_counter() - t) * 1000, 1), results=results)
            except Exception as e:  # как 500 демона: запрос в метриках идёт с пустой выдачей
                rec.update(http=500, ms=round((time.perf_counter() - t) * 1000, 1), error=repr(e)[:500])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def check_against(run_dir: Path, ref: str) -> bool:
    """Сверка text с выдачей демона: пути и оценки файлов (допуск 1e-6).

    Допустимы только расхождения среди равных оценок BM25 — их порядок зависит
    от адресов строк: другой порядок файлов с равной оценкой и другой файл на
    границе limit, когда 10-й и 11-й чанки равны (одинаковые таблицы xlsx,
    отчёты соседних лет)."""
    def load(p: Path) -> dict[str, dict]:
        return {r["qid"]: r for r in read_jsonl(p)}

    def files(r: dict) -> list[tuple[str, float]]:
        return [(x["absolute_path"], x["score"]) for x in r.get("results", [])]

    def chunk_scores(r: dict) -> list[float]:
        return sorted(c["score"] for x in r.get("results", []) for c in x["chunks"])

    mine, theirs = load(run_dir / "responses_text.jsonl"), load(EVAL / "runs" / ref / "responses_text.jsonl")
    exact = ties = cut = 0
    bad = []
    for qid, rb in theirs.items():
        ra = mine.get(qid, {})
        a, b = files(ra), files(rb)
        sa, sb = chunk_scores(ra), chunk_scores(rb)
        same_chunks = len(sa) == len(sb) and all(abs(x - y) <= 1e-6 for x, y in zip(sa, sb))
        same_files = len(a) == len(b) and all(abs(x[1] - y[1]) <= 1e-6 for x, y in zip(a, b))
        if same_files and [x[0] for x in a] == [y[0] for y in b]:
            exact += 1
        elif same_files and _same_up_to_ties(a, b):
            ties += 1
        elif same_chunks and sa and all(abs(s - sa[0]) <= 1e-6 for _, s in set(a) ^ set(b)):
            cut += 1
        else:
            bad.append(qid)
    print(f"сверка text с {ref}: точно {exact}, другой порядок равных оценок {ties}, равные оценки на "
          f"границе limit {cut}, расходятся {len(bad)} из {len(theirs)}"
          + (f"; первые: {bad[:10]}" if bad else ""))
    return not bad


def overlap(run_dir: Path, ref: str, mode: str) -> None:
    """Близость к выдаче демона там, где точного совпадения не ждём (hybrid: IVF-PQ у демона)."""
    def paths(p: Path) -> dict[str, list[str]]:
        return {r["qid"]: [x["absolute_path"] for x in r.get("results", [])] for r in read_jsonl(p)}

    a, b = paths(run_dir / f"responses_{mode}.jsonl"), paths(EVAL / "runs" / ref / f"responses_{mode}.jsonl")
    first = sum(1 for q in b if a.get(q, [])[:1] == b[q][:1])
    same = sum(1 for q in b if set(a.get(q, [])) == set(b[q]))
    print(f"{mode} против {ref}: первый документ совпал в {first}, набор топ-10 — в {same} из {len(b)}")


def _same_up_to_ties(a: list[tuple[str, float]], b: list[tuple[str, float]]) -> bool:
    """Одинаковые наборы файлов с одинаковыми оценками, порядок может разниться
    только внутри групп равных оценок."""
    def groups(xs):
        out: list[tuple[float, set[str]]] = []
        for path, score in xs:
            if out and abs(out[-1][0] - score) <= 1e-6:
                out[-1][1].add(path)
            else:
                out.append((score, {path}))
        return [g[1] for g in out]

    return groups(a) == groups(b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="R7", help="прогон, чей индекс берём как чанки (только чтение)")
    ap.add_argument("--variants", default="", help="через запятую: " + ",".join(VARIANTS))
    ap.add_argument("--modes", default="text", help="text и/или hybrid через запятую")
    ap.add_argument("--run-prefix", default="FTS_", help="прогоны пишутся в runs/<префикс><вариант>")
    ap.add_argument("--work", type=Path, default=EVAL / "work" / "fts", help="каталог лемм V2")
    ap.add_argument("--queries", default=str(DATA / "queries.jsonl"))
    ap.add_argument("--check-against", help="сверить выдачу с этим прогоном: text — точно для варианта с "
                                            "параметрами FTS источника, hybrid — близость")
    ap.add_argument("--dump-chunks", type=Path, help="выгрузить чанки источника для лемматизатора")
    ap.add_argument("--long-tokens-out", type=Path, help="срез запросов с токеном ≥40 байт")
    ap.add_argument("--force", action="store_true", help="перезаписать существующие прогоны")
    a = ap.parse_args()
    queries = read_jsonl(Path(a.queries))
    if a.long_tokens_out:
        long_token_slice(queries, a.long_tokens_out)
    variants = [v for v in a.variants.split(",") if v]
    modes = [m for m in a.modes.split(",") if m]
    if not variants and not a.dump_chunks:
        return
    chunks = source_chunks(a.source, with_vectors="hybrid" in modes)
    if a.dump_chunks:
        dump_chunks(chunks, a.dump_chunks)
    if not variants:
        return
    model = load_model()
    qvecs = query_vectors(queries) if "hybrid" in modes else None
    src_state = json.loads((EVAL / "runs" / a.source / "index_state.json").read_text(encoding="utf-8"))
    src_fts = source_fts(a.source)
    ok = True
    for name in variants:
        spec = VARIANTS[name]
        run_id = f"{a.run_prefix}{name}"
        run_dir = EVAL / "runs" / run_id
        if run_dir.exists() and any(run_dir.iterdir()):
            if not a.force:
                raise SystemExit(f"{run_dir} уже есть: возьмите --force, чтобы перезаписать")
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        data, lemmas = (lemma_inputs(chunks, a.work) if spec["column"] == "text_fts" else (chunks, None))
        table, index, seconds = build(data, name, spec)
        for mode in modes:
            t = time.perf_counter()
            run_queries(table, model, queries, mode, spec["column"], lemmas, qvecs,
                        run_dir / f"responses_{mode}.jsonl")
            print(f"{name} {mode}: индекс {seconds} с, запросы {time.perf_counter() - t:.1f} с → {run_dir}")
        state = {**src_state, "run_id": run_id,
                 "emulation": {"source": a.source, "variant": name, "fts": spec["fts"], "modes": modes,
                               "preset": spec.get("preset"), "index": index, "build_seconds": seconds,
                               "hybrid": "точный перебор векторов, без IVF-PQ" if qvecs else None,
                               "dup_fetch_factor": dup_fetch_factor(), "project": str(PROJECT),
                               "lancedb": lancedb.__version__, "pylance": lance.__version__}}
        (run_dir / "index_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=1, default=str),
                                                  encoding="utf-8")
        if a.check_against:
            if "text" in modes and spec["column"] == "text" and all(src_fts.get(k) == v
                                                                    for k, v in spec["fts"].items()):
                ok = check_against(run_dir, a.check_against) and ok
            if "hybrid" in modes:
                overlap(run_dir, a.check_against, "hybrid")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
