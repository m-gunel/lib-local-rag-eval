"""Векторный поиск: что теряет приближённый индекс IVF-PQ против точного перебора.

Берёт готовую таблицу прогона (только чтение), кодирует запросы моделью
самого индексатора и ищет двумя способами:
- как в продукте: IVF-PQ, nprobes=32, без refine (src/indexer/indexer.py, _vector_search);
- точный перебор по тем же векторам (bypass_vector_index).
Чанки группируются в документы так же, как в ответе API (по пути, в порядке
близости лучшего чанка). Метрики — по общей разметке, неоценённые = неправильные.

    PYTHONDONTWRITEBYTECODE=1 daemon-venv/bin/python src/vector_probe.py --run R7
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parent.parent
PROJECT = Path(os.environ.get("LIB_LOCAL_RAG", "/Users/gunel30/Downloads/pip_local_rag_0110"))
sys.path.insert(0, str(EVAL / "src"))
from metrics import METRICS, per_query, relevant  # noqa: E402
from report import corpus_rel, load_qrels, load_run  # noqa: E402

TOP_K = 10


def docs_from_chunks(df) -> list[str]:
    out = []
    for p in df.sort("_distance")["path"].to_list():
        d = corpus_rel(p)
        if d not in out:
            out.append(d)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="R7")
    ap.add_argument("--nprobes", type=int, default=32)
    a = ap.parse_args()

    import lancedb

    sys.path.insert(0, str(PROJECT))
    from src.utils.openvino_encoder import OpenVINOEmbedder  # код индексатора

    enc = OpenVINOEmbedder(str(PROJECT / "models" / "rubert-tiny2_002"))
    table = lancedb.connect(str(EVAL / "runs" / a.run / "home" / ".files-search" / "lance")).open_table("files")
    queries = [json.loads(l) for l in open(EVAL / "data" / "queries.jsonl", encoding="utf-8")]
    qrels = load_qrels()
    ids = [q["qid"] for q in queries if relevant(qrels.get(q["qid"], {}))]

    ann, exact, chunk_overlap = {}, {}, []
    for q in queries:
        vec = enc.encode([q["query"]], task="query")[0]
        base = lambda: table.search(vec, query_type="vector", vector_column_name="vector").distance_type("cosine").limit(TOP_K)  # noqa: E731
        df_a = base().nprobes(a.nprobes).to_polars()
        df_e = base().bypass_vector_index().to_polars()
        ann[q["qid"]] = docs_from_chunks(df_a)
        exact[q["qid"]] = docs_from_chunks(df_e)
        ea, ee = set(df_a["idx"].to_list()), set(df_e["idx"].to_list())
        chunk_overlap.append(len(ea & ee) / max(1, len(ee)))

    product, _ = load_run(EVAL / "runs" / a.run, "vector")
    same = sum(1 for i in ids if product.get(i) == ann.get(i))
    print(f"проверка: наш приближённый поиск совпал с ответом API ({a.run}, vector) в {same} из {len(ids)} запросов")
    print(f"доля точных топ-10 чанков, которые находит IVF-PQ: {statistics.mean(chunk_overlap):.3f}")
    for name, ranked in (("IVF-PQ (как в продукте)", ann), ("точный перебор", exact)):
        pq = per_query(ranked, qrels, ids)
        unj = sum(1 for i in ids for d in ranked.get(i, [])[:10] if d not in qrels.get(i, {}))
        print(f"{name:24s} неоценённых {unj:5d} | "
              + "  ".join(f"{m} {statistics.mean(pq[i][m] for i in ids):.3f}" for m in METRICS))


if __name__ == "__main__":
    main()
