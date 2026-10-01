"""Состояние индекса прогона: только чтение таблицы LanceDB демона.

Запускается интерпретатором окружения демона (там lancedb нужной версии):
    daemon-venv/bin/python src/lance_probe.py <HOME прогона> [--paths]

Печатает JSON: число строк, индексы и их num_indexed/num_unindexed_rows,
версии библиотек; с --paths — число чанков по каждому файлу.
"""

import json
import sys
from pathlib import Path

import lancedb


def main() -> None:
    home = Path(sys.argv[1])
    with_paths = "--paths" in sys.argv
    db = lancedb.connect(str(home / ".files-search" / "lance"))
    out = {"lancedb": lancedb.__version__}
    try:
        import lance

        out["pylance"] = lance.__version__
    except Exception:
        pass
    names = list(db.table_names())
    out["tables"] = names
    if "files" not in names:
        print(json.dumps(out))
        return
    t = db.open_table("files")
    out["rows"] = t.count_rows()
    idx = []
    for i in t.list_indices():
        rec = {"name": i.name, "type": str(i.index_type), "columns": list(i.columns),
               "details": i.index_details}
        try:
            st = t.index_stats(i.name)
            rec["indexed"] = st.num_indexed_rows
            rec["unindexed"] = st.num_unindexed_rows
            rec["segments"] = getattr(st, "num_indices", None)
        except Exception as e:
            rec["error"] = str(e)
        idx.append(rec)
    out["indices"] = idx
    if with_paths:
        col = t.to_arrow().column("path").to_pylist() if out["rows"] else []
        counts: dict[str, int] = {}
        for p in col:
            counts[p] = counts.get(p, 0) + 1
        out["paths"] = counts
    print(json.dumps(out, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
