"""Текст файла глазами индексатора: чанки, которые выдают его собственные парсеры.

Запускать интерпретатором окружения lib-local-rag (там стоят его зависимости):
    <venv>/bin/python src/indexer_view.py <файл> [<файл> ...] [--out DIR]

Код проекта только импортируется, ничего в нём не меняется. Нужен, чтобы
запросы писались по тому, что индексатор реально видит (листы xlsx он может
пропустить, pptx читает без заметок), и чтобы проверить, что эталонный
фрагмент попадает в индекс.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # не оставлять __pycache__ в дереве проекта
PROJECT = Path(os.environ.get("LIB_LOCAL_RAG", "/Users/gunel30/Downloads/pip_local_rag_0110"))
os.environ.setdefault("CONFIG_PATH", str(PROJECT / "cfg" / "dev.yml"))
# Config() читает ~/.ragsearch и заводит ~/.files-search — уводим HOME во
# временный каталог, чтобы не трогать настоящие данные пользователя.
_TMP_HOME = Path(os.environ.get("INDEXER_VIEW_HOME", "/tmp/indexer_view_home"))
_TMP_HOME.mkdir(parents=True, exist_ok=True)
os.environ["HOME"] = str(_TMP_HOME)
sys.path.insert(0, str(PROJECT))
CALLER_CWD = Path.cwd()
os.chdir(PROJECT)  # модель задана относительным путём

from src.parsers.base import file_extension, get_parser  # noqa: E402
import src.parsers  # noqa: E402,F401  регистрирует парсеры
from src.utils.config_mixin import set_config  # noqa: E402

set_config()


def _jsonable(v):
    """Метаданные проекта (datetime, вложенные dict) → то, что ложится в JSON."""
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def records_of(path: Path) -> tuple[list[dict], dict | None]:
    """Чанки файла с полями метаданных и метаданные документа — как их пишет индексатор.

    Если в проекте есть общий синхронный конвейер `src.parsers.file_chunks` (тот же путь,
    что `_process_path`: нормализация, обогащение текста, поля чанка), берём его — иначе
    проверки не увидят префикс и метаданные. Старые версии проекта: парсер напрямую.
    """
    parser = get_parser(file_extension(path))
    if parser is None:
        return [], None
    import src.parsers as parsers_pkg

    file_chunks = getattr(parsers_pkg, "file_chunks", None)
    if file_chunks is not None:
        rows = list(file_chunks(str(path)))
        meta = _jsonable(rows[0].get("meta")) if rows else None
        chunks = [_jsonable({k: v for k, v in r.items() if k not in ("meta", "file_meta", "path")}) for r in rows]
        return chunks, meta
    chunks = [dict(c) if isinstance(c, dict) else {"idx": c.idx, "text": c.text} for c in parser.iter_chunks(str(path))]
    return _jsonable(chunks), _jsonable(dict(parser.get_metadata(str(path))))


def chunks_of(path: Path) -> list[str]:
    return [c["text"] for c in records_of(path)[0]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", help="каталог для <имя>.chunks.json")
    a = ap.parse_args()
    for f in a.files:
        p = (CALLER_CWD / f).resolve()
        try:
            chunks = chunks_of(p)
            err = None
        except Exception as e:  # парсер упал — это тоже результат
            chunks, err = [], f"{type(e).__name__}: {e}"
        info = {"file": str(p), "chunks": len(chunks), "chars": sum(map(len, chunks)), "error": err}
        print(json.dumps(info, ensure_ascii=False))
        if a.out:
            out = CALLER_CWD / a.out
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{p.name}.chunks.json").write_text(
                json.dumps({"file": str(p), "error": err, "chunks": chunks}, ensure_ascii=False, indent=0),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
