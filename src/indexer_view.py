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
PROJECT = Path(os.environ.get("LIB_LOCAL_RAG", "/Users/gunel30/Downloads/lib_local_rag"))
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


def chunks_of(path: Path) -> list[str]:
    parser = get_parser(file_extension(path))
    if parser is None:
        return []
    return [c["text"] if isinstance(c, dict) else c.text for c in parser.iter_chunks(str(path))]


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
