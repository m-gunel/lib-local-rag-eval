"""Прогон замера: поднять демон lib-local-rag на корпусе, дождаться готовности
индексов, прогнать запросы в трёх режимах, сохранить ответы.

    .venv/bin/python src/harness.py run --run-id R1 [--port 8077] [--modes hybrid,vector,text]

Изоляция (методика, раздел «Протокол и сценарии»):
- отдельный HOME на прогон: runs/<id>/home — там ~/.files-search и ~/.ragsearch
  демона (пути вычисляются через expanduser при импорте src/config.py);
- свой конфиг с явным путём к корпусу и выключенной почтой;
- демон запускается интерпретатором daemon-venv (версии из uv.lock) из корня
  проекта (модель задана относительным путём), проект при этом не меняется.

Готовность: статус idle И проход завершён (total_files = числу файлов корпуса,
очередь пуста) И у полнотекстового и векторного индексов num_unindexed_rows = 0
(векторного нет, если строк < 256) — два одинаковых опроса подряд.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

EVAL = Path(__file__).resolve().parent.parent
PROJECT = Path(os.environ.get("LIB_LOCAL_RAG", "/Users/gunel30/Downloads/pip_local_rag_0110"))
# Интерпретатор индексатора: на другой машине — .venv проекта (uv sync --frozen).
DAEMON_PY = Path(os.environ.get("DAEMON_PY", EVAL / "daemon-venv" / "bin" / "python"))
CORPUS = EVAL / "corpus"
DATA = EVAL / "data"
SUPPORTED = {".txt", ".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".xls"}
MIN_ROWS_FOR_VECTOR_INDEX = 256

CONFIG = """# Конфиг прогона замера качества ранжирования (сгенерирован harness.py)
file_scan_cfg:
  paths:
    - "{corpus}"
embed_model_cfg:
  model_path: {model}
email:
  enabled: false
full_rescan_interval: 86400
app_logging_level: INFO
"""


def corpus_files(corpus: Path) -> list[Path]:
    return sorted(p for p in corpus.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)


def probe(home: Path, paths: bool = False) -> dict:
    args = [str(DAEMON_PY), str(EVAL / "src" / "lance_probe.py"), str(home)]
    if paths:
        args.append("--paths")
    r = subprocess.run(args, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        return {"error": r.stderr[-2000:]}
    return json.loads(r.stdout.strip().splitlines()[-1])


def indexes_ready(p: dict) -> tuple[bool, str]:
    if "rows" not in p:
        return False, "таблицы ещё нет"
    idx = p.get("indices", [])
    fts = [i for i in idx if "FTS" in i["type"].upper() or "INVERTED" in i["type"].upper()]
    vec = [i for i in idx if "PQ" in i["type"].upper() or "IVF" in i["type"].upper()]
    if not fts:
        return False, "нет полнотекстового индекса"
    if p["rows"] >= MIN_ROWS_FOR_VECTOR_INDEX and not vec:
        return False, "нет векторного индекса"
    for i in fts + vec:
        if i.get("unindexed", 1) != 0:
            return False, f"{i['name']}: непроиндексировано {i.get('unindexed')}"
    return True, "ok"


def wait_ready(client: httpx.Client, home: Path, expected_files: int, log, timeout: float) -> dict:
    t0 = time.time()
    last_sig = None
    stable = 0
    while time.time() - t0 < timeout:
        try:
            st = client.get("/api/v1/status").json()
        except Exception as e:
            log(f"статус недоступен: {e}")
            time.sleep(10)
            continue
        idle = st.get("type") == "idle" and not st.get("is_indexing")
        passed = st.get("total_files", 0) >= expected_files and st.get("queued_files", 0) == 0
        if idle and passed:
            p = probe(home)
            ok, why = indexes_ready(p)
            sig = (p.get("rows"), json.dumps(p.get("indices"), sort_keys=True))
            if ok and sig == last_sig:
                stable += 1
            else:
                stable = 0
            last_sig = sig
            log(f"idle; строк {p.get('rows')}; индексы: {why}; стабильно {stable}")
            if ok and stable >= 1:
                return {"status": st, "probe": p, "seconds": round(time.time() - t0)}
        else:
            log(f"индексация: {st.get('type')} total={st.get('total_files')} queued={st.get('queued_files')}")
        time.sleep(20)
    raise TimeoutError("индексы не достроились за отведённое время")


def run_queries(client: httpx.Client, queries: list[dict], mode: str, out: Path, log) -> None:
    done = set()
    if out.exists():
        done = {json.loads(l)["qid"] for l in out.open(encoding="utf-8")}
    with out.open("a", encoding="utf-8") as f:
        for i, q in enumerate(queries):
            if q["qid"] in done:
                continue
            body = {"query": q["query"], "limit": 10, "query_type": mode}
            t = time.perf_counter()
            r = client.post("/api/v1/search", json=body)
            ms = (time.perf_counter() - t) * 1000
            rec = {"qid": q["qid"], "mode": mode, "http": r.status_code, "ms": round(ms, 1)}
            if r.status_code == 200:
                rec["results"] = r.json()["results"]
            else:
                rec["error"] = r.text[:500]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 200 == 0:
                log(f"{mode}: {i + 1}/{len(queries)}")


def cmd_run(a) -> None:
    run_dir = EVAL / "runs" / a.run_id
    # Повтор того же run-id молча дописывал бы ответы к старым: run_queries
    # пропускает уже записанные qid, а HOME со старым индексом остаётся.
    if run_dir.exists() and any(run_dir.iterdir()) and not a.resume:
        raise SystemExit(
            f"прогон {a.run_id} уже есть ({run_dir}): возьмите новый --run-id "
            "или добавьте --resume, чтобы дописать прерванный прогон"
        )
    home = run_dir / "home"
    home.mkdir(parents=True, exist_ok=True)
    logf = (run_dir / "harness.log").open("a", encoding="utf-8")

    def log(msg: str) -> None:
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    corpus = Path(a.corpus).resolve()
    files = corpus_files(corpus)
    cfg = run_dir / "config.yml"
    extra = "".join(f"{line}\n" for line in a.extra_config)
    cfg.write_text(
        CONFIG.format(corpus=corpus, model=os.environ.get("RAG_MODEL_PATH", PROJECT / "models" / "rubert-tiny2_002")) + extra, encoding="utf-8"
    )
    env = {
        **os.environ,
        "HOME": str(home),
        "CONFIG_PATH": str(cfg),
        "RAG_PORT": str(a.port),
        "RAG_HOST": "127.0.0.1",
        # Не оставлять __pycache__ в дереве проекта.
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    env.pop("DEV", None)
    daemon_log = (run_dir / "daemon.log").open("a", encoding="utf-8")
    log(f"корпус: {len(files)} файлов; запуск демона на :{a.port}")
    proc = subprocess.Popen(
        [str(DAEMON_PY), "main.py"], cwd=PROJECT, env=env, stdout=daemon_log, stderr=subprocess.STDOUT
    )
    meta = {"run_id": a.run_id, "corpus_files": len(files), "port": a.port, "extra_config": a.extra_config}
    try:
        base = f"http://127.0.0.1:{a.port}"
        with httpx.Client(base_url=base, timeout=120) as client:
            for _ in range(300):
                try:
                    if client.get("/api/v1/health").status_code == 200:
                        break
                except Exception:
                    pass
                if proc.poll() is not None:
                    raise RuntimeError("демон завершился на старте, см. daemon.log")
                time.sleep(2)
            ready = wait_ready(client, home, len(files), log, a.timeout)
            meta.update(ready)
            meta["probe_paths"] = probe(home, paths=True)
            (run_dir / "index_state.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
            queries = [json.loads(l) for l in Path(a.queries).open(encoding="utf-8")]
            # Контрольные запросы методики: фраза целиком в кавычках и пустой
            # запрос — уходит ли гибрид в тихий фолбэк на вектор (score < 0).
            controls = [
                {"qid": "ctl_quoted", "query": '"требования к системе управления операционным риском"'},
                {"qid": "ctl_empty", "query": ""},
                {"qid": "ctl_spaces", "query": "   "},
            ]
            run_queries(client, controls, "hybrid", run_dir / "responses_control.jsonl", log)
            for mode in a.modes.split(","):
                log(f"прогон {mode}: {len(queries)} запросов")
                run_queries(client, queries, mode, run_dir / f"responses_{mode}.jsonl", log)
            log("готово")
    finally:
        if not a.keep:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run-id", required=True)
    r.add_argument("--port", type=int, default=8077)
    r.add_argument("--corpus", default=str(CORPUS))
    r.add_argument("--queries", default=str(DATA / "queries.jsonl"))
    r.add_argument("--modes", default="hybrid,vector,text")
    r.add_argument("--timeout", type=float, default=6 * 3600)
    r.add_argument("--keep", action="store_true", help="не останавливать демон после прогона")
    r.add_argument("--resume", action="store_true",
                   help="дописать прерванный прогон с тем же run-id (иначе существующий run-id — ошибка)")
    r.add_argument("--extra-config", action="append", default=[],
                   help="строка YAML, дописываемая в конфиг прогона (можно несколько раз)")
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
