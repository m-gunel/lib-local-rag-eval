"""Окружение для демона lib-local-rag с версиями ровно из его uv.lock.

uv на машине нет, поэтому берём замкнутое множество зависимостей пакета
files-search-back (основная группа, без dev/test/build) из uv.lock с учётом
маркеров платформы и ставим pip-ом с --no-deps точными версиями.
Окружение создаётся в директории стенда: проект lib-local-rag не меняется.
"""

import json
import subprocess
import sys
import tomllib
from pathlib import Path

from packaging.markers import Marker

PROJECT = Path("/Users/gunel30/Downloads/lib_local_rag")
EVAL = Path(__file__).resolve().parent.parent
VENV = EVAL / "daemon-venv"
PY = "/opt/local/bin/python3.12"
ROOT_NAME = "files-search-back"


def marker_ok(m: str | None) -> bool:
    if not m:
        return True
    try:
        return Marker(m).evaluate({"extra": ""})
    except Exception:
        return True


def closure() -> dict[str, str]:
    lock = tomllib.loads((PROJECT / "uv.lock").read_text(encoding="utf-8"))
    pkgs = {p["name"]: p for p in lock["package"]}
    root = pkgs[ROOT_NAME]
    todo = [d for d in root.get("dependencies", []) if marker_ok(d.get("marker"))]
    pins: dict[str, str] = {}
    while todo:
        dep = todo.pop()
        name = dep["name"]
        p = pkgs.get(name)
        if p is None or name in pins:
            continue
        pins[name] = p["version"]
        extras = dep.get("extra", [])
        for d in p.get("dependencies", []):
            if marker_ok(d.get("marker")):
                todo.append(d)
        for ex in extras:
            for d in p.get("optional-dependencies", {}).get(ex, []):
                if marker_ok(d.get("marker")):
                    todo.append(d)
    return pins


def main() -> None:
    pins = closure()
    (EVAL / "work").mkdir(exist_ok=True)
    (EVAL / "work" / "daemon_pins.json").write_text(json.dumps(pins, indent=1, sort_keys=True))
    print(f"пакетов: {len(pins)}; lancedb={pins.get('lancedb')} pylance={pins.get('pylance')} openvino={pins.get('openvino')}")
    if not VENV.exists():
        subprocess.check_call([PY, "-m", "venv", str(VENV)])
    reqs = [f"{n}=={v}" for n, v in sorted(pins.items())]
    subprocess.check_call(
        [str(VENV / "bin" / "pip"), "install", "-q", "--no-deps", "--index-url", "https://pypi.org/simple", *reqs]
    )
    print("готово:", VENV)


if __name__ == "__main__":
    sys.exit(main())
