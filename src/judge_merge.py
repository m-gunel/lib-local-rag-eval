"""Сведение оценок асессоров: взвешенная каппа Коэна, расхождения, итог.

Вход: work/judge/<раунд>/grades_a.json, grades_b.json (от workflow разметки)
и, после третьего асессора, grades_c.json. Выход: work/judge/<раунд>/final.tsv
и строка в work/judge/judged.tsv; сводка согласия — agreement.json.

Итоговая оценка: первичная, а там, где вторая оценка расходится с ней на ≥ 1
балл в «зоне релевантности» (одна из оценок ≥ 2), — оценка третьего асессора.
Каппа считается по парам, где хотя бы один поставил ≥ 1: очевидные нули её
завышают (методика).
"""

import argparse
import json
from pathlib import Path

EVAL = Path(__file__).resolve().parent.parent
JUDGE = EVAL / "work" / "judge"


def weighted_kappa(a: list[int], b: list[int], k: int = 4) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[x][y] += 1
    pa = [sum(obs[i]) / n for i in range(k)]
    pb = [sum(obs[i][j] for i in range(k)) / n for j in range(k)]
    w = [[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)]
    do = sum(w[i][j] * obs[i][j] / n for i in range(k) for j in range(k))
    de = sum(w[i][j] * pa[i] * pb[j] for i in range(k) for j in range(k))
    return 1 - do / de if de else float("nan")


def load(path: Path) -> dict[str, dict]:
    return {g["pid"]: g for g in json.load(open(path, encoding="utf-8"))} if path.exists() else {}


def conflicts(round_name: str) -> list[str]:
    a = load(JUDGE / round_name / "grades_a.json")
    b = load(JUDGE / round_name / "grades_b.json")
    return sorted(p for p in b if p in a and a[p]["grade"] != b[p]["grade"]
                  and max(a[p]["grade"], b[p]["grade"]) >= 2)


def cmd_merge(round_name: str) -> None:
    d = JUDGE / round_name
    pairs = {}
    for f in sorted(d.glob("a_*.json")):
        for p in json.load(open(f, encoding="utf-8")):
            pairs[p["pid"]] = p
    a, b, c = load(d / "grades_a.json"), load(d / "grades_b.json"), load(d / "grades_c.json")
    both = [p for p in b if p in a]
    ka = [a[p]["grade"] for p in both]
    kb = [b[p]["grade"] for p in both]
    nz = [(x, y) for x, y in zip(ka, kb) if max(x, y) >= 1]
    agree = {
        "pairs_double": len(both),
        "kappa_all": weighted_kappa(ka, kb),
        "kappa_nonzero": weighted_kappa([x for x, _ in nz], [y for _, y in nz]),
        "exact_agreement": sum(x == y for x, y in zip(ka, kb)) / len(both) if both else None,
        "conflicts": len(conflicts(round_name)),
        "resolved_by_third": len(c),
        "missing_primary": sorted(set(pairs) - set(a)),
    }
    (d / "agreement.json").write_text(json.dumps(agree, ensure_ascii=False, indent=1))
    rows = []
    for pid, p in pairs.items():
        if pid not in a:
            continue
        g = c[pid]["grade"] if pid in c else a[pid]["grade"]
        rows.append((p["qid"], p["doc_id"], g, round_name))
    (d / "final.tsv").write_text("".join(f"{q}\t{doc}\t{g}\t{r}\n" for q, doc, g, r in rows), encoding="utf-8")
    judged = JUDGE / "judged.tsv"
    old = [l for l in judged.read_text(encoding="utf-8").splitlines() if not l.endswith("\t" + round_name)] if judged.exists() else []
    judged.write_text("\n".join(old + [f"{q}\t{doc}\t{g}\t{r}" for q, doc, g, r in rows]) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in agree.items() if k != "missing_primary"}, ensure_ascii=False),
          "без первичной оценки:", len(agree["missing_primary"]))


def cmd_args(round_name: str) -> None:
    d = JUDGE / round_name
    out = {"round": round_name, "dir": str(d), "a": [], "b": []}
    for f in sorted(d.glob("a_*.json")):
        out["a"].append({"file": f.name, "n": len(json.load(open(f, encoding="utf-8")))})
    for f in sorted(d.glob("b_*.json")):
        out["b"].append({"file": f.name, "n": len(json.load(open(f, encoding="utf-8")))})
    print(json.dumps(out, ensure_ascii=False))


def cmd_save(round_name: str, wf_output: str) -> None:
    res = json.load(open(wf_output, encoding="utf-8"))
    res = res.get("result", res)
    d = JUDGE / round_name
    for k in ("A", "B", "C"):
        (d / f"grades_{k.lower()}.json").write_text(json.dumps(res.get(k, []), ensure_ascii=False), encoding="utf-8")
    print({k: len(res.get(k, [])) for k in ("A", "B", "C")})


def cmd_collect(round_name: str, outputs: list[str]) -> None:
    """Добавить оценки из результатов workflow к уже собранным (по pid,
    первая полученная оценка не перезаписывается). Нужен, когда раунд идёт
    несколькими запусками (например, после лимита сессии)."""
    d = JUDGE / round_name
    total = {}
    for k in ("A", "B", "C"):
        f = d / f"grades_{k.lower()}.json"
        have = load(f)
        for out in outputs:
            res = json.load(open(out, encoding="utf-8"))
            res = res.get("result", res)
            for g in res.get(k, []) or []:
                have.setdefault(g["pid"], g)
        f.write_text(json.dumps(list(have.values()), ensure_ascii=False), encoding="utf-8")
        total[k] = len(have)
    print(round_name, total)


def cmd_todo(round_name: str) -> None:
    """Пакеты, в которых есть пары без оценки: параметры для judge_workflow.js."""
    d = JUDGE / round_name
    a = load(d / "grades_a.json")
    b = load(d / "grades_b.json")
    out = {"round": round_name, "dir": str(d), "a": [], "b": [], "skip_c": True}
    for prefix, have in (("a", a), ("b", b)):
        for f in sorted(d.glob(f"{prefix}_*.json")):
            pairs = json.load(open(f, encoding="utf-8"))
            if any(p["pid"] not in have for p in pairs):
                out[prefix].append({"file": f.name, "n": len(pairs)})
    print(json.dumps(out, ensure_ascii=False))


def cmd_rebatch(round_name: str, runs: list[str], batch: int = 40) -> None:
    """Переупаковать пары без первичной оценки: сначала те, что есть в топ-10
    hybrid прогонов runs (p_XXX.json), затем остальные (q_XXX.json). pid не
    меняются, a_*.json остаются полным списком пар раунда (их читают merge и C)."""
    d = JUDGE / round_name
    a = load(d / "grades_a.json")
    need = set()
    for run in runs:
        for l in open(EVAL / "runs" / run / "responses_hybrid.jsonl", encoding="utf-8"):
            r = json.loads(l)
            for it in r.get("results", [])[:10]:
                need.add((r["qid"], it["absolute_path"].split("/corpus/", 1)[-1]))
    todo = [p for f in sorted(d.glob("a_*.json")) for p in json.load(open(f, encoding="utf-8")) if p["pid"] not in a]
    for old in list(d.glob("p_*.json")) + list(d.glob("q_*.json")):
        old.unlink()
    out = {}
    for prefix, part in (("p", [p for p in todo if (p["qid"], p["doc_id"]) in need]),
                         ("q", [p for p in todo if (p["qid"], p["doc_id"]) not in need])):
        out[prefix] = []
        for n, i in enumerate(range(0, len(part), batch)):
            chunk = part[i:i + batch]
            name = f"{prefix}_{n:03d}.json"
            (d / name).write_text(json.dumps(chunk, ensure_ascii=False, indent=0), encoding="utf-8")
            out[prefix].append({"file": name, "n": len(chunk)})
    print(json.dumps({k: {"batches": len(v), "pairs": sum(x["n"] for x in v)} for k, v in out.items()}, ensure_ascii=False))


def cmd_verify(round_name: str, batch: int = 40) -> None:
    """Пакеты v_XXX.json для перепроверки основной моделью всех пар, где первичный
    асессор поставил ≥ 2, а второй оценки ещё нет: первичный асессор Sonnet мягче
    в зоне релевантности. Оценки идут в grades_b.json, расхождения решает C.
    Каппа согласия считается только по случайной выборке b_*.json."""
    d = JUDGE / round_name
    a, b = load(d / "grades_a.json"), load(d / "grades_b.json")
    pairs = {p["pid"]: p for f in sorted(d.glob("a_*.json")) for p in json.load(open(f, encoding="utf-8"))}
    # opus_a_pids.json — пары, где первичную оценку уже ставила основная модель
    f = d / "opus_a_pids.json"
    opus = set(json.load(open(f, encoding="utf-8"))) if f.exists() else set()
    todo = [pairs[p] for p, g in a.items() if g["grade"] >= 2 and p not in b and p not in opus]
    for old in d.glob("v_*.json"):
        old.unlink()
    for n, i in enumerate(range(0, len(todo), batch)):
        (d / f"v_{n:03d}.json").write_text(json.dumps(todo[i:i + batch], ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"пар на перепроверку: {len(todo)}, пакетов: {-(-len(todo) // batch)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["conflicts", "merge", "args", "save", "collect", "todo", "rebatch", "verify"])
    ap.add_argument("round")
    ap.add_argument("file", nargs="*")
    a = ap.parse_args()
    if a.mode == "collect":
        cmd_collect(a.round, a.file)
        return
    if a.mode == "todo":
        cmd_todo(a.round)
        return
    if a.mode == "verify":
        cmd_verify(a.round)
        return
    if a.mode == "rebatch":
        cmd_rebatch(a.round, a.file)
        return
    a.file = a.file[0] if a.file else None
    if a.mode == "conflicts":
        print(json.dumps(conflicts(a.round)))
    elif a.mode == "args":
        cmd_args(a.round)
    elif a.mode == "save":
        cmd_save(a.round, a.file)
    else:
        cmd_merge(a.round)


if __name__ == "__main__":
    main()
