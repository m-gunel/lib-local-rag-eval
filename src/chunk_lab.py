"""Лаборатория нарезки и метаданных в поиске: варианты из реестра, офлайн, точный перебор.

Вариант — запись в data/chunk_variants.yaml: копия проекта (worktree), настройки
сплиттера, выдача (пул чанков на канал, 10 чанков или 10 документов), метаданные в
ранжировании (карточка документа, файл по коду из имени, раздел и название полями FTS).
Прогон: чанки кодом проекта (кэш parse_checks) → векторы (кэш offline_ab) → таблица LanceDB
с FTS как в демоне → обычные запросы и навигационный набор → выдачи. Отчёт против базы на
рабочей или контрольной части запросов:
- ранжирование: обычные запросы по разметке (две версии неоценённых: «неправильные» и
  «правильные» — на объединении неоценённых обеих выдач), навигационные — по известному ответу;
- «ответ во фрагменте»: эталонные фрагменты parse_checks.reference_passages() и эталон xlsx;
  текст, отданный по документу, — найденные чанки, окно с соседями или родитель (страница,
  раздел, слайд, таблица), всё в одном бюджете символов.

  PY="env PYTHONDONTWRITEBYTECODE=1 daemon-venv/bin/python"
  $PY src/chunk_lab.py split                      # один раз: data/split.json
  $PY src/chunk_lab.py nav                        # один раз: data/nav_queries.jsonl (по базе C0)
  $PY src/chunk_lab.py run C0                     # вариант из реестра → work/chunking/runs/C0.json
  $PY src/chunk_lab.py report C0 D50 D100 --part dev   # первый — база
Прогоны — строго по одному: кэш векторов offline_ab пишется целиком.
"""

import argparse
import collections
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path, PurePath

sys.dont_write_bytecode = True
EVAL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL / "src"))
LAB = EVAL / "work" / "chunking"
REGISTRY = EVAL / "data" / "chunk_variants.yaml"  # реестр — в git, прогоны и таблицы — в work/
RUNS, ROWS, DB, CONFIGS = LAB / "runs", LAB / "rows", LAB / "db", LAB / "configs"
SPLIT, NAV = EVAL / "data" / "split.json", EVAL / "data" / "nav_queries.jsonl"
BUDGET = 1800  # символов текста на документ для «ответа во фрагменте» (≈ 3 фрагмента MCP по 600)
MAX_FRAGS = 3  # фрагментов на документ в выдаче, как в MCP
WORD = re.compile(r"[0-9a-zа-я]+")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ntok(s: str) -> str:
    return " ".join(WORD.findall(str(s or "").lower().replace("ё", "е")))


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def strip_prefix(text: str) -> str:
    """Текст чанка без строки «Документ: …» конвейера."""
    return text.split("\n", 1)[1] if text.startswith("Документ: ") and "\n" in text else text


# ---------- реестр ----------

DEFAULTS = {"project": "work/project_wt", "splitter": None, "pool": 10, "docs": False,
            "card": False, "name_code": False, "fields": None}


def variant(name: str) -> dict:
    import yaml

    reg = yaml.safe_load(open(REGISTRY, encoding="utf-8")) or {}
    if name not in reg:
        raise SystemExit(f"варианта {name} нет в {REGISTRY}")
    v = {**DEFAULTS, **(reg[name] or {})}
    p = Path(v["project"])
    v["project"] = str(p if p.is_absolute() else EVAL / p)
    return v


def variant_env(name: str, v: dict) -> dict:
    """Окружение разбора: копия проекта и её конфиг; настройки сплиттера — YAML поверх cfg/dev.yml."""
    import yaml

    env = dict(os.environ)
    env.pop("RAG_EMBED_SPLITTER_CFG", None)  # перебивает YAML — не даём протечь из оболочки
    env["LIB_LOCAL_RAG"] = v["project"]
    env.setdefault("INDEXER_VIEW_HOME", str(EVAL / "work" / "indexer_view_home"))
    cfg_path = Path(v["project"]) / "cfg" / "dev.yml"
    if v["splitter"]:
        cfg = yaml.safe_load(open(cfg_path, encoding="utf-8")) or {}
        cfg["embed_splitter_cfg"] = v["splitter"]
        CONFIGS.mkdir(parents=True, exist_ok=True)
        cfg_path = CONFIGS / f"{name}.yml"
        cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    env["CONFIG_PATH"] = str(cfg_path)
    return env


def git_head(path: str) -> str:
    def git(*a):
        return subprocess.run(["git", "-C", path, *a], capture_output=True, text=True).stdout.strip()

    return git("rev-parse", "--short", "HEAD") + ("+правки" if git("status", "--porcelain", "--", "src") else "")


# ---------- деление запросов и навигационный набор ----------

def part_of(key: str) -> str:
    """Рабочая (2/3) или контрольная (1/3) часть — по хэшу кластера, без случайности."""
    return "test" if int(hashlib.sha1(f"split\0{key}".encode()).hexdigest(), 16) % 3 == 0 else "dev"


def cmd_split(a):
    queries = jl(EVAL / "data" / "queries.jsonl")
    import parse_checks as pc

    out = {"rule": "sha1('split\\0' + кластер) % 3 == 0 → test; кластер — base запроса, для навигации — doc_id",
           "main": collections.defaultdict(list), "nav_docs": collections.defaultdict(list)}
    for q in queries:
        out["main"][part_of(q.get("base") or q["qid"])].append(q["qid"])
    for m in pc.MANIFEST:
        out["nav_docs"][part_of(m["doc_id"])].append(m["doc_id"])
    SPLIT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"запросы: dev {len(out['main']['dev'])}, test {len(out['main']['test'])}; документы навигации: "
        f"dev {len(out['nav_docs']['dev'])}, test {len(out['nav_docs']['test'])} → {SPLIT}")


NAV_TEMPLATES = {
    "name": ["Найди файл {x}", "Где документ {x}", "Открой {x}", "{x}"],
    "code": ["Найди файл {x}", "Где файл {x}", "Открой документ {x}", "{x}"],
    "title": ["Найди документ {x}", "Где документ «{x}»", "{x}"],
    "heading": ["Найди раздел {x}", "Где раздел «{x}»", "{x}"],
}


def is_code(tok: str) -> bool:
    """Код в имени файла: цифры вместе с буквами или подчёркиванием («ЦБ_21», «5348-У», «obs_286»)."""
    return bool(re.search(r"\d", tok)) and bool(re.search(r"[^\W\d_]", tok) or "_" in tok)


def cmd_nav(a):
    """Навигационный набор по чанкам и метаданным базы: имя, код из имени, название, заголовок."""
    rows = load_rows(json.loads((RUNS / f"{a.base}.json").read_text(encoding="utf-8"))["table"])
    meta = json.loads((ROWS / f"{a.base}.meta.json").read_text(encoding="utf-8"))
    heads = collections.defaultdict(list)
    for r in rows:
        if r["kind"] != "card" and r.get("section") and meta[r["doc"]].get("ext") in ("pdf", "docx", "pptx", "txt"):
            h = r["section"].split(" › ")[-1].strip()
            if h not in heads[r["doc"]]:
                heads[r["doc"]].append(h)
    owners = collections.defaultdict(set)
    for d, hs in heads.items():
        for h in hs:
            owners[ntok(h)].add(d)
    stems = {d: PurePath(m.get("original_name") or d).stem for d, m in meta.items()}
    code_owners = collections.defaultdict(set)
    for d, s in stems.items():
        for t in s.split():
            if is_code(t):
                code_owners[ntok(t)].add(d)
    rnd = random.Random(20261002)
    out = []
    for d in sorted(meta):
        stem, title = stems[d], (meta[d].get("title") or "").strip()
        items = [("name", stem)]
        codes = [t for t in stem.split() if is_code(t) and len(code_owners[ntok(t)]) == 1]
        if codes:
            items.append(("code", codes[-1]))
        if title and ntok(title) not in ntok(stem):
            items.append(("title", " ".join(title.split()[:15])))
        cand = sorted(h for h in heads.get(d, ()) if len(h.split()) >= 3 and len(owners[ntok(h)]) == 1)
        if cand:
            items.append(("heading", " ".join(rnd.choice(cand).split()[:15])))
        for kind, x in items:
            out.append({"nid": f"nav{len(out):04d}", "type": kind, "query": rnd.choice(NAV_TEMPLATES[kind]).format(x=x),
                        "doc_id": d})
    NAV.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")
    log(f"навигационных запросов {len(out)}: {dict(collections.Counter(x['type'] for x in out))} → {NAV}")


# ---------- прогон ----------

def card_text(meta: dict) -> str:
    """Карточка документа: имя файла и свойства, которых нет в имени."""
    lines = [f"Документ: {PurePath(meta.get('original_name') or '').stem}"]
    for label, key in (("Название", "title"), ("Автор", "author"), ("Изменил", "last_modified_by"),
                       ("Тема", "subject"), ("Ключевые слова", "keywords"), ("Описание", "description")):
        v = " ".join(str(meta.get(key) or "").split())
        if v and ntok(v) not in ntok("\n".join(lines)):
            lines.append(f"{label}: {v[:300]}")
    return "\n".join(lines)


def load_rows(key: str) -> list[dict]:
    return jl(ROWS / f"{key}.jsonl")


def cmd_run(a):
    v = variant(a.name)
    if (RUNS / f"{a.name}.json").exists() and not a.force:
        raise SystemExit(f"прогон {a.name} уже есть — --force, чтобы пересчитать")
    env = variant_env(a.name, v)
    log(f"{a.name}: проект {v['project']} ({git_head(v['project'])}), конфиг {env['CONFIG_PATH']}")
    subprocess.run([sys.executable, __file__, "_run", a.name], env=env, check=True, cwd=EVAL)


def cmd_child_run(a):
    t0 = time.time()
    v = variant(a.name)
    import lancedb
    import numpy as np
    import pyarrow as pa
    from lancedb.index import FTS
    from lancedb.query import BooleanQuery, MatchQuery, MultiMatchQuery, Occur

    import offline_ab as oa
    import parse_checks as pc

    src = pc.LiveSource()
    from src.common.fts import FtsConfig  # проект уже на sys.path (indexer_view)

    emb = oa.Embedder(src)
    rows, meta = [], {}
    for m in pc.MANIFEST:
        chunks, dmeta, err, _ = src.chunks(m)
        meta[m["doc_id"]] = dmeta or {}
        for c in chunks:
            rows.append({"doc": m["doc_id"], "text": c["text"], "kind": c.get("kind") or "text",
                         "section": c.get("section") or "", "idx": c.get("idx"), **{k: c.get(k) for k in (
                             "page_start", "page_end", "char_start", "char_end", "table_id")}})
        if v["card"]:
            rows.append({"doc": m["doc_id"], "text": card_text(meta[m["doc_id"]]), "kind": "card", "section": "",
                         "idx": -1})
    for i, r in enumerate(rows):
        r["cid"] = i
    splitter = src.cfg.embed_splitter_cfg.model_dump()
    key = hashlib.sha1(json.dumps([sorted(src.keys.items()), splitter, v["card"], emb.fp, git_head(v["project"])],
                                  ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    log(f"строк {len(rows)} (карточек {sum(r['kind'] == 'card' for r in rows)}), документов {len(meta)}; "
        f"сплиттер {splitter}; таблица {key}")
    t1 = time.time()
    vecs = emb.get([r["text"] for r in rows], "document")
    embed_s = round(time.time() - t1)
    db = lancedb.connect(str(DB))
    if key in db.table_names():
        table = db.open_table(key)
    else:
        titles = {d: (m.get("title") or "") for d, m in meta.items()}
        data = pa.table({"cid": [r["cid"] for r in rows], "path": [r["doc"] for r in rows],
                         "text": [r["text"] for r in rows], "kind": [r["kind"] for r in rows],
                         "section": [r["section"] for r in rows], "title": [titles[r["doc"]] for r in rows],
                         "vector": pa.FixedSizeListArray.from_arrays(pa.array(vecs.ravel()), vecs.shape[1])})
        table = db.create_table(key, data, mode="overwrite")
        for col in ("text", "section", "title"):
            table.create_index(col, config=FTS(**FtsConfig().model_dump()))
        ROWS.mkdir(parents=True, exist_ok=True)
        with open(ROWS / f"{key}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: r[k] for k in r}, ensure_ascii=False) + "\n")
    (ROWS / f"{a.name}.meta.json").write_text(json.dumps({d: {k: m.get(k) for k in (
        "original_name", "ext", "title", "author", "last_modified_by", "subject", "keywords")} for d, m in meta.items()},
        ensure_ascii=False), encoding="utf-8")
    names = {d: f" {ntok(PurePath(m.get('original_name') or d).stem)} " for d, m in meta.items()}
    kind_of = [r["kind"] for r in rows]

    def name_hits(q: str) -> list[str]:
        hits = []
        for raw in q.split():
            t = raw.strip(".,;:!?«»\"'()[]")
            if not is_code(t):
                continue
            found = sorted(d for d, n in names.items() if f" {ntok(t)} " in n)
            if 0 < len(found) <= 3:
                hits += [d for d in found if d not in hits]
        return hits

    fields = dict(v["fields"] or {})
    mode = fields.pop("mode", "sum")
    weights = {c: float(w) for c, w in fields.items() if float(w) > 0}

    def text_query(q: str):
        """FTS-часть гибрида. Без полей — только text, как в продукте: при нескольких
        FTS-индексах строковый запрос гибрида LanceDB ищет по всем полям сразу.
        sum — сумма оценок полей с весами (BooleanQuery SHOULD), max — максимум (MultiMatchQuery)."""
        if not weights:
            return MatchQuery(q, column="text")
        if mode == "max":
            return MultiMatchQuery(q, columns=["text", *weights], boosts=[1.0, *weights.values()])
        return BooleanQuery([(Occur.SHOULD, MatchQuery(q, column="text")),
                             *((Occur.SHOULD, MatchQuery(q, column=c, boost=w)) for c, w in weights.items())])

    def search(q: str, vec) -> dict:
        tq = text_query(q)
        try:
            r = (table.search(query_type="hybrid", vector_column_name="vector", fts_columns="text")
                 .distance_type("cosine").vector(vec).text(tq).limit(v["pool"]).to_arrow())
            cids, fb = r.column("cid").to_pylist(), False
        except Exception:
            r = (table.search(vec, query_type="vector", vector_column_name="vector").distance_type("cosine")
                 .limit(v["pool"]).to_arrow())
            cids, fb = r.column("cid").to_pylist(), True
        if not v["docs"]:
            cids = cids[:10]  # выдача как сейчас: 10 чанков, документы — по первому вхождению
        docs, frags = [], collections.defaultdict(list)
        for c in cids:  # строки уже по убыванию оценки; документ — по лучшему чанку
            d = rows[c]["doc"]
            if d not in docs:
                docs.append(d)
            if kind_of[c] != "card" and len(frags[d]) < MAX_FRAGS:
                frags[d].append(c)
        if v["name_code"]:
            hits = name_hits(q)
            docs = [d for d in docs if d in hits] + [d for d in hits if d not in docs] + [d for d in docs if d not in hits]
        docs = docs[:10]
        return {"docs": docs, "frags": {d: frags[d] for d in docs if frags.get(d)}, "fb": fb,
                "rule": bool(v["name_code"] and name_hits(q))}

    queries = jl(EVAL / "data" / "queries.jsonl")
    nav = jl(NAV) if NAV.exists() else []
    out = {"variant": {**v, "name": a.name}, "project_head": git_head(v["project"]), "config": os.environ["CONFIG_PATH"],
           "splitter": splitter, "embedder": emb.fp, "parser_keys": src.keys, "table": key, "rows": len(rows),
           "cards": sum(k == "card" for k in kind_of),
           "doc_ids": sorted({r["doc"] for r in rows if r["kind"] != "card"}),
           "mean_chars": round(statistics.mean(len(strip_prefix(r["text"])) for r in rows if r["kind"] != "card")),
           "embed_seconds": embed_s}
    sample = random.Random(0).sample([r["text"] for r in rows if r["kind"] != "card"], min(2000, len(rows)))
    out["mean_tokens"] = round(statistics.mean(emb.ov.count_tokens(strip_prefix(t)) - 2 for t in sample))
    for part, qs, field in (("main", queries, "qid"), ("nav", nav, "nid")):
        qv = emb.get([q["query"] for q in qs], "query") if qs else []
        t2 = time.time()
        out[part] = {q[field]: search(q["query"], vec) for q, vec in zip(qs, qv)}
        out[f"{part}_ms"] = round(1000 * (time.time() - t2) / max(1, len(qs)), 1)
    out["seconds"] = round(time.time() - t0)
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{a.name}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    fb = sum(x["fb"] for p in ("main", "nav") for x in out[p].values())
    log(f"{a.name}: строк {len(rows)}, ~{out['mean_tokens']} ток. на чанк, эмбеддинг {embed_s} c, "
        f"запросов {len(out['main'])} + {len(out['nav'])}, откатов на вектор {fb}, всего {out['seconds']} c")


# ---------- отчёт ----------

def parent_key(r: dict, ext: str):
    if r["kind"] in ("table", "table_row", "notes") or r.get("table_id"):
        return ("t", r.get("table_id") or r["cid"])
    if ext in ("pdf", "pptx") or r["kind"] == "ocr":
        return ("p", r.get("page_start"))
    if r.get("section"):
        return ("s", r["section"])
    return None


def stitch(parts: list[dict]) -> tuple[str, dict]:
    """Текст подряд идущих чанков по их позициям в документе; смещение каждого чанка в нём."""
    parts = sorted(parts, key=lambda r: (r.get("char_start") if r.get("char_start") is not None else 10**12, r["cid"]))
    text, end, offs = "", None, {}
    for r in parts:
        t, s = strip_prefix(r["text"]), r.get("char_start")
        if end is not None and s is not None and s < end:  # перекрытие с уже собранным
            cut = min(len(t), end - s)
            offs[r["cid"]] = len(text) - cut
            text += t[cut:]
        else:
            if text:
                text += "\n"
            offs[r["cid"]] = len(text)
            text += t
        end = max(end or 0, s + len(t)) if s is not None else None
    return text, offs


def window(text: str, start: int, length: int) -> str:
    c = start + length // 2
    lo = max(0, min(len(text) - BUDGET, c - BUDGET // 2))
    return text[lo:lo + BUDGET]


class Units:
    """Текст, который выдача отдаёт по документу: найденные чанки, окно или родитель."""

    def __init__(self, key: str, meta: dict):
        self.by_cid, self.by_doc = {}, collections.defaultdict(list)
        for r in load_rows(key):
            if r["kind"] == "card":
                continue
            self.by_cid[r["cid"]] = r
            self.by_doc[r["doc"]].append(r)
        for d in self.by_doc:
            self.by_doc[d].sort(key=lambda r: (r["idx"] if r["idx"] is not None else 0))
        self.ext = {d: (m.get("ext") or "") for d, m in meta.items()}

    def text(self, unit: str, doc: str, cids: list[int]) -> str:
        if not cids:
            return ""
        if unit == "chunk":
            return "\n".join(strip_prefix(self.by_cid[c]["text"]) for c in cids)[:BUDGET]
        best = self.by_cid[cids[0]]
        rows = self.by_doc[doc]
        i = next(k for k, r in enumerate(rows) if r["cid"] == best["cid"])
        pk = parent_key(best, self.ext.get(doc, ""))
        if unit == "parent" and pk is not None:
            if pk[0] == "p":
                lo, hi = best.get("page_start"), best.get("page_end") or best.get("page_start")
                group = [r for r in rows if r.get("page_start") is not None and r["page_start"] <= hi
                         and (r.get("page_end") or r["page_start"]) >= lo and r["kind"] == best["kind"]]
            else:
                group = [r for r in rows if parent_key(r, self.ext.get(doc, "")) == pk]
        else:  # окно: найденный чанк и соседи по порядку
            group = rows[max(0, i - 1):i + 2]
        text, offs = stitch(group)
        return window(text, offs[best["cid"]], len(strip_prefix(best["text"])))


def answer_refs(qids: set) -> dict:
    """Эталоны «ответа во фрагменте»: qid → [(doc_id, проверка текста)]."""
    import parse_checks as pc

    refs = collections.defaultdict(list)
    passages, _ = pc.reference_passages()
    for ds, qid, doc, text in passages:
        if qid in qids:
            ref = pc.grams(pc.tok(text))
            if ref:
                refs[qid].append((doc, lambda t, ref=ref: len(ref & pc.grams(pc.tok(t))) / len(ref) >= 0.8))
    for g in pc.xlsx_gold():
        if g["qid"] not in qids:
            continue
        label = pc.norm(g["label"].translate(pc._SUPERSCRIPTS))
        tol = max(0.051, 1e-6 * abs(g["value"]))
        human = None
        if re.fullmatch(r"\d{4}-\d\d-\d\d", g["period"]):
            y, mo, dd = map(int, g["period"].split("-"))
            human = f"{dd} {pc.MONTHS[mo - 1]} {y}"

        def ok(t, label=label, tol=tol, human=human, value=g["value"]):
            n = pc.norm(t.translate(pc._SUPERSCRIPTS))
            return (label in n and any(abs(x - value) <= tol for x in pc.numbers_in(t))
                    and (human is None or human in n))
        refs[g["qid"]].append((g["doc_id"], ok))
    return refs


def fmt_ci(pt, lo, hi) -> str:
    return f"{pt:+.3f} [{lo:+.3f}…{hi:+.3f}]"


def cmd_report(a):
    import parse_checks as pc
    from metrics import bootstrap_ci, per_query, relevant
    from report import load_qrels

    runs = {n: json.loads((RUNS / f"{n}.json").read_text(encoding="utf-8")) for n in a.names}
    base_name, base = a.names[0], runs[a.names[0]]
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    queries = {q["qid"]: q for q in jl(EVAL / "data" / "queries.jsonl")}
    qrels = load_qrels()
    part_q = set(queries) if a.part == "all" else set(split["main"][a.part])
    part_d = None if a.part == "all" else set(split["nav_docs"][a.part])
    in_index = set.intersection(*(set(r["doc_ids"]) for r in runs.values()))
    ans = sorted(q for q in part_q if queries[q]["scenario"] != 6 and relevant(qrels.get(q, {})) & in_index)
    clusters = {q: queries[q].get("base") or q for q in queries}
    metrics = a.metrics.split(",")
    lines = [f"# Сравнение вариантов: {', '.join(a.names)} (база {base_name}), часть «{a.part}»", "",
             f"Обычных запросов с ответом: {len(ans)}.", "", "## Варианты", "",
             "| Вариант | Проект | Строк (карточек) | ~токенов на чанк | Пул | Выдача | Метаданные | Документов в выдаче | Неоценённых в топ-10 | Эмбеддинг, с | Поиск, мс |",
             "|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|"]
    for n, r in runs.items():
        v = r["variant"]
        md = ", ".join(x for x, on in (("карточка", v["card"]), ("код из имени", v["name_code"]),
                                       (f"поля {v['fields']}", v["fields"])) if on) or "—"
        nd = statistics.mean(len(r["main"][q]["docs"]) for q in ans)
        unj = sum(1 for q in ans for d in r["main"][q]["docs"][:10] if d not in qrels.get(q, {}))
        lines.append(f"| {n} | {r['project_head']} | {r['rows']} ({r['cards']}) | {r['mean_tokens']} | {v['pool']} | "
                     f"{'10 документов' if v['docs'] else '10 чанков'} | {md} | {nd:.1f} | {unj} | {r['embed_seconds']} | "
                     f"{r['main_ms']} |")
    # ранжирование обычных запросов: две версии неоценённых
    lines += ["", "## Обычные запросы: ранжирование документов", "",
              "«Правильные» — неоценённые документы топ-10 обеих выдач получают оценку 2 (одинаково для обеих сторон).", ""]
    slices = pc.query_slices(ans)
    keep = [s for s in slices if s == "Итого" or s.startswith("сценарий") or s.startswith("набор")]
    for n in a.names[1:]:
        run = {q: runs[n]["main"][q]["docs"] for q in ans}
        bas = {q: base["main"][q]["docs"] for q in ans}
        up = {q: {**qrels.get(q, {}), **{d: 2 for d in set(run[q][:10]) | set(bas[q][:10]) if d not in qrels.get(q, {})}}
              for q in ans}
        pr, pb, pru, pbu = per_query(run, qrels, ans), per_query(bas, qrels, ans), per_query(run, up, ans), per_query(bas, up, ans)
        lines += [f"### {n} против {base_name}", "",
                  "| Срез | Запросов | Метрика | " + base_name + " | " + n + " | Разница [95% ДИ] | Если неоценённые правильные |",
                  "|---|---:|---|---:|---:|---:|---:|"]
        for s in keep:
            qs = slices[s]
            for m in metrics:
                d = {q: {m: pr[q][m] - pb[q][m]} for q in qs}
                du = {q: {m: pru[q][m] - pbu[q][m]} for q in qs}
                mb, mr = statistics.mean(pb[q][m] for q in qs), statistics.mean(pr[q][m] for q in qs)
                lines.append(f"| {s} | {len(qs)} | {m} | {mb:.3f} | {mr:.3f} | {fmt_ci(*bootstrap_ci(d, m, clusters))} | "
                             f"{fmt_ci(*bootstrap_ci(du, m, clusters))} |")
        fired = [q for q in ans if runs[n]["main"][q].get("rule")]
        if fired:
            cells = ", ".join(f"{m} {statistics.mean(pb[q][m] for q in fired):.3f} → {statistics.mean(pr[q][m] for q in fired):.3f}"
                              for m in ("Hit@1", "MRR@10", "nDCG@10"))
            lines += ["", f"Правило «код из имени» сработало в {len(fired)} из {len(ans)} запросов: {cells}."]
        lines.append("")
    # навигационные запросы
    nav = [x for x in (jl(NAV) if NAV.exists() else []) if part_d is None or x["doc_id"] in part_d]
    if nav and all(r.get("nav") for r in runs.values()):
        lines += ["## Навигационные запросы (известный ответ — сам документ): Hit@1 / Hit@10 / MRR@10", "",
                  "| Тип | Запросов | " + " | ".join(a.names) + " | Разница Hit@1 с базой [95% ДИ] |",
                  "|---|---:|" + "---:|" * len(a.names) + "---:|"]
        for t in ["все", *sorted({x["type"] for x in nav})]:
            xs = [x for x in nav if t == "все" or x["type"] == t]
            cells, h1 = [], {}
            for n, r in runs.items():
                res = []
                for x in xs:
                    docs = r["nav"][x["nid"]]["docs"]
                    rank = docs.index(x["doc_id"]) + 1 if x["doc_id"] in docs else None
                    res.append((rank == 1, rank is not None and rank <= 10, 1 / rank if rank else 0.0))
                h1[n] = {x["nid"]: {"h": float(z[0])} for x, z in zip(xs, res)}
                cells.append(" / ".join(f"{statistics.mean(z[i] for z in res):.2f}" for i in range(3)))
            last = a.names[-1]
            diff = {k: {"h": h1[last][k]["h"] - h1[base_name][k]["h"]} for k in h1[base_name]}
            lines.append(f"| {t} | {len(xs)} | " + " | ".join(cells) + f" | {last}: "
                         f"{fmt_ci(*bootstrap_ci(diff, 'h', {x['nid']: x['doc_id'] for x in xs}))} |")
        lines.append("")
    # ответ во фрагменте
    refs = answer_refs(set(ans))
    if refs:
        units = a.units.split(",")
        lines += [f"## Ответ во фрагменте ({len(refs)} запросов с эталоном; бюджет {BUDGET} символов на документ)", "",
                  "Засчитано, если эталонный документ в топ-1 / топ-3 и в отданном по нему тексте есть ответ.", "",
                  "| Вариант | Единица выдачи | Ответ@1 | Ответ@3 | Разница Ответ@3 с базой (тот же вид) [95% ДИ] |",
                  "|---|---|---:|---:|---:|"]
        stats = {}
        for n, r in runs.items():
            u = Units(r["table"], load_meta(n))
            for unit in units:
                per = {}
                for q, rs in refs.items():
                    res = r["main"][q]
                    h1 = h3 = 0.0
                    for k, d in enumerate(res["docs"][:3]):
                        if any(doc == d and ok(u.text(unit, d, res["frags"].get(d, []))) for doc, ok in rs):
                            h3 = 1.0
                            h1 = 1.0 if k == 0 else h1
                    per[q] = {"a1": h1, "a3": h3}
                stats[(n, unit)] = per
        for n in a.names:
            for unit in units:
                per = stats[(n, unit)]
                diff = ""
                if n != base_name:
                    d = {q: {"a3": per[q]["a3"] - stats[(base_name, unit)][q]["a3"]} for q in per}
                    diff = fmt_ci(*bootstrap_ci(d, "a3", clusters))
                lines.append(f"| {n} | {unit} | {statistics.mean(x['a1'] for x in per.values()):.3f} | "
                             f"{statistics.mean(x['a3'] for x in per.values()):.3f} | {diff} |")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


def load_meta(name: str) -> dict:
    return json.loads((ROWS / f"{name}.meta.json").read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("split")
    n = sub.add_parser("nav")
    n.add_argument("--base", default="C0")
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("--force", action="store_true")
    c = sub.add_parser("_run")
    c.add_argument("name")
    p = sub.add_parser("report")
    p.add_argument("names", nargs="+")
    p.add_argument("--part", choices=("dev", "test", "all"), default="dev")
    p.add_argument("--metrics", default="Hit@1,MRR@10,nDCG@10,Hit@10,Recall@10")
    p.add_argument("--units", default="chunk,window,parent")
    p.add_argument("--out")
    a = ap.parse_args()
    {"split": cmd_split, "nav": cmd_nav, "run": cmd_run, "_run": cmd_child_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
