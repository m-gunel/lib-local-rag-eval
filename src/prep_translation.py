"""Подготовка единиц перевода: документы ZX Bank и главы ObliQA, плюс запросы.

Пишет work/translate/:
  docs/<unit_id>.en.md   — исходный текст документа (перевести в <unit_id>.ru.md)
  units.jsonl            — метаданные документов (набор, формат в корпусе, источник)
  queries_zx.jsonl       — запросы ZX Bank с эталоном (документ + фрагменты)
  queries_obliqa.jsonl   — выборка вопросов ObliQA с эталоном (главы)
"""

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
OUT = ROOT / "work" / "translate"
DOCS = OUT / "docs"

# Побайтные дубли в ZX Bank: оставляем первый, запросы ко второму переназначаем.
ZX_DUPLICATES = {
    "Personal Loan Information.md": "Personal Loan.md",
    "ZX Bank Agriculture Loan.md": "Agriculture Loan.md",
}
ZX_FORMATS = ("docx", "pdf", "pptx")

# ObliQA: своды, ближайшие к банку (номера DocumentID из DocumentMap.rtf).
OBLIQA_DOCS = {
    1: "AML",  # Anti-Money Laundering and Sanctions Rules and Guidance
    14: "BRR",  # Bank Recovery and Resolution Regulations 2018
    36: "CLIMATE",  # Principles for the Effective Management of Climate-related Financial Risks
    15: "CRS",  # Common Reporting Standard Regulations 2017
    16: "FATCA",  # Foreign Account Tax Compliance Regulations
}
OBLIQA_QUESTIONS = 250
SEED = 20260925


def slug(name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return base[:60] or hashlib.md5(name.encode()).hexdigest()[:8]


def prep_zx(units: list[dict]) -> None:
    zx_dir = RAW / "rag-multi-corpus" / "datasets" / "ZX Bank"
    md_files = sorted(
        p.name for p in (zx_dir / "md").iterdir() if p.suffix == ".md"
    )
    for dup, keep in ZX_DUPLICATES.items():
        a = (zx_dir / "md" / dup).read_bytes()
        b = (zx_dir / "md" / keep).read_bytes()
        assert a == b, f"{dup} и {keep} не совпадают побайтно"
    kept = [n for n in md_files if n not in ZX_DUPLICATES]
    # Формат в корпусе — по кругу в алфавитном порядке: каждый документ
    # присутствует ровно в одном формате, без триплетов одного текста.
    name_to_unit = {}
    for i, name in enumerate(kept):
        uid = f"zx_{slug(name[:-3])}"
        fmt = ZX_FORMATS[i % len(ZX_FORMATS)]
        (DOCS / f"{uid}.en.md").write_text(
            (zx_dir / "md" / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
        units.append(
            {
                "unit_id": uid,
                "dataset": "zx",
                "source_name": name,
                "corpus_format": fmt,
                "title_en": name[:-3],
            }
        )
        name_to_unit[name] = uid
    for dup, keep in ZX_DUPLICATES.items():
        name_to_unit[dup] = name_to_unit[keep]

    rows = list(
        csv.DictReader(
            open(
                RAW / "rag-multi-corpus" / "datasets" / "Dataset categories - queries_01122025.csv",
                encoding="utf-8",
            )
        )
    )
    dropped = 0
    with (OUT / "queries_zx.jsonl").open("w", encoding="utf-8") as f:
        n = 0
        for r in rows:
            if r["Enterprise Name"].strip() != "ZX Bank":
                continue
            facts = json.loads(r["Supporting Facts"])
            files = {x["filename"] for x in facts}
            if any(fn not in name_to_unit for fn in files):
                dropped += 1
                continue
            n += 1
            f.write(
                json.dumps(
                    {
                        "qid": f"zx{n:04d}",
                        "query_en": r["Query"].strip(),
                        "query_type": r["Query Type"].strip(),
                        "gold_units": sorted({name_to_unit[fn] for fn in files}),
                        "facts_en": [
                            {"unit_id": name_to_unit[x["filename"]], "text": x["text"]}
                            for x in facts
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"ZX: документов {len(kept)}, запросов {n}, выброшено (нет файла) {dropped}")
    print("ZX форматы:", Counter(u["corpus_format"] for u in units if u["dataset"] == "zx"))


def passage_heading(pid: str) -> str:
    return pid.rstrip(".")


def prep_obliqa(units: list[dict]) -> None:
    obl = RAW / "obliqa"
    chapters: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for did, code in OBLIQA_DOCS.items():
        passages = json.load(open(obl / "StructuredRegulatoryDocuments" / f"{did}.json"))
        for p in passages:
            chap = p["PassageID"].split(".")[0]
            chapters[(did, chap)].append(p)
    chap_unit = {}
    for (did, chap), passages in sorted(chapters.items(), key=lambda x: (x[0][0], int(x[0][1]) if x[0][1].isdigit() else 999)):
        code = OBLIQA_DOCS[did]
        uid = f"obl_{code.lower()}_ch{slug(chap)}"
        lines = []
        for p in passages:
            text = p["Passage"].strip()
            if not text:
                continue
            lines.append(f"[{passage_heading(p['PassageID'])}] {text}")
        (DOCS / f"{uid}.en.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
        chap_unit[(did, chap)] = uid
        units.append(
            {
                "unit_id": uid,
                "dataset": "obliqa",
                "source_name": f"{code} chapter {chap}",
                "corpus_format": "txt",
                "title_en": f"{code} chapter {chap}: {passages[0]['Passage'][:80]}",
                "words": sum(len(p["Passage"].split()) for p in passages),
            }
        )

    test = json.load(open(obl / "ObliQA_test.json"))
    eligible = []
    for q in test:
        keys = {(p["DocumentID"], p["PassageID"].split(".")[0]) for p in q["Passages"]}
        if keys and all(k in chap_unit for k in keys):
            eligible.append((q, sorted({chap_unit[k] for k in keys})))
    rnd = random.Random(SEED)
    # Стратифицируем по своду, чтобы маленькие своды не пропали из выборки.
    by_doc = defaultdict(list)
    for q, us in eligible:
        by_doc[us[0].split("_ch")[0]].append((q, us))
    total = len(eligible)
    picked = []
    for code, items in sorted(by_doc.items()):
        k = max(5, round(OBLIQA_QUESTIONS * len(items) / total))
        picked.extend(rnd.sample(items, min(k, len(items))))
    with (OUT / "queries_obliqa.jsonl").open("w", encoding="utf-8") as f:
        for n, (q, us) in enumerate(picked, 1):
            f.write(
                json.dumps(
                    {
                        "qid": f"obl{n:04d}",
                        "source_qid": q["QuestionID"],
                        "query_en": q["Question"].strip(),
                        "gold_units": us,
                        "facts_en": [
                            {
                                "unit_id": chap_unit[(p["DocumentID"], p["PassageID"].split(".")[0])],
                                "passage_id": p["PassageID"],
                                "text": p["Passage"],
                            }
                            for p in q["Passages"]
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    words = sum(u.get("words", 0) for u in units if u["dataset"] == "obliqa")
    print(
        f"ObliQA: глав {len(chap_unit)}, слов {words}, подходящих вопросов {total}, "
        f"в выборке {len(picked)} {Counter(us[0].split('_ch')[0] for _, us in picked)}"
    )


PART_WORDS = 3500


def split_parts(units: list[dict]) -> None:
    """Режет длинные тексты на части по границам абзацев для перевода по кускам."""
    parts_dir = OUT / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for old in parts_dir.glob("*.en.md"):
        old.unlink()
    total = 0
    for u in units:
        text = (DOCS / f"{u['unit_id']}.en.md").read_text(encoding="utf-8")
        paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks, cur, words = [], [], 0
        for p in paras:
            w = len(p.split())
            if cur and words + w > PART_WORDS:
                chunks.append(cur)
                cur, words = [], 0
            cur.append(p)
            words += w
        if cur:
            chunks.append(cur)
        u["parts"] = []
        for i, c in enumerate(chunks, 1):
            pid = f"{u['unit_id']}__p{i:02d}"
            (parts_dir / f"{pid}.en.md").write_text("\n\n".join(c) + "\n", encoding="utf-8")
            u["parts"].append(pid)
        total += len(chunks)
    print(f"частей перевода: {total}")


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    for old in DOCS.glob("*.en.md"):
        old.unlink()
    units: list[dict] = []
    prep_zx(units)
    prep_obliqa(units)
    split_parts(units)
    with (OUT / "units.jsonl").open("w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")
    print(f"единиц перевода: {len(units)}")


if __name__ == "__main__":
    main()
