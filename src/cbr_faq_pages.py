"""FAQ Банка России для граждан (набор Qessia/cbr_rag): скачать актуальные
страницы cbr.ru и проверить, что ответ на каждый вопрос на странице ещё есть.

Набор собран в 2024 году; страницы с тех пор менялись. Пары, ответ которых
на странице не находится, в эталон не попадают (иначе разметка врёт).

Пишет raw/cbr_faq/pages/<slug>.html, pages/<slug>.txt и raw/cbr_faq/pairs.jsonl.
"""

import csv
import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
OUT = RAW / "cbr_faq"
PAGES = OUT / "pages"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126 Safari/537.36"
)


def slug(url: str) -> str:
    path = url.split("cbr.ru/", 1)[1].strip("/")
    return re.sub(r"[^A-Za-z0-9_-]+", "_", path).strip("_").lower()


def page_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "form", "header", "footer", "nav"]):
        tag.decompose()
    h1 = soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else ""
    main = soup.select_one("main") or soup.select_one(".page-content") or soup.body or soup
    for sel in [".breadcrumbs", ".page-info", ".b-share", ".subscribe", ".page-rating"]:
        for n in main.select(sel):
            n.decompose()
    blocks = []
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "div"], recursive=True):
        if el.name == "div" and el.find(["p", "li", "div", "h2", "h3", "td"]):
            continue
        t = el.get_text(" ", strip=True)
        if t and (not blocks or blocks[-1] != t):
            blocks.append(t)
    noise = {
        "Да Нет", "Отправить", "Ответ не помог решить проблему",
        "Ответ ссылается на неактуальные данные", "Спасибо, что помогаете нам стать лучше!",
        "Пожалуйста, расскажите, почему вам не подошел этот ответ?",
    }
    blocks = [b for b in blocks if b not in noise and not re.fullmatch(r"\d{1,3}", b)]
    text = "\n\n".join(blocks)
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return title, text


def norm(s: str) -> str:
    s = s.lower().replace("ё", "е")
    return re.sub(r"[^0-9a-zа-я]+", " ", s).strip()


def answer_found(answer: str, page: str) -> float:
    """Доля предложений ответа (≥40 символов), найденных на странице."""
    p = norm(page)
    sents = [norm(x) for x in re.split(r"(?<=[.!?])\s+", answer) if len(x) >= 40]
    if not sents:
        sents = [norm(answer)]
    hit = 0
    for s in sents:
        if s in p:
            hit += 1
            continue
        # Мелкие правки формулировок: ищем близкий фрагмент той же длины.
        probe = s[:60]
        i = p.find(probe[:30])
        if i >= 0 and SequenceMatcher(None, s, p[i : i + len(s) + 20]).ratio() > 0.85:
            hit += 1
    return hit / len(sents)


def main() -> None:
    PAGES.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(RAW / "cbr_rag" / "data.csv", encoding="utf-8")))
    urls = sorted({r["link"].strip() for r in rows})
    texts = {}
    with httpx.Client(headers={"User-Agent": UA}, timeout=90, follow_redirects=True) as c:
        for u in urls:
            s = slug(u)
            hp = PAGES / f"{s}.html"
            if not hp.exists():
                r = c.get(u)
                if r.status_code != 200:
                    print(f"{r.status_code} {u}")
                    continue
                hp.write_text(r.text, encoding="utf-8")
                time.sleep(1.0)
            title, text = page_text(hp.read_text(encoding="utf-8"))
            (PAGES / f"{s}.txt").write_text(f"{title}\n\n{text}\n", encoding="utf-8")
            texts[u] = (s, title, text)
    kept = 0
    with (OUT / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows, 1):
            u = r["link"].strip()
            if u not in texts:
                continue
            s, title, text = texts[u]
            score = answer_found(r["answer"], text)
            q_on_page = norm(r["text"])[:80] in norm(text)
            rec = {
                "fid": f"faq{i:04d}",
                "question": r["text"].strip(),
                "answer": r["answer"].strip(),
                "url": u,
                "page": s,
                "page_title": title,
                "answer_on_page": round(score, 2),
                "question_on_page": q_on_page,
            }
            rec["relevant"] = bool(q_on_page or score >= 0.6)
            kept += rec["relevant"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"страниц {len(texts)} из {len(urls)}; пар, подтверждённых страницей (вопрос или ответ ≥60%): {kept} из {len(rows)}")


if __name__ == "__main__":
    main()
