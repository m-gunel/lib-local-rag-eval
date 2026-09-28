"""Сбор вопросов кредитных организаций к Банку России (cbr.ru/faq_ufr/dbrnfaq).

Скачивает оглавление и страницы разделов (по одному акту на раздел), разбирает
блоки «Вопрос / Ответ» и пишет raw/cbr_dbrnfaq/questions.jsonl.

Из текста ответа извлекаются ссылки на другие акты — они станут
документами с оценкой 2 в разметке. Условия cbr.ru: воспроизведение
разрешено со ссылкой на источник, поэтому у каждой записи сохраняется url.
"""

import json
import re
import time
from pathlib import Path
from urllib.parse import quote, urljoin

import httpx
from bs4 import BeautifulSoup

BASE = "https://www.cbr.ru"
ROOT = Path(__file__).resolve().parent.parent / "raw" / "cbr_dbrnfaq"
HTML = ROOT / "html"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126 Safari/537.36"
)
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

# Номер акта ЦБ: «№ 716-П», «№ 4927-У», «№ 199-И», «№ 12-МР»; закон: «№ 395-1», «№ 115-ФЗ».
ACT_RE = re.compile(r"№\s*(\d{1,5})\s*[-‑–]\s*(ПР|МР|П|У|И|Т)(?![а-яА-ЯёЁ])")
LAW_RE = re.compile(r"№\s*(\d{1,4}(?:-\d)?)\s*[-‑–]\s*ФЗ|№\s*(395-1)\b")


def fetch(client: httpx.Client, url: str, dest: Path) -> str:
    if dest.exists():
        return dest.read_text(encoding="utf-8")
    r = client.get(url)
    r.raise_for_status()
    dest.write_text(r.text, encoding="utf-8")
    time.sleep(1.0)
    return r.text


def text_of(nodes) -> str:
    parts = []
    for n in nodes:
        if getattr(n, "name", None) is None:
            s = str(n).strip()
            if s:
                parts.append(s)
            continue
        for sup in n.find_all("sup"):
            sup.replace_with(sup.get_text().translate(SUP))
        s = n.get_text(" ", strip=True)
        if s:
            parts.append(s)
    text = "\n".join(parts)
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def norm_act(num: str, kind: str) -> str:
    return f"{int(num)}-{kind}"


def acts_in(text: str) -> list[str]:
    return sorted({norm_act(n, k) for n, k in ACT_RE.findall(text)})


def laws_in(text: str) -> list[str]:
    out = set()
    for a, b in LAW_RE.findall(text):
        out.add(f"{a}-ФЗ" if a else b)
    return sorted(out)


def parse_section(html: str, number: str, url: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    h1 = soup.find("h1")
    act_title = h1.get_text(" ", strip=True) if h1 else number
    out = []
    for block in soup.select("div.dropdown.question"):
        num = block.select_one(".question_num")
        topic = block.select_one(".question_title")
        content = block.select_one(".dropdown_content")
        if content is None:
            continue
        # Делим содержимое на часть «Вопрос» и часть «Ответ» по заголовку «Ответ».
        q_nodes, a_nodes, in_answer = [], [], False
        for child in content.children:
            name = getattr(child, "name", None)
            cls = child.get("class", []) if name else []
            if name and "dropdown_content_title" in cls:
                if "Ответ" in child.get_text():
                    in_answer = True
                continue
            (a_nodes if in_answer else q_nodes).append(child)
        q_text = text_of(q_nodes)
        a_text = text_of(a_nodes)
        q_date = re.match(r"от\s+(\d{2}\.\d{2}\.\d{4})", q_text)
        a_head = re.match(r"от\s+(\d{2}\.\d{2}\.\d{4})\s*(№\s*\S+)?", a_text)
        if q_date:
            q_text = q_text[q_date.end():].strip()
        a_number = None
        if a_head:
            a_number = (a_head.group(2) or "").replace("№", "").strip() or None
            a_text = a_text[a_head.end():].strip()
        ref_acts = acts_in(a_text)
        out.append(
            {
                "section": number,
                "act_title": act_title,
                "num": num.get_text(strip=True) if num else None,
                "topic": topic.get_text(" ", strip=True) if topic else None,
                "question_date": q_date.group(1) if q_date else None,
                "question": q_text,
                "answer_date": a_head.group(1) if a_head else None,
                "answer_number": a_number,
                "answer": a_text,
                "ref_acts": ref_acts,
                "ref_laws": laws_in(a_text),
                "url": url,
            }
        )
    return out


def main() -> None:
    HTML.mkdir(parents=True, exist_ok=True)
    rows = []
    with httpx.Client(headers={"User-Agent": UA}, timeout=90, follow_redirects=True) as client:
        index = fetch(client, f"{BASE}/faq_ufr/dbrnfaq/", HTML / "index.html")
        soup = BeautifulSoup(index, "lxml")
        numbers = []
        for a in soup.find_all("a", href=True):
            m = re.search(r"dbrnfaq/doc\?number=([^&#]+)", a["href"])
            if m:
                numbers.append(httpx.URL(a["href"]).params.get("number") or m.group(1))
        numbers = list(dict.fromkeys(numbers))
        print(f"разделов: {len(numbers)}")
        for number in numbers:
            url = f"{BASE}/faq_ufr/dbrnfaq/doc?number={quote(number)}"
            fname = number.replace("/", "_") + ".html"
            html = fetch(client, url, HTML / fname)
            items = parse_section(html, number, url)
            m = re.search(r"(\d+)\s*вопрос", BeautifulSoup(html, "lxml").select_one(".counter").get_text() if BeautifulSoup(html, "lxml").select_one(".counter") else "")
            expected = int(m.group(1)) if m else None
            print(f"{number}: разобрано {len(items)}, на странице {expected}")
            rows.extend(items)
    for i, r in enumerate(rows):
        r["qid"] = f"cbr{i + 1:04d}"
    out = ROOT / "questions.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"всего вопросов: {len(rows)} → {out}")


if __name__ == "__main__":
    main()
