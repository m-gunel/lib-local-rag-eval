"""Дополнительные документы cbr.ru для корпуса (расширение до ~400 документов).

- «Разъяснения» Банка России по банковскому регулированию и формам отчётности
  кредитных организаций (HTML → текст) — близкие по теме «соседи» для вопросов
  кредитных организаций к ЦБ;
- разделы FAQ, которых нет в наборе FAQ ЦБ;
- ежемесячные обзоры «О развитии банковского сектора» (цифровые PDF);
- ещё два выпуска xlsx «Статистические показатели банковского сектора».

Пишет raw/cbr_extra/{pages,analytics}/ и raw/xlsx/. Страницы без текста или с
ошибкой пропускаются и перечисляются в выводе.
"""

import json
import re
import sys
import time
from pathlib import Path

import httpx
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cbr_faq_pages import page_text, slug  # noqa: E402

EVAL = Path(__file__).resolve().parent.parent
OUT = EVAL / "raw" / "cbr_extra"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126 Safari/537.36"
)
BASE = "https://www.cbr.ru"

# Банковское регулирование и отчётность КО. Страхование, НПФ, МФО, биржи,
# коллективные инвестиции и формы БКИ не берём — это не банковская среда.
EXPLAN = [
    "0409212", "0409909_01042023", "353-fz", "590-p", "845-p", "Psystem", "accounts_budgets",
    "acts_bko", "acts_bnko", "acts_bs", "acts_psv", "ckki", "corporate_rel", "dopusk",
    "form_0403301", "form_0409112", "form_0409126", "form_0409128", "form_0409129",
    "form_0409135", "form_0409260", "form_0409264", "form_0409265", "form_0409302",
    "form_0409303", "form_0409310", "form_0409316", "form_0409702", "form_0409704",
    "form_0409707", "form_0409711", "form_0409724", "ko_0409708", "likvid_ko",
    "macroprudential_limits", "measures_support_citizens_economy", "mery-podderzhki-fin-sektora",
    "obschie-voprosy", "oper_br/reserve_requirements", "osbu", "pcod", "pips_br", "r_0409119",
]
FAQ = ["bank_s", "credit_h", "foreign_exchange_market", "nps", "oper_br", "ucbr", "w_fin_sector"]
ANALYTICS_PAGE = "/analytics/bank_sector/develop/"
ANALYTICS_N = 8
XLSX_ISSUES = [280, 281]


def get(c: httpx.Client, url: str) -> httpx.Response:
    r = c.get(url)
    time.sleep(1.0)
    return r


def main() -> None:
    (OUT / "pages").mkdir(parents=True, exist_ok=True)
    (OUT / "analytics").mkdir(parents=True, exist_ok=True)
    report = {"pages": [], "skipped": [], "analytics": [], "xlsx": []}
    with httpx.Client(headers={"User-Agent": UA}, timeout=120, follow_redirects=True) as c:
        for kind, items in (("explan", EXPLAN), ("faq", FAQ)):
            for it in items:
                url = f"{BASE}/{kind}/{it}/"
                s = slug(url)
                hp = OUT / "pages" / f"{s}.html"
                if not hp.exists():
                    r = get(c, url)
                    if r.status_code != 200:
                        report["skipped"].append((url, r.status_code))
                        continue
                    hp.write_text(r.text, encoding="utf-8")
                title, text = page_text(hp.read_text(encoding="utf-8"))
                if len(text) < 800:
                    report["skipped"].append((url, f"мало текста: {len(text)}"))
                    continue
                (OUT / "pages" / f"{s}.txt").write_text(f"{title}\n\n{text}\n", encoding="utf-8")
                report["pages"].append({"slug": s, "url": url, "title": title, "chars": len(text)})
        html = get(c, BASE + ANALYTICS_PAGE).text
        pdfs = re.findall(r'href="(/Collection/Collection/File/\d+/razv_bs_(\d\d)_(\d\d)\.pdf)"', html)
        pdfs = sorted(set(pdfs), key=lambda x: (x[1], x[2]), reverse=True)[:ANALYTICS_N]
        for href, yy, mm in pdfs:
            dest = OUT / "analytics" / f"razv_bs_{yy}_{mm}.pdf"
            if not dest.exists():
                r = get(c, BASE + href)
                if r.status_code != 200 or not r.content.startswith(b"%PDF"):
                    report["skipped"].append((href, r.status_code))
                    continue
                dest.write_bytes(r.content)
            doc = pymupdf.open(dest)
            chars = sum(len(p.get_text()) for p in doc)
            report["analytics"].append({"file": dest.name, "pages": len(doc), "chars_per_page": chars // max(len(doc), 1),
                                        "period": f"20{yy}-{mm}"})
        review = (EVAL / "raw" / "xlsx" / "review.html").read_text(encoding="utf-8")
        for n in XLSX_ISSUES:
            m = re.search(rf'href="([^"]*obs_{n}\.xlsx)"', review)
            dest = EVAL / "raw" / "xlsx" / f"obs_{n}.xlsx"
            if m and not dest.exists():
                r = get(c, BASE + m.group(1))
                if r.status_code == 200 and r.content[:2] == b"PK":
                    dest.write_bytes(r.content)
            report["xlsx"].append({"file": dest.name, "ok": dest.exists()})
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"страниц: {len(report['pages'])}, аналитики: {len(report['analytics'])}, xlsx: "
          f"{sum(x['ok'] for x in report['xlsx'])}, пропущено: {len(report['skipped'])}")
    for s in report["skipped"]:
        print("  пропущено:", s)
    for a in report["analytics"]:
        print("  ", a)


if __name__ == "__main__":
    main()
