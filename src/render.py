"""Сборка файлов корпуса из markdown тем же способом, что у оригиналов ZX Bank.

Оригиналы RAG-Multi-Corpus сделаны pandoc из md: docx (стили pandoc, таблицы
настоящие), pptx (слайд на заголовок, таблицы — объекты-таблицы), pdf через
LaTeX. Здесь: docx и pptx — тем же pandoc (pypandoc-binary), pdf — LibreOffice
из docx (LaTeX на машине нет; текстовый слой сохраняется).
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pypandoc

SOFFICE = "/Applications/LibreOffice.app/Contents/MacOS/soffice"


def md_to_docx(md: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    pypandoc.convert_file(str(md), "docx", format="markdown", outputfile=str(out))


def md_to_pptx(md: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    pypandoc.convert_file(str(md), "pptx", format="markdown", outputfile=str(out))


def docx_to_pdf(docx: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        # Отдельный профиль LibreOffice: не трогаем пользовательский и не
        # конфликтуем с открытым приложением.
        profile = Path(tmp) / "profile"
        subprocess.run(
            [
                SOFFICE,
                f"-env:UserInstallation=file://{profile}",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                tmp,
                str(docx),
            ],
            check=True,
            capture_output=True,
            timeout=300,
        )
        shutil.move(str(Path(tmp) / (docx.stem + ".pdf")), str(out))


def md_to_pdf(md: Path, out: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        docx = Path(tmp) / (out.stem + ".docx")
        md_to_docx(md, docx)
        docx_to_pdf(docx, out)


def render(md: Path, fmt: str, out: Path) -> None:
    if fmt == "docx":
        md_to_docx(md, out)
    elif fmt == "pptx":
        md_to_pptx(md, out)
    elif fmt == "pdf":
        md_to_pdf(md, out)
    elif fmt == "txt":
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        raise ValueError(fmt)
