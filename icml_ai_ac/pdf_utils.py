from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from icml_ai_ac.parser import pdf_page_count


def write_pdf_page_excerpt(
    *,
    source_pdf: Path,
    out_pdf: Path,
    max_pages: int = 10,
    overwrite: bool = False,
    optimize_if_larger_than_bytes: int | None = None,
    pdf_settings: str = "/ebook",
) -> dict[str, object]:
    """Write a PDF containing pages 1..max_pages from source_pdf."""

    if max_pages <= 0:
        raise ValueError("max_pages must be positive")
    if not source_pdf.exists():
        raise FileNotFoundError(source_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    page_count = pdf_page_count(source_pdf)
    pages_written = min(page_count or max_pages, max_pages)
    optimized = False
    if out_pdf.exists() and not overwrite:
        optimized = optimize_pdf_if_needed(
            out_pdf,
            threshold_bytes=optimize_if_larger_than_bytes,
            pdf_settings=pdf_settings,
        )
        return {
            "source_pdf": str(source_pdf),
            "out_pdf": str(out_pdf),
            "source_pages": page_count,
            "pages_written": pages_written,
            "skipped": True,
            "optimized": optimized,
            "bytes": out_pdf.stat().st_size,
        }
    if page_count is not None and page_count <= max_pages:
        shutil.copyfile(source_pdf, out_pdf)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            pattern = str(Path(tmp) / "page-%d.pdf")
            subprocess.run(
                ["pdfseparate", "-f", "1", "-l", str(max_pages), str(source_pdf), pattern],
                check=True,
                capture_output=True,
                text=True,
            )
            page_paths = sorted(Path(tmp).glob("page-*.pdf"), key=_page_number)
            if not page_paths:
                raise RuntimeError(f"pdfseparate wrote no pages for {source_pdf}")
            subprocess.run(
                ["pdfunite", *[str(path) for path in page_paths], str(out_pdf)],
                check=True,
                capture_output=True,
                text=True,
            )
            pages_written = len(page_paths)
    optimized = optimize_pdf_if_needed(
        out_pdf,
        threshold_bytes=optimize_if_larger_than_bytes,
        pdf_settings=pdf_settings,
    )
    return {
        "source_pdf": str(source_pdf),
        "out_pdf": str(out_pdf),
        "source_pages": page_count,
        "pages_written": pages_written,
        "skipped": False,
        "optimized": optimized,
        "bytes": out_pdf.stat().st_size,
    }


def _page_number(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.rsplit("-", 1)[-1])
    except ValueError:
        return 10**9


def optimize_pdf_if_needed(
    path: Path,
    *,
    threshold_bytes: int | None,
    pdf_settings: str = "/ebook",
) -> bool:
    if threshold_bytes is None or threshold_bytes <= 0:
        return False
    if not path.exists() or path.stat().st_size <= threshold_bytes:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        compressed = Path(tmp) / path.name
        subprocess.run(
            [
                "gs",
                "-sDEVICE=pdfwrite",
                "-dCompatibilityLevel=1.4",
                f"-dPDFSETTINGS={pdf_settings}",
                "-dNOPAUSE",
                "-dQUIET",
                "-dBATCH",
                f"-sOutputFile={compressed}",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if compressed.exists() and compressed.stat().st_size < path.stat().st_size:
            shutil.copyfile(compressed, path)
            return True
    return False
