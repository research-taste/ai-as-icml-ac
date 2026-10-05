from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from icml_ai_ac.models import PaperRecord
from icml_ai_ac.storage import ensure_parent, write_json


SECTION_ALIASES = {
    "abstract": ("abstract",),
    "introduction": ("introduction", "intro"),
    "method": ("method", "methods", "methodology", "approach", "model", "proposed method"),
    "experiments": (
        "experiments",
        "experiment",
        "experimental results",
        "experimental evaluation",
        "evaluation",
        "empirical evaluation",
        "results",
    ),
    "limitations": ("limitations", "limitation", "discussion", "broader impact", "impact statement"),
    "conclusion": ("conclusion", "conclusions", "discussion and conclusion", "concluding remarks"),
    "references": ("references", "bibliography"),
}


@dataclass(slots=True)
class ParsedPaper:
    record: PaperRecord
    raw_text: str
    clean_text: str
    sections: dict[str, str]
    compact_repr: str
    full_repr: str
    scoring_repr: str
    main_limit_text: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


def parse_pdf_record(
    record: PaperRecord,
    *,
    text_dir: Path,
    full_char_budget: int = 120_000,
    compact_char_budget: int = 24_000,
    scoring_char_budget: int = 80_000,
    main_paper_max_pages: int = 9,
    strip_references: bool = True,
) -> PaperRecord:
    if not record.pdf_path:
        record.parse_status = "missing_pdf_path"
        record.extra.setdefault("parse_diagnostics", {})["error"] = "record has no pdf_path"
        return record

    pdf_path = Path(record.pdf_path)
    if not pdf_path.exists():
        record.parse_status = "missing_pdf_file"
        record.extra.setdefault("parse_diagnostics", {})["error"] = f"pdf_path does not exist: {pdf_path}"
        return record

    parsed = parse_pdf(
        record,
        pdf_path=pdf_path,
        full_char_budget=full_char_budget,
        compact_char_budget=compact_char_budget,
        scoring_char_budget=scoring_char_budget,
        main_paper_max_pages=main_paper_max_pages,
        strip_references=strip_references,
    )
    artifact_paths = write_text_artifacts(parsed, text_dir=text_dir)
    record.text_full = str(artifact_paths["full_repr"])
    record.text_compact = str(artifact_paths["compact_repr"])
    record.text_scoring = str(artifact_paths["scoring_repr"])
    record.parse_status = parsed.diagnostics["parse_status"]
    record.extra["text_artifacts"] = {key: str(value) for key, value in artifact_paths.items()}
    record.extra["parse_diagnostics"] = parsed.diagnostics
    return record


def parse_pdf(
    record: PaperRecord,
    *,
    pdf_path: Path,
    full_char_budget: int = 120_000,
    compact_char_budget: int = 24_000,
    scoring_char_budget: int = 80_000,
    main_paper_max_pages: int = 10,
    strip_references: bool = True,
) -> ParsedPaper:
    raw_text = extract_text_pdftotext(pdf_path)
    clean = normalize_extracted_text(raw_text)
    limited_clean = normalize_extracted_text(
        extract_text_pdftotext(pdf_path, first_page=1, last_page=main_paper_max_pages)
    ) if main_paper_max_pages > 0 else clean
    if strip_references:
        body_text, references_text = split_references(clean)
        limited_body_text, limited_references_text = split_references(limited_clean)
    else:
        body_text, references_text = clean, ""
        limited_body_text, limited_references_text = limited_clean, ""
    main_body_text, appendix_text = split_appendix(body_text)
    limited_main_body_text, limited_appendix_text = split_appendix(limited_body_text)
    sections = segment_sections(main_body_text)
    compact = build_compact_repr(record, sections, compact_char_budget=compact_char_budget)
    full = build_full_repr(record, body_text, full_char_budget=full_char_budget)
    scoring = build_scoring_repr(
        record,
        limited_main_body_text,
        scoring_char_budget=scoring_char_budget,
        main_paper_max_pages=main_paper_max_pages,
    )
    diagnostics = build_parse_diagnostics(
        pdf_path=pdf_path,
        raw_text=raw_text,
        clean_text=clean,
        body_text=body_text,
        main_body_text=main_body_text,
        appendix_text=appendix_text,
        limited_clean_text=limited_clean,
        limited_main_body_text=limited_main_body_text,
        limited_appendix_text=limited_appendix_text,
        limited_references_text=limited_references_text,
        references_text=references_text,
        sections=sections,
        compact_repr=compact,
        full_repr=full,
        scoring_repr=scoring,
        main_paper_max_pages=main_paper_max_pages,
    )
    return ParsedPaper(
        record=record,
        raw_text=raw_text,
        clean_text=clean,
        sections=sections,
        compact_repr=compact,
        full_repr=full,
        scoring_repr=scoring,
        main_limit_text=limited_clean,
        diagnostics=diagnostics,
    )


def extract_text_pdftotext(pdf_path: Path, *, first_page: int | None = None, last_page: int | None = None) -> str:
    command = ["pdftotext", "-raw", "-enc", "UTF-8"]
    if first_page is not None:
        command.extend(["-f", str(first_page)])
    if last_page is not None:
        command.extend(["-l", str(last_page)])
    command.extend([str(pdf_path), "-"])
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def pdf_page_count(pdf_path: Path) -> int | None:
    try:
        result = subprocess.run(["pdfinfo", str(pdf_path)], check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, flags=re.MULTILINE)
    return int(match.group(1)) if match else None


def normalize_extracted_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\f+", "\n\n", text)
    lines = [normalize_line(line) for line in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def normalize_line(line: str) -> str:
    line = line.replace("\u00ad", "")
    line = re.sub(r"[ \t]+$", "", line)
    return line


def split_references(text: str) -> tuple[str, str]:
    matches = list(re.finditer(r"(?im)^references\s*$|^bibliography\s*$", text))
    if not matches:
        return text, ""
    match = matches[-1]
    return text[: match.start()].rstrip(), text[match.start() :].strip()


def split_appendix(text: str) -> tuple[str, str]:
    offset = 0
    previous_line = ""
    for line in text.splitlines(keepends=True):
        if is_appendix_heading(line, previous_line=previous_line):
            return text[:offset].rstrip(), text[offset:].strip()
        offset += len(line)
        previous_line = line
    return text, ""


def is_appendix_heading(line: str, *, previous_line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 90:
        return False
    if previous_line.strip():
        return False
    normalized = re.sub(r"\s+", " ", stripped.lower())
    return bool(
        re.match(
            r"^(appendix|appendices|supplementary material|supplemental material|supplementary information)(?:\b|$)",
            normalized,
        )
    )


def segment_sections(text: str) -> dict[str, str]:
    headings = find_section_headings(text)
    sections: dict[str, str] = {}
    if not headings:
        abstract = extract_abstract_block(text)
        if abstract:
            sections["abstract"] = abstract
        return sections

    for idx, (start, end, canonical, heading_text) in enumerate(headings):
        next_start = headings[idx + 1][0] if idx + 1 < len(headings) else len(text)
        content = text[end:next_start].strip()
        if canonical and canonical not in sections and content:
            sections[canonical] = content
        sections.setdefault(f"heading:{heading_text}", content)

    if "abstract" not in sections:
        abstract = extract_abstract_block(text)
        if abstract:
            sections["abstract"] = abstract
    return sections


def find_section_headings(text: str) -> list[tuple[int, int, str | None, str]]:
    headings: list[tuple[int, int, str | None, str]] = []
    pattern = re.compile(
        r"(?m)^(?P<prefix>(?:\d+(?:\.\d+)*\.?|[IVX]+\.?)\s+)?(?P<title>[A-Z][A-Za-z0-9 ,:/()&'\\-]{2,80})\s*$"
    )
    for match in pattern.finditer(text):
        title = re.sub(r"\s+", " ", match.group("title")).strip()
        canonical = canonical_section_name(title)
        if canonical or match.group("prefix"):
            headings.append((match.start(), match.end(), canonical, title))
    return headings


def canonical_section_name(title: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
    for canonical, aliases in SECTION_ALIASES.items():
        if normalized in aliases:
            return canonical
    return None


def extract_abstract_block(text: str) -> str | None:
    match = re.search(r"(?is)\babstract\b\.?\s*(?P<body>.+?)(?:\n\s*(?:1\.?\s+)?introduction\b|\n\s*keywords\b)", text)
    if not match:
        return None
    return compact_whitespace(match.group("body"))[:6000].strip()


def build_compact_repr(record: PaperRecord, sections: dict[str, str], *, compact_char_budget: int) -> str:
    parts = [
        f"Title: {record.title or ''}".strip(),
    ]
    abstract = sections.get("abstract") or record.abstract
    if abstract:
        parts.append(f"\nAbstract:\n{clean_section_text(abstract)}")
    if sections.get("introduction"):
        parts.append(f"\nIntroduction:\n{clean_section_text(sections['introduction'])}")
    if sections.get("conclusion"):
        parts.append(f"\nConclusion:\n{clean_section_text(sections['conclusion'])}")
    compact = "\n".join(part for part in parts if part)
    return truncate_text(compact, compact_char_budget)


def build_full_repr(record: PaperRecord, body_text: str, *, full_char_budget: int) -> str:
    header = f"Title: {record.title or ''}\n\n"
    return truncate_text(header + body_text, full_char_budget)


def build_scoring_repr(
    record: PaperRecord,
    main_body_text: str,
    *,
    scoring_char_budget: int,
    main_paper_max_pages: int | None = 9,
) -> str:
    page_note = (
        f"Extracted from the first {main_paper_max_pages} PDF pages before reference/appendix cleanup. "
        if main_paper_max_pages and main_paper_max_pages > 0
        else ""
    )
    header = (
        f"Title: {record.title or ''}\n\n"
        "Representation: main paper body extracted from the PDF. "
        f"{page_note}"
        "References are removed. "
        "Appendix or supplementary material is removed when detected.\n\n"
    )
    return truncate_text(header + main_body_text, scoring_char_budget)


def clean_section_text(text: str) -> str:
    return compact_whitespace(text)


def compact_whitespace(text: str) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def truncate_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    truncated = text[:limit].rstrip()
    return f"{truncated}\n\n[TRUNCATED at {limit} characters]"


def build_parse_diagnostics(
    *,
    pdf_path: Path,
    raw_text: str,
    clean_text: str,
    body_text: str,
    main_body_text: str,
    appendix_text: str,
    limited_clean_text: str,
    limited_main_body_text: str,
    limited_appendix_text: str,
    limited_references_text: str,
    references_text: str,
    sections: dict[str, str],
    compact_repr: str,
    full_repr: str,
    scoring_repr: str,
    main_paper_max_pages: int,
) -> dict[str, Any]:
    page_count = pdf_page_count(pdf_path)
    flags: list[str] = []
    if len(clean_text) < 10_000:
        flags.append("short_extraction")
    if "abstract" not in sections:
        flags.append("missing_abstract")
    if "introduction" not in sections:
        flags.append("missing_introduction")
    if "conclusion" not in sections:
        flags.append("missing_conclusion")
    if not references_text:
        flags.append("missing_references_split")
    parse_status = "ok" if not {"short_extraction", "missing_abstract"}.intersection(flags) else "needs_review"
    return {
        "parse_status": parse_status,
        "parser_backend": "pdftotext-raw",
        "page_count": page_count,
        "raw_chars": len(raw_text),
        "clean_chars": len(clean_text),
        "body_chars": len(body_text),
        "main_body_chars": len(main_body_text),
        "appendix_chars": len(appendix_text),
        "references_chars": len(references_text),
        "main_paper_max_pages": main_paper_max_pages,
        "limited_clean_chars": len(limited_clean_text),
        "limited_main_body_chars": len(limited_main_body_text),
        "limited_appendix_chars": len(limited_appendix_text),
        "limited_references_chars": len(limited_references_text),
        "compact_repr_chars": len(compact_repr),
        "full_repr_chars": len(full_repr),
        "scoring_repr_chars": len(scoring_repr),
        "section_keys": sorted(key for key in sections if not key.startswith("heading:")),
        "flags": flags,
    }


def write_text_artifacts(parsed: ParsedPaper, *, text_dir: Path) -> dict[str, Path]:
    paper_dir = text_dir / parsed.record.paper_id
    paper_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "raw_text": paper_dir / "raw.txt",
        "clean_text": paper_dir / "clean.txt",
        "main_limit_text": paper_dir / "main_limit.txt",
        "compact_repr": paper_dir / "compact_repr.txt",
        "full_repr": paper_dir / "full_repr.txt",
        "scoring_repr": paper_dir / "scoring_repr.txt",
        "sections": paper_dir / "sections.json",
    }
    paths["raw_text"].write_text(parsed.raw_text, encoding="utf-8")
    paths["clean_text"].write_text(parsed.clean_text, encoding="utf-8")
    paths["main_limit_text"].write_text(parsed.main_limit_text, encoding="utf-8")
    paths["compact_repr"].write_text(parsed.compact_repr, encoding="utf-8")
    paths["full_repr"].write_text(parsed.full_repr, encoding="utf-8")
    paths["scoring_repr"].write_text(parsed.scoring_repr, encoding="utf-8")
    write_json(paths["sections"], {key: value for key, value in parsed.sections.items() if not key.startswith("heading:")})
    return paths


def write_parse_report(path: Path, records: list[PaperRecord]) -> None:
    rows = []
    for record in records:
        diagnostics = record.extra.get("parse_diagnostics", {})
        rows.append(
            {
                "paper_id": record.paper_id,
                "title": record.title,
                "parse_status": record.parse_status,
                "page_count": diagnostics.get("page_count"),
                "clean_chars": diagnostics.get("clean_chars"),
                "compact_repr_chars": diagnostics.get("compact_repr_chars"),
                "full_repr_chars": diagnostics.get("full_repr_chars"),
                "scoring_repr_chars": diagnostics.get("scoring_repr_chars"),
                "appendix_chars": diagnostics.get("appendix_chars"),
                "main_paper_max_pages": diagnostics.get("main_paper_max_pages"),
                "limited_main_body_chars": diagnostics.get("limited_main_body_chars"),
                "limited_references_chars": diagnostics.get("limited_references_chars"),
                "flags": diagnostics.get("flags", []),
                "text_compact": record.text_compact,
                "text_full": record.text_full,
                "text_scoring": record.text_scoring,
            }
        )
    ensure_parent(path)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
