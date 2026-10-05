from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf

from icml_ai_ac.models import PaperRecord
from icml_ai_ac.storage import append_jsonl, read_jsonl_if_exists, write_json, write_jsonl


ANONYMIZATION_VERSION = "direct_identity_redaction_v33"
TEXT_DICT_FLAGS = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES
EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b")
URL_PATTERN = re.compile(
    r"(?i)(?:"
    r"https?://[^\s<>()\[\]{}]*|"
    r"\bwww\.[^\s<>()\[\]{}]+|"
    r"\b(?:[a-z0-9-]+\.)+(?:com|org|net|edu|ai|io|dev|co)"
    r"(?:/[^\s<>()\[\]{}]*)?"
    r")"
)
URL_ONLY_LINE_PATTERN = re.compile(r"(?i)^\s*\d*\s*https?://")
URL_CONTINUATION_PATTERN = re.compile(r"^[A-Za-z0-9._~/?#=&%+\-]+$")
TEXT_IDENTITY_NOTE_PATTERN = re.compile(
    r"(?i)(?:"
    r"\b(?:research\s+)?internship\s+at\b|"
    r"\b(?:co-?)?corresponding\s+authors?\b|"
    r"\b(?:equal\s+contributions?|core\s+contributors?)\b|"
    r"\bauthors?\s+are\s+members\b|"
    r"(?:\b|[1-9])work\s+(?:was\s+)?(?:done|completed)\s+"
    r"(?:during\s+an?\s+internship|while\s+at|"
    r"at\s+(?:their\s+)?previous\s+affiliation)\b|"
    r"\bwork\s+done\s+during\b"
    r")"
)
ACKNOWLEDGEMENTS_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:(?:\d+(?:\.\d+)*\.?|[A-Z])\s+)?"
    r"(?:acknowledg(?:e)?ments?|acknowledgement)[ \t]*$"
)
IDENTITY_LINE_PATTERN = re.compile(
    r"(?i)(?:\bcorrespondence\s+to\b|\bequal\s+contribution\b|"
    r"\bdepartment\s+of\b|\buniversity\b|\binstitute\b|\blaboratory\b|"
    r"\baffiliation\b|\bproceedings\s+of\s+the\b|\bpmlr\s+\d+\b|"
    r"\bcopyright\s+\d{4}\b|\bby\s+the\s+author(?:s)?\b|@)"
)
TEXT_IDENTITY_LINE_PATTERN = re.compile(
    r"(?i)(?:"
    r"^\s*\[identity\s+redacted\]\s*$|"
    r"^\s*(?:anonymous|anonymized)\s+author(?:s)?(?:\s*\([^)]*\))?\s*$|"
    r"^\s*[*†‡]?\s*(?:equal\s*contribution|correspondence\s+to|"
    r"(?:co-?)?corresponding\s+authors?|(?:co-?)?senior\s+authors?|"
    r"lead\s+author|project\s+leader|"
    r"(?:part\s+of\s+)?(?:this\s+)?work\s+(?:was\s+)?"
    r"(?:done|performed|conducted)\b|now\s+at\b|"
    r"all\s+authors?\s+have\s+significantly\s+contributed|"
    r"the\s+order\s+does\s+not\s+indicate\b|department\b|"
    r"school\s+of\b|faculty\s+of\b|laboratory\s+of\b|institute\b|"
    r"international\s+conference\s+on\s+machine\b)|"
    r"\bcorrespondence\s+to\s*:|"
    r"^\s*of\b.{0,120}\buniversity\b|"
    r"\bproceedings\s+of\s+the\b|\bpmlr\s+\d+\b|"
    r"\bcopyright\s+\d{4}\b|\bby\s+the\s+author(?:s)?\b|"
    r"^\s*.{0,90}\buniversity\s*,.{0,80}$"
    r")"
)
AFFILIATION_FRAGMENT_PATTERN = re.compile(
    r"(?i)(?:"
    r"\buniversity\b|\bdepartment\b|\binstitute\b|\blaborator(?:y|ies)\b|"
    r"\bschool\s+of\b|\bfaculty\s+of\b|\bcorrespondence\b|"
    r"\bauthors?\s+are\s+ordered\b|\binternship\s+at\b|"
    r"\b(?:USA|U\.S\.A\.|UK|U\.K\.|Canada|China|France|Germany|"
    r"Switzerland|Belgium|Australia|Japan|Korea|Singapore)\b"
    r")"
)
ANONYMOUS_AUTHOR_PATTERN = re.compile(
    r"(?i)^\s*(?:anonymous|anonymized)\s+author(?:s)?(?:\s*\([^)]*\))?"
    r"(?:\s*[*†‡]?\d+)*\s*$"
)
SOURCE_CUE_PATTERN = re.compile(
    r"(?i)^\s*(?:"
    r"arxiv:\d{4}\.\d+|"
    r"for\s+icml\s+\d{4}\s+reviewers:|"
    r"\.?authorerr:"
    r")"
)
REVIEWER_FOOTER_PATTERN = re.compile(
    r"(?i)^\s*for\s+icml\s+\d{4}\s+reviewers:"
)
REVIEW_STATUS_FOOTER_PATTERN = re.compile(
    r"(?i)^\s*(?:preliminary\s+work\.\s*)?under\s+review\s+by\b"
)
FOOTER_IDENTITY_ANCHOR_PATTERN = re.compile(
    r"(?i)(?:"
    r"\bcorrespondence\b|\bproceedings\s+of\s+the\b|\bpmlr\s+\d+\b|"
    r"\bcopyright\s+\d{4}\b|\bby\s+the\s+author(?:s)?\b"
    r")"
)


@dataclass(frozen=True, slots=True)
class AnonymizationConfig:
    pdf_dir: Path
    text_dir: Path
    max_pages: int = 9
    limit: int | None = None
    overwrite: bool = False


def anonymize_manifest(
    *,
    records: list[PaperRecord],
    out: Path,
    report_path: Path,
    config: AnonymizationConfig,
) -> dict[str, Any]:
    if config.max_pages <= 0:
        raise ValueError("max_pages must be positive")
    journal_path = out.with_suffix(out.suffix + ".journal.jsonl")
    journal_rows = {
        str(row.get("paper_id")): row
        for row in read_jsonl_if_exists(journal_path)
        if row.get("paper_id")
    }
    selected = records[: config.limit] if config.limit is not None else records
    completed: list[PaperRecord] = []
    audit_rows: list[dict[str, Any]] = []
    resumed = 0
    for record in selected:
        fingerprint = anonymization_fingerprint(record, max_pages=config.max_pages)
        prior = journal_rows.get(record.paper_id)
        if (
            prior
            and prior.get("fingerprint") == fingerprint
            and prior.get("status") == "ok"
            and not config.overwrite
        ):
            resumed_record = PaperRecord.from_dict(prior["record"])
            if scoring_artifacts_exist(resumed_record):
                completed.append(resumed_record)
                audit_rows.append(prior["audit"])
                resumed += 1
                continue
        anonymized_record, audit = anonymize_record(
            record,
            config=config,
            fingerprint=fingerprint,
        )
        journal_row = {
            "paper_id": record.paper_id,
            "fingerprint": fingerprint,
            "status": audit["status"],
            "audit": audit,
            "record": anonymized_record.to_dict(),
        }
        append_jsonl(journal_path, [journal_row])
        journal_rows[record.paper_id] = journal_row
        audit_rows.append(audit)
        if audit["status"] == "ok":
            completed.append(anonymized_record)

    write_jsonl(out, (record.to_dict() for record in completed))
    status_counts: dict[str, int] = {}
    for row in audit_rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    report = {
        "anonymization_version": ANONYMIZATION_VERSION,
        "input_records": len(selected),
        "output_records": len(completed),
        "excluded_records": len(selected) - len(completed),
        "resumed_records": resumed,
        "max_pages": config.max_pages,
        "pdf_dir": str(config.pdf_dir),
        "text_dir": str(config.text_dir),
        "journal": str(journal_path),
        "status_counts": status_counts,
        "excluded_paper_ids": [
            row["paper_id"] for row in audit_rows if row.get("status") != "ok"
        ],
        "records": audit_rows,
    }
    write_json(report_path, report)
    return report


def anonymize_record(
    record: PaperRecord,
    *,
    config: AnonymizationConfig,
    fingerprint: str | None = None,
) -> tuple[PaperRecord, dict[str, Any]]:
    active_authors, ignored_authors = filter_author_identities(record.authors)
    source_pdf = Path(record.pdf_path) if record.pdf_path else None
    audit: dict[str, Any] = {
        "paper_id": record.paper_id,
        "status": "failed",
        "anonymization_version": ANONYMIZATION_VERSION,
        "fingerprint": fingerprint,
        "authors_supplied": len(record.authors),
        "authors_used": len(active_authors),
        "ignored_author_identities": ignored_authors,
        "error": None,
    }
    if source_pdf is None or not source_pdf.exists():
        audit["error"] = "missing source PDF"
        return record, audit

    source_text_paths = representation_paths(record)
    missing_text = sorted(name for name, path in source_text_paths.items() if path is None or not path.exists())
    if missing_text:
        audit["error"] = f"missing text representations: {', '.join(missing_text)}"
        return record, audit

    config.pdf_dir.mkdir(parents=True, exist_ok=True)
    paper_text_dir = config.text_dir / record.paper_id
    paper_text_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = config.pdf_dir / f"{record.paper_id}.first{config.max_pages}.anonymized.pdf"
    out_text_paths = {
        name: paper_text_dir / f"{name}.anonymized.txt"
        for name in ("compact_repr", "full_repr", "scoring_repr")
    }
    try:
        pdf_audit = anonymize_pdf(
            source_pdf=source_pdf,
            out_pdf=out_pdf,
            title=record.title or "",
            authors=active_authors,
            max_pages=config.max_pages,
        )
        text_audits: dict[str, Any] = {}
        for name, source_path in source_text_paths.items():
            assert source_path is not None
            source_text = source_path.read_text(encoding="utf-8")
            sanitized, text_audit = anonymize_representation(
                source_text,
                authors=active_authors,
                strip_front_matter=name in {"full_repr", "scoring_repr"},
            )
            out_text_paths[name].write_text(sanitized, encoding="utf-8")
            text_audits[name] = text_audit
        residual_text_authors = sorted(
            {
                author
                for text_audit in text_audits.values()
                for author in text_audit["residual_authors"]
            }
        )
        status = (
            "ok"
            if pdf_audit["status"] == "ok" and not residual_text_authors
            else "needs_review"
        )
        audit.update(
            {
                "status": status,
                "source_pdf": str(source_pdf),
                "source_pdf_sha256": sha256_file(source_pdf),
                "anonymized_pdf": str(out_pdf),
                "anonymized_pdf_sha256": sha256_file(out_pdf),
                "pdf": pdf_audit,
                "text": text_audits,
                "residual_text_authors": residual_text_authors,
            }
        )
        if status == "ok":
            record = build_blinded_record(
                source=record,
                out_pdf=out_pdf,
                out_text_paths=out_text_paths,
                audit=audit,
            )
    except Exception as exc:  # noqa: BLE001 - preserve row-local failures for review.
        audit["error"] = repr(exc)
    return record, audit


def build_blinded_record(
    *,
    source: PaperRecord,
    out_pdf: Path,
    out_text_paths: dict[str, Path],
    audit: dict[str, Any],
) -> PaperRecord:
    scoring_artifacts = {
        "artifact_kind": "anonymized_derivative",
        "anonymization_version": ANONYMIZATION_VERSION,
        "pdf": str(out_pdf),
        **{name: str(path) for name, path in out_text_paths.items()},
    }
    safe_anonymization = {
        key: audit[key]
        for key in (
            "paper_id",
            "status",
            "anonymization_version",
            "fingerprint",
            "source_pdf_sha256",
            "anonymized_pdf_sha256",
        )
    }
    return PaperRecord(
        paper_id=source.paper_id,
        source="blinded_scoring",
        title=source.title,
        pdf_path=str(out_pdf),
        text_full=str(out_text_paths["full_repr"]),
        text_compact=str(out_text_paths["compact_repr"]),
        text_scoring=str(out_text_paths["scoring_repr"]),
        parse_status=source.parse_status,
        extra={
            "anonymization": safe_anonymization,
            "scoring_artifacts": scoring_artifacts,
            "blind_manifest": {
                "human_outcomes_available_to_models": False,
                "identity_metadata_available_to_models": False,
            },
        },
    )


def anonymize_pdf(
    *,
    source_pdf: Path,
    out_pdf: Path,
    title: str,
    authors: list[str],
    max_pages: int,
) -> dict[str, Any]:
    authors, ignored_author_identities = filter_author_identities(authors)
    previous_error_display = pymupdf.TOOLS.mupdf_display_errors()
    pymupdf.TOOLS.mupdf_warnings(reset=1)
    pymupdf.TOOLS.mupdf_display_errors(False)
    output: pymupdf.Document | None = None
    first_page_author_band_redacted = False
    first_page_identity_band_reason: list[str] = []
    try:
        source = pymupdf.open(source_pdf)
        try:
            if len(source) == 0:
                raise ValueError(f"source PDF has no pages: {source_pdf}")
            output = pymupdf.open()
            output.insert_pdf(source, from_page=0, to_page=min(len(source), max_pages) - 1)
        finally:
            source.close()

        author_hits: dict[str, int] = {author: 0 for author in authors}
        redaction_count = 0
        identity_line_count = 0
        mail_link_count = 0
        external_link_count = 0
        url_redaction_count = 0
        for page_index, page in enumerate(output):
            page_rects: list[pymupdf.Rect] = []
            first_page_author_rects: list[pymupdf.Rect] = []
            words = page.get_text("words", sort=True)
            page_lines = iter_page_lines(page)
            if page_index == 0:
                identity_band, identity_reasons = find_first_page_identity_band(
                    page=page,
                    lines=page_lines,
                    title=title,
                    authors=authors,
                )
                if identity_band is not None:
                    page_rects.append(identity_band)
                    first_page_author_band_redacted = True
                    first_page_identity_band_reason = identity_reasons
                footer_blocks = find_first_page_footer_identity_blocks(
                    page=page,
                    lines=page_lines,
                )
                page_rects.extend(footer_blocks)
                identity_line_count += len(footer_blocks)
            for author in authors:
                rects: list[pymupdf.Rect] = []
                for variant in author_name_variants(author):
                    rects.extend(find_name_rects(words, variant))
                if not rects:
                    rects = list(page.search_for(author))
                rects = dedupe_rects(rects)
                if rects:
                    author_hits[author] += len(rects)
                    page_rects.extend(rects)
                    if page_index == 0:
                        first_page_author_rects.extend(
                            rect for rect in rects if rect.y0 < page.rect.height * 0.42
                        )
            if first_page_author_rects and not first_page_author_band_redacted:
                y0 = max(0.0, min(rect.y0 for rect in first_page_author_rects) - 2.0)
                y1 = min(page.rect.height, max(rect.y1 for rect in first_page_author_rects) + 2.0)
                page_rects.append(pymupdf.Rect(0.0, y0, page.rect.width, y1))
                first_page_author_band_redacted = True
                first_page_identity_band_reason = ["known_author_rect"]
            author_band_bottom = (
                max(rect.y1 for rect in first_page_author_rects)
                if first_page_author_rects
                else None
            )
            abstract_hits = list(page.search_for("Abstract")) if page_index == 0 else []
            abstract_top = min((rect.y0 for rect in abstract_hits), default=None)
            wrapped_author_lines = wrapped_author_line_indexes(
                [line_text for line_text, _ in page_lines],
                authors,
            )

            for line_index, (line_text, line_rect) in enumerate(page_lines):
                has_author_identity = any(
                    contains_author_identity(line_text, author)
                    for author in authors
                )
                if line_index in wrapped_author_lines:
                    page_rects.append(
                        expand_rect(line_rect, page=page, x_padding=2.0, y_padding=2.0)
                    )
                    identity_line_count += 1
                elif page_index == 0 and REVIEWER_FOOTER_PATTERN.search(line_text):
                    page_rects.append(
                        pymupdf.Rect(
                            0.0,
                            max(0.0, line_rect.y0 - 2.0),
                            page.rect.width,
                            page.rect.height,
                        )
                    )
                    identity_line_count += 1
                elif (
                    page_index == 0
                    and line_rect.y0 > page.rect.height * 0.70
                    and REVIEW_STATUS_FOOTER_PATTERN.search(line_text)
                ):
                    midpoint = page.rect.width / 2.0
                    x0, x1 = (
                        (0.0, midpoint)
                        if (line_rect.x0 + line_rect.x1) / 2.0 < midpoint
                        else (midpoint, page.rect.width)
                    )
                    page_rects.append(
                        pymupdf.Rect(
                            x0,
                            max(0.0, line_rect.y0 - 2.0),
                            x1,
                            page.rect.height,
                        )
                    )
                    identity_line_count += 1
                elif (
                    has_author_identity
                    or EMAIL_PATTERN.search(line_text)
                    or (page_index == 0 and ANONYMOUS_AUTHOR_PATTERN.search(line_text))
                    or (page_index == 0 and SOURCE_CUE_PATTERN.search(line_text))
                ):
                    page_rects.append(
                        expand_rect(line_rect, page=page, x_padding=2.0, y_padding=2.0)
                    )
                    identity_line_count += 1
                elif (
                    page_index == 0
                    and (
                        line_rect.y0 > page.rect.height * 0.72
                        or (
                            author_band_bottom is not None
                            and abstract_top is not None
                            and line_rect.y0 >= author_band_bottom
                            and line_rect.y1 <= abstract_top
                        )
                    )
                    and IDENTITY_LINE_PATTERN.search(line_text)
                ):
                    page_rects.append(
                        expand_rect(line_rect, page=page, x_padding=2.0, y_padding=9.0)
                    )
                    identity_line_count += 1

            uri_link_groups: dict[str, list[tuple[pymupdf.Rect, str]]] = {}
            for link in page.get_links():
                uri = str(link.get("uri") or "")
                link_rect = pymupdf.Rect(link.get("from") or pymupdf.Rect())
                visible_text = page.get_textbox(link_rect) if link_rect.is_valid else ""
                if uri.lower().startswith("mailto:"):
                    page.delete_link(link)
                    mail_link_count += 1
                    if link_rect.is_valid:
                        page_rects.append(
                            expand_rect(
                                link_rect,
                                page=page,
                                x_padding=2.0,
                                y_padding=2.0,
                            )
                        )
                elif uri:
                    page.delete_link(link)
                    external_link_count += 1
                    if link_rect.is_valid:
                        uri_link_groups.setdefault(uri, []).append(
                            (link_rect, visible_text)
                        )

            for link_fragments in uri_link_groups.values():
                if not any(URL_PATTERN.search(text) for _, text in link_fragments):
                    continue
                for link_rect, _ in link_fragments:
                    page_rects.append(
                        expand_rect(
                            link_rect,
                            page=page,
                            x_padding=2.0,
                            y_padding=2.0,
                        )
                    )
                    url_redaction_count += 1

            url_line_rects = find_url_only_line_rects(
                page=page,
                lines=page_lines,
            )
            page_rects.extend(url_line_rects)
            url_redaction_count += len(url_line_rects)

            for url_match in URL_PATTERN.finditer(page.get_text()):
                url = url_match.group(0).rstrip(".,;:!?)]}")
                url_rects = list(page.search_for(url))
                if not url_rects:
                    url_rects = find_layout_fragment_rects(page, url)
                page_rects.extend(url_rects)
                url_redaction_count += len(url_rects)

            for rect in dedupe_rects(page_rects):
                page.add_redact_annot(rect, fill=(1, 1, 1), cross_out=False)
                redaction_count += 1
            if page_rects:
                page.apply_redactions(
                    images=pymupdf.PDF_REDACT_IMAGE_NONE,
                    graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                    text=pymupdf.PDF_REDACT_TEXT_REMOVE,
                )

        output.set_metadata({})
        if hasattr(output, "del_xml_metadata"):
            output.del_xml_metadata()
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        output.save(out_pdf, garbage=4, clean=True, deflate=True)
    finally:
        try:
            if output is not None:
                output.close()
        finally:
            diagnostics_text = pymupdf.TOOLS.mupdf_warnings(reset=1)
            pymupdf.TOOLS.mupdf_display_errors(previous_error_display)

    diagnostics = list(
        dict.fromkeys(
            line.strip()
            for line in str(diagnostics_text or "").splitlines()
            if line.strip()
        )
    )

    validation = validate_anonymized_pdf(out_pdf, authors=authors)
    authors_not_located = sorted(author for author, count in author_hits.items() if count == 0)
    status = (
        "ok"
        if first_page_author_band_redacted
        and not validation["residual_authors"]
        and validation["residual_email_count"] == 0
        and validation["residual_url_count"] == 0
        and not validation["metadata_author"]
        and not validation["xml_metadata_present"]
        else "needs_review"
    )
    return {
        "status": status,
        "pages_written": validation["page_count"],
        "redaction_count": redaction_count,
        "identity_line_count": identity_line_count,
        "mail_link_count": mail_link_count,
        "external_link_count": external_link_count,
        "url_redaction_count": url_redaction_count,
        "author_hits": author_hits,
        "authors_not_located": authors_not_located,
        "ignored_author_identities": ignored_author_identities,
        "first_page_author_band_redacted": first_page_author_band_redacted,
        "first_page_identity_band_reason": first_page_identity_band_reason,
        "source_repair_diagnostic_count": len(diagnostics),
        "source_repair_diagnostics": diagnostics[:20],
        **validation,
    }


def validate_anonymized_pdf(path: Path, *, authors: list[str]) -> dict[str, Any]:
    document = pymupdf.open(path)
    text = "\n".join(page.get_text() for page in document)
    metadata_author = str(document.metadata.get("author") or "")
    xml_metadata_present = bool(document.get_xml_metadata())
    page_count = len(document)
    document.close()
    residual_authors = [
        author
        for author in authors
        if contains_author_identity(text, author)
    ]
    return {
        "page_count": page_count,
        "residual_authors": residual_authors,
        "residual_email_count": len(EMAIL_PATTERN.findall(text)),
        "residual_url_count": len(URL_PATTERN.findall(text)),
        "metadata_author": metadata_author,
        "xml_metadata_present": xml_metadata_present,
    }


def anonymize_representation(
    text: str,
    *,
    authors: list[str],
    strip_front_matter: bool,
) -> tuple[str, dict[str, Any]]:
    authors, ignored_author_identities = filter_author_identities(authors)
    sanitized = text
    front_matter_removed = False
    if strip_front_matter:
        sanitized, front_matter_removed = remove_front_matter(sanitized)
    sanitized, acknowledgements_removed = remove_acknowledgements(sanitized)
    sanitized, email_count = EMAIL_PATTERN.subn("[EMAIL REDACTED]", sanitized)
    sanitized, url_count = URL_PATTERN.subn("[URL REDACTED]", sanitized)
    replacement_count = 0
    identity_line_replacement_count = 0
    lines = sanitized.splitlines()
    author_line_indexes = {
        index
        for author in authors
        for index in author_identity_line_indexes(sanitized, author)
    }
    author_flags = [index in author_line_indexes for index in range(len(lines))]
    identity_flags = [
        is_text_identity_line(line)
        for line in lines
    ]
    for index in correspondence_identity_line_indexes(lines):
        identity_flags[index] = True
    for index in identity_note_line_indexes(lines):
        identity_flags[index] = True
    direct_flags = [
        author_flag or identity_flag
        for author_flag, identity_flag in zip(
            author_flags,
            identity_flags,
            strict=True,
        )
    ]
    identity_block_lines: set[int] = set()
    for _ in range(len(lines) + 1):
        expanded = find_identity_block_lines(
            lines,
            direct_flags=[
                direct or index in identity_block_lines
                for index, direct in enumerate(direct_flags)
            ],
        )
        if expanded <= identity_block_lines:
            break
        identity_block_lines.update(expanded)
    else:
        raise RuntimeError("identity block redaction did not converge")
    sanitized_lines: list[str] = []
    for index, line in enumerate(lines):
        if author_flags[index]:
            sanitized_lines.append("[IDENTITY REDACTED]")
            replacement_count += 1
        elif identity_flags[index] or index in identity_block_lines:
            sanitized_lines.append("[IDENTITY REDACTED]")
            identity_line_replacement_count += 1
        else:
            sanitized_lines.append(line)
    sanitized = "\n".join(sanitized_lines)
    residual_authors = [
        author
        for author in authors
        if contains_author_identity(sanitized, author)
    ]
    return sanitized, {
        "front_matter_removed": front_matter_removed,
        "acknowledgements_removed": acknowledgements_removed,
        "author_replacement_count": replacement_count,
        "identity_line_replacement_count": identity_line_replacement_count,
        "email_replacement_count": email_count,
        "url_replacement_count": url_count,
        "residual_authors": residual_authors,
        "ignored_author_identities": ignored_author_identities,
    }


def is_text_identity_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped and len(stripped) <= 220 and TEXT_IDENTITY_LINE_PATTERN.search(stripped))


def is_text_affiliation_cue_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 220:
        return False
    body = re.sub(r"^\s*(?:\d+|[*†‡]+)\s*", "", stripped)
    if re.match(
        r"(?i)^(?:"
        r"universit(?:y|e|é)\s+(?:of|de)|univ\.\s+|"
        r"department\s+of|school\s+of|faculty\s+of|"
        r"college\s+of|laboratory\s+(?:of|for)|institute\s+(?:of|for)|"
        r"(?:state\s+|national\s+)?key\s+laboratory|"
        r"(?:engineering\s+)?research\s+cent(?:er|re)\s+(?:of|for)|"
        r"ministry\s+of"
        r")\b",
        body,
    ):
        return True
    if re.fullmatch(
        r"(?i)(?:Google(?:\s+(?:DeepMind|Research))?|Microsoft(?: Research)?|"
        r"Meta(?:\s+AI)?|NVIDIA|OpenAI|Anthropic|Amazon(?:\s+Web Services)?|"
        r"IBM Research|Adobe Research|ByteDance|Tencent|ServiceNow AI Research|"
        r"RIKEN|Mila|CNRS|INRIA)(?:[.,]?\s+\d+)?",
        body,
    ):
        return True
    organization_term = re.search(
        r"(?i)\b(?:university|universit(?:e|é)|univ\.|institute|"
        r"laborator(?:y|ies)|laboratoire|department|"
        r"school\s+of|faculty\s+of|college\s+of|research\s+cent(?:er|re)|"
        r"cent(?:er|re)\s+for|academy\s+of\s+sciences|ministry\s+of)\b",
        body,
    )
    if organization_term:
        words = re.findall(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", body, flags=re.UNICODE)
        significant = [
            word
            for word in words
            if word.casefold()
            not in {"a", "an", "and", "at", "for", "in", "of", "on", "the", "to"}
        ]
        capitalized = [
            word
            for word in significant
            if word[:1].isupper() or word.isupper()
        ]
        if significant and len(capitalized) / len(significant) >= 0.6:
            return True
    if re.search(
        r"(?i)(?:,\s*)?(?:USA|U\.S\.A\.|UK|U\.K\.|Canada|China|France|"
        r"Germany|Switzerland|Belgium|Australia|Japan|Korea|Singapore|"
        r"Netherlands|Israel)(?:[.,]?\s*\d+)?\s*$",
        body,
    ):
        words = re.findall(r"[^\W\d_]+", body, flags=re.UNICODE)
        if 1 < len(words) <= 20:
            return True
    return bool(
        re.fullmatch(
            r"(?:[A-Z][\w&.'’()/-]*\s+){1,10}"
            r"(?:University|Laborator(?:y|ies)|Institute)"
            r"(?:\s*[,;]?\s*(?:\d+|USA|U\.S\.A\.|UK|U\.K\.|Canada|"
            r"China|France|Germany|Switzerland|Belgium|Australia|Japan|"
            r"Korea|Singapore))*",
            body,
        )
    )


def is_titlecase_affiliation_fragment(line: str) -> bool:
    if "," not in line or len(line) > 220:
        return False
    words = re.findall(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", line, flags=re.UNICODE)
    significant = [
        word
        for word in words
        if word.casefold()
        not in {"a", "an", "and", "at", "for", "in", "of", "on", "the", "to"}
    ]
    capitalized = [
        word
        for word in significant
        if word[:1].isupper() or word.isupper()
    ]
    return bool(
        1 < len(significant) <= 20
        and len(capitalized) / len(significant) >= 0.6
    )


def find_identity_block_lines(
    lines: list[str],
    *,
    direct_flags: list[bool],
) -> set[int]:
    block_lines: set[int] = set()
    for start, line in enumerate(lines):
        if not re.fullmatch(r"\s*[*†‡]+\s*", line):
            continue
        lookahead_end = min(len(lines), start + 13)
        if not any(direct_flags[start + 1 : lookahead_end]):
            continue
        block_end = start
        seen_identity = False
        for index in range(start + 1, lookahead_end):
            if not lines[index].strip():
                if seen_identity:
                    break
                continue
            if direct_flags[index] or is_text_affiliation_cue_line(lines[index]):
                block_end = index
                seen_identity = True
                continue
            if seen_identity or index - start > 3:
                break
        if seen_identity:
            block_lines.update(range(start, block_end + 1))
    early_limit = min(len(lines), 120)
    for direct_index in range(early_limit):
        if not direct_flags[direct_index]:
            continue
        window_start = max(0, direct_index - 15)
        window_end = min(early_limit, direct_index + 21)
        cue_indices = [
            index
            for index in range(window_start, window_end)
            if is_text_affiliation_cue_line(lines[index])
        ]
        if not cue_indices:
            continue
        related_indices = cue_indices + [
            index
            for index in range(window_start, window_end)
            if direct_flags[index]
        ]
        block_start = min(related_indices)
        for marker_index in range(
            block_start - 1,
            max(-1, block_start - 7),
            -1,
        ):
            if not lines[marker_index].strip():
                break
            if re.fullmatch(
                r"\s*(?:\d+|[*†‡]+)\s*",
                lines[marker_index],
            ):
                block_start = marker_index
                break
        block_end = max(related_indices)
        block_lines.update(range(block_start, block_end + 1))
    run_start = 0
    structural_limit = min(len(lines), 75)
    while run_start < structural_limit:
        if not direct_flags[run_start]:
            run_start += 1
            continue
        run_end = run_start
        while run_end < len(lines) and direct_flags[run_end]:
            run_end += 1
        if run_end - run_start >= 3:
            marker_candidates = [
                index
                for index in range(max(0, run_start - 15), run_start)
                if re.fullmatch(r"\s*(?:[1-9]|[*†‡]+)\s*", lines[index])
            ]
            if marker_candidates:
                marker_index = max(marker_candidates)
                candidate_lines = lines[marker_index:run_start]
                has_affiliation_evidence = any(
                    is_text_affiliation_cue_line(line)
                    or is_titlecase_affiliation_fragment(line)
                    or is_text_identity_line(line)
                    or re.search(
                        r"(?i)(?:department|univ|institut|laborator|"
                        r"\blab\b|school|college|research|academy|"
                        r"correspon|internship|\bauthors?\b|"
                        r"\bsch\.|\bfaculty\b|\bprogram\b|\bdivision\b|"
                        r"\bprofessorship\b|\bcluster\s+of\s+excellence\b|"
                        r"\bartificial\s+intelligence\b|\bengineering\b|"
                        r"\btechnolog(?:y|ies)\b|\bcorporation\b|\bcompany\b|"
                        r"\binc\.?\b|\bgroup\b|\bAI\b|\bCNRS\b|\bINRIA\b|"
                        r"\bETH\b|\bEPFL\b|\bMIT\b|\bKAIST\b|\bTU\b|"
                        r"\bC2\b|\bMATS\b|\bUCSD\b|\bApple\b|\bAmazon\b|"
                        r"\bAlibaba\b|\bGoogle\b|\bMeta\b|\bHuawei\b|"
                        r"\bToyota\b|\bUber\b|\bYandex\b|\bKRAFTON\b)",
                        line,
                    )
                    or re.search(r"(?:\s|[,.;])[1-9]\s*$", line)
                    or re.fullmatch(
                        r"\s*[A-Z][A-Z0-9&.'/-]{1,15}"
                        r"(?:\s+[A-Z][A-Z0-9&.'/-]{1,15})?"
                        r"(?:\s+[1-9])?\s*",
                        line,
                    )
                    for line in candidate_lines[1:]
                )
                if has_affiliation_evidence:
                    block_lines.update(range(marker_index, run_end))
        run_start = run_end
    return block_lines


def remove_front_matter(text: str) -> tuple[str, bool]:
    header_match = re.match(
        r"(?s)\A(?P<header>Title:[^\n]*\n\n(?:Representation:.*?\n\n)?)",
        text,
    )
    header = header_match.group("header") if header_match else ""
    body = text[len(header) :]
    abstract = re.search(r"(?im)^[ \t]*abstract[ \t]*$", body)
    if not abstract:
        return text, False
    return header + body[abstract.start() :], bool(body[: abstract.start()].strip())


def remove_acknowledgements(text: str) -> tuple[str, bool]:
    match = ACKNOWLEDGEMENTS_PATTERN.search(text)
    if not match:
        return text, False
    return text[: match.start()].rstrip() + "\n", True


def find_name_rects(words: list[tuple[Any, ...]], name: str) -> list[pymupdf.Rect]:
    target = normalize_identity_text(name)
    if not target:
        return []
    normalized_words = [normalize_identity_token(str(word[4])) for word in words]
    results: list[pymupdf.Rect] = []
    for start, token in enumerate(normalized_words):
        if not token or not target.startswith(token):
            continue
        matched_indices = [start]
        accumulated = token
        word_index = start + 1
        while word_index < len(words) and accumulated != target:
            candidate = normalized_words[word_index]
            proposed = accumulated + candidate
            if candidate and target.startswith(proposed):
                matched_indices.append(word_index)
                accumulated = proposed
            elif not candidate or candidate.isdigit() or len(candidate) == 1:
                matched_indices.append(word_index)
            else:
                break
            word_index += 1
        if accumulated == target:
            line_groups: dict[tuple[Any, Any], list[int]] = {}
            for index in matched_indices:
                word = words[index]
                line_groups.setdefault((word[5], word[6]), []).append(index)
            for indices in line_groups.values():
                rect = pymupdf.Rect(words[indices[0]][:4])
                for index in indices[1:]:
                    rect.include_rect(pymupdf.Rect(words[index][:4]))
                results.append(rect)
    return dedupe_rects(results)


def iter_page_lines(page: pymupdf.Page) -> list[tuple[str, pymupdf.Rect]]:
    lines: list[tuple[str, pymupdf.Rect]] = []
    page_dict = page.get_text("dict", sort=True, flags=TEXT_DICT_FLAGS)
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(str(span.get("text", "")) for span in spans).strip()
            if text:
                lines.append((text, pymupdf.Rect(line["bbox"])))
    return lines


def find_layout_fragment_rects(
    page: pymupdf.Page,
    fragment: str,
) -> list[pymupdf.Rect]:
    results: list[pymupdf.Rect] = []
    page_dict = page.get_text("dict", sort=True, flags=TEXT_DICT_FLAGS)
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(str(span.get("text", "")) for span in spans)
            search_start = 0
            while True:
                match_start = line_text.find(fragment, search_start)
                if match_start < 0:
                    break
                match_end = match_start + len(fragment)
                span_offset = 0
                match_rect: pymupdf.Rect | None = None
                for span in spans:
                    span_text = str(span.get("text", ""))
                    span_end = span_offset + len(span_text)
                    overlap_start = max(match_start, span_offset)
                    overlap_end = min(match_end, span_end)
                    if overlap_start < overlap_end:
                        span_rect = pymupdf.Rect(span["bbox"])
                        local_start = overlap_start - span_offset
                        local_end = overlap_end - span_offset
                        visible_fragment = span_text[local_start:local_end]
                        visible_rects = list(
                            page.search_for(
                                visible_fragment,
                                clip=expand_rect(
                                    span_rect,
                                    page=page,
                                    x_padding=1.0,
                                    y_padding=1.0,
                                ),
                            )
                        )
                        if not visible_rects:
                            visible_rects = [span_rect]
                        for visible_rect in visible_rects:
                            if match_rect is None:
                                match_rect = pymupdf.Rect(visible_rect)
                            else:
                                match_rect.include_rect(visible_rect)
                    span_offset = span_end
                if match_rect is not None:
                    results.append(match_rect)
                search_start = match_end
    return dedupe_rects(results)


def find_url_only_line_rects(
    *,
    page: pymupdf.Page,
    lines: list[tuple[str, pymupdf.Rect]],
) -> list[pymupdf.Rect]:
    ordered = sorted(lines, key=lambda item: (item[1].y0, item[1].x0))
    results: list[pymupdf.Rect] = []
    for line_text, line_rect in ordered:
        if not URL_ONLY_LINE_PATTERN.search(line_text):
            continue
        results.append(
            expand_rect(
                line_rect,
                page=page,
                x_padding=2.0,
                y_padding=1.5,
            )
        )
        previous_rect = line_rect
        while True:
            candidates = [
                (text, rect)
                for text, rect in ordered
                if 0.0 <= rect.y0 - previous_rect.y1 <= 2.5
                and rect.x0 <= previous_rect.x1 + 3.0
                and rect.x1 >= previous_rect.x0 - 18.0
            ]
            if not candidates:
                break
            continuation_text, continuation_rect = min(
                candidates,
                key=lambda item: (item[1].y0, item[1].x0),
            )
            stripped = continuation_text.strip()
            if URL_ONLY_LINE_PATTERN.search(stripped):
                break
            if not URL_CONTINUATION_PATTERN.fullmatch(stripped):
                break
            results.append(
                expand_rect(
                    continuation_rect,
                    page=page,
                    x_padding=2.0,
                    y_padding=1.5,
                )
            )
            previous_rect = continuation_rect
    return dedupe_rects(results)


def find_first_page_identity_band(
    *,
    page: pymupdf.Page,
    lines: list[tuple[str, pymupdf.Rect]],
    title: str,
    authors: list[str],
) -> tuple[pymupdf.Rect | None, list[str]]:
    normalized_title = normalize_identity_text(title)
    if not normalized_title:
        return None, []
    abstract_rects = [
        rect
        for text, rect in lines
        if normalize_identity_text(text) == "abstract"
    ]
    if not abstract_rects:
        return None, []
    abstract_top = min(rect.y0 for rect in abstract_rects)
    title_rects: list[pymupdf.Rect] = []
    title_candidates: list[tuple[str, pymupdf.Rect]] = []
    for text, rect in lines:
        if rect.y1 >= abstract_top:
            continue
        normalized_text = normalize_identity_text(text)
        if normalized_text:
            title_candidates.append((normalized_text, rect))
    title_candidates.sort(key=lambda item: (item[1].y0, item[1].x0))
    for start, (normalized_line, rect) in enumerate(title_candidates):
        if len(normalized_line) < 8:
            continue
        if not (
            normalized_title.startswith(normalized_line)
            or normalized_line.startswith(normalized_title)
        ):
            continue
        candidate_rects = [rect]
        accumulated = normalized_line
        previous_rect = rect
        for next_line, next_rect in title_candidates[start + 1 :]:
            max_line_gap = max(8.0, previous_rect.height * 1.25)
            if next_rect.y0 - previous_rect.y1 > max_line_gap:
                break
            proposed = accumulated + next_line
            if not (
                normalized_title.startswith(proposed)
                or (
                    len(next_line) >= 8
                    and next_rect.height >= previous_rect.height * 0.8
                    and proposed.startswith(normalized_title)
                )
            ):
                break
            candidate_rects.append(next_rect)
            accumulated = proposed
            previous_rect = next_rect
        title_rects = candidate_rects
        break
    if not title_rects:
        fallback_lines: list[pymupdf.Rect] = []
        fallback_reasons: list[str] = []
        for text, rect in lines:
            if rect.y1 >= abstract_top:
                continue
            if any(contains_author_identity(text, author) for author in authors):
                fallback_lines.append(rect)
                if "known_author_line" not in fallback_reasons:
                    fallback_reasons.append("known_author_line")
            if EMAIL_PATTERN.search(text):
                fallback_lines.append(rect)
                if "email" not in fallback_reasons:
                    fallback_reasons.append("email")
            if ANONYMOUS_AUTHOR_PATTERN.search(text):
                fallback_lines.append(rect)
                if "anonymous_author" not in fallback_reasons:
                    fallback_reasons.append("anonymous_author")
        if not fallback_lines:
            return None, []
        y0 = max(0.0, min(rect.y0 for rect in fallback_lines) - 3.0)
        y1 = min(page.rect.height, abstract_top - 2.0)
        if y1 <= y0:
            return None, []
        return pymupdf.Rect(0.0, y0, page.rect.width, y1), fallback_reasons
    title_bottom = max(rect.y1 for rect in title_rects)
    header_lines = [
        (text, rect)
        for text, rect in lines
        if rect.y0 >= title_bottom - 0.5 and rect.y1 < abstract_top
    ]
    if not header_lines:
        return None, []

    reasons: list[str] = []
    if any(
        contains_author_identity(text, author)
        for text, _ in header_lines
        for author in authors
    ):
        reasons.append("known_author")
    if any(EMAIL_PATTERN.search(text) for text, _ in header_lines):
        reasons.append("email")
    if any(AFFILIATION_FRAGMENT_PATTERN.search(text) for text, _ in header_lines):
        reasons.append("affiliation")
    if any(ANONYMOUS_AUTHOR_PATTERN.search(text) for text, _ in header_lines):
        reasons.append("anonymous_author")
    if not reasons:
        return None, []

    y0 = max(
        title_bottom + 1.0,
        min(rect.y0 for _, rect in header_lines) - 3.0,
    )
    y1 = min(page.rect.height, abstract_top - 2.0)
    if y1 <= y0:
        return None, []
    return pymupdf.Rect(0.0, y0, page.rect.width, y1), reasons


def find_first_page_footer_identity_blocks(
    *,
    page: pymupdf.Page,
    lines: list[tuple[str, pymupdf.Rect]],
) -> list[pymupdf.Rect]:
    lower_lines = [
        (text, rect)
        for text, rect in lines
        if rect.y0 > page.rect.height * 0.70
    ]
    blocks: list[pymupdf.Rect] = []
    midpoint = page.rect.width / 2.0
    for column_index, (column_left, column_right) in enumerate(
        ((0.0, midpoint), (midpoint, page.rect.width))
    ):
        column_lines = [
            (text, rect)
            for text, rect in lower_lines
            if int((rect.x0 + rect.x1) / 2.0 >= midpoint) == column_index
        ]
        anchors = [
            (text, rect)
            for text, rect in column_lines
            if EMAIL_PATTERN.search(text)
            or FOOTER_IDENTITY_ANCHOR_PATTERN.search(text)
        ]
        if not anchors:
            continue
        identity_lines = [
            (text, rect)
            for text, rect in column_lines
            if EMAIL_PATTERN.search(text)
            or IDENTITY_LINE_PATTERN.search(text)
            or AFFILIATION_FRAGMENT_PATTERN.search(text)
        ]
        if not identity_lines:
            continue
        y0 = min(rect.y0 for _, rect in identity_lines)
        y1 = max(rect.y1 for _, rect in identity_lines)
        block_lines = [
            rect
            for _, rect in column_lines
            if rect.y1 >= y0 and rect.y0 <= y1
        ]
        if not block_lines:
            continue
        x0 = max(column_left, min(rect.x0 for rect in block_lines) - 3.0)
        x1 = min(column_right, max(rect.x1 for rect in block_lines) + 3.0)
        blocks.append(
            pymupdf.Rect(
                x0,
                max(0.0, y0 - 3.0),
                x1,
                min(page.rect.height, y1 + 3.0),
            )
        )
    return blocks


def dedupe_rects(rects: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
    unique: list[pymupdf.Rect] = []
    seen: set[tuple[int, int, int, int]] = set()
    for rect in rects:
        key = tuple(round(value * 10) for value in (rect.x0, rect.y0, rect.x1, rect.y1))
        if key not in seen and rect.width > 0 and rect.height > 0:
            seen.add(key)
            unique.append(rect)
    return unique


def expand_rect(
    rect: pymupdf.Rect,
    *,
    page: pymupdf.Page,
    x_padding: float,
    y_padding: float,
) -> pymupdf.Rect:
    return pymupdf.Rect(
        max(0.0, rect.x0 - x_padding),
        max(0.0, rect.y0 - y_padding),
        min(page.rect.width, rect.x1 + x_padding),
        min(page.rect.height, rect.y1 + y_padding),
    )


def representation_paths(record: PaperRecord) -> dict[str, Path | None]:
    artifacts = record.extra.get("text_artifacts")
    artifacts = artifacts if isinstance(artifacts, dict) else {}
    values = {
        "compact_repr": record.text_compact or artifacts.get("compact_repr"),
        "full_repr": record.text_full or artifacts.get("full_repr"),
        "scoring_repr": record.text_scoring or artifacts.get("scoring_repr"),
    }
    return {name: Path(value) if value else None for name, value in values.items()}


def scoring_artifacts_exist(record: PaperRecord) -> bool:
    artifacts = record.extra.get("scoring_artifacts")
    if not isinstance(artifacts, dict):
        return False
    return all(
        isinstance(artifacts.get(name), str) and Path(artifacts[name]).exists()
        for name in ("pdf", "compact_repr", "full_repr", "scoring_repr")
    )


def anonymization_fingerprint(record: PaperRecord, *, max_pages: int) -> str:
    text_paths = representation_paths(record)
    payload = "\n".join(
        [
            ANONYMIZATION_VERSION,
            str(max_pages),
            record.paper_id,
            record.title or "",
            "\x1f".join(record.authors),
            str(record.pdf_path or ""),
            sha256_file(Path(record.pdf_path)) if record.pdf_path and Path(record.pdf_path).exists() else "",
            *[
                f"{name}:{sha256_file(path) if path is not None and path.exists() else ''}"
                for name, path in sorted(text_paths.items())
            ],
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_identity_token(value: str) -> str:
    return normalize_identity_text(value)


def normalize_identity_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    normalized_parts: list[str] = []
    current_part: list[str] = []
    for character in decomposed.casefold():
        if is_spacing_diacritic(character):
            continue
        if character.isalnum():
            current_part.append(character)
            continue
        if current_part:
            normalized_parts.append(normalize_transliteration("".join(current_part)))
            current_part = []
    if current_part:
        normalized_parts.append(normalize_transliteration("".join(current_part)))
    return "".join(normalized_parts)


def is_spacing_diacritic(character: str) -> bool:
    if unicodedata.category(character) != "Lm":
        return False
    name = unicodedata.name(character, "")
    return any(
        term in name
        for term in ("ACCENT", "CARON", "MACRON", "BREVE", "TILDE", "RING")
    )


def normalize_transliteration(value: str) -> str:
    value = value.translate(
        str.maketrans(
            {
                "æ": "a",
                "đ": "d",
                "ð": "d",
                "ı": "i",
                "ł": "l",
                "œ": "o",
                "ø": "o",
                "þ": "th",
            }
        )
    )
    return value.replace("ae", "a").replace("oe", "o").replace("ue", "u")


def filter_author_identities(authors: list[str]) -> tuple[list[str], list[str]]:
    active: list[str] = []
    ignored: list[str] = []
    seen: set[str] = set()
    for author in authors:
        key = normalize_identity_text(author)
        letters = "".join(character for character in author if character.isalpha())
        looks_like_short_acronym = len(letters) <= 3 and letters.isupper()
        if not key or looks_like_short_acronym:
            ignored.append(author)
            continue
        if key not in seen:
            seen.add(key)
            active.append(author)
    return active, ignored


def author_name_variants(author: str) -> list[str]:
    variants = [author]
    tokens = re.findall(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", author, flags=re.UNICODE)
    if len(tokens) > 2:
        variants.append(f"{tokens[0]} {tokens[-1]}")
    return list(dict.fromkeys(variants))


def author_identity_keys(author: str) -> list[str]:
    return [
        key
        for variant in author_name_variants(author)
        if (key := normalize_identity_text(variant))
    ]


def contains_author_identity(text: str, author: str) -> bool:
    return bool(author_identity_line_indexes(text, author))


def author_identity_line_indexes(text: str, author: str) -> set[int]:
    lines = text.splitlines()
    matches = {
        index
        for index, line in enumerate(lines)
        if any(
            line_contains_name_variant(line, variant)
            for variant in author_name_variants(author)
        )
    }
    matches.update(wrapped_author_line_indexes(lines, [author]))
    return matches


def wrapped_author_line_indexes(lines: list[str], authors: list[str]) -> set[int]:
    matches: set[int] = set()
    for index, line in enumerate(lines[:-1]):
        if not re.search(r"[-‐‑‒–]\s*$", line):
            continue
        joined = re.sub(r"[-‐‑‒–]\s*$", "", line) + lines[index + 1].lstrip()
        if any(
            line_contains_name_variant(joined, variant)
            for author in authors
            for variant in author_name_variants(author)
        ):
            matches.update((index, index + 1))
    return matches


def correspondence_identity_line_indexes(lines: list[str]) -> set[int]:
    matches: set[int] = set()
    for index, line in enumerate(lines[:-1]):
        joined = line.rstrip() + " " + lines[index + 1].lstrip()
        if re.search(r"(?i)\bcorrespondence\s+to\s*:", joined):
            matches.update((index, index + 1))
    return matches


def identity_note_line_indexes(lines: list[str]) -> set[int]:
    matches = {
        index
        for index, line in enumerate(lines)
        if TEXT_IDENTITY_NOTE_PATTERN.search(line)
    }
    for index, line in enumerate(lines[:-1]):
        joined = line.rstrip() + " " + lines[index + 1].lstrip()
        if TEXT_IDENTITY_NOTE_PATTERN.search(joined):
            matches.update((index, index + 1))
    return matches


def line_contains_name_variant(line: str, variant: str) -> bool:
    target = normalize_identity_text(variant)
    tokens = [
        normalize_identity_token(token)
        for token in re.findall(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", line, flags=re.UNICODE)
    ]
    for start, token in enumerate(tokens):
        if not token or not target.startswith(token):
            continue
        accumulated = token
        for candidate in tokens[start + 1 :]:
            if accumulated == target:
                return True
            proposed = accumulated + candidate
            if candidate and target.startswith(proposed):
                accumulated = proposed
            elif candidate.isdigit() or len(candidate) == 1:
                continue
            else:
                break
        if accumulated == target:
            return True
    name_tokens = [
        normalize_identity_token(token)
        for token in re.findall(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", variant, flags=re.UNICODE)
    ]
    name_tokens = [token for token in name_tokens if token]
    if len(name_tokens) >= 2:
        normalized_line = "".join(token for token in tokens if token)
        token_boundaries = {0}
        offset = 0
        for token in tokens:
            if token:
                offset += len(token)
                token_boundaries.add(offset)
        first_name = name_tokens[0]
        last_name = name_tokens[-1]
        for first_at in sorted(token_boundaries):
            first_end = first_at + len(first_name)
            if (
                first_end not in token_boundaries
                or not normalized_line.startswith(first_name, first_at)
            ):
                continue
            for last_at in sorted(
                boundary for boundary in token_boundaries if boundary >= first_end
            ):
                last_end = last_at + len(last_name)
                if last_at - first_end > 40:
                    break
                if (
                    last_end in token_boundaries
                    and normalized_line.startswith(last_name, last_at)
                ):
                    return True
    return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
