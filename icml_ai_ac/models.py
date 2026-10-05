from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PaperRecord:
    """Canonical per-paper metadata row.

    The fields mirror the README's recommended data model. Crawler-specific
    details that are useful for auditing live scrapes belong in ``extra``.
    """

    paper_id: str
    source: str
    decision_label: str | None = None
    title: str | None = None
    abstract: str | None = None
    authors: list[str] = field(default_factory=list)
    forum_url: str | None = None
    pdf_url: str | None = None
    pdf_path: str | None = None
    text_full: str | None = None
    text_compact: str | None = None
    text_scoring: str | None = None
    parse_status: str | None = None
    topic_cluster: str | None = None
    scores_pass1: dict[str, Any] | None = None
    scores_pass2: dict[str, Any] | None = None
    pairwise_results: dict[str, Any] | None = None
    final_rank_scores: dict[str, Any] | None = None
    notes: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "PaperRecord":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        normalized = {key: value for key, value in row.items() if key in known}
        authors = normalized.get("authors")
        if isinstance(authors, str):
            normalized["authors"] = [authors]
        elif not isinstance(authors, list):
            normalized["authors"] = []
        extra = normalized.get("extra")
        if not isinstance(extra, dict):
            normalized["extra"] = {}
        return cls(**normalized)

    def pdf_filename(self) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in self.paper_id)
        return f"{safe}.pdf"

    def with_pdf_path(self, root: Path) -> "PaperRecord":
        self.pdf_path = str(root / self.pdf_filename())
        return self
