from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from icml_ai_ac.scraper.openreview import content_value, forum_id_from_url
from icml_ai_ac.storage import read_json, read_jsonl


SCORE_FIELD_HINTS = (
    "rating",
    "overall",
    "recommendation",
    "soundness",
    "confidence",
    "contribution",
    "presentation",
    "novelty",
    "significance",
)

CANONICAL_SCORE_ALIASES = {
    "overall": ("overall_assessment", "overall_score", "rating", "recommendation"),
    "soundness": ("soundness",),
    "confidence": ("confidence",),
    "contribution": ("contribution",),
    "presentation": ("presentation",),
    "novelty": ("novelty",),
    "significance": ("significance",),
}


def normalize_title(value: str | None) -> str:
    text = unicodedata.normalize("NFKD", value or "").casefold()
    text = "".join(character for character in text if not unicodedata.combining(character))
    return " ".join(re.findall(r"[a-z0-9]+", text))


def summarize_openreview_scores(note: dict[str, Any]) -> dict[str, Any]:
    details = note.get("details") if isinstance(note.get("details"), dict) else {}
    replies = details.get("directReplies") if isinstance(details.get("directReplies"), list) else []
    reviews: list[dict[str, Any]] = []
    values_by_field: dict[str, list[float]] = defaultdict(list)

    for reply in replies:
        if not isinstance(reply, dict) or not is_official_review(reply):
            continue
        content = reply.get("content") if isinstance(reply.get("content"), dict) else {}
        scores: dict[str, float] = {}
        for raw_key, raw_value in content.items():
            key = normalize_score_key(str(raw_key))
            if not is_score_field(key):
                continue
            score = numeric_score(raw_value)
            if score is None:
                continue
            scores[key] = score
            values_by_field[key].append(score)
        reviews.append(
            {
                "review_id": reply.get("id"),
                "invitation": review_invitation(reply),
                "cdate": reply.get("cdate"),
                "mdate": reply.get("mdate"),
                "tcdate": reply.get("tcdate"),
                "tmdate": reply.get("tmdate"),
                "scores": scores,
            }
        )

    content = note.get("content") if isinstance(note.get("content"), dict) else {}
    summary: dict[str, Any] = {
        "openreview_forum_id": str(note.get("forum") or note.get("id") or ""),
        "title": content_value(content, "title"),
        "venue": content_value(content, "venue"),
        "venueid": content_value(content, "venueid"),
        "reply_count": len(replies),
        "review_count": len(reviews),
        "reviews": reviews,
        "score_fields": {
            key: {"values": values, "mean": mean(values)}
            for key, values in sorted(values_by_field.items())
        },
        "snapshot_kind": "published_current_note_state",
    }
    for canonical_name, aliases in CANONICAL_SCORE_ALIASES.items():
        values = canonical_values(reviews, aliases)
        if values:
            summary[f"{canonical_name}_values"] = values
            summary[f"{canonical_name}_mean"] = mean(values)
    return summary


def is_official_review(reply: dict[str, Any]) -> bool:
    invitations = reply.get("invitations")
    if isinstance(invitations, str):
        values = [invitations]
    elif isinstance(invitations, list):
        values = [str(value) for value in invitations]
    else:
        invitation = reply.get("invitation")
        values = [str(invitation)] if invitation else []
    return any("official_review" in value.casefold() for value in values)


def review_invitation(reply: dict[str, Any]) -> str | None:
    invitation = reply.get("invitation")
    if isinstance(invitation, str):
        return invitation
    invitations = reply.get("invitations")
    if isinstance(invitations, list):
        return next((str(value) for value in invitations if "official_review" in str(value).casefold()), None)
    return None


def normalize_score_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def is_score_field(key: str) -> bool:
    return any(hint in key for hint in SCORE_FIELD_HINTS)


def numeric_score(value: Any) -> float | None:
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    match = re.match(r"^\s*(-?\d+(?:\.\d+)?)", value)
    return float(match.group(1)) if match else None


def canonical_values(reviews: list[dict[str, Any]], aliases: tuple[str, ...]) -> list[float]:
    values: list[float] = []
    for review in reviews:
        scores = review.get("scores") if isinstance(review.get("scores"), dict) else {}
        for alias in aliases:
            if alias in scores:
                values.append(float(scores[alias]))
                break
    return values


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6)


def enrich_metadata_rows(
    rows: list[dict[str, Any]],
    *,
    ratings_path: Path | None = None,
    awards_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ratings = list(read_jsonl(ratings_path)) if ratings_path else []
    ratings_by_forum = {
        str(row.get("openreview_forum_id")): row
        for row in ratings
        if row.get("openreview_forum_id")
    }
    ratings_by_title = unique_rows_by_title(ratings)

    awards_payload = read_json(awards_path) if awards_path else {}
    awards = awards_payload.get("awards", []) if isinstance(awards_payload, dict) else []
    awards_source_url = awards_payload.get("source_url") if isinstance(awards_payload, dict) else None
    awards_announced_at = awards_payload.get("announced_at") if isinstance(awards_payload, dict) else None
    awards_by_title: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for award in awards:
        if isinstance(award, dict) and normalize_title(str(award.get("title") or "")):
            label = dict(award)
            label.setdefault("source_url", awards_source_url)
            label.setdefault("announced_at", awards_announced_at)
            awards_by_title[normalize_title(str(award.get("title") or ""))].append(label)

    enriched: list[dict[str, Any]] = []
    matched_rating_ids: set[str] = set()
    matched_award_titles: set[str] = set()
    for input_row in rows:
        row = dict(input_row)
        extra = dict(row.get("extra")) if isinstance(row.get("extra"), dict) else {}
        row["extra"] = extra
        forum_id = str(extra.get("openreview_forum_id") or forum_id_from_url(row.get("forum_url")) or "")
        title_key = normalize_title(str(row.get("title") or ""))

        rating = ratings_by_forum.get(forum_id) or ratings_by_title.get(title_key)
        if rating:
            extra["openreview_scores"] = rating
            rating_id = str(rating.get("openreview_forum_id") or "")
            if rating_id:
                matched_rating_ids.add(rating_id)

        award_labels = awards_by_title.get(title_key, [])
        if award_labels:
            extra["award_labels"] = award_labels
            extra["is_award_paper"] = True
            extra["is_outstanding_paper"] = any(
                award.get("award_type") == "outstanding_paper" and award.get("track") == "main"
                for award in award_labels
            )
            matched_award_titles.add(title_key)
        else:
            extra.setdefault("is_award_paper", False)
            extra.setdefault("is_outstanding_paper", False)
        enriched.append(row)

    unmatched_awards = [
        award
        for award in awards
        if isinstance(award, dict) and normalize_title(str(award.get("title") or "")) not in matched_award_titles
    ]
    report = {
        "records": len(enriched),
        "ratings_source": str(ratings_path) if ratings_path else None,
        "rating_rows": len(ratings),
        "records_with_ratings": sum(bool(row.get("extra", {}).get("openreview_scores")) for row in enriched),
        "unmatched_rating_rows": len(ratings_by_forum.keys() - matched_rating_ids),
        "awards_source": str(awards_path) if awards_path else None,
        "award_rows": len(awards),
        "records_with_awards": sum(bool(row.get("extra", {}).get("award_labels")) for row in enriched),
        "unmatched_awards": unmatched_awards,
    }
    return enriched, report


def unique_rows_by_title(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = normalize_title(str(row.get("title") or ""))
        if key:
            grouped[key].append(row)
    return {key: matches[0] for key, matches in grouped.items() if len(matches) == 1}
