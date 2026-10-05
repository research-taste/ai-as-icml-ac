"""Export, verify, and rebuild the frozen ICML 2026 experiment release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from icml_ai_ac.storage import read_jsonl, write_json

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = REPO_ROOT / "reproducibility/icml_2026_release.json"
DEFAULT_RELEASE_DIR = REPO_ROOT / "dist/icml_2026_reproducibility"
ARCHIVE_TIMESTAMP = (2026, 7, 30, 0, 0, 0)
TEXT_SUFFIXES = {".json", ".jsonl", ".js", ".log", ".md", ".plist", ".py", ".txt"}
SECRET_PATTERNS = (
    re.compile(rb"(?<![A-Za-z0-9])sk-(?:or-v1-)?[A-Za-z0-9_-]{24,}"),
    re.compile(rb"(?<![0-9A-Za-z_-])AIza[0-9A-Za-z_-]{35}(?![0-9A-Za-z_-])"),
    re.compile(rb"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{24,}(?![A-Za-z0-9])"),
    re.compile(
        rb"(?:OPENAI|OPENROUTER|ANTHROPIC|GOOGLE)_API_KEY[\"']?\s*[:=]\s*[\"'][^\"'\r\n]{12,}"
    ),
    re.compile(rb"Authorization\s*[:=]\s*[\"']?Bearer\s+[A-Za-z0-9._-]{20,}"),
)


@dataclass(frozen=True, slots=True)
class ReleaseSpec:
    schema_version: int
    release_id: str
    release_date: str
    analysis_files: tuple[str, ...]
    analysis_globs: tuple[str, ...]
    audit_directories: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> ReleaseSpec:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            schema_version=int(payload["schema_version"]),
            release_id=str(payload["release_id"]),
            release_date=str(payload["release_date"]),
            analysis_files=tuple(payload["analysis_files"]),
            analysis_globs=tuple(payload.get("analysis_globs") or ()),
            audit_directories=tuple(payload.get("audit_directories") or ()),
        )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"release path must be repository-relative: {value}")
    if any(part == ".env" or part.startswith(".env.") for part in path.parts):
        raise ValueError(f"credential file cannot enter a release: {value}")
    if path.suffix.casefold() == ".pdf":
        raise ValueError(f"PDFs belong outside the results release: {value}")
    return path


def _analysis_paths(repo_root: Path, spec: ReleaseSpec) -> list[Path]:
    relative_paths = {_safe_relative_path(value) for value in spec.analysis_files}
    for pattern in spec.analysis_globs:
        _safe_relative_path(pattern)
        matches = [path for path in repo_root.glob(pattern) if path.is_file()]
        if not matches:
            raise FileNotFoundError(f"analysis glob matched no files: {pattern}")
        relative_paths.update(path.relative_to(repo_root) for path in matches)
    missing = [path for path in sorted(relative_paths) if not (repo_root / path).is_file()]
    if missing:
        raise FileNotFoundError("missing release artifacts: " + ", ".join(map(str, missing)))
    return [repo_root / path for path in sorted(relative_paths)]


def _audit_paths(repo_root: Path, spec: ReleaseSpec) -> list[Path]:
    paths: set[Path] = set()
    for value in spec.audit_directories:
        relative = _safe_relative_path(value)
        directory = repo_root / relative
        if not directory.is_dir():
            raise FileNotFoundError(f"missing audit directory: {relative}")
        for path in directory.rglob("*"):
            if path.is_file():
                _safe_relative_path(str(path.relative_to(repo_root)))
                paths.add(path)
    return sorted(paths, key=lambda path: str(path.relative_to(repo_root)))


def _sanitized_bytes(path: Path, *, repo_root: Path) -> tuple[bytes, bytes, bool]:
    source_data = path.read_bytes()
    if path.suffix.casefold() not in TEXT_SUFFIXES:
        return source_data, source_data, False
    source = str(repo_root).encode("utf-8")
    sanitized = source_data.replace(source, b"<REPO_ROOT>")
    matches = [pattern.pattern.decode("ascii", "replace") for pattern in SECRET_PATTERNS if pattern.search(sanitized)]
    if matches:
        raise ValueError(f"possible credential material in {path}: {matches}")
    return sanitized, source_data, sanitized != source_data


def _entry(data: bytes, source_data: bytes, *, relative: Path, sanitized: bool) -> dict[str, Any]:
    return {
        "path": relative.as_posix(),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
        "source_sha256": sha256_bytes(source_data),
        "sanitized_local_paths": sanitized,
    }


def _manifest(spec: ReleaseSpec, *, bundle_kind: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": spec.schema_version,
        "release_id": spec.release_id,
        "release_date": spec.release_date,
        "bundle_kind": bundle_kind,
        "file_count": len(entries),
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "files": entries,
        "notes": {
            "credentials_included": False,
            "pdfs_included": False,
            "local_repository_paths_replaced_with": "<REPO_ROOT>",
        },
    }


def _checksum_text(entries: Iterable[dict[str, Any]]) -> str:
    return "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in entries)


def _analysis_readme(spec: ReleaseSpec) -> str:
    return f"""# ICML 2026 Analysis Bundle

Release: `{spec.release_id}` ({spec.release_date})

This bundle contains the canonical metadata, every parsed model judgment used
by the published pipeline, frozen intermediate rankings, tournament matches,
and deterministic human-comparison outputs. It intentionally excludes PDFs,
extracted paper text, credentials, and raw provider request/response payloads.

From a checkout of the repository:

```bash
python -m icml_ai_ac.reproducibility verify /path/to/analysis_bundle
python -m icml_ai_ac.reproducibility rebuild \\
  --bundle /path/to/analysis_bundle \\
  --output /tmp/icml_2026_rebuilt
```

The rebuild command reruns all Stage 7 quantitative analyses without model
calls, regenerates the website data file, and checks the canonical result
content against the frozen references in this bundle.
"""


def _write_zip_member(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=ARCHIVE_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)


def _archive_directory(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    _write_zip_member(archive, path.relative_to(source).as_posix(), path.read_bytes())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_output_directory(path: Path, *, force: bool, protected: Iterable[Path]) -> None:
    resolved = path.resolve()
    protected_paths = {Path("/").resolve(), Path.home().resolve(), *(item.resolve() for item in protected)}
    if resolved in protected_paths or len(resolved.parts) < 3:
        raise ValueError(f"refusing to replace protected output directory: {resolved}")
    if resolved.exists():
        if not force:
            raise FileExistsError(f"output already exists: {resolved}; pass --force to replace it")
        shutil.rmtree(resolved)


def export_release(
    *,
    repo_root: Path,
    output_dir: Path,
    spec: ReleaseSpec,
    include_audit: bool,
    create_archives: bool,
    force: bool,
) -> dict[str, Any]:
    analysis_dir = output_dir / "analysis_bundle"
    _prepare_output_directory(output_dir, force=force, protected=(repo_root,))
    analysis_dir.mkdir(parents=True)

    entries: list[dict[str, Any]] = []
    for source in _analysis_paths(repo_root, spec):
        relative = source.relative_to(repo_root)
        data, source_data, sanitized = _sanitized_bytes(source, repo_root=repo_root)
        destination = analysis_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        entries.append(_entry(data, source_data, relative=relative, sanitized=sanitized))

    manifest = _manifest(spec, bundle_kind="analysis", entries=entries)
    write_json(analysis_dir / "MANIFEST.json", manifest)
    (analysis_dir / "CHECKSUMS.sha256").write_text(_checksum_text(entries), encoding="utf-8")
    (analysis_dir / "README.md").write_text(_analysis_readme(spec), encoding="utf-8")
    verify_bundle(analysis_dir)

    result: dict[str, Any] = {
        "analysis_bundle": "analysis_bundle",
        "analysis_files": len(entries),
        "analysis_bytes": manifest["total_bytes"],
    }
    if create_archives:
        analysis_archive = output_dir / f"{spec.release_id}_analysis.zip"
        _archive_directory(analysis_dir, analysis_archive)
        result["analysis_archive"] = analysis_archive.name
        result["analysis_archive_bytes"] = analysis_archive.stat().st_size

    if include_audit:
        audit_archive = output_dir / f"{spec.release_id}_audit.zip"
        temporary_audit = audit_archive.with_name(f".{audit_archive.name}.tmp")
        audit_entries: list[dict[str, Any]] = []
        try:
            with zipfile.ZipFile(temporary_audit, "w", allowZip64=True) as archive:
                for source in _audit_paths(repo_root, spec):
                    relative = source.relative_to(repo_root)
                    data, source_data, sanitized = _sanitized_bytes(source, repo_root=repo_root)
                    audit_entries.append(
                        _entry(data, source_data, relative=relative, sanitized=sanitized)
                    )
                    _write_zip_member(archive, relative.as_posix(), data)
                audit_manifest = _manifest(spec, bundle_kind="audit", entries=audit_entries)
                _write_zip_member(
                    archive,
                    "MANIFEST.json",
                    (json.dumps(audit_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
                )
                _write_zip_member(archive, "CHECKSUMS.sha256", _checksum_text(audit_entries).encode("utf-8"))
            os.replace(temporary_audit, audit_archive)
        finally:
            temporary_audit.unlink(missing_ok=True)
        verify_archive(audit_archive)
        result.update(
            {
                "audit_archive": audit_archive.name,
                "audit_files": len(audit_entries),
                "audit_bytes": audit_manifest["total_bytes"],
                "audit_archive_bytes": audit_archive.stat().st_size,
            }
        )
    write_json(output_dir / "release_report.json", result)
    return result


def verify_bundle(bundle: Path) -> dict[str, Any]:
    manifest = json.loads((bundle / "MANIFEST.json").read_text(encoding="utf-8"))
    entries = manifest["files"]
    _validate_manifest(manifest)
    expected = {entry["path"] for entry in entries}
    actual = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and path.name not in {"MANIFEST.json", "CHECKSUMS.sha256", "README.md"}
    }
    if actual != expected:
        raise ValueError(
            f"bundle inventory mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    failures = []
    for entry in entries:
        path = bundle / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            failures.append(entry["path"])
    if failures:
        raise ValueError("bundle checksum failures: " + ", ".join(failures))
    if (bundle / "CHECKSUMS.sha256").read_text(encoding="utf-8") != _checksum_text(entries):
        raise ValueError("CHECKSUMS.sha256 does not match MANIFEST.json")
    return {"status": "ok", "bundle_kind": manifest["bundle_kind"], "files": len(entries)}


def _validate_manifest(manifest: dict[str, Any]) -> None:
    entries = manifest.get("files") or []
    paths = [entry["path"] for entry in entries]
    if len(paths) != len(set(paths)):
        raise ValueError("manifest contains duplicate paths")
    if manifest.get("file_count") != len(entries):
        raise ValueError("manifest file_count is inconsistent")
    if manifest.get("total_bytes") != sum(int(entry["bytes"]) for entry in entries):
        raise ValueError("manifest total_bytes is inconsistent")


def verify_archive(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("MANIFEST.json"))
        _validate_manifest(manifest)
        expected = {entry["path"] for entry in manifest["files"]}
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("archive contains duplicate member names")
        actual = set(names) - {"MANIFEST.json", "CHECKSUMS.sha256", "README.md"}
        if actual != expected:
            raise ValueError("archive inventory does not match MANIFEST.json")
        failures = []
        for entry in manifest["files"]:
            data = archive.read(entry["path"])
            if len(data) != entry["bytes"] or sha256_bytes(data) != entry["sha256"]:
                failures.append(entry["path"])
        if archive.read("CHECKSUMS.sha256").decode("utf-8") != _checksum_text(manifest["files"]):
            raise ValueError("CHECKSUMS.sha256 does not match MANIFEST.json")
    if failures:
        raise ValueError("archive checksum failures: " + ", ".join(failures))
    return {"status": "ok", "bundle_kind": manifest["bundle_kind"], "files": len(expected)}


def _reference_equal(generated: Path, reference: Path) -> bool:
    if not reference.is_file():
        return False
    if generated.suffix == ".json":
        return json.loads(generated.read_text(encoding="utf-8")) == json.loads(
            reference.read_text(encoding="utf-8")
        )
    if generated.suffix == ".jsonl":
        return list(read_jsonl(generated)) == list(read_jsonl(reference))
    if generated.suffix == ".js":
        prefix = "window.ICML_AI_AC_DATA = "

        def payload(path: Path) -> Any:
            text = path.read_text(encoding="utf-8")
            if not text.startswith(prefix):
                return None
            return json.loads(text[len(prefix) :].rstrip().removesuffix(";"))

        return payload(generated) == payload(reference)
    return generated.read_text(encoding="utf-8").rstrip("\n") == reference.read_text(
        encoding="utf-8"
    ).rstrip("\n")


def _run_cli(arguments: Sequence[str], *, repo_root: Path) -> None:
    command = [sys.executable, "-m", "icml_ai_ac.cli", *arguments]
    subprocess.run(command, cwd=repo_root, check=True)


def _strong_ranking(tournament: dict[str, Any], *, name: str, count: int) -> dict[str, Any]:
    return {
        "analysis": f"stage7_strong_ranking_{name}",
        "source": "data/scores/icml_2026_frontier_tournament_sol56_union172_swiss10_playoff60.json",
        "ranked_papers": tournament["ranked_papers"][:count],
    }


def _track_name(extra: dict[str, Any]) -> str:
    source = str(extra.get("sourceurl") or "")
    if source.endswith("/Conference"):
        return "main_track"
    if "Position_Paper_Track" in source:
        return "position_track"
    return "journal_presentations"


def _human_tier(extra: dict[str, Any]) -> str:
    if extra.get("is_oral"):
        return "oral"
    if extra.get("is_spotlight"):
        return "spotlight"
    return "poster"


def _reviewer_overall(extra: dict[str, Any]) -> float | None:
    value = (extra.get("openreview_scores") or {}).get("overall_mean")
    return float(value) if value is not None else None


def _spearman(ids_a: list[str], ids_b: list[str]) -> float:
    if (
        set(ids_a) != set(ids_b)
        or len(ids_a) != len(ids_b)
        or len(set(ids_a)) != len(ids_a)
    ):
        raise ValueError("Spearman inputs must contain the same unique paper IDs")
    rank_b = {paper_id: rank for rank, paper_id in enumerate(ids_b, start=1)}
    squared = sum((rank - rank_b[paper_id]) ** 2 for rank, paper_id in enumerate(ids_a, start=1))
    n = len(ids_a)
    return round(1 - (6 * squared) / (n * (n * n - 1)), 4)


def _pipeline_set(
    name: str,
    paper_ids: list[str],
    manifest: dict[str, dict[str, Any]],
    *,
    main_honor_rate: float,
) -> dict[str, Any]:
    rows = [manifest[paper_id] for paper_id in paper_ids]
    tracks = [_track_name(row.get("extra") or {}) for row in rows]
    extras = [row.get("extra") or {} for row in rows]
    honored = sum(_human_tier(extra) in {"oral", "spotlight"} for extra in extras)
    reviewer_scores = [value for extra in extras if (value := _reviewer_overall(extra)) is not None]
    honored_rate = honored / len(rows)
    return {
        "name": name,
        "papers": len(rows),
        "main_track": tracks.count("main_track"),
        "position_track": tracks.count("position_track"),
        "journal_presentations": tracks.count("journal_presentations"),
        "honored": honored,
        "honored_rate": round(honored_rate, 4),
        "main_track_honor_enrichment": round(honored_rate / main_honor_rate, 3),
        "reviewer_score_coverage": len(reviewer_scores),
        "reviewer_overall_mean": round(statistics.mean(reviewer_scores), 4),
        "award_papers": sum(bool(extra.get("is_award_paper")) for extra in extras),
    }


def build_stage7_summary(*, data_root: Path, eval_root: Path, release_date: str) -> dict[str, Any]:
    metadata = data_root / "metadata"
    scores = data_root / "scores"
    manifest_rows = list(read_jsonl(metadata / "icml_2026_scoring_manifest.jsonl"))
    manifest = {str(row["paper_id"]): row for row in manifest_rows}
    tracks: dict[str, list[dict[str, Any]]] = {
        "main_track": [],
        "position_track": [],
        "journal_presentations": [],
    }
    for row in manifest_rows:
        tracks[_track_name(row.get("extra") or {})].append(row)

    def population(rows: list[dict[str, Any]]) -> dict[str, Any]:
        extras = [row.get("extra") or {} for row in rows]
        tiers = [_human_tier(extra) for extra in extras]
        return {
            "papers": len(rows),
            "oral": tiers.count("oral"),
            "spotlight": tiers.count("spotlight"),
            "poster": tiers.count("poster"),
            "reviewer_score_coverage": sum(_reviewer_overall(extra) is not None for extra in extras),
        }

    main_population = population(tracks["main_track"])
    main_honor_rate = (main_population["oral"] + main_population["spotlight"]) / main_population["papers"]
    main_reviewer_scores = [
        value
        for row in tracks["main_track"]
        if (value := _reviewer_overall(row.get("extra") or {})) is not None
    ]
    tournament = json.loads(
        (scores / "icml_2026_frontier_tournament_sol56_union172_swiss10_playoff60.json").read_text()
    )
    seed = json.loads((scores / "icml_2026_tournament_seed_union172.json").read_text())
    finalists = [str(row["paper_id"]) for row in read_jsonl(scores / "icml_2026_finalists250.jsonl")]
    swiss = [str(row["paper_id"]) for row in seed["ranked_papers"] if row.get("tournament_pool_reasons")]
    direct = [str(row["paper_id"]) for row in tournament["ranked_papers"][:60]]
    bt = json.loads((scores / "icml_2026_frontier_tournament_bt_playoff.json").read_text())
    bt_ids = [str(row["paper_id"]) for row in bt["ranked_papers"]]

    main_agreement = json.loads(
        (eval_root / "icml_2026_human_agreement_main_track_finalists250.json").read_text()
    )["full_coverage"]
    position_agreement = json.loads(
        (eval_root / "icml_2026_human_agreement_position_track_finalists250.json").read_text()
    )["full_coverage"]
    direct_metrics = json.loads(
        (eval_root / "icml_2026_human_agreement_main_track_playoff60.json").read_text()
    )["strong_subset"]["strong_on_same_subset"]
    bt_metrics = json.loads(
        (eval_root / "icml_2026_human_agreement_playoff60_bt.json").read_text()
    )["strong_subset"]["strong_on_same_subset"]
    main_divergence = json.loads(
        (eval_root / "icml_2026_human_divergence_main_track_tier.json").read_text()
    )
    reviewer_divergence = json.loads(
        (eval_root / "icml_2026_human_divergence_reviewer.json").read_text()
    )

    return {
        "analysis": "icml_2026_stage7_human_comparison_summary",
        "status": "complete",
        "generated_at": release_date,
        "model_calls": 0,
        "inputs": {
            "canonical_human_manifest": "data/metadata/icml_2026_scoring_manifest.jsonl",
            "full_coverage_ai_ranking": "data/scores/icml_2026_pass1_cheap_ensemble_signal.jsonl",
            "final_tournament": "data/scores/icml_2026_frontier_tournament_sol56_union172_swiss10_playoff60.json",
            "playoff_bradley_terry": "data/scores/icml_2026_frontier_tournament_bt_playoff.json",
        },
        "populations": {
            "all_scored": len(manifest_rows),
            "main_track": main_population,
            "position_track": population(tracks["position_track"]),
            "journal_presentations": population(tracks["journal_presentations"]),
        },
        "main_track_full_coverage": {
            "kendall_tau_b_vs_tier": main_agreement["kendall_tau_b"]["vs_tier"],
            "kendall_tau_b_vs_reviewer_overall": main_agreement["kendall_tau_b"]["vs_reviewer_overall"],
            "roc_auc_honored_vs_poster": main_agreement["roc_auc"]["honored_vs_poster"],
            "roc_auc_honored_vs_poster_ci95": main_agreement["roc_auc_ci95"]["honored_vs_poster"],
            "roc_auc_oral_vs_rest": main_agreement["roc_auc"]["oral_vs_rest"],
            "roc_auc_oral_vs_rest_ci95": main_agreement["roc_auc_ci95"]["oral_vs_rest"],
            "honored_rate": round(main_honor_rate, 6),
            "reviewer_overall_mean": round(statistics.mean(main_reviewer_scores), 4),
        },
        "position_track_full_coverage": {
            "kendall_tau_b_vs_tier": position_agreement["kendall_tau_b"]["vs_tier"],
            "roc_auc_honored_vs_poster": position_agreement["roc_auc"]["honored_vs_poster"],
            "roc_auc_honored_vs_poster_ci95": position_agreement["roc_auc_ci95"]["honored_vs_poster"],
        },
        "pipeline_sets": [
            _pipeline_set("finalists", finalists, manifest, main_honor_rate=main_honor_rate),
            _pipeline_set("swiss_pool", swiss, manifest, main_honor_rate=main_honor_rate),
            _pipeline_set("all_pairs_playoff", direct, manifest, main_honor_rate=main_honor_rate),
            _pipeline_set("final_top_20", direct[:20], manifest, main_honor_rate=main_honor_rate),
            _pipeline_set("final_top_10", direct[:10], manifest, main_honor_rate=main_honor_rate),
        ],
        "playoff_ordering": {
            "direct_wins": {
                "kendall_tau_b_vs_tier": direct_metrics["kendall_tau_b"]["vs_tier"],
                "kendall_tau_b_vs_reviewer_overall": direct_metrics["kendall_tau_b"]["vs_reviewer_overall"],
                "roc_auc_honored_vs_poster": direct_metrics["roc_auc"]["honored_vs_poster"],
                "roc_auc_oral_vs_rest": direct_metrics["roc_auc"]["oral_vs_rest"],
            },
            "unweighted_bradley_terry": {
                "kendall_tau_b_vs_tier": bt_metrics["kendall_tau_b"]["vs_tier"],
                "kendall_tau_b_vs_reviewer_overall": bt_metrics["kendall_tau_b"]["vs_reviewer_overall"],
                "roc_auc_honored_vs_poster": bt_metrics["roc_auc"]["honored_vs_poster"],
                "roc_auc_oral_vs_rest": bt_metrics["roc_auc"]["oral_vs_rest"],
                "spearman_vs_direct_wins": _spearman(direct, bt_ids),
                "top_10_overlap": f"{len(set(direct[:10]) & set(bt_ids[:10]))}/10",
            },
        },
        "divergence_counts": {
            "main_track_ai_top_decile_posters": main_divergence["overlooked_gems"]["total_matching"],
            "main_track_ai_bottom_decile_oral_or_spotlight": main_divergence["blind_spots"]["total_matching"],
            "ai_top_reviewer_bottom_decile": reviewer_divergence["overlooked_gems"]["total_matching"],
            "reviewer_top_ai_bottom_decile": reviewer_divergence["blind_spots"]["total_matching"],
        },
        "primary_interpretation": (
            "The independent AI pipeline is enriched for human-recognized papers but imposes a "
            "materially different ordering and contribution-type preference."
        ),
    }


def _agreement_args(
    manifest: Path,
    scores: Path,
    strong: Path,
    out: Path,
    k_values: Sequence[int],
) -> list[str]:
    args = [
        "human-agreement-report",
        "--manifest", str(manifest),
        "--ai-scores", str(scores),
        "--strong-ranking", str(strong),
        "--out", str(out),
        "--honored-includes-award",
        "--min-coverage", "1.0",
        "--bootstrap-samples", "2000",
        "--bootstrap-seed", "20260730",
    ]
    for value in k_values:
        args.extend(("--k", str(value)))
    return args


def rebuild_analysis(
    *,
    bundle: Path,
    output: Path,
    repo_root: Path,
    spec: ReleaseSpec,
    check_reference: bool,
    force: bool,
) -> dict[str, Any]:
    verify_bundle(bundle)
    _prepare_output_directory(output, force=force, protected=(repo_root, bundle))
    eval_out = output / "data/evals"
    score_out = output / "data/scores"
    docs_out = output / "docs"
    site_out = output / "site"
    for directory in (eval_out, score_out, docs_out, site_out):
        directory.mkdir(parents=True, exist_ok=True)

    data = bundle / "data"
    metadata = data / "metadata"
    scores = data / "scores"
    cheap = scores / "icml_2026_pass1_cheap_ensemble_signal.jsonl"
    tournament_path = scores / "icml_2026_frontier_tournament_sol56_union172_swiss10_playoff60.json"
    tournament = json.loads(tournament_path.read_text(encoding="utf-8"))
    strong_specs = (("finalists250", 250), ("tournament172", 172), ("playoff60", 60))
    for name, count in strong_specs:
        write_json(score_out / f"icml_2026_stage7_strong_{name}.json", _strong_ranking(tournament, name=name, count=count))

    agreement_runs = (
        ("icml_2026_human_agreement_finalists250.json", "icml_2026_scoring_manifest.jsonl", "finalists250", (10, 20, 50, 60, 100, 172, 250)),
        ("icml_2026_human_agreement_tournament172.json", "icml_2026_scoring_manifest.jsonl", "tournament172", (10, 20, 50, 60, 100, 172)),
        ("icml_2026_human_agreement_playoff60.json", "icml_2026_scoring_manifest.jsonl", "playoff60", (10, 20, 50, 60)),
        ("icml_2026_human_agreement_main_track_finalists250.json", "icml_2026_scoring_manifest_main_track.jsonl", "finalists250", (10, 20, 50, 60, 100, 172, 250)),
        ("icml_2026_human_agreement_main_track_tournament172.json", "icml_2026_scoring_manifest_main_track.jsonl", "tournament172", (10, 20, 50, 60, 100, 172)),
        ("icml_2026_human_agreement_main_track_playoff60.json", "icml_2026_scoring_manifest_main_track.jsonl", "playoff60", (10, 20, 50, 60)),
        ("icml_2026_human_agreement_position_track_finalists250.json", "icml_2026_scoring_manifest_position_track.jsonl", "finalists250", (5, 10, 20, 50)),
    )
    for filename, manifest_name, strong_name, k_values in agreement_runs:
        _run_cli(
            _agreement_args(
                metadata / manifest_name,
                cheap,
                score_out / f"icml_2026_stage7_strong_{strong_name}.json",
                eval_out / filename,
                k_values,
            ),
            repo_root=repo_root,
        )
    for filename, strong_name in (
        ("icml_2026_human_agreement_playoff60_bt.json", "icml_2026_frontier_tournament_bt_playoff.json"),
        ("icml_2026_human_agreement_playoff60_bt_confidence.json", "icml_2026_frontier_tournament_bt_playoff_confidence.json"),
    ):
        _run_cli(
            _agreement_args(
                metadata / "icml_2026_scoring_manifest.jsonl",
                cheap,
                scores / strong_name,
                eval_out / filename,
                (10, 20, 50, 60),
            ),
            repo_root=repo_root,
        )

    divergence_runs = (
        ("tier", "icml_2026_scoring_manifest.jsonl", "icml_2026_human_divergence_tier", 25, 0),
        ("tier", "icml_2026_scoring_manifest_main_track.jsonl", "icml_2026_human_divergence_main_track_tier", 25, 0),
        ("tier", "icml_2026_scoring_manifest_position_track.jsonl", "icml_2026_human_divergence_position_track_tier", 20, 0),
        ("reviewer", "icml_2026_scoring_manifest.jsonl", "icml_2026_human_divergence_reviewer", 25, 3),
    )
    for axis, manifest_name, stem, cases, min_reviews in divergence_runs:
        args = [
            "human-divergence-report",
            "--manifest", str(metadata / manifest_name),
            "--ai-scores", str(cheap),
            "--strong-ranking", str(score_out / "icml_2026_stage7_strong_finalists250.json"),
            "--out", str(eval_out / f"{stem}.json"),
            "--markdown", str(docs_out / f"{stem}.md"),
            "--human-axis", axis,
            "--top-frac", "0.1",
            "--cases-per-direction", str(cases),
            "--min-reviews", str(min_reviews),
            "--min-coverage", "1.0",
        ]
        _run_cli(args, repo_root=repo_root)

    for manifest_name, filename in (
        ("icml_2026_scoring_manifest.jsonl", "icml_2026_ai_axis_decomposition.json"),
        ("icml_2026_scoring_manifest_main_track.jsonl", "icml_2026_ai_axis_decomposition_main_track.json"),
    ):
        _run_cli(
            [
                "ai-axis-decomposition",
                "--manifest", str(metadata / manifest_name),
                "--ai-scores", str(cheap),
                "--out", str(eval_out / filename),
                "--honored-includes-award",
                "--min-coverage", "1.0",
            ],
            repo_root=repo_root,
        )

    summary = build_stage7_summary(data_root=data, eval_root=eval_out, release_date=spec.release_date)
    write_json(eval_out / "icml_2026_stage7_summary.json", summary)
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "site/build_data.py"),
            "--data-root", str(data),
            "--summary", str(eval_out / "icml_2026_stage7_summary.json"),
            "--overrides", str(bundle / "site/arxiv_overrides.json"),
            "--output", str(site_out / "data.js"),
        ],
        cwd=repo_root,
        check=True,
    )

    generated = [
        *sorted(path for path in eval_out.glob("icml_2026_*.json")),
        *sorted(path for path in score_out.glob("icml_2026_*.json")),
        *sorted(path for path in docs_out.glob("icml_2026_*.md")),
        site_out / "data.js",
    ]
    mismatches: list[str] = []
    if check_reference:
        for path in generated:
            relative = path.relative_to(output)
            reference = bundle / relative
            if not _reference_equal(path, reference):
                mismatches.append(relative.as_posix())
        if mismatches:
            raise ValueError("rebuilt artifacts differ from frozen references: " + ", ".join(mismatches))
    report = {
        "status": "ok",
        "model_calls": 0,
        "generated_files": len(generated),
        "reference_check": "passed" if check_reference else "skipped",
        "mismatches": mismatches,
    }
    write_json(output / "rebuild_report.json", report)
    return report


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="Build checksummed analysis and optional raw-audit bundles.")
    export.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    export.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    export.add_argument("--output", type=Path, default=DEFAULT_RELEASE_DIR)
    export.add_argument("--include-audit", action="store_true")
    export.add_argument("--no-archives", action="store_true")
    export.add_argument("--force", action="store_true")

    verify = subparsers.add_parser("verify", help="Verify a bundle directory or ZIP against its manifest.")
    verify.add_argument("path", type=Path)

    rebuild = subparsers.add_parser("rebuild", help="Regenerate Stage 7 analyses and website data without model calls.")
    rebuild.add_argument("--bundle", type=Path, required=True)
    rebuild.add_argument("--output", type=Path, required=True)
    rebuild.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    rebuild.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    rebuild.add_argument("--no-reference-check", action="store_true")
    rebuild.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "export":
        result = export_release(
            repo_root=args.repo_root.resolve(),
            output_dir=args.output.resolve(),
            spec=ReleaseSpec.load(args.spec),
            include_audit=args.include_audit,
            create_archives=not args.no_archives,
            force=args.force,
        )
        print(
            f"Analysis bundle: {result['analysis_files']} files, "
            f"{_format_bytes(result['analysis_bytes'])}"
        )
        if "audit_files" in result:
            print(f"Audit archive: {result['audit_files']} files, {_format_bytes(result['audit_bytes'])}")
        return 0
    if args.command == "verify":
        result = verify_archive(args.path) if args.path.suffix.casefold() == ".zip" else verify_bundle(args.path)
        print(f"Verified {result['bundle_kind']} bundle: {result['files']} files")
        return 0
    if args.command == "rebuild":
        result = rebuild_analysis(
            bundle=args.bundle.resolve(),
            output=args.output.resolve(),
            repo_root=args.repo_root.resolve(),
            spec=ReleaseSpec.load(args.spec),
            check_reference=not args.no_reference_check,
            force=args.force,
        )
        print(
            f"Rebuilt {result['generated_files']} files without model calls; "
            f"reference check {result['reference_check']}"
        )
        return 0
    raise AssertionError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
