from __future__ import annotations

import json
from pathlib import Path

import pytest

from icml_ai_ac.reproducibility import (
    ReleaseSpec,
    _spearman,
    export_release,
    verify_bundle,
)


def test_export_sanitizes_local_paths_and_verifies(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    artifact = repo / "data/results.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps({"path": f"{repo}/data/source.txt", "result": 7}),
        encoding="utf-8",
    )
    spec = ReleaseSpec(
        schema_version=1,
        release_id="test_release",
        release_date="2026-07-30",
        analysis_files=("data/results.json",),
        analysis_globs=(),
        audit_directories=(),
    )

    result = export_release(
        repo_root=repo,
        output_dir=tmp_path / "release",
        spec=spec,
        include_audit=False,
        create_archives=False,
        force=False,
    )

    bundle = tmp_path / "release" / result["analysis_bundle"]
    payload = (bundle / "data/results.json").read_text(encoding="utf-8")
    assert str(repo) not in payload
    assert "<REPO_ROOT>/data/source.txt" in payload
    assert verify_bundle(bundle) == {"status": "ok", "bundle_kind": "analysis", "files": 1}


def test_verify_bundle_rejects_modified_payload(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    artifact = repo / "data/results.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")
    spec = ReleaseSpec(1, "test", "2026-07-30", ("data/results.json",), (), ())
    result = export_release(
        repo_root=repo,
        output_dir=tmp_path / "release",
        spec=spec,
        include_audit=False,
        create_archives=False,
        force=False,
    )
    bundle = tmp_path / "release" / result["analysis_bundle"]
    (bundle / "data/results.json").write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="checksum failures"):
        verify_bundle(bundle)


def test_export_rejects_recognizable_credentials(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    artifact = repo / "data/results.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"token": "sk-or-v1-abcdefghijklmnopqrstuvwxyz123456"}', encoding="utf-8")
    spec = ReleaseSpec(1, "test", "2026-07-30", ("data/results.json",), (), ())
    with pytest.raises(ValueError, match="credential material"):
        export_release(
            repo_root=repo,
            output_dir=tmp_path / "release",
            spec=spec,
            include_audit=False,
            create_archives=False,
            force=False,
        )


def test_secret_scan_does_not_flag_hyphenated_paper_text(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    artifact = repo / "data/results.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"text": "learningtask-relevantpatternswithoutspaces"}', encoding="utf-8")
    spec = ReleaseSpec(1, "test", "2026-07-30", ("data/results.json",), (), ())
    result = export_release(
        repo_root=repo,
        output_dir=tmp_path / "release",
        spec=spec,
        include_audit=False,
        create_archives=False,
        force=False,
    )
    assert result["analysis_files"] == 1


def test_secret_scan_does_not_flag_ai_prefix_inside_encoded_payload(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    artifact = repo / "data/results.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"encoded": "prefixAIza' + "a" * 100 + 'suffix"}', encoding="utf-8")
    spec = ReleaseSpec(1, "test", "2026-07-30", ("data/results.json",), (), ())
    result = export_release(
        repo_root=repo,
        output_dir=tmp_path / "release",
        spec=spec,
        include_audit=False,
        create_archives=False,
        force=False,
    )
    assert result["analysis_files"] == 1


def test_spearman_uses_the_same_paper_population() -> None:
    assert _spearman(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert _spearman(["a", "b", "c"], ["c", "b", "a"]) == -1.0
    with pytest.raises(ValueError, match="same unique paper IDs"):
        _spearman(["a", "b"], ["a", "c"])
    with pytest.raises(ValueError, match="same unique paper IDs"):
        _spearman(["a", "a"], ["a", "a"])


def test_force_refuses_to_replace_repository_root(tmp_path: Path) -> None:
    spec = ReleaseSpec(1, "test", "2026-07-30", (), (), ())
    with pytest.raises(ValueError, match="protected output directory"):
        export_release(
            repo_root=tmp_path,
            output_dir=tmp_path,
            spec=spec,
            include_audit=False,
            create_archives=False,
            force=True,
        )
