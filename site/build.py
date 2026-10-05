"""Build a self-contained, ready-to-serve copy of the ICML 2026 results blog."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SITE_SOURCE = ROOT / "site"
DEFAULT_OUTPUT = ROOT / "build/icml-ai-ac-blog"

SITE_FILES = ("index.html", "styles.css", "app.js", "data.js")
DOC_FILES = (
    "docs/stage7_human_comparison.md",
    "docs/final_pipeline_methodology.md",
)


def build_blog(
    *,
    output: Path = DEFAULT_OUTPUT,
    data_root: Path | None = None,
    overwrite: bool = False,
    create_archive: bool = True,
) -> dict[str, Any]:
    output = output.resolve()
    archive = output.with_suffix(".zip") if create_archive else None
    ensure_available(output, overwrite=overwrite)
    if archive is not None:
        ensure_available(archive, overwrite=overwrite)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="icml-blog-", dir=output.parent) as temporary:
        package = Path(temporary) / output.name
        site_output = package / "site"
        docs_output = package / "docs"
        site_output.mkdir(parents=True)
        docs_output.mkdir(parents=True)

        for filename in SITE_FILES:
            if filename == "data.js" and data_root is not None:
                continue
            shutil.copy2(SITE_SOURCE / filename, site_output / filename)
        shutil.copytree(SITE_SOURCE / "assets", site_output / "assets")
        for relative in DOC_FILES:
            source = ROOT / relative
            shutil.copy2(source, docs_output / source.name)

        if data_root is not None:
            rebuild_data(data_root=data_root.resolve(), output=site_output / "data.js")

        write_package_readme(package)
        report = validate_blog(package, used_committed_data=data_root is None)
        if overwrite and output.exists():
            shutil.rmtree(output)
        shutil.move(str(package), output)

    if archive is not None:
        if overwrite and archive.exists():
            archive.unlink()
        shutil.make_archive(
            str(output),
            "zip",
            root_dir=output.parent,
            base_dir=output.name,
        )
        report["archive"] = str(archive)
        report["archive_bytes"] = archive.stat().st_size
    report["output"] = str(output)
    return report


def rebuild_data(*, data_root: Path, output: Path) -> None:
    if not data_root.is_dir():
        raise FileNotFoundError(f"analysis data root not found: {data_root}")
    subprocess.run(
        [
            sys.executable,
            str(SITE_SOURCE / "build_data.py"),
            "--data-root",
            str(data_root),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )


def ensure_available(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}; pass --overwrite to replace it")


def validate_blog(package: Path, *, used_committed_data: bool) -> dict[str, Any]:
    site_root = package / "site"
    required = [site_root / filename for filename in SITE_FILES]
    required.extend(package / relative for relative in DOC_FILES)
    missing = [str(path.relative_to(package)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"blog package is missing required files: {missing}")
    empty = [str(path.relative_to(package)) for path in required if path.stat().st_size == 0]
    if empty:
        raise ValueError(f"blog package contains empty required files: {empty}")

    data_text = (site_root / "data.js").read_text(encoding="utf-8")
    prefix = "window.ICML_AI_AC_DATA = "
    if not data_text.startswith(prefix) or not data_text.rstrip().endswith(";"):
        raise ValueError("site/data.js is not a valid ICML blog data assignment")
    payload = json.loads(data_text[len(prefix) :].strip().removesuffix(";"))
    papers = payload.get("papers")
    if not isinstance(papers, list) or len(papers) != 60:
        raise ValueError("site/data.js must contain the frozen 60-paper AI oral program")

    assets = sorted((site_root / "assets").glob("*"))
    if not assets or any(not path.is_file() or path.stat().st_size == 0 for path in assets):
        raise ValueError("site/assets must contain non-empty preview files")
    return {
        "status": "ok",
        "papers": len(papers),
        "assets": len(assets),
        "used_committed_data": used_committed_data,
    }


def write_package_readme(package: Path) -> None:
    text = """# ICML 2026 Results Blog

This directory is a self-contained static build of the results blog.

From this directory, run:

```bash
python3 -m http.server 8080
```

Then open `http://localhost:8080/site/`.

The committed `site/data.js` contains the frozen production results. The page links
its methodology and human-comparison documents on GitHub; copies are also included
under `docs/` for offline reading.
"""
    (package / "README.md").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional analysis-bundle data directory used to regenerate site/data.js.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = build_blog(
            output=args.output,
            data_root=args.data_root,
            overwrite=args.overwrite,
            create_archive=not args.no_archive,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Cannot build blog: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
