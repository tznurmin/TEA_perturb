"""

Bootstrap the local work directory.

This script:
  1: creates the workdir scaffold used by the pipeline
  2: validates TEA_curated_data is present and non-empty
  3: optionally downloads and installs TEA_curated_data (v1.1) if missing

Usage (from repo root):
  python tea_spec/src/setup_workdir.py

"""

from __future__ import annotations

import argparse
import os
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TEA_URL = (
    "https://github.com/tznurmin/TEA_curated_data/archive/refs/tags/v1.1.tar.gz"
)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    issues: list[str]


def _repo_root() -> Path:
    # <repo>/tea_spec/src/setup_workdir.py
    return Path(__file__).resolve().parents[2]


def ensure_scaffold(workdir: Path) -> None:
    """Create expected workdir subdirectories (idempotent)."""
    (workdir / "TEA_curated_data").mkdir(parents=True, exist_ok=True)
    (workdir / "species").mkdir(parents=True, exist_ok=True)
    (workdir / "templates").mkdir(parents=True, exist_ok=True)
    (workdir / "cache").mkdir(parents=True, exist_ok=True)
    (workdir / "summaries_from_cache").mkdir(parents=True, exist_ok=True)
    (workdir / "_downloads").mkdir(parents=True, exist_ok=True)


def validate_tea_curated_data(tea_root: Path) -> ValidationResult:
    issues: list[str] = []

    if not tea_root.exists() or not tea_root.is_dir():
        return ValidationResult(False, [f"Missing directory: {tea_root}"])

    req_dirs = [
        tea_root / "curation_data" / "pathogens",
        tea_root / "curation_data" / "strains",
        tea_root / "source_articles",
    ]

    for d in req_dirs:
        if not d.exists() or not d.is_dir():
            issues.append(f"Missing directory: {d}")

    # basic sanity check
    pathogens = tea_root / "curation_data" / "pathogens"
    strains = tea_root / "curation_data" / "strains"
    articles = tea_root / "source_articles"

    if pathogens.exists() and not any(pathogens.glob("*.json")):
        issues.append(f"No JSON files found under: {pathogens}")
    if strains.exists() and not any(strains.glob("*.json")):
        issues.append(f"No JSON files found under: {strains}")
    if articles.exists() and not any(articles.rglob("*.txt")):
        issues.append(f"No article .txt files found under: {articles}")

    return ValidationResult(len(issues) == 0, issues)


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    tmp.replace(out_path)


def _safe_extract(tf: tarfile.TarFile, dest: Path) -> None:
    """Extract tar into dest while blocking path traversal."""
    dest_abs = dest.resolve()

    for member in tf.getmembers():
        member_path = (dest_abs / member.name).resolve()
        if (
            not str(member_path).startswith(str(dest_abs) + os.sep)
            and member_path != dest_abs
        ):
            raise RuntimeError(
                f"Refusing to extract path outside destination: {member.name}"
            )

    tf.extractall(dest_abs)


def _find_extracted_root(extract_dir: Path) -> Path:
    # GitHub tag archives extract to a single top level folder
    cands = [
        p
        for p in extract_dir.iterdir()
        if p.is_dir() and p.name.startswith("TEA_curated_data-")
    ]
    if len(cands) != 1:
        raise RuntimeError(
            "Unexpected archive layout. "
            f"Found {len(cands)} candidate roots: {[c.name for c in cands]}"
        )
    return cands[0]


def install_tea_curated_data(workdir: Path, url: str, force: bool) -> None:
    tea_root = workdir / "TEA_curated_data"
    res = validate_tea_curated_data(tea_root)
    if res.ok and not force:
        return

    dl_path = workdir / "_downloads" / "TEA_curated_data-v1.1.tar.gz"
    _download(url, dl_path)

    with tempfile.TemporaryDirectory(prefix="tea_curated_extract_") as tmp:
        tmpdir = Path(tmp)
        with tarfile.open(dl_path, "r:gz") as tf:
            _safe_extract(tf, tmpdir)

        extracted_root = _find_extracted_root(tmpdir)

        staging = workdir / "_downloads" / "_staging_TEA_curated_data"
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(extracted_root, staging)

        if tea_root.exists():
            shutil.rmtree(tea_root)
        staging.replace(tea_root)

    res2 = validate_tea_curated_data(tea_root)
    if not res2.ok:
        raise SystemExit("\n".join(res2.issues))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create/validate the work/ scaffold and TEA_curated_data."
    )
    ap.add_argument(
        "--workdir",
        type=str,
        default=None,
        help="Workspace directory (default: <repo_root>/work)",
    )
    ap.add_argument(
        "--tea-url",
        type=str,
        default=DEFAULT_TEA_URL,
        help="TEA_curated_data tarball URL (default: v1.1 release)",
    )
    ap.add_argument(
        "--validate-only",
        action="store_true",
        help="Create scaffold and validate only (never download).",
    )
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not download; fail if curated data is missing.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-download and replace TEA_curated_data even if present.",
    )
    args = ap.parse_args()

    repo_root = _repo_root()
    workdir = Path(args.workdir) if args.workdir else (repo_root / "work")

    ensure_scaffold(workdir)

    tea_root = workdir / "TEA_curated_data"
    res = validate_tea_curated_data(tea_root)

    if args.validate_only:
        if not res.ok:
            raise SystemExit("\n".join(res.issues))
        return

    if not res.ok and args.skip_download:
        raise SystemExit("\n".join(res.issues))

    if not res.ok or args.force:
        install_tea_curated_data(workdir=workdir, url=args.tea_url, force=args.force)


if __name__ == "__main__":
    main()
