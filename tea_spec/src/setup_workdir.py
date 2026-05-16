# Copyright 2026 tznurmin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
import hashlib
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
DEFAULT_TEA_SHA256 = "bb811506fdaaf78a1fe9f1c4e0f3b89cefd75c65c70c421dbc309ec76946aec4"


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


def _download(url: str, out_path: Path, timeout: float = 60.0) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    tmp.replace(out_path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_sha256(path: Path, expected: str) -> None:
    expected = (expected or "").strip().lower()
    if not expected:
        return
    actual = _sha256_file(path)
    if actual.lower() != expected:
        raise RuntimeError(
            f"Checksum mismatch for {path}: expected {expected}, got {actual}"
        )


def _safe_extract(tf: tarfile.TarFile, dest: Path) -> None:
    """Extract tar into dest while blocking unsafe archive entries."""
    dest_abs = dest.resolve()

    for member in tf.getmembers():
        if member.issym() or member.islnk() or member.isdev():
            raise RuntimeError(f"Refusing unsafe archive member: {member.name}")
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


def install_tea_curated_data(
    workdir: Path, url: str, sha256: str, force: bool
) -> None:
    tea_root = workdir / "TEA_curated_data"
    res = validate_tea_curated_data(tea_root)
    if res.ok and not force:
        return

    dl_path = workdir / "_downloads" / "TEA_curated_data-v1.1.tar.gz"
    _download(url, dl_path)
    _verify_sha256(dl_path, sha256)

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
        help="Work directory (default: <repo_root>/work)",
    )
    ap.add_argument(
        "--tea-url",
        type=str,
        default=DEFAULT_TEA_URL,
        help="TEA_curated_data tarball URL (default: v1.1 release)",
    )
    ap.add_argument(
        "--tea-sha256",
        type=str,
        default=None,
        help=(
            "Expected SHA-256 checksum for the TEA_curated_data tarball. "
            "Defaults to the pinned checksum for the built-in v1.1 URL."
        ),
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
        tea_sha256 = args.tea_sha256
        if tea_sha256 is None:
            tea_sha256 = DEFAULT_TEA_SHA256 if args.tea_url == DEFAULT_TEA_URL else ""
        install_tea_curated_data(
            workdir=workdir,
            url=args.tea_url,
            sha256=tea_sha256,
            force=args.force,
        )


if __name__ == "__main__":
    main()
