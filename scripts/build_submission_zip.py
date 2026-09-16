"""Build the submission ZIP for a literal cold run + final handoff.

The cold run must unpack this ZIP somewhere fresh, follow README's quick
start (venv, pip install requirements, pytest, platform, run_two_weeks), and
reproduce the two-week loop-closure run from nothing but the repository. So
the ZIP contains the SOURCE TREE ONLY (code, prompts, templates, tests,
brief, docs) and excludes anything generated: .venv, logs/, *.db, __pycache__,
.pyc, .env (secrets), and the trace/report artifacts (they are produced by
the run itself).

Usage:  python scripts/build_submission_zip.py [out.zip]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return [ROOT / p for p in out.stdout.splitlines()]

EXPLICIT_EXTRA = [
    "scripts/run_two_weeks.py",
    "scripts/launch_twoweeks.cmd",
    "writeup/writeup.md",
    "writeup/week1_vs_week2_results.md",
]

EXCLUDE_DIRS = {"logs", ".venv", "writeup_assets", ".git"}
EXCLUDE_SUFFIXES = {".pyc", ".db", ".log", ".zip"}
EXCLUDE_NAMES = {".env"}


def build(out_path: Path, dry_run: bool = False) -> list[Path]:
    files = {p.resolve() for p in tracked_files()}
    files |= {ROOT / e for e in EXPLICIT_EXTRA}

    selected: list[Path] = []
    for p in sorted(files, key=lambda x: x.as_posix()):
        try:
            rel = p.relative_to(ROOT)
        except ValueError:
            continue
        parts = set(rel.parts)
        if parts & EXCLUDE_DIRS:
            continue
        if p.suffix in EXCLUDE_SUFFIXES or p.name in EXCLUDE_NAMES or p.suffix == ".pyc":
            continue
        if not p.is_file():
            continue
        selected.append(p)

    if dry_run:
        return selected

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in selected:
            zf.write(p, arcname=p.relative_to(ROOT).as_posix())
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default=str(ROOT / "Prodigal_AI_Task1_submission.zip"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    selected = build(Path(args.out), dry_run=args.dry_run)
    for p in selected:
        print(f" + {p.relative_to(ROOT).as_posix()}")
    print(f"\n{len(selected)} files")
    if not args.dry_run:
        print(f"zip: {Path(args.out)}")


if __name__ == "__main__":
    main()