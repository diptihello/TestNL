"""Fail if a file that git tracks (or would track) contains a Groq-style key.

Run from anywhere:  python scripts\\check_no_secrets.py
Files ignored by git, such as .env, are not scanned. Matches are reported by file
and line only, never by value.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PATTERN = re.compile(r"gsk_[A-Za-z0-9]{10,}")
MAX_BYTES = 5_000_000
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def candidate_files(root: Path) -> list[Path]:
    """Tracked files plus untracked files that are not ignored."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    names = [n for n in out.decode("utf-8", "replace").split("\0") if n]
    return [root / n for n in names]


def scan(paths: list[Path]) -> list[tuple[Path, int]]:
    """Return (file, line number) for every line that contains a key-shaped string."""
    hits: list[tuple[Path, int]] = []
    for path in paths:
        try:
            if not path.is_file() or path.stat().st_size > MAX_BYTES:
                continue
            text = path.read_bytes().decode("utf-8", "ignore")
        except OSError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if PATTERN.search(line):
                hits.append((path, number))
    return hits


def main() -> int:
    try:
        files = candidate_files(PROJECT_ROOT)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Could not list files with git: {exc}", file=sys.stderr)
        return 2
    hits = scan(files)
    for path, number in hits:
        print(f"possible secret: {path.relative_to(PROJECT_ROOT)}:{number}")
    if hits:
        print("Remove the key, rotate it, and keep it only in .env.", file=sys.stderr)
        return 1
    print(f"No key-shaped strings in {len(files)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
