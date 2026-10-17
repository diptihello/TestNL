"""Install a git pre-commit hook that blocks commits containing a Groq-style key.

Run once from the project folder:  python scripts\\install_hooks.py

The hook runs ``scripts/check_no_secrets.py`` and aborts the commit when it finds
a string beginning with ``gsk_`` in a tracked or about-to-be-tracked file. The
hook is written to the repository's real hooks folder (found with
``git rev-parse --git-path hooks``), so it works even when this project lives in
a sub-folder of a larger repository. An existing hook that this script did not
write is left alone unless ``--force`` is given; it is then backed up first.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKER = PROJECT_ROOT / "scripts" / "check_no_secrets.py"
MARKER = "# zomato-reco secret check"


def hooks_dir() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--git-path", "hooks"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    path = Path(out)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def hook_script() -> str:
    # Git for Windows runs hooks with its bundled sh, so use forward slashes.
    python = Path(sys.executable).as_posix()
    checker = CHECKER.as_posix()
    return (
        "#!/bin/sh\n"
        f"{MARKER}\n"
        f'"{python}" "{checker}" || {{\n'
        '  echo "pre-commit: commit blocked by the secret check." >&2\n'
        "  exit 1\n"
        "}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="Replace an existing hook (backed up).")
    args = parser.parse_args(argv)

    try:
        target_dir = hooks_dir()
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Not inside a git repository, or git is unavailable: {exc}", file=sys.stderr)
        return 2

    target_dir.mkdir(parents=True, exist_ok=True)
    hook = target_dir / "pre-commit"
    if hook.exists():
        existing = hook.read_text(encoding="utf-8", errors="ignore")
        if MARKER not in existing:
            if not args.force:
                print(
                    f"A different pre-commit hook already exists at {hook}. "
                    "Re-run with --force to back it up and replace it.",
                    file=sys.stderr,
                )
                return 1
            backup = hook.with_name("pre-commit.bak")
            hook.replace(backup)
            print(f"Existing hook backed up to {backup}")

    hook.write_text(hook_script(), encoding="utf-8", newline="\n")
    try:
        hook.chmod(0o755)
    except OSError:
        pass
    print(f"Installed pre-commit secret check at {hook}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
