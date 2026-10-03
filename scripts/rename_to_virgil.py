#!/usr/bin/env python3
"""Deterministic, idempotent drpangloss -> virgil rename.

Import package: `virgil`. PyPI distribution: `virgil-astro`.

Run from the repository root:

    python3 scripts/rename_to_virgil.py                   # apply in place
    python3 scripts/rename_to_virgil.py --check           # exit 1 if incomplete
    python3 scripts/rename_to_virgil.py --migrate-branch  # bring a feature
                                                           # branch across

Two tags on origin mark the rename on main:

    virgil-rename-base  last commit before the rename (adds this script)
    virgil-rename       the rename itself: exactly this script's output

Because `virgil-rename` is pure script output, a feature branch that already
contains `virgil-rename-base` can run the script itself and then record the
rename with `git merge -s ours virgil-rename` - no conflicts from lines that
both sides renamed. --migrate-branch does this with safety checks.

Stdlib only. Touches tracked, non-binary files only.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

OLD_PKG = Path("src/drpangloss")
NEW_PKG = Path("src/virgil")

# Files never rewritten: this script, the migration doc, historical logs,
# and anything binary (filtered separately).
SKIP_FILES = {
    "scripts/rename_to_virgil.py",
    "MIGRATION_VIRGIL.md",
    # The project's history names the original package; its present-tense
    # mentions are edited by hand in the rename's commit B.
    "docs/contributors.md",
    # The README (and the docs landing page generated from it) says what the
    # package used to be called, and points to its final release.
    "README.md",
    "docs/index.md",
}
SKIP_SUFFIXES = {".log", ".npy", ".oifits", ".fits", ".png", ".pdf"}

# Import name vs distribution name. The import package is `virgil`; the
# PyPI distribution is `virgil-astro` (`virgil` is taken on PyPI). Rules are
# applied in order: distribution-name contexts first, then everything else.
DIST = "virgil-astro"
PKG = "virgil"

# Applied only to the named files.
FILE_RULES = {
    "pyproject.toml": [
        (re.compile(r'(?m)^name = "drpangloss"$'), f'name = "{DIST}"'),
    ],
    "uv.lock": [
        (re.compile(r'name = "drpangloss"'), f'name = "{DIST}"'),
    ],
}

# Applied to every text file, in order.
RULES = [
    # Install commands: pip install [-U ...] drpangloss, uv add drpangloss.
    (
        re.compile(
            r"((?:pip|pip3|uv pip|uv) (?:install|add)(?: +-[-\w]+)* +)"
            r"drpangloss\b"
        ),
        rf"\g<1>{DIST}",
    ),
    # Extras: drpangloss[nufft] -> virgil-astro[nufft].
    (re.compile(r"\bdrpangloss\["), f"{DIST}["),
    # PyPI URLs and badges.
    (
        re.compile(r"(pypi\.org/project/|pypi/[a-z]+/)drpangloss\b"),
        rf"\g<1>{DIST}",
    ),
    # importlib.metadata lookups by distribution name.
    (
        re.compile(
            r"""((?:version|metadata|distribution)\(\s*["'])drpangloss(["'])"""
        ),
        rf"\g<1>{DIST}\g<2>",
    ),
    # Everything else is the import / repo / docs name. The bundled fixture
    # data/calibrated_visibility.npy carries the tag
    # "drpangloss-synthetic-mixed-disco-v1" inside the pickle; text quoting
    # that tag must keep matching it, so it is excluded by the lookahead.
    (re.compile(r"\bdrpangloss\b(?!-synthetic)"), PKG),
    (re.compile(r"\bDRPANGLOSS\b"), PKG.upper()),
]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def move_package() -> list[str]:
    """git mv src/drpangloss -> src/virgil, file by file (idempotent)."""
    moved = []
    tracked = [
        p for p in git("ls-files", "--", str(OLD_PKG)).splitlines() if p
    ]
    for old in tracked:
        new = str(NEW_PKG / Path(old).relative_to(OLD_PKG))
        if Path(new).exists():
            sys.exit(
                f"ERROR: both {old} and {new} exist. Merge them by hand, "
                "delete the drpangloss copy, then rerun."
            )
        Path(new).parent.mkdir(parents=True, exist_ok=True)
        git("mv", old, new)
        moved.append(f"{old} -> {new}")
    return moved


def is_binary(path: Path) -> bool:
    try:
        return b"\0" in path.read_bytes()[:8192]
    except OSError:
        return True


def rewrite_text(check: bool) -> list[str]:
    changed = []
    for name in git("ls-files").splitlines():
        p = Path(name)
        if (
            name in SKIP_FILES
            or p.suffix in SKIP_SUFFIXES
            or not p.is_file()
            or is_binary(p)
        ):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        new = text
        for pattern, repl in FILE_RULES.get(name, []) + RULES:
            new = pattern.sub(repl, new)
        if new != text:
            changed.append(name)
            if not check:
                p.write_text(new, encoding="utf-8")
    return changed


BASE_TAG = "virgil-rename-base"
RENAME_TAG = "virgil-rename"


def ok(*args: str) -> bool:
    return subprocess.run(["git", *args], capture_output=True).returncode == 0


def tree(rev: str) -> str:
    return git("rev-parse", f"{rev}^{{tree}}").strip()


def verify_rename_tag() -> None:
    """Prove RENAME_TAG == this script applied to BASE_TAG, byte for byte."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        wt = str(Path(tmp) / "wt")
        git("worktree", "add", "--detach", wt, BASE_TAG)
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(wt) / "scripts/rename_to_virgil.py"),
                ],
                cwd=wt,
                check=True,
                capture_output=True,
            )
            subprocess.run(["git", "add", "-A"], cwd=wt, check=True)
            got = subprocess.run(
                ["git", "write-tree"],
                cwd=wt,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        finally:
            git("worktree", "remove", "--force", wt)
    if got != tree(RENAME_TAG):
        sys.exit(
            f"STOP: {RENAME_TAG} is not pure script output of {BASE_TAG}. "
            "Do not use -s ours. Ask the human coordinator."
        )


def migrate_branch() -> None:
    if git("status", "--porcelain", "--untracked-files=no").strip():
        sys.exit("STOP: commit or stash your changes first.")
    branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    if branch in ("main", "HEAD", "gh-pages"):
        sys.exit(f"STOP: run this on a feature branch, not {branch}.")
    git("fetch", "origin", "--tags", "--quiet")
    for t in (BASE_TAG, RENAME_TAG):
        if not ok("rev-parse", "--verify", "--quiet", f"refs/tags/{t}"):
            sys.exit(f"STOP: tag {t} not found; the rename has not landed.")

    if ok("merge-base", "--is-ancestor", RENAME_TAG, "HEAD"):
        print(f"{branch} already contains {RENAME_TAG}.")
    else:
        # 1. Bring in everything on main up to the rename (normal merge).
        if not ok("merge-base", "--is-ancestor", BASE_TAG, "HEAD"):
            print(f"merging {BASE_TAG} ...")
            r = subprocess.run(["git", "merge", "--no-edit", BASE_TAG])
            if r.returncode:
                sys.exit(
                    "STOP: conflicts merging pre-rename main. Resolve them, "
                    "run the tests, `git commit`, then rerun this command."
                )
        # 2. Apply the rename on this branch with the identical script.
        verify_rename_tag()
        move_package()
        if OLD_PKG.exists():
            junk = [
                f
                for f in OLD_PKG.rglob("*")
                if f.is_file() and "__pycache__" not in f.parts
            ]
            if junk:
                sys.exit(
                    f"STOP: untracked files in {OLD_PKG}: {junk}. "
                    "Move them into src/virgil/ or delete, then rerun."
                )
            shutil.rmtree(OLD_PKG)
        rewrite_text(check=False)
        # Stage only tracked files (the rewrites; git mv staged the moves),
        # so that untracked files in the worktree never enter the commit.
        git("add", "-u")
        if git("status", "--porcelain", "--untracked-files=no").strip():
            git(
                "commit",
                "-q",
                "-m",
                "Apply drpangloss -> virgil rename "
                "(scripts/rename_to_virgil.py)",
            )
        # 3. Record the rename commit as merged; its content is already here.
        git(
            "merge",
            "-q",
            "-s",
            "ours",
            "--no-edit",
            "-m",
            f"Merge {RENAME_TAG} (applied by scripts/rename_to_virgil.py)",
            RENAME_TAG,
        )
        print(f"recorded {RENAME_TAG} on {branch}.")

    # 4. Bring in whatever landed on main after the rename (normal merge).
    print("merging origin/main ...")
    r = subprocess.run(["git", "merge", "--no-edit", "origin/main"])
    if r.returncode:
        sys.exit(
            "STOP: conflicts merging origin/main. These are ordinary content "
            "conflicts, not rename conflicts. Resolve, run the tests, "
            "`git commit`, then rerun this command to finish."
        )
    # 5. Check with the script as it now is on this branch (just merged from
    # main), not the copy running here: main may skip more files since.
    r = subprocess.run(
        [sys.executable, "scripts/rename_to_virgil.py", "--check"],
        capture_output=True,
        text=True,
    )
    if r.returncode:
        sys.exit(f"STOP: still mentions drpangloss:\n{r.stdout}")
    print(
        "done. Next: reinstall, run ruff and pytest, then push "
        "(plain `git push`, never --force)."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--check",
        action="store_true",
        help="report only; exit 1 if the rename is incomplete",
    )
    ap.add_argument(
        "--migrate-branch",
        action="store_true",
        help="bring the current feature branch across the rename",
    )
    args = ap.parse_args()

    if not Path(".git").exists() or not Path("pyproject.toml").exists():
        sys.exit("Run from the repository root.")

    if args.migrate_branch:
        migrate_branch()
        return

    if args.check:
        leftovers = rewrite_text(check=True)
        old_files = git("ls-files", "--", str(OLD_PKG)).split()
        for f in old_files + leftovers:
            print(f"needs rename: {f}")
        sys.exit(1 if (old_files or leftovers) else 0)

    moved = move_package()
    # Remove the old directory if only untracked caches are left in it;
    # otherwise `import drpangloss` would still succeed as an empty
    # namespace package and hide missed imports.
    if OLD_PKG.exists():
        junk = [
            f
            for f in OLD_PKG.rglob("*")
            if f.is_file() and "__pycache__" not in f.parts
        ]
        if junk:
            print(f"WARNING: untracked files left in {OLD_PKG}: {junk}")
        else:
            shutil.rmtree(OLD_PKG)
    changed = rewrite_text(check=False)
    print(f"moved {len(moved)} files, rewrote {len(changed)} files")
    for line in moved + changed:
        print("  " + line)
    # Stale build metadata from the old name confuses editable installs.
    for stale in Path("src").glob("drpangloss.egg-info"):
        print(f"note: delete stale {stale} (untracked) and reinstall")


if __name__ == "__main__":
    main()
