#!/usr/bin/env python3
"""Wiki-sync gate — fail a commit that changes code without changing the wiki.

WHY THIS EXISTS
---------------
`audit/continuity.py::check_wiki` reads exactly three files:

    audit/scientific_flaws.md
    agent_wiki/README.md
    agent_wiki/development/common_pitfalls.md

It therefore does **not** see `agent_wiki/compositional/` or `agent_wiki/thmc/` at
all — currently **1,087 KB** of Rust-engine specification with zero automated
drift checking. Nothing in this repository (no pre-commit hook, no CI workflow
other than `docs.yml`) would notice an agent changing code and never updating
the docs.

DESIGN RULES (from the repo's own invariants)
---------------------------------------------
* **INV-1 fail loudly.** A missing doc update is an error, not a warning. A gate
  that only warns is a gate that does not exist.
* **INV-1 no silent degradation.** The escape hatch is a *visible commit-marker*
  (`Docs-Skip:`), never a flag, never a config file, never a silent pass. Every
  skip is greppable in the history and therefore auditable.
* **Derive-then-assert.** The source/wiki classification is derived from the
  changed-path list, not from a hand-maintained inventory.

ESCAPE HATCH
------------
    Docs-Skip: <reason>

The marker is required to be non-empty. A bare `Docs-Skip:` with no reason is
rejected. Skips are counted and reported so they cannot accumulate unnoticed.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Path classification ────────────────────────────────────────────────────────
# Source roots: the Python engine plus, once it exists, the Rust engine.
SOURCE_DIRS = ("core/", "ui/", "evaluation/", "utils/", "analysis/", "tests/",
               "audit/", "src/", "crates/", "engine/")
SOURCE_SUFFIXES = (".rs", ".toml")
# Docs: the wiki is the single source of truth, and `3D_THMC_docs/` is immutable
# reference material that must never be edited to match code.
DOC_PREFIXES = ("agent_wiki/",)

DOC_SKIP = re.compile(r"^\s*Docs-Skip:\s*(\S.*)$", re.M)


@dataclass
class Result:
    source_changed: list[str] = field(default_factory=list)
    wiki_changed: list[str] = field(default_factory=list)
    immutable_touched: list[str] = field(default_factory=list)
    skips: dict[str, str] = field(default_factory=dict)
    commits: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.source_changed and not self.wiki_changed
                    and not self.skips)


def git(*args: str) -> str:
    return subprocess.run(("git",) + args, cwd=REPO_ROOT, check=True,
                          capture_output=True, text=True).stdout


def changed_paths(base: str, head: str) -> tuple[list[str], dict[str, str]]:
    """Return (paths, {commit_sha: subject}) for base..head."""
    names = git("diff", "--name-only", f"{base}..{head}").splitlines()
    shas = git("log", "--format=%H", f"{base}..{head}").splitlines()
    subjects: dict[str, str] = {}
    for sha in shas:
        subjects[sha] = git("log", "-1", "--format=%s", sha).strip()
        body = git("log", "-1", "--format=%B", sha)
        m = DOC_SKIP.search(body)
        if m and m.group(1).strip():
            subjects[sha] = subjects[sha] + "  [Docs-Skip: " + m.group(1).strip() + "]"
    return [n for n in names if n.strip()], subjects


def classify(paths: list[str]) -> tuple[list[str], list[str], list[str]]:
    src, doc, immutable = [], [], []
    for p in paths:
        norm = p.replace("\\", "/")
        if norm.startswith("3D_THMC_docs/"):
            immutable.append(norm)
        elif norm.startswith(DOC_PREFIXES):
            doc.append(norm)
        elif norm.startswith(SOURCE_DIRS) or norm.endswith(SOURCE_SUFFIXES):
            src.append(norm)
    return src, doc, immutable


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="HEAD~1", help="base ref (default HEAD~1)")
    ap.add_argument("--head", default="HEAD", help="head ref (default HEAD)")
    ap.add_argument("--allow-immutable", action="store_true",
                    help="permit edits to the immutable 3D_THMC_docs/ reference set")
    args = ap.parse_args()

    paths, subjects = changed_paths(args.base, args.head)
    r = Result(commits=[f"{s[:8]} {t}" for s, t in subjects.items()])
    r.source_changed, r.wiki_changed, r.immutable_touched = classify(paths)

    for sha, subj in subjects.items():
        m = DOC_SKIP.search(git("log", "-1", "--format=%B", sha))
        if m and m.group(1).strip():
            r.skips[sha] = m.group(1).strip()

    print("=" * 72)
    print("WIKI-SYNC GATE")
    print("=" * 72)
    print(f"  range    : {args.base}..{args.head}")
    print(f"  commits  : {len(subjects)}")
    print(f"  source   : {len(r.source_changed)} file(s)")
    print(f"  wiki     : {len(r.wiki_changed)} file(s)")
    print()

    if r.immutable_touched and not args.allow_immutable:
        print("FAIL — immutable reference set was modified:")
        for p in r.immutable_touched:
            print(f"  {p}")
        print("\n  `3D_THMC_docs/` is declared IMMUTABLE. The corrected spec lives")
        print("  in agent_wiki/. Correct the wiki, never the source document.")
        return 2

    for c in r.commits:
        print(f"  {c}")

    if not r.source_changed:
        print("\nPASS — no source change in range; nothing to sync.")
        return 0

    if r.wiki_changed:
        print("\nPASS — source changed and the wiki changed in the same range.")
        for p in r.wiki_changed[:12]:
            print(f"  + {p}")
        if len(r.wiki_changed) > 12:
            print(f"  ... and {len(r.wiki_changed) - 12} more")
        return 0

    if r.skips:
        print("\nPASS (EXPLICIT SKIP) — source changed with no wiki change, but a")
        print("  non-empty `Docs-Skip:` marker was found. Recorded, not silent:")
        for sha, why in r.skips.items():
            print(f"  {sha[:8]}  Docs-Skip: {why}")
        return 0

    print("\nFAIL — source changed and NO wiki file changed in the same range.")
    for p in r.source_changed[:20]:
        print(f"  M {p}")
    if len(r.source_changed) > 20:
        print(f"  ... and {len(r.source_changed) - 20} more")
    print("\n  The wiki is the source of truth. An agent that changes behaviour")
    print("  without recording it leaves the next agent working from a lie.")
    print("\n  Fix: update agent_wiki/ in this commit, or state why with")
    print("       Docs-Skip: <non-empty reason>")
    return 1


if __name__ == "__main__":
    sys.exit(main())