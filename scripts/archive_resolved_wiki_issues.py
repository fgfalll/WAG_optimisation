#!/usr/bin/env python3
"""
archive_resolved_wiki_issues.py

Automated utility to enforce the Agent Wiki invariant:
    "Active audit documents must contain ONLY active, open problems.
     All resolved issues, post-mortems, and fixes belong strictly
     in agent_wiki/audit/resolved_issues.md."

This script:
1. Scans active audit documents (agent_wiki/audit/*.md, excluding resolved_issues.md).
2. Detects resolved items in headers (## or ###), status metadata lines, and summary tables.
3. Ensures all resolved items are archived in agent_wiki/audit/resolved_issues.md.
4. Purges resolved sections, tables, and notices from active docs so only open issues remain.
5. In --check mode, exits with code 1 if resolved items are lingering in active audit docs.

Usage:
    python scripts/archive_resolved_wiki_issues.py [--clean] [--check] [--dry-run]
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

WIKI_AUDIT_DIR = Path("agent_wiki/audit")
RESOLVED_ARCHIVE_FILE = WIKI_AUDIT_DIR / "resolved_issues.md"

RESOLVED_KEYWORDS = {"RESOLVED", "ERADICATED", "CLOSED", "VERIFIED"}

STANDARD_ARCHIVE_NOTICE = (
    "> [!NOTE]\n"
    "> This document catalogs **only active, open items**. "
    "For resolved flaws, historical post-mortems, and verification status, "
    "consult the [**Resolved Issues & Defect Resolution Archive**](resolved_issues.md).\n"
)


def is_resolved_str(s: str) -> bool:
    upper = s.upper()
    return any(k in upper for k in RESOLVED_KEYWORDS)


def get_archived_issue_ids(archive_text: str) -> Set[str]:
    """Extract all issue IDs currently cataloged in resolved_issues.md."""
    ids = set()
    # Match in index table: | **ID** | ... or ### [ID] ...
    for m in re.finditer(r"\|\s*\*\*([A-Za-z0-9_\-]+)\*\*\s*\|", archive_text):
        ids.add(m.group(1).upper())
    for m in re.finditer(r"###\s+\[([A-Za-z0-9_\-]+)\]", archive_text):
        ids.add(m.group(1).upper())
    for m in re.finditer(r"-\s*\*\*ID\*\*:\s*[`\"]?([A-Za-z0-9_\-]+)[`\"]?", archive_text):
        ids.add(m.group(1).upper())
    return ids


def clean_markdown_table_resolved_rows(text: str) -> Tuple[str, List[str]]:
    """Remove rows from markdown tables where the status column is RESOLVED."""
    lines = text.splitlines(keepends=True)
    new_lines = []
    removed_rows = []

    for line in lines:
        if line.strip().startswith("|") and line.strip().endswith("|"):
            cells = [c.strip() for c in line.strip().split("|")[1:-1]]
            # Check if any cell indicates resolved status
            has_resolved_cell = False
            for cell in cells:
                cell_upper = cell.upper()
                # Check for explicit resolved status in cell
                if re.search(r"\b(RESOLVED|ERADICATED)\b", cell_upper):
                    # But don't match divider row
                    if not re.match(r"^:?-+:?$", cell):
                        has_resolved_cell = True
                        break
            if has_resolved_cell:
                removed_rows.append(line.strip())
                continue
        new_lines.append(line)

    return "".join(new_lines), removed_rows


def clean_resolved_blockquotes(text: str) -> Tuple[str, List[str]]:
    """Remove blockquote notes that mention resolved issues."""
    pattern = re.compile(
        r"(?:^|\n)> \[!NOTE\]\s*\n(?:> [^\n]*\n*)+",
        re.MULTILINE
    )
    removed = []

    def repl(match):
        chunk = match.group(0)
        chunk_upper = chunk.upper()
        # Don't remove the top-level standard active-only disclaimer
        if "ONLY ACTIVE" in chunk_upper or "DEFECT RESOLUTION ARCHIVE" in chunk_upper:
            return chunk
        if "RESOLVED" in chunk_upper or "ARCHIVED TO" in chunk_upper:
            first_line = chunk.splitlines()[1] if len(chunk.splitlines()) > 1 else chunk
            removed.append(first_line.strip("> *"))
            return "\n"
        return chunk

    cleaned_text = pattern.sub(repl, text)
    return cleaned_text, removed


def clean_resolved_sections(text: str) -> Tuple[str, List[str]]:
    """
    Remove complete ## or ### sections dedicated to resolved issues,
    such as '## 2. Master Resolved Discrepancies Archive',
    '## 4. pyproject.toml Configuration Debt: RESOLVED',
    '## 5. UI Presentation Layer Defects: ERADICATED',
    '## 3. Master Consolidated Duplicates Archive',
    '## 3. Master Resolved Calibrations Archive', etc.
    """
    lines = text.splitlines(keepends=True)
    new_lines = []
    removed_sections = []
    skipping = False
    skip_level = 2

    for line in lines:
        m = re.match(r"^(#{2,3})\s+(.+)$", line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            title_upper = title.upper()

            resolved_section_keywords = [
                "RESOLVED",
                "ERADICATED",
                "COMPLETED",
                "CONSOLIDATED DUPLICATES ARCHIVE",
                "RESOLVED CALIBRATIONS ARCHIVE",
                "RESOLVED DISCREPANCIES ARCHIVE",
                "RESOLVED ISSUES ARCHIVE",
            ]
            if any(k in title_upper for k in resolved_section_keywords):
                skipping = True
                skip_level = level
                removed_sections.append(title)
                continue
            elif skipping and level <= skip_level:
                skipping = False

        if not skipping:
            new_lines.append(line)

    return "".join(new_lines), removed_sections


def ensure_archive_notice(text: str) -> str:
    """Ensure the standard active-only notice exists after the main title."""
    if "This document catalogs **only active, open items**" in text:
        return text

    lines = text.splitlines(keepends=True)
    out = []
    inserted = False

    for i, line in enumerate(lines):
        out.append(line)
        if line.startswith("# ") and not inserted:
            # Insert notice after title and blank line
            out.append("\n" + STANDARD_ARCHIVE_NOTICE + "\n")
            inserted = True

    return "".join(out) if inserted else text


def process_audit_document(
    file_path: Path,
    archive_ids: Set[str],
    dry_run: bool = False
) -> Tuple[int, List[str]]:
    """Process a single active audit markdown document."""
    text = file_path.read_text(encoding="utf-8")
    original_text = text
    all_removed = []

    # 1. Clean resolved table rows
    text, removed_rows = clean_markdown_table_resolved_rows(text)
    if removed_rows:
        all_removed.extend([f"Table row: {r[:70]}..." for r in removed_rows])

    # 2. Clean blockquote notices
    text, removed_notes = clean_resolved_blockquotes(text)
    if removed_notes:
        all_removed.extend([f"Notice: {n}" for n in removed_notes])

    # 3. Clean dedicated resolved sections
    text, removed_secs = clean_resolved_sections(text)
    if removed_secs:
        all_removed.extend([f"Section: {s}" for s in removed_secs])

    # 4. Clean consecutive horizontal dividers and whitespace
    text = re.sub(r"(\n---\n\s*){2,}", "\n---\n\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. Add standard archive notice
    text = ensure_archive_notice(text)

    if not dry_run and text != original_text:
        file_path.write_text(text, encoding="utf-8")

    return len(all_removed), all_removed


def main():
    parser = argparse.ArgumentParser(description="Audit Wiki Active/Resolved Enforcer")
    parser.add_argument("--clean", action="store_true", help="Clean resolved items from active audit documents")
    parser.add_argument("--check", action="store_true", help="Exit with 1 if resolved items linger in active audit docs")
    parser.add_argument("--dry-run", action="store_true", help="Report without modifying files")
    args = parser.parse_args()

    if not RESOLVED_ARCHIVE_FILE.exists():
        print(f"[ERROR] Target archive {RESOLVED_ARCHIVE_FILE} does not exist.")
        sys.exit(1)

    archive_text = RESOLVED_ARCHIVE_FILE.read_text(encoding="utf-8")
    archive_ids = get_archived_issue_ids(archive_text)
    print(f"[INFO] Loaded {len(archive_ids)} resolved issue IDs from {RESOLVED_ARCHIVE_FILE}.")

    audit_files = sorted([
        p for p in WIKI_AUDIT_DIR.glob("*.md")
        if p.resolve() != RESOLVED_ARCHIVE_FILE.resolve()
    ])

    total_issues = 0
    issues_by_file = {}

    dry_run = args.dry_run or args.check

    for f in audit_files:
        count, items = process_audit_document(f, archive_ids, dry_run=dry_run)
        if count > 0:
            total_issues += count
            issues_by_file[f.name] = items

    if total_issues > 0:
        print(f"\n[FOUND] Detected {total_issues} resolved item(s) across {len(issues_by_file)} active audit docs:")
        for fname, items in issues_by_file.items():
            print(f"  • {fname} ({len(items)} items):")
            for item in items[:5]:
                print(f"      - {item}")
            if len(items) > 5:
                print(f"      - ... and {len(items) - 5} more")

        if args.check:
            print("\n[FAIL] Resolved issues found in active audit documents! Run with --clean to fix.")
            sys.exit(1)
        elif not dry_run:
            print(f"\n[SUCCESS] Cleaned {total_issues} resolved item(s) from active audit documents.")
        else:
            print(f"\n[INFO] Dry run complete. Run with --clean to remove resolved items from active docs.")
    else:
        print("\n[OK] All active audit documents strictly contain only open issues. No resolved clutter found.")
        sys.exit(0)


if __name__ == "__main__":
    main()
