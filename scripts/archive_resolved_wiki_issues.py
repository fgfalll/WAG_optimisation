#!/usr/bin/env python3
"""
archive_resolved_wiki_issues.py

Automated utility to detect, extract, and migrate resolved audit issues,
flaws, and calibrations from active wiki documents into the centralized
archive: agent_wiki/audit/resolved_issues.md.

Usage:
    python scripts/archive_resolved_wiki_issues.py [--dry-run] [--source <path>] [--target <path>]
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


WIKI_AUDIT_DIR = Path("agent_wiki/audit")
DEFAULT_TARGET = WIKI_AUDIT_DIR / "resolved_issues.md"

SECTION_HEADER_RE = re.compile(r"^###\s+(.+)$", re.MULTILINE)
STATUS_LINE_RE = re.compile(r"^\s*-\s*\*\*Status\*\*:\s*([^\n\r]+)", re.MULTILINE | re.IGNORECASE)
ID_LINE_RE = re.compile(r"^\s*-\s*\*\*ID\*\*:\s*[`\"]?([^`\"\n\r]+)[`\"]?", re.MULTILINE | re.IGNORECASE)


class IssueSection:
    def __init__(self, title: str, content: str, start_pos: int, end_pos: int):
        self.title = title.strip()
        self.content = content.strip()
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.is_resolved = False
        self.issue_id = ""
        self._analyze()

    def _analyze(self):
        # Check header
        if "[RESOLVED]" in self.title.upper() or "RESOLVED" in self.title.upper():
            self.is_resolved = True

        # Check status line
        m_status = STATUS_LINE_RE.search(self.content)
        if m_status:
            status_val = m_status.group(1).strip().upper()
            if "RESOLVED" in status_val or "ERADICATED" in status_val or "CLOSED" in status_val:
                self.is_resolved = True

        # Check ID
        m_id = ID_LINE_RE.search(self.content)
        if m_id:
            self.issue_id = m_id.group(1).strip()
        else:
            # Try to infer ID from title like [TAG] or SCI-FLAW-XX or A.
            m_tag = re.search(r"\[([A-Za-z0-9_\-]+)\]", self.title)
            if m_tag:
                self.issue_id = m_tag.group(1)
            else:
                m_code = re.match(r"^([A-Za-z0-9_\-]+)[:\.]\s*", self.title)
                if m_code:
                    self.issue_id = m_code.group(1)
                else:
                    self.issue_id = self.title[:20].strip()


def parse_sections(text: str) -> List[IssueSection]:
    """Parse text into distinct ### sections."""
    matches = list(SECTION_HEADER_RE.finditer(text))
    if not matches:
        return []

    sections = []
    for i, match in enumerate(matches):
        header_text = match.group(1)
        start_idx = match.start()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start_idx:end_idx]
        sections.append(IssueSection(header_text, chunk, start_idx, end_idx))

    return sections


def slugify(text: str) -> str:
    """Convert header or ID to github-flavored markdown anchor slug."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s_]+", "-", text).strip("-")


def migrate_resolved_from_file(
    source_path: Path,
    target_path: Path,
    dry_run: bool = False
) -> Tuple[int, List[str]]:
    """Scan source markdown file, extract resolved sections, and append to target."""
    if not source_path.exists():
        print(f"Warning: Source file {source_path} does not exist.")
        return 0, []

    if source_path.resolve() == target_path.resolve():
        return 0, []

    text = source_path.read_text(encoding="utf-8")
    sections = parse_sections(text)

    resolved_sections = [s for s in sections if s.is_resolved]
    if not resolved_sections:
        return 0, []

    target_content = target_path.read_text(encoding="utf-8") if target_path.exists() else ""

    migrated_count = 0
    migrated_titles = []

    # New source text assembly
    new_text_chunks = []
    last_idx = 0

    for sec in sections:
        if sec.is_resolved:
            # Check if already present in target
            slug = slugify(sec.issue_id or sec.title)
            if sec.issue_id and sec.issue_id in target_content:
                already_in_target = True
            else:
                already_in_target = False

            if not already_in_target and not dry_run:
                # Append to target under Suspicious / General
                target_content += f"\n\n---\n\n{sec.content}\n"
                migrated_count += 1
                migrated_titles.append(sec.title)
            elif already_in_target:
                migrated_titles.append(f"{sec.title} (already in archive)")

            # Add prefix chunk up to sec.start_pos
            new_text_chunks.append(text[last_idx:sec.start_pos])

            # Replace resolved section in source with a clean one-line notice
            replacement_notice = (
                f"> [!NOTE]\n"
                f"> **{sec.title}** has been resolved and archived to "
                f"[`resolved_issues.md`](resolved_issues.md#{slug}).\n\n"
            )
            new_text_chunks.append(replacement_notice)
            last_idx = sec.end_pos
        else:
            # Keep open section intact
            pass

    new_text_chunks.append(text[last_idx:])
    updated_source_text = "".join(new_text_chunks)

    # Clean up multiple divider lines
    updated_source_text = re.sub(r"(\n---\n\s*){2,}", "\n---\n\n", updated_source_text)

    if not dry_run:
        source_path.write_text(updated_source_text, encoding="utf-8")
        if migrated_count > 0:
            target_path.write_text(target_content, encoding="utf-8")

    return len(resolved_sections), migrated_titles


def main():
    parser = argparse.ArgumentParser(description="Archive resolved wiki issues to resolved_issues.md")
    parser.add_argument("--dry-run", action="store_true", help="Scan and report without modifying files")
    parser.add_argument("--source", type=str, default=None, help="Specific markdown file to process")
    parser.add_argument("--target", type=str, default=str(DEFAULT_TARGET), help="Target resolved issues file")

    args = parser.parse_args()

    target_path = Path(args.target)
    if not target_path.exists() and not args.dry_run:
        print(f"Error: Target archive {target_path} does not exist.")
        sys.exit(1)

    if args.source:
        source_files = [Path(args.source)]
    else:
        # Scan all markdown files in agent_wiki/audit except target
        source_files = [p for p in WIKI_AUDIT_DIR.glob("*.md") if p.resolve() != target_path.resolve()]

    print(f"Scanning {len(source_files)} audit documents for resolved issues...")
    total_found = 0

    for src in source_files:
        found, titles = migrate_resolved_from_file(src, target_path, dry_run=args.dry_run)
        if found > 0:
            total_found += found
            print(f"  [{src.name}] Found {found} resolved issue(s):")
            for t in titles:
                print(f"    - {t}")

    action = "Identified" if args.dry_run else "Processed"
    print(f"\n{action} {total_found} resolved issue section(s).")


if __name__ == "__main__":
    main()
