"""
Finding Registry — canonical writer, validator, and GitHub bridge.

WHY THIS EXISTS
---------------
Findings were being hand-written into `audit/scientific_flaws.md`, and the
hand-written records diverged from each other:

  * two layouts — `### CRIT-14 - title` bullet blocks and `| **MED-13** | ... |` tables
  * `**Status:**` sometimes on its own line, sometimes appended to the
    previous field
  * em-dash and hyphen both used as the title separator
  * severity/category free text, so `CRITICAL`/`Critical`/`crit` all appeared

The continuity gate's parser then silently missed the variants it did not
recognise, which is exactly how a finding can be "resolved" in the register and
still look open to every other tool.

This module makes the register **machine-generated and machine-checked**:

  * `finding new`   writes a canonical block (the only sanctioned way to add)
  * `finding validate` lints every existing record against the schema
  * `finding issue` creates the GitHub issue from the record, never by hand

A malformed record is a hard error, not a warning. An unknown severity, a
missing field, a location that does not exist on disk, or a duplicate ID all
fail with a non-zero exit code.

SCHEMA (exactly these eight fields, in this order)
--------------------------------------------------
    ### <ID> — <title>

    - **Severity:** <enum>
    - **Category:** <enum>
    - **Location:** `<path>:<line>`
    - **Observed:** <what the code actually does>
    - **Expected:** <what the physics/maths requires>
    - **Scientific Impact:** <consequence>
    - **Evidence & Citation:** <reproduction + literature>
    - **Status:** <enum>

USAGE
-----
    python -m audit registry validate
    python -m audit registry list [--status OPEN]
    python -m audit registry new CRIT-22 --severity CRITICAL --category MATHEMATICAL \\
        --location core/engine_surrogate/pvt_state.py:415 \\
        --observed "..." --expected "..." --impact "..." --evidence "..."
    python -m audit registry issue CRIT-22
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# The register contains mojibake-repaired unicode (µ, ², —, CO₂). Windows
# consoles default to cp1251 here and raise UnicodeEncodeError on print, which
# would make the CLI unusable for exactly the content it exists to handle.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover - non-tty streams
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTER = REPO_ROOT / "audit" / "scientific_flaws.md"

# --------------------------------------------------------------------------- #
# Canonical vocabularies. A value outside these sets is a hard error.
# --------------------------------------------------------------------------- #

SEVERITIES = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFORMATIONAL")
CATEGORIES = ("MATHEMATICAL", "PHYSICAL", "NUMERICAL", "SOFTWARE", "PROVENANCE")

# Status vocabulary is shared with audit.continuity so the two cannot diverge.
STATUSES = (
    "NEW",
    "OPEN",
    "CONFIRMED",
    "PARTIALLY_RESOLVED",
    "CONFIRMED_BUT_INERT",
    "REGRESSED",
    "RECURRED",              # was fixed, the same defect class returned
    "STILL_OPEN",
    "RESOLVED",
    "STALE",
    "UNKNOWN_EVIDENCE_REQUIRED",
    "SUPERSEDED",
)

ID_RE = re.compile(r"^(CRIT|HIGH|MED|LOW|INFO)-(\d{2,})$")
TITLE_SEP = "—"          # em dash, enforced
# The eight required fields, in canonical order. `Note` is an OPTIONAL ninth
# field that carries relocated prose (e.g. a Status line's original wording).
FIELDS = (
    "Severity",
    "Category",
    "Location",
    "Observed",
    "Expected",
    "Scientific Impact",
    "Evidence & Citation",
    "Status",
)
OPTIONAL_FIELDS = ("Note",)
ALL_FIELDS = FIELDS + OPTIONAL_FIELDS
FIELD_RE = re.compile(r"^-\s+\*\*([^*]+?):\*\*\s*(.*)$")
LOCATION_RE = re.compile(r"`([\w/\\.\-]+\.py)(?::(\d+)(?:\s*[-–,]\s*\d+)*)?`")

# Legacy label spellings seen in the register, mapped to the canonical name.
# The validator normalises through this table BEFORE checking, so historical
# prose is accepted while any NEW record must use the canonical labels.
CANON_LABEL = {
    "severity": "Severity",
    "category": "Category",
    "location": "Location",
    "observed": "Observed",
    "observed behavior": "Observed",
    "observed behaviour": "Observed",
    "observed (measured)": "Observed",
    "expected": "Expected",
    "expected behavior": "Expected",
    "expected behaviour": "Expected",
    "scientific impact": "Scientific Impact",
    "impact": "Scientific Impact",
    "evidence & citation": "Evidence & Citation",
    "evidence and citation": "Evidence & Citation",
    "evidence": "Evidence & Citation",
    "status": "Status",
    "note": "Note",
}


@dataclass
class Finding:
    """One parsed register record."""

    fid: str
    title: str
    severity: str
    category: str
    location: str
    observed: str
    expected: str
    impact: str
    evidence: str
    status: str
    line_no: int = 0
    raw: str = ""
    problems: list[str] = field(default_factory=list)

    @property
    def severity_tier(self) -> str:
        """GitHub label for this severity."""
        return f"severity-{self.severity.lower()}"

    @property
    def is_resolved(self) -> bool:
        return self.status.upper().startswith(("RESOLVED", "CONFIRMED", "STALE", "SUPERSEDED"))


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #

def parse_findings(register: Path = REGISTER) -> list[Finding]:
    """Parse canonical bullet-block findings.

    Deliberately strict: anything not matching the schema is returned with a
    populated `problems` list rather than being silently accepted or silently
    dropped. Historical table-form rows are reported as `legacy_table_row` so
    they can be migrated deliberately instead of rotting unnoticed.
    """
    if not register.exists():
        return []
    text = register.read_text(encoding="utf-8")
    lines = text.split("\n")

    out: list[Finding] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = re.match(r"^###\s+((?:CRIT|HIGH|MED|LOW|INFO)-\d+)\s*(.*)$", line)
        if not m:
            # legacy table row -> record as a stub with a migration problem
            mt = re.match(r"^\|\s*\*\*((?:CRIT|HIGH|MED|LOW|SCI-FLAW)-\w+)\*\*\s*\|(.*)\|\s*$", line)
            if mt:
                out.append(Finding(
                    fid=mt.group(1), title="", severity="", category="", location="",
                    observed="", expected="", impact="", evidence="", status="",
                    line_no=i + 1, raw=line,
                    problems=["legacy_table_row: migrate to the canonical 8-field block "
                              "with `audit registry new`"],
                ))
            i += 1
            continue

        fid = m.group(1)
        tail = m.group(2).strip()
        problems: list[str] = []

        # Title separator must be an em dash with single-space padding.
        if tail.startswith("- "):
            problems.append(
                "title separator is a hyphen; the canonical form is an em dash "
                f"'### {fid}{TITLE_SEP}<title>'"
            )
        title = tail.lstrip("-—").strip()

        if not ID_RE.match(fid):
            problems.append(f"malformed ID {fid!r}; expected CRIT-nn / HIGH-nn / MED-nn / LOW-nn / INFO-nn")

        # A field's value spans until the next field, the next heading, or a
        # horizontal rule. Values legitimately contain fenced code blocks,
        # tables and prose, so the terminator must be structural - not "any line
        # that is not a bullet", which truncated multi-line values.
        fields: dict[str, list[str]] = {}
        order: list[str] = []
        j = i + 1
        while j < n:
            ln = lines[j]
            if re.match(r"^###\s+", ln) or re.match(r"^---\s*$", ln):
                break
            fm = FIELD_RE.match(ln)
            if fm:
                label = fm.group(1).strip()
                fields[label] = [fm.group(2).strip()]
                order.append(label)
            elif ln.strip() in fields:
                # continuation of the previous field's value
                fields[ln.strip()].append(ln.rstrip())
            elif ln.strip():
                # blank, or a stray line inside the block: keep it if a field is open
                if order:
                    fields[order[-1]].append(ln.rstrip())
            j += 1

        # normalise labels through the alias table before validating
        normalised: dict[str, list[str]] = {}
        norm_order: list[str] = []
        for label in order:
            canon = CANON_LABEL.get(label.strip().lower().strip(" :"), label.strip())
            if canon not in normalised:
                normalised[canon] = []
                norm_order.append(canon)
            normalised[canon].extend(fields[label])
        fields = normalised
        order = norm_order

        def val(name: str) -> str:
            parts = fields.get(name, [])
            return "\n".join(parts).strip()

        missing = [f for f in FIELDS if f not in fields]
        extra = [f for f in fields if f not in ALL_FIELDS]
        if missing:
            problems.append(f"missing field(s): {', '.join(missing)}")
        if extra:
            problems.append(
                f"unrecognised field(s): {', '.join(extra)}; allowed: "
                f"{', '.join(ALL_FIELDS)}"
            )

        canonical_order = [f for f in ALL_FIELDS if f in fields]
        if order != canonical_order:
            problems.append(f"field order is {order}, expected {canonical_order}")

        sev = val("Severity")
        cat = val("Category")
        sta = val("Status").split()[0].upper() if val("Status") else ""

        if sev and sev.upper() not in SEVERITIES:
            problems.append(f"severity {sev!r} not in {SEVERITIES}")
        if cat and cat.upper().strip("`") not in CATEGORIES:
            problems.append(f"category {cat!r} not in {CATEGORIES}")
        if sta and sta not in STATUSES:
            problems.append(f"status {sta!r} not in {STATUSES}")

        loc = val("Location")
        # `repo-wide` is a legitimate scope for cross-cutting findings
        # (toolchain metrics, doc drift). It must not be forced to cite a file.
        repo_wide = bool(re.search(r"`repo-wide`", loc))
        if loc and not repo_wide:
            lm = LOCATION_RE.search(loc)
            if not lm:
                problems.append(
                    f"location {loc!r} cites neither a .py file:line nor `repo-wide`"
                )
            else:
                target = lm.group(1).replace("\\", "/")
                if target.startswith(("core/", "ui/", "analysis/", "evaluation/",
                                      "utils/", "tests/", "audit/")):
                    if not (REPO_ROOT / target).exists():
                        problems.append(f"location cites {target}, which does not exist")

        out.append(Finding(
            fid=fid, title=title,
            severity=sev.upper().strip("`"), category=cat.upper().strip("` "),
            location=loc,
            observed=val("Observed"),
            expected=val("Expected"),
            impact=val("Scientific Impact"),
            evidence=val("Evidence & Citation"),
            status=sta, line_no=i + 1,
            raw="\n".join(lines[i:j]),
            problems=problems,
        ))
        i = j
    return out


# --------------------------------------------------------------------------- #
# Canonical renderer
# --------------------------------------------------------------------------- #

def render_finding(f: Finding) -> str:
    """Render a finding in canonical form. The ONLY sanctioned serialiser."""
    return "\n".join([
        f"### {f.fid}{TITLE_SEP}{f.title}",
        "",
        f"- **Severity:** {f.severity}",
        f"- **Category:** `{f.category}`",
        f"- **Location:** {f.location}",
        f"- **Observed:** {f.observed}",
        f"- **Expected:** {f.expected}",
        f"- **Scientific Impact:** {f.impact}",
        f"- **Evidence & Citation:** {f.evidence}",
        f"- **Status:** {f.status}",
    ])


def next_id(findings: list[Finding], prefix: str) -> str:
    """Lowest unused ID for a severity prefix."""
    used = {int(m.group(2)) for f in findings if (m := ID_RE.match(f.fid)) and m.group(1) == prefix}
    return f"{prefix}-{(max(used) + 1 if used else 1):02d}"


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #

@dataclass
class ValidationReport:
    total: int = 0
    canonical: int = 0
    legacy: int = 0
    invalid: list[tuple[str, list[str]]] = field(default_factory=list)
    duplicate_ids: dict[str, list[int]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.invalid and not self.duplicate_ids


def validate(findings: Optional[list[Finding]] = None) -> ValidationReport:
    findings = findings if findings is not None else parse_findings()
    rep = ValidationReport(total=len(findings))

    seen: dict[str, list[int]] = {}
    for f in findings:
        seen.setdefault(f.fid, []).append(f.line_no)
        if "legacy_table_row" in " ".join(f.problems):
            rep.legacy += 1
            continue
        if f.problems:
            rep.invalid.append((f.fid, f.problems))
        else:
            rep.canonical += 1
    rep.duplicate_ids = {k: v for k, v in seen.items() if len(v) > 1}
    return rep


# --------------------------------------------------------------------------- #
# GitHub bridge
# --------------------------------------------------------------------------- #

def render_issue_body(f: Finding, repo: str = "fgfalll/WAG_optimisation") -> str:
    """Render a GitHub issue body FROM the register record.

    Generated, never hand-written: a divergence between the register and the
    tracker is then impossible by construction.
    """
    return "\n".join([
        f"## {f.fid} {TITLE_SEP} {f.title}",
        "",
        f"- **Severity:** {f.severity}",
        f"- **Category:** `{f.category}`",
        f"- **Location:** `{f.location}`",
        f"- **Status in register:** `{f.status}`",
        "",
        "### Observed",
        "",
        f.observed,
        "",
        "### Expected",
        "",
        f.expected,
        "",
        "### Scientific impact",
        "",
        f.impact,
        "",
        "### Evidence & citation",
        "",
        f.evidence,
        "",
        "---",
        "",
        f"_Generated by `python -m audit registry issue {f.fid}` from "
        f"[`audit/scientific_flaws.md`](https://github.com/{repo}/blob/main/audit/scientific_flaws.md). "
        f"Do not edit by hand; edit the register and regenerate._",
        "",
        "**Closure policy:** this issue closes only via a commit containing "
        f"`Closes #{'<number>'}`. One commit, one issue — see "
        "[`agent_wiki/development/continuity_gate.md`](https://github.com/"
        f"{repo}/blob/main/agent_wiki/development/continuity_gate.md).",
    ])


def _gh(*args: str, stdin: Optional[str] = None) -> tuple[int, str]:
    r = subprocess.run(["gh", *args], cwd=REPO_ROOT, capture_output=True,
                       text=True, encoding="utf-8", input=stdin)
    return r.returncode, (r.stdout or r.stderr).strip()


def ensure_labels() -> None:
    wanted = {
        "scientific-audit": ("5319E7", "Forensic audit finding"),
        "severity-critical": ("B60205", "CRITICAL"),
        "severity-high": ("D93F0B", "HIGH"),
        "severity-medium": ("FBCA04", "MEDIUM"),
        "severity-low": ("C2E0C6", "LOW"),
        "physics": ("1D76DB", "Reservoir physics defect"),
        "needs-evidence": ("F9D0C4", "Evidence required"),
    }
    rc, out = _gh("label", "list")
    have = {l.split("\t")[0].strip() for l in out.splitlines()}
    for name, (color, desc) in wanted.items():
        if name not in have:
            _gh("label", "create", name, "--color", color, "--description", desc)


def create_issue(fid: str, dry_run: bool = False) -> tuple[int, str]:
    findings = {f.fid: f for f in parse_findings()}
    if fid not in findings:
        return 2, f"{fid} not found in the register"
    f = findings[fid]
    if f.problems:
        return 2, f"{fid} is not schema-valid; run `audit registry validate` first:\n  " + \
                  "\n  ".join(f.problems)
    if dry_run:
        return 0, render_issue_body(f)
    ensure_labels()
    body = render_issue_body(f)
    labels = ["scientific-audit", f.severity_tier]
    if f.category == "PROVENANCE":
        labels.append("needs-evidence")
    if f.category in ("MATHEMATICAL", "PHYSICAL"):
        labels.append("physics")
    title = f"[{f.fid}] [{f.severity}] {f.title}"
    return _gh("issue", "create", "--title", title, "--body", body,
               *sum((["--label", l] for l in labels), []))


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def cmd_validate(_args: list[str]) -> int:
    findings = parse_findings()
    rep = validate(findings)
    print("=" * 78)
    print("REGISTER VALIDATION")
    print("=" * 78)
    print(f"  records parsed        : {rep.total}")
    print(f"  canonical             : {rep.canonical}")
    print(f"  legacy table rows     : {rep.legacy}  (migrate with `registry new`)")
    print(f"  schema violations     : {len(rep.invalid)}")
    if rep.duplicate_ids:
        print(f"  duplicate IDs         : {len(rep.duplicate_ids)}")
    print()
    for fid, probs in rep.invalid:
        print(f"  [INVALID] {fid}")
        for p in probs:
            print(f"      - {p}")
    for fid, lines in rep.duplicate_ids.items():
        print(f"  [DUPLICATE] {fid} appears at lines {lines}")
    print()
    print("PASS" if rep.ok else "FAIL")
    return 0 if rep.ok else 1


def cmd_list(args: list[str]) -> int:
    findings = parse_findings()
    want = None
    if "--status" in args:
        want = args[args.index("--status") + 1].upper()
    if "--severity" in args:
        sev = args[args.index("--severity") + 1].upper()
        findings = [f for f in findings if f.severity == sev]
    if want:
        findings = [f for f in findings if f.status.startswith(want)]

    rows = []
    for f in sorted(findings, key=lambda x: (SEVERITIES.index(x.severity)
                                             if x.severity in SEVERITIES else 99, x.fid)):
        flag = "!" if f.problems else " "
        rows.append((flag, f.fid, f.severity, f.status, f.title[:52], str(f.line_no)))

    print(f"{'':<1}{'ID':<10} {'SEVERITY':<9} {'STATUS':<26} {'TITLE':<52} LINE")
    print("-" * 106)
    for r in rows:
        print(" {}{:<10} {:<9} {:<26} {:<52} {}".format(*r))
    print(f"\n{len(rows)} records; '!' marks a schema violation "
          f"({sum(1 for f in findings if f.problems)} flagged).")
    return 0


def cmd_new(args: list[str]) -> int:
    findings = parse_findings()

    def opt(name: str, required: bool = True) -> Optional[str]:
        if name in args:
            i = args.index(name)
            if i + 1 < len(args):
                return args[i + 1]
        if required:
            raise SystemExit(f"error: {name} is required")
        return None

    fid = args[0] if args and not args[0].startswith("--") else None
    if not fid:
        pref = opt("--severity-prefix", required=False) or "MED"
        fid = next_id(findings, pref)
        print(f"auto-assigned ID: {fid}")

    title = opt("--title")
    sev = opt("--severity").upper()
    cat = opt("--category").upper()
    location = opt("--location")
    observed = opt("--observed")
    expected = opt("--expected")
    impact = opt("--impact")
    evidence = opt("--evidence")
    status = (opt("--status", required=False) or "NEW").upper()

    cand = Finding(fid=fid, title=title, severity=sev, category=cat,
                   location=location, observed=observed, expected=expected,
                   impact=impact, evidence=evidence, status=status)
    findings_c = parse_findings() + [cand]
    rep = validate([cand])
    if not rep.ok:
        print(f"error: new finding {fid} fails schema:")
        for fid_, probs in rep.invalid:
            for p in probs:
                print(f"  - {p}")
        return 1
    if any(f.fid == fid for f in findings):
        print(f"error: {fid} already exists")
        return 1

    if "--dry-run" in args:
        print(render_finding(cand))
        return 0

    block = render_finding(cand)
    text = REGISTER.read_text(encoding="utf-8")
    anchor = "\n---\n\n## 3. MEDIUM findings"
    if anchor in text:
        text = text.replace(anchor, f"\n---\n\n{block}\n{anchor}", 1)
    else:
        text = text.rstrip() + "\n\n---\n\n" + block + "\n"
    REGISTER.write_text(text, encoding="utf-8")

    mirror = REPO_ROOT / "agent_wiki" / "audit" / "scientific_flaws.md"
    if mirror.exists():
        mirror.write_text(text, encoding="utf-8")

    print(f"wrote {fid} to audit/scientific_flaws.md")
    print(f"next: python -m audit registry issue {fid}")
    return 0


def cmd_issue(args: list[str]) -> int:
    if not args:
        print("usage: registry issue <FINDING_ID> [--dry-run]")
        return 1
    rc, out = create_issue(args[0], dry_run="--dry-run" in args)
    print(out)
    return rc


def cmd_json(_args: list[str]) -> int:
    findings = parse_findings()
    payload = {
        "total": len(findings),
        "canonical": sum(1 for f in findings if not f.problems),
        "invalid": {f.fid: f.problems for f in findings if f.problems},
        "findings": [
            {"id": f.fid, "severity": f.severity, "category": f.category,
             "status": f.status, "location": f.location, "title": f.title,
             "line": f.line_no}
            for f in findings
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 1
    cmd, rest = argv[0], argv[1:]
    table = {
        "validate": cmd_validate,
        "list": cmd_list,
        "new": cmd_new,
        "issue": cmd_issue,
        "json": cmd_json,
    }
    if cmd not in table:
        print(f"unknown command: {cmd}")
        print(__doc__)
        return 1
    return table[cmd](rest)


if __name__ == "__main__":
    raise SystemExit(main())