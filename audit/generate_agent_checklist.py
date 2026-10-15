"""
CO2 EOR Optimizer - Automated Agent Showcase Checklist Generator
================================================================

This module parses:
1. audit/scientific_flaws.md (53 master scientific, physical, and software findings)
2. audit/parameter_provenance.csv (91 cataloged parameters and calibration statuses)
3. Active codebase AST (silent exception fallbacks, hardcoded numbers)
4. Static analysis reports (Ruff F821 undefined names, F811 redefinitions, F841 unused assignments)

Outputs:
- agent_wiki/audit/agent_checklist.md (Agent-facing interactive showcase checklist)
- audit/agent_checklist.md (Root mirror for audit artifacts)
- audit/agent_checklist.json (Machine-readable JSON schema for autonomous agents & CI)
- agent_wiki/audit/scientific_flaws.md (Synchronized 53-item register)
"""

import ast
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = REPO_ROOT / "audit"
WIKI_AUDIT_DIR = REPO_ROOT / "agent_wiki" / "audit"


def parse_scientific_flaws_md(flaws_file: Path) -> List[Dict[str, Any]]:
    """Parse audit/scientific_flaws.md and extract all structured flaw records."""
    if not flaws_file.exists():
        print(f"Warning: {flaws_file} not found.")
        return []

    content = flaws_file.read_text(encoding="utf-8")
    lines = content.splitlines()

    findings: Dict[str, Dict[str, Any]] = {}
    current_id: Optional[str] = None
    current_finding: Dict[str, Any] = {}

    def commit_current():
        nonlocal current_id, current_finding
        if current_id and current_finding:
            findings[current_id] = current_finding
        current_id = None
        current_finding = {}

    for line in lines:
        # Match markdown headers: ### CRIT-01 — Title
        m_head = re.match(r"^###\s+(CRIT|HIGH|MED|LOW)-(\d+)\s*[—-]\s*(.*)$", line)
        if m_head:
            commit_current()
            prefix, num_str, title = m_head.groups()
            current_id = f"{prefix}-{int(num_str):02d}"
            severity_map = {
                "CRIT": "CRITICAL",
                "HIGH": "HIGH",
                "MED": "MEDIUM",
                "LOW": "LOW",
            }
            current_finding = {
                "id": current_id,
                "severity": severity_map.get(prefix, prefix),
                "category": "UNKNOWN",
                "title": title.strip(),
                "location": "",
                "observed": "",
                "expected": "",
                "impact": "",
                "evidence": "",
                "status": "OPEN",
                "details": [],
            }
            continue

        # Match table rows in sections 3 and 4: | **MED-01** | ...
        m_row = re.match(r"^\|\s*\*\*(CRIT|HIGH|MED|LOW)-(\d+)\*\*\s*\|", line)
        if m_row:
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 4:
                prefix = m_row.group(1)
                num = int(m_row.group(2))
                fid = f"{prefix}-{num:02d}"
                cat = parts[1].strip()
                loc = parts[2].strip().replace("`", "")
                desc = parts[3].strip()
                if len(parts) >= 6:
                    evi = parts[4].strip()
                    st = parts[5].strip()
                elif len(parts) == 5:
                    evi = "Documented in flaw register"
                    st = parts[4].strip()
                else:
                    evi = "Documented in flaw register"
                    st = "OPEN"

                # Split Observed -> Expected -> Impact
                sub_parts = re.split(r"\s*(?:→|->)\s*", desc)
                obs = sub_parts[0].strip() if len(sub_parts) > 0 else desc
                exp = sub_parts[1].strip() if len(sub_parts) > 1 else ""
                imp = sub_parts[2].strip() if len(sub_parts) > 2 else ""

                # Title extraction
                title_candidate = obs.split(". ")[0].strip() if ". " in obs else obs.strip()
                if len(title_candidate) > 100:
                    title_candidate = title_candidate[:97] + "..."

                severity_map = {
                    "CRIT": "CRITICAL",
                    "HIGH": "HIGH",
                    "MED": "MEDIUM",
                    "LOW": "LOW",
                }
                findings[fid] = {
                    "id": fid,
                    "severity": severity_map.get(prefix, prefix),
                    "category": cat,
                    "title": title_candidate,
                    "location": loc,
                    "observed": obs,
                    "expected": exp,
                    "impact": imp,
                    "evidence": evi,
                    "status": st,
                    "details": [desc],
                }
            continue

        if current_id:
            # Parse field bullet points
            m_sev = re.match(r"^- \*\*Severity:\*\*\s*([A-Z]+)", line)
            if m_sev:
                current_finding["severity"] = m_sev.group(1).strip()
                m_cat = re.search(r"\*\*Category:\*\*\s*`?([A-Z]+)`?", line)
                if m_cat:
                    current_finding["category"] = m_cat.group(1).strip()
                continue

            m_cat2 = re.match(r"^- \*\*Category:\*\*\s*`?([A-Z]+)`?", line)
            if m_cat2:
                current_finding["category"] = m_cat2.group(1).strip()
                continue

            m_loc = re.match(r"^- \*\*Location:\*\*\s*(.*)$", line)
            if m_loc:
                current_finding["location"] = m_loc.group(1).strip()
                continue

            m_obs = re.match(r"^- \*\*Observed Behavior:\*\*\s*(.*)$", line)
            if m_obs:
                current_finding["observed"] = m_obs.group(1).strip()
                continue

            m_exp = re.match(r"^- \*\*Expected Behavior:\*\*\s*(.*)$", line)
            if m_exp:
                current_finding["expected"] = m_exp.group(1).strip()
                continue

            m_imp = re.match(r"^- \*\*Scientific Impact:\*\*\s*(.*)$", line)
            if m_imp:
                current_finding["impact"] = m_imp.group(1).strip()
                continue

            m_evi = re.match(r"^- \*\*Evidence & Citation:\*\*\s*(.*)$", line)
            if m_evi:
                current_finding["evidence"] = m_evi.group(1).strip()
                continue

            m_st = re.match(r"^- \*\*Status:\*\*\s*(.*)$", line)
            if m_st:
                current_finding["status"] = m_st.group(1).strip()
                continue

            # Capture additional context lines
            if line.strip() and not line.startswith("---") and not line.startswith("##"):
                current_finding["details"].append(line)

    commit_current()

    # Sort in canonical order
    sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    sorted_flaws = sorted(
        findings.values(),
        key=lambda x: (
            sev_order.get(x["severity"], 9),
            int(x["id"].split("-")[1]),
        ),
    )
    return sorted_flaws


def parse_parameter_provenance_csv(csv_file: Path) -> List[Dict[str, Any]]:
    """Parse audit/parameter_provenance.csv."""
    if not csv_file.exists():
        return []

    records = []
    with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "parameter": row.get("Parameter", "").strip(),
                "value": row.get("Value", "").strip(),
                "unit": row.get("Unit", "").strip(),
                "location": row.get("Source Code Location", "").strip(),
                "provenance": row.get("Provenance", "").strip(),
                "status": row.get("Status", "").strip(),
            })
    return records


def parse_safe_harbor_items(flaws_file: Path) -> List[str]:
    """Parse section 6: Verified correct — do not flag."""
    if not flaws_file.exists():
        return []

    content = flaws_file.read_text(encoding="utf-8")
    safe_harbor = []
    in_section = False
    for line in content.splitlines():
        if line.startswith("## 6. Findings explicitly **not** raised"):
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if in_section:
            m = re.match(r"^\d+\.\s+\*\*(.*?)\*\*\s*(.*)$", line)
            if m:
                safe_harbor.append(f"**{m.group(1)}**: {m.group(2)}")
    return safe_harbor


def scan_silent_fallbacks() -> List[Dict[str, Any]]:
    """Scan AST for silent fallbacks across core/ modules."""
    fallbacks = []
    core_dir = REPO_ROOT / "core"
    for py_file in core_dir.rglob("*.py"):
        rel = py_file.relative_to(REPO_ROOT).as_posix()
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="ignore"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    for handler in node.handlers:
                        is_bare = handler.type is None
                        exc_type = "bare except" if is_bare else ast.unparse(handler.type)
                        body_stmts = handler.body
                        is_pass = len(body_stmts) == 1 and isinstance(body_stmts[0], ast.Pass)
                        has_return_default = any(
                            isinstance(s, ast.Return) and not isinstance(s.value, ast.Name) for s in body_stmts
                        )
                        has_reraise = any(isinstance(s, ast.Raise) for s in body_stmts)
                        if not has_reraise and (is_pass or is_bare or has_return_default or len(body_stmts) <= 2):
                            fallbacks.append({
                                "module": rel,
                                "line": handler.lineno,
                                "type": exc_type,
                                "behavior": "SILENT_PASS" if is_pass else "FALLBACK_VALUE",
                            })
        except Exception:
            continue
    return fallbacks


def scan_ruff_diagnostics() -> Dict[str, List[Dict[str, Any]]]:
    """Extract dangerous static analysis rules (F821, F811, F841) from ruff output."""
    ruff_json = AUDIT_DIR / "ruff_output.json"
    diagnostics: Dict[str, List[Dict[str, Any]]] = {"F821": [], "F811": [], "F841": []}
    if not ruff_json.exists():
        return diagnostics

    try:
        with open(ruff_json, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        repo_str = str(REPO_ROOT).lower().replace("\\", "/").rstrip("/")
        for item in data:
            code = item.get("code")
            if code in diagnostics:
                raw_fn = item.get("filename", "")
                fn_norm = raw_fn.replace("\\", "/")
                if fn_norm.lower().startswith(repo_str):
                    rel = fn_norm[len(repo_str):].lstrip("/")
                else:
                    rel = raw_fn
                diagnostics[code].append({
                    "code": code,
                    "module": rel,
                    "line": item.get("location", {}).get("row", 0),
                    "message": item.get("message", ""),
                })
    except Exception as e:
        print(f"Warning: Failed to parse ruff diagnostics: {e}")
    return diagnostics


def generate_agent_checklist_markdown(
    flaws: List[Dict[str, Any]],
    parameters: List[Dict[str, Any]],
    safe_harbor: List[str],
    fallbacks: List[Dict[str, Any]],
    diagnostics: Dict[str, List[Dict[str, Any]]],
) -> str:
    """Generate the complete Agent Showcase Checklist Markdown."""
    sev_counts = Counter(f["severity"] for f in flaws)
    cat_counts = Counter(f["category"] for f in flaws)
    st_counts = Counter(f["status"].replace("*", "").split()[0] for f in flaws)

    lines: List[str] = []
    lines.append("# Master Agent Flaw & Scientific Showcase Checklist")
    lines.append("")
    lines.append("> [!IMPORTANT]")
    lines.append("> **MANDATORY AI AGENT PRE-FLIGHT DIRECTIVE**:")
    lines.append("> Before scanning source code, running repetitive grep queries, or attempting model refactors,")
    lines.append("> future AI agents MUST consult this showcase checklist. All 53 scientific defects, silent fallbacks,")
    lines.append("> dead data-paths, and hardcoded calibration factors are already forensically cataloged with file:line")
    lines.append("> anchors, measured evidence, and safe modification boundaries.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 📊 Repository Forensic Health Dashboard")
    lines.append("")
    lines.append("| Metric | Count | Assessment |")
    lines.append("|---|---|---|")
    lines.append(f"| **Total Confirmed Scientific Flaws** | **{len(flaws)}** | 13 CRITICAL · 19 HIGH · 16 MEDIUM · 5 LOW |")
    lines.append(f"| **Active Defects Status** | **{st_counts.get('NEW', 0) + st_counts.get('CONFIRMED', 0) + st_counts.get('OPEN', 0)} OPEN** | {st_counts.get('RESOLVED', 0)} RESOLVED |")
    lines.append(f"| **Cataloged Parameter Provenance** | **{len(parameters)} parameters** | Sources: Literature, Empirical, Calibrated, Unknown |")
    lines.append(f"| **Undocumented / Arbitrary Constants** | **{sum(1 for p in parameters if 'UNKNOWN' in p['provenance'] or 'CALIBRATED' in p['provenance'])}** | Require physical calibration or field bounds |")
    lines.append(f"| **Silent Fallback Exception Handlers** | **{len(fallbacks)} blocks** | Masking numerical divergences in `core/` |")
    lines.append(f"| **Undefined Names (F821)** | **{len(diagnostics['F821'])}** | Runtime `NameError` crash hazards |")
    lines.append(f"| **Shadowed Redefinitions (F811)** | **{len(diagnostics['F811'])}** | Overwritten functions/classes |")
    lines.append(f"| **Discarded Calculations (F841)** | **{len(diagnostics['F841'])}** | Computed variables never read |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 🛡️ Safe Harbor: Verified Correct (DO NOT FLAG)")
    lines.append("")
    lines.append("The following 13 mechanisms have been mathematically verified against analytical solutions. Agents **must NOT** waste tokens flagging them:")
    lines.append("")
    for item in safe_harbor:
        lines.append(f"- [x] {item}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 📋 Master Flaw Showcase Checklist")
    lines.append("")
    lines.append("| ID | Sev | Category | File:Line | Title / Headline Defect | Status |")
    lines.append("|---|---|---|---|---|---|")
    for f in flaws:
        st_badge = f"`{f['status']}`"
        loc_str = f"`{f['location']}`" if f["location"] else "—"
        lines.append(f"| **{f['id']}** | **{f['severity']}** | {f['category']} | {loc_str} | {f['title']} | {st_badge} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Detailed Flaw Records grouped by Severity
    for target_sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        group = [f for f in flaws if f["severity"] == target_sev]
        lines.append(f"## 🚨 {target_sev} Severity Flaw Register ({len(group)} items)")
        lines.append("")
        for f in group:
            st_box = "[x]" if "RESOLVED" in f["status"].upper() else "[ ]"
            lines.append(f"### {st_box} {f['id']} — {f['title']}")
            lines.append(f"- **Severity**: `{f['severity']}` | **Category**: `{f['category']}` | **Status**: `{f['status']}`")
            lines.append(f"- **Location**: `{f['location']}`")
            if f.get("observed"):
                lines.append(f"- **Observed Behavior**: {f['observed']}")
            if f.get("expected"):
                lines.append(f"- **Expected Behavior**: {f['expected']}")
            if f.get("impact"):
                lines.append(f"- **Scientific Impact**: {f['impact']}")
            if f.get("evidence"):
                lines.append(f"- **Evidence**: {f['evidence']}")
            lines.append("")
        lines.append("---")
        lines.append("")

    # Parameter Provenance Showcase
    lines.append("## 🔍 Parameter Provenance & Hidden Calibration Showcase")
    lines.append("")
    lines.append("| Parameter | Value | Unit | Source Code Location | Provenance | Assessment |")
    lines.append("|---|---|---|---|---|---|")
    for p in parameters[:35]:  # Display top critical parameters in table
        lines.append(f"| `{p['parameter']}` | {p['value']} | {p['unit']} | `{p['location']}` | **{p['provenance']}** | {p['status'][:80]} |")
    lines.append("")
    lines.append(f"*... {len(parameters) - 35} additional parameters cataloged in `audit/parameter_provenance.csv`.*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Static Analysis Disconnects Showcase
    lines.append("## ⚠️ Software Disconnects & Dangerous Anti-Patterns")
    lines.append("")
    lines.append("### 1. Undefined Names (Ruff F821) — Immediate Crash Hazards")
    if diagnostics["F821"]:
        lines.append("| Module | Line | Error Message | Risk |")
        lines.append("|---|---|---|---|")
        for d in diagnostics["F821"]:
            lines.append(f"| `{d['module']}` | {d['line']} | `{d['message']}` | Unhandled runtime NameError |")
    else:
        lines.append("None currently reported.")
    lines.append("")

    lines.append("### 2. Discarded Computations (Ruff F841) in Scientific Engines")
    sci_f841 = [d for d in diagnostics["F841"] if "core" in d["module"]]
    lines.append(f"Found **{len(sci_f841)}** computed variables assigned and never read in core simulation paths:")
    for d in sci_f841[:15]:
        lines.append(f"- [ ] `{d['module']}:{d['line']}`: {d['message']}")
    if len(sci_f841) > 15:
        lines.append(f"- *... and {len(sci_f841) - 15} more in `audit/ruff_output.json`*")
    lines.append("")

    lines.append("### 3. Silent Exception Fallbacks in Core Physics")
    lines.append(f"Found **{len(fallbacks)}** silent exception handlers. Top critical sites:")
    for fb in fallbacks[:15]:
        lines.append(f"- [ ] `{fb['module']}:{fb['line']}`: catches `{fb['type']}` ({fb['behavior']})")
    lines.append("")
    lines.append("---")
    lines.append("")

    # AI Agent Safe Modification Protocols
    lines.append("## 🛑 Agent Action Protocol & Change-Safety Invariants")
    lines.append("")
    lines.append("1. **NEVER modify equations without checking the flaw register**: All 53 items above are known. Modifying a clamp or formula without a corresponding verification test will cause regressions.")
    lines.append("2. **Active vs Legacy Rule**: Only `core/engine_surrogate/` executes. Do not attempt to fix dormant modules (`core/unified_engine/`, `core/simulation/recovery_models.py`) under the impression that it will fix optimization runs.")
    lines.append("3. **Two-Evaluations Trap (CRIT-02)**: Be aware that `recovery_factor` (RF₂) and `npv` (f(RF₁)) are derived from two distinct states. Any fix to the engine must reconcile these two into one single evaluation.")
    lines.append("4. **Dimensional Consistency**: HCPVI must be `RB / RB` (dimensionless), not `MSCF / STB` (CRIT-01). B_g must be `5.035 * Z * T / P` in RB/MSCF (CRIT-04).")
    lines.append("5. **Mandatory Save/Load Test**: Any edit to data models or UI widgets requires running:")
    lines.append("   ```bash")
    lines.append("   python -m pytest tests/test_project_save_load.py -v")
    lines.append("   ```")
    lines.append("")
    return "\n".join(lines)


def run_checklist_pipeline() -> Dict[str, Any]:
    """Execute complete checklist generation and sync pipeline."""
    print("=" * 80)
    print("EXECUTING AUTOMATED AGENT SHOWCASE CHECKLIST PIPELINE")
    print("=" * 80)

    flaws_md = AUDIT_DIR / "scientific_flaws.md"
    params_csv = AUDIT_DIR / "parameter_provenance.csv"

    print("[1/5] Parsing Master Flaw Register from audit/scientific_flaws.md...")
    flaws = parse_scientific_flaws_md(flaws_md)
    print(f"      Parsed {len(flaws)} scientific and software findings.")

    print("[2/5] Parsing Parameter Provenance Registry...")
    parameters = parse_parameter_provenance_csv(params_csv)
    print(f"      Parsed {len(parameters)} parameters.")

    print("[3/5] Parsing Safe Harbor items & scanning AST fallbacks...")
    safe_harbor = parse_safe_harbor_items(flaws_md)
    fallbacks = scan_silent_fallbacks()
    diagnostics = scan_ruff_diagnostics()
    print(f"      Extracted {len(safe_harbor)} safe harbor rules, {len(fallbacks)} silent fallbacks.")

    print("[4/5] Building Agent Showcase Checklist Markdown & JSON...")
    checklist_md = generate_agent_checklist_markdown(
        flaws, parameters, safe_harbor, fallbacks, diagnostics
    )

    # Write agent_wiki/audit/agent_checklist.md
    WIKI_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    wiki_checklist_path = WIKI_AUDIT_DIR / "agent_checklist.md"
    wiki_checklist_path.write_text(checklist_md, encoding="utf-8")
    print(f"      Wrote Wiki Showcase: {wiki_checklist_path}")

    # Write audit/agent_checklist.md
    audit_checklist_path = AUDIT_DIR / "agent_checklist.md"
    audit_checklist_path.write_text(checklist_md, encoding="utf-8")
    print(f"      Wrote Audit Mirror: {audit_checklist_path}")

    # Write audit/agent_checklist.json
    checklist_json_path = AUDIT_DIR / "agent_checklist.json"
    json_data = {
        "summary": {
            "total_flaws": len(flaws),
            "critical": sum(1 for f in flaws if f["severity"] == "CRITICAL"),
            "high": sum(1 for f in flaws if f["severity"] == "HIGH"),
            "medium": sum(1 for f in flaws if f["severity"] == "MEDIUM"),
            "low": sum(1 for f in flaws if f["severity"] == "LOW"),
            "total_parameters": len(parameters),
            "silent_fallbacks": len(fallbacks),
        },
        "flaws": flaws,
        "parameters": parameters,
        "safe_harbor": safe_harbor,
        "fallbacks": fallbacks,
        "diagnostics": diagnostics,
    }
    with open(checklist_json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"      Wrote Machine JSON: {checklist_json_path}")

    # Synchronize agent_wiki/audit/scientific_flaws.md with full register
    wiki_flaws_path = WIKI_AUDIT_DIR / "scientific_flaws.md"
    if flaws_md.exists():
        wiki_flaws_path.write_text(flaws_md.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"      Synchronized Wiki Flaw Register: {wiki_flaws_path}")

    print("[5/5] Checklist Generation Complete!")
    print("=" * 80)
    return json_data


if __name__ == "__main__":
    run_checklist_pipeline()
