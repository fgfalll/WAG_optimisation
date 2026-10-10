# Finding Registry — Canonical Schema, CLI, and Issue Discipline

**Module:** [`audit/registry.py`](../../audit/registry.py)
**Register:** [`audit/scientific_flaws.md`](../../audit/scientific_flaws.md)
**Gate:** `python -m audit --register validate`

---

## 1. Why this exists

Findings used to be hand-written straight into `audit/scientific_flaws.md`. The hand-written records
diverged from each other in ways no tool could see:

| Divergence | Consequence |
|---|---|
| Two layouts: `### CRIT-14 — …` bullet blocks vs `\| **MED-13** \| … \|` table rows | the parser read 19 of 73 records; **54 were invisible** to the continuity gate |
| `**Status:**` on its own line, or appended to the previous field | the same finding read as resolved or open depending on layout |
| `Observed Behavior` / `Expected Behavior` / `Impact` / `Evidence` vs canonical names | field extraction returned nothing |
| Em dash and hyphen both used as the title separator | heading matching failed |
| Free-text severity (`HIGH \| **Category:** \`SOFTWARE\``) | severity parsing returned junk |

The consequence was concrete: the register could say `RESOLVED` while every consumer — the
continuity gate, the checklist generator, a human reader — saw something else.

**The register is now generated and machine-validated.** Hand-edits that break the schema fail with
a non-zero exit code.

---

## 2. Commands

```bash
python -m audit --register validate                    # schema-lint every record
python -m audit --register list                        # inventory, severity-ordered
python -m audit --register list --severity CRITICAL    # filter by severity
python -m audit --register list --status OPEN          # filter by status
python -m audit --register json                        # machine-readable inventory

# add a finding (the ONLY sanctioned way)
python -m audit --register new CRIT-22 \
    --severity CRITICAL --category MATHEMATICAL \
    --location core/engine_surrogate/pvt_state.py:415 \
    --observed "what the code actually does" \
    --expected "what the physics requires" \
    --impact "consequence for RF / pressure / economics / containment" \
    --evidence "reproduction command + measured value + literature" \
    --dry-run                                          # preview without writing

python -m audit --register issue CRIT-22               # generate the GitHub twin
python -m audit --register issue CRIT-22 --dry-run     # preview the issue body
```

`new` auto-assigns the next free ID when the first argument is omitted
(`python -m audit --register new --severity-prefix HIGH --title "…"`). It writes both
`audit/scientific_flaws.md` and the wiki mirror `agent_wiki/audit/scientific_flaws.md`.

---

## 3. Schema

Every record is exactly these fields, in this order:

```markdown
### CRIT-22 — <title>

- **Severity:** CRITICAL
- **Category:** `MATHEMATICAL`
- **Location:** `core/engine_surrogate/pvt_state.py:415`
- **Observed:** <what the code actually calculates>
- **Expected:** <what the governing physics/mathematics requires>
- **Scientific Impact:** <effect on pressure, saturation, RF, economics, containment>
- **Evidence & Citation:** <reproduction + measured value + literature>
- **Status:** NEW
```

An optional ninth field, **`Note`**, carries relocated prose — for example the original wording of a
Status line that now has a canonical token. It must come last.

### Closed enumerations

| Field | Allowed values |
|---|---|
| **Severity** | `CRITICAL` `HIGH` `MEDIUM` `LOW` `INFORMATIONAL` |
| **Category** | `MATHEMATICAL` `PHYSICAL` `NUMERICAL` `SOFTWARE` `PROVENANCE` |
| **Status** | `NEW` `OPEN` `CONFIRMED` `PARTIALLY_RESOLVED` `CONFIRMED_BUT_INERT` `REGRESSED` `RECURRED` `STILL_OPEN` `RESOLVED` `STALE` `SUPERSEDED` `UNKNOWN_EVIDENCE_REQUIRED` |

`Status` is the same vocabulary as [`audit/continuity.py`](continuity_gate.md), deliberately: the two
modules cannot disagree about what a status means.

### Validation rules

`validate` fails (exit 1) on any of:

1. missing required field
2. unrecognised field
3. fields out of canonical order
4. malformed ID (`CRIT-nn` / `HIGH-nn` / `MED-nn` / `LOW-nn` / `INFO-nn`)
5. hyphen instead of em dash as the title separator
6. severity or category outside its enumeration
7. status outside its enumeration
8. `Location` citing neither a `file.py:line` nor the literal `` `repo-wide` ``
9. `Location` citing a repository path that **does not exist on disk**
10. duplicate finding IDs

Rule 9 is the one that catches silent drift: renaming or deleting a module without updating the
register breaks the gate.

---

## 4. `repo-wide` is a valid scope

Cross-cutting findings (toolchain metrics, documentation drift, repository-wide lint counts) may cite
`` `repo-wide` `` instead of a file. Do **not** force a file path onto a finding that genuinely has
none.

---

## 5. Issues are generated, never hand-written

```bash
python -m audit --register issue CRIT-22
```

The body is rendered **from the register record**: severity, category, location, observed, expected,
impact, evidence — plus a footer stating that the issue was generated and must not be edited by hand,
and restating the closure policy.

Why this matters: a hand-written issue is a second source of truth. On 05-10-2026 the register and
the tracker could drift silently, and did — issues were opened for findings the register already
described, then closed by hand while the register still said the defect was live.

### Registry ↔ tracker

| | Source of truth |
|---|---|
| Finding content | `audit/scientific_flaws.md` (validated) |
| GitHub issue | generated from the record; the footer says so |
| Status change | edit the register, then `--register issue <ID>` to refresh |

Issues **#16–#20** were closed on this basis in the 05-10-2026 round. Each closure comment names the
register entry, states plainly that the defect is **not** fixed, and lists the remaining work. The
findings remain open in the register with their measured status.

---

## 6. Closure policy — one commit, one issue

An issue closes **only** via a commit containing a closing keyword. The commit is the auditable
record that the fix landed; a tracker click is not.

```bash
python -m audit.continuity check-commit "fix(pvt): derive c_g from PR-EOS" "Closes #21"
python -m audit --issue-gate            # gate the staged message
python -m audit.continuity selftest     # 8 cases, 3 must FAIL
```

| Message | Verdict | Why |
|---|---|---|
| `Closes #21` | PASS | single close |
| `Closes #21, #22` | **FAIL** | stacked close |
| `Closes #21 and #22` | **FAIL** | stacked close |
| `Fixes #21. Also Closes #22` | **FAIL** | two closing clauses |
| `see #21 for the closure` | **FAIL** | a reference is not a closure |
| `Refactors #21, no behaviour change` | **FAIL** | bare reference, disposition unstated |
| `chore: bump pin` | PASS | no issue involved |

The parser stops at the first token that is neither a separator nor a reference, so trailing prose is
never misread as a second issue:

```
"Closes #21 for the saturation closure"  -> 1 reference (prose ignored)
```

---

## 7. Workflow for an agent

```
1. find a defect, reproduce it, get a NUMBER
2. python -m audit --register new <ID> --severity … --category … \
       --location file.py:line --observed … --expected … --impact … --evidence …
3. python -m audit --register validate          # must PASS
4. python -m audit --register issue <ID>        # generate the tracker twin
5. fix the code (one issue per commit)
6. python -m audit.continuity check <ID>        # must show CONFIRMED
7. edit the Status line to CONFIRMED / RESOLVED with the measured value
8. python -m audit --register validate
9. git commit -m "fix(...): …" -m "Closes #<N>"
```

Step 6 before step 7 is not optional. Writing `RESOLVED` from intent is what produced 17 unchecked
marks on 05-10-2026.

---

## 8. Adding a probe

```python
def probe_my_defect() -> tuple[str, str, Optional[str]]:
    """CRIT-nn: one sentence on what 'fixed' means."""
    src = (REPO_ROOT / "core/.../module.py").read_text(encoding="utf-8")
    if "good pattern" in src:
        return CONFIRMED, "why this proves absence", "measured value"
    return STILL_OPEN, "what the code does", "measured value"
```

Register it in `CHECKS` in `audit/continuity.py`, and add the finding to `RESOLVED_CLAIMS` if it
re-verifies a resolution claim.

Requirements:

- **Read-only.** Never mutate repository state.
- **Return a measurement.** `"evidence"` alone is not evidence.
- **Never raise.** A raising probe reports `UNKNOWN`, not a pass.
- **Accept every valid placement** when a fix could reasonably land in more than one place. The
  CRIT-11 probe initially failed correct code because it checked only one of two equivalent places
  the `×1000` conversion could appear — a false positive that had to be fixed.

---

## 9. Trap: a gate can be wrong

The first CRIT-11 probe searched for `* 1000` on the water-rate line and reported `FAIL` against
correct code, because the factor had been applied at the `B_g` definition instead:

```python
# analytical_models.py / profile_generator_fast.py:1109
default_b_gas = params.get("bg_rb_per_mscf", params.get("default_gas_fvf", 0.005) * 1000.0)
```

Both placements are dimensionally valid. Measured after the probe was corrected: 5 000 MSCFD →
**25 000 bpd** of WAG water, exactly 1000× the pre-remediation 25 bpd.

Every surprising verdict must be confirmed by reading the source before it is allowed to reopen an
issue. A gate that cannot be wrong is not a gate.