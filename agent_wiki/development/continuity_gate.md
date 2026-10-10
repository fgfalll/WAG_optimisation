# Continuity Gate — Re-verifying "RESOLVED" Claims

**Module:** [`audit/continuity.py`](../../audit/continuity.py)
**Invoked:** `python -m audit.continuity check` (also `python -m audit --continuity`)
**Verdict of last run:** see [`audit/continuity_report.json`](../../audit/continuity_report.json)

---

## 1. Why this gate exists

On 05-10-2026, 17 findings were marked **RESOLVED** in the same uncommitted working tree in which
they were edited. No measurement backed any of those marks. An adversarial re-audit found:

| Verdict | Count |
|---|---:|
| CONFIRMED | 11 |
| PARTIALLY_RESOLVED | 4 |
| REGRESSED | 1 |
| CONFIRMED_BUT_INERT | 1 |

Throughout that audit the suite reported **`335 passed / 0 failed`**. Nothing in the repository
could have detected the difference between "fixed" and "marked fixed".

**This gate closes that gap.** It re-measures every `RESOLVED` claim on demand and reports drift
between the documentation and the code.

---

## 2. Hard rules

1. **Verify, don't trust.** A `**Status:** RESOLVED` line is a *claim*, not a fact. Every verdict
   comes from a measurement or a source citation.
2. **No composite score.** Results are bucketed by evidence tier and reported separately. This
   module never produces a "model accuracy" number.
3. **Evidence or it did not happen.** A probe that cannot execute reports
   `UNKNOWN — EVIDENCE REQUIRED`. It never silently passes.
4. **Audit-only.** The gate writes reports and issues. It does not edit scientific logic.
5. **Categorical separation is preserved.** A `PASS` at `BENCHMARK_AGREEMENT` never compensates for
   a `FAIL` at `MATHEMATICAL_CORRECTNESS`.

---

## 3. Verdict vocabulary

| Verdict | Meaning |
|---|---|
| `CONFIRMED` | defect measured **absent** |
| `PARTIALLY_RESOLVED` | reduced, residual remains |
| `REGRESSED` | was fixed, is broken again |
| `STILL_OPEN` | never fixed; the status label is wrong |
| `CONFIRMED_BUT_INERT` | correct in isolation, has no effect in the shipped configuration |
| `UNKNOWN — EVIDENCE REQUIRED` | cannot be verified automatically |

Any of `PARTIALLY_RESOLVED`, `REGRESSED`, `STILL_OPEN`, `CONFIRMED_BUT_INERT` on a finding labelled
`RESOLVED` sets `should_reopen`, which makes the CLI exit **1**.

---

## 4. Commands

```bash
# verify every RESOLVED claim + check wiki/code drift
python -m audit.continuity check

# verify a single finding
python -m audit.continuity check CRIT-14

# machine-readable summary
python -m audit.continuity status

# propose wiki corrections (dry run, writes nothing)
python -m audit.continuity sync

# validate a candidate commit message against the 1-commit-1-issue policy
python -m audit.continuity check-commit "fix(pvt): derive c_g from PR-EOS" "Closes #10"

# prove the issue gate works (7 cases, 3 of which must FAIL)
python -m audit.continuity selftest

# pre-commit hook over the staged message
python -m audit --issue-gate
```

Exit codes: `0` clean · `1` reopens required · `2` issue-discipline violation.

> **Companion gate.** The gate above re-verifies claims; the
> [Finding Registry](finding_registry.md) (`python -m audit --register validate`) guarantees the
> register those claims live in is schema-valid, and that GitHub issues are *generated* from it
> rather than hand-written. Run both before committing.

---

## 5. Evidence tiers

Probes are grouped by the tier of the evidence hierarchy they exercise, highest priority first:

```
MATHEMATICAL_CORRECTNESS   units, algebra, closure identities
PHYSICAL_CONSISTENCY      PVT thermodynamics, mass balance, leakage
NUMERICAL_DISCRETIZATION  continuity, monotonicity, clip saturation
IMPLEMENTATION            data flow, key wiring, provenance guards
PARAMETER_PROVENANCE      hard-coded constants and fudge multipliers
ANALYTICAL_VERIFICATION   limiting cases, unit analysis
EXPERIMENTAL_VALIDATION   (no probe; not established in-repo)
BENCHMARK_AGREEMENT       (no probe; explicitly never credited as proof)
EXECUTION_SPEED           (no probe; lowest tier)
```

A `PASS` in the top tier does **not** imply anything about the tiers below. The report says so
explicitly.

---

## 6. Issue discipline — 1 commit = 1 issue

### The rule

An issue is closed **only** by a commit, because the commit is the auditable record that the fix
landed. The gate never closes an issue programmatically.

### What is rejected

| Commit message | Verdict | Why |
|---|---|---|
| `Closes #10` | PASS | single close |
| `Closes #10, #12` | **FAIL** | stacked close |
| `Closes #7 and #8` | **FAIL** | stacked close |
| `Fixes #3. Also Closes #4` | **FAIL** | two closing clauses |
| `see #9 for the closure` | **FAIL** | a reference is not a closure |
| `Refactors #14, no behaviour change` | **FAIL** | bare reference, disposition unstated |
| `chore: bump pin` | PASS | no issue involved |

The parser stops at the first token that is not a separator or a reference, so trailing prose is
never misread as a second issue:

```
"Closes #10, #12"        -> 2 refs  (stacking)
"Closes #9 for the closure" -> 1 ref (prose ignored)
```

### Why this matters scientifically

Without the rule, a single commit that touches four PVT functions closes four issues at once and the
per-finding evidence becomes unrecoverable. That is exactly how the 05-10-2026 batch remediation
happened: one edit, seventeen status marks, zero measurements. **One commit per issue keeps the
audit trail per finding.**

---

## 7. Current gate status (05-10-2026)

```
PASS ( 8)          CRIT-02, CRIT-04, CRIT-05, CRIT-07, CRIT-08, CRIT-11,
                   INV-LEDGER, INV-RECYCLE
PASS-BUT-INERT ( 1) CRIT-06
PARTIAL ( 2)       CRIT-01, CRIT-18
REGRESSED ( 2)     CRIT-12, CRIT-14
FAIL ( 5)          CRIT-03, CRIT-13, CRIT-16, CRIT-21, HIGH-23

REQUIRES REOPENING: 8
```

### Findings genuinely fixed — keep them

- **CRIT-04** `B_g = 5.0351` derived vs `5.035` used.
- **CRIT-05** Papay (1968); `Z = 0.849 / 0.827 / 0.868 / 0.972` at Ppr 2.24→6.73, a proper dense-gas
  dip replacing the old monotone `Z > 1`.
- **CRIT-07** immiscible limb has a real gradient (24 distinct values over 45 evaluations, was 1).
- **CRIT-11** WAG water now 25 000 bpd at 5 000 MSCFD (was 25 bpd) — exactly 1000×.
- **CRIT-08** containment score decays to 0 and **can** prune (was structurally unable to).
- **CRIT-02** `cumulative_oil_stb == OOIP × RF` to 5.8e-10 STB.
- **INV-LEDGER / INV-RECYCLE** closed-loop CO₂ accounting closes to machine precision.

### Findings to reopen

| ID | Verdict | Where it lives now |
|---|---|---|
| CRIT-01 | `PARTIAL` | register — denominator is total PV; the stated formula carries a spurious `B_o` |
| CRIT-03 | `FAIL` | register — `P_b` is a literal constant, no Standing correlation exists |
| CRIT-06 | `PASS-BUT-INERT` | register — correct, but saturated at its clip at the shipped HCPVI |
| CRIT-12 | `REGRESSED` | register + [#9](https://github.com/fgfalll/WAG_optimisation/issues/9) — saturation closure now broken |
| CRIT-13 | `FAIL` | register — `gravity_factor` is now an active fudge; also CRIT-19, CRIT-20 |
| CRIT-14 | `REGRESSED` | register + [#7](https://github.com/fgfalll/WAG_optimisation/issues/7) — viscosity severed from recovery |
| CRIT-16 | `FAIL` | register — `getattr` guards name non-existent fields |
| CRIT-18 | `PARTIAL` | register — gas revenue computed but not in the cash flow |
| CRIT-21 | `FAIL` | register + [#10](https://github.com/fgfalll/WAG_optimisation/issues/10) — `c_g` power law vs in-class PR EOS |
| HIGH-23 | `FAIL` | register + [#13](https://github.com/fgfalll/WAG_optimisation/issues/13) — leakage zero, storage credit leakage-blind |
| HIGH-11 | `RECURRED` | register — three `F821` undefined names (gate-detected) |

> **Tracker consolidation (05-10-2026).** Issues #16–#20 duplicated register records and were closed
> with a comment naming the register entry, stating that the defect is **not** fixed, and listing the
> remaining work. The register is the single source of truth; issues are generated from it with
> `python -m audit --register issue <ID>`. See [`finding_registry.md`](finding_registry.md).

---

## 8. A documented false positive

The gate keeps a reviewed allowlist so a blanket "fix every `F821`" cannot corrupt working control
flow:

```python
F821_FALSE_POSITIVE = {
    ("core/geology/petrophysical_distribution.py", "prev_field"),
}
```

`prev_field` **is** bound at `:405` on the previous loop iteration and read only under `if k > 0`
(`:402`). Ruff's flow-insensitive analysis cannot see that. Documented rather than deleted, so the
gate stays strict everywhere else.

---

## 9. Trap: a probe can be wrong even when the code is right

The first revision of the CRIT-11 probe searched for `* 1000` on the *water-rate* line and reported
`FAIL` — against code that was in fact correct, because the factor had been applied at the `B_g`
definition instead. Both placements are dimensionally valid.

The probe was fixed to accept either, and the trap is recorded in the function docstring:

```python
# The x1000 scf->MSCF factor may be applied either at the B_g *definition*
# (`default_gas_fvf * 1000`) or at the *use site* (`* 1000`). Both are
# dimensionally valid, so the probe must accept either. An earlier revision of
# this probe only checked the use site and produced a false FAIL against code
# that was in fact correct - which is exactly the error class this module
# exists to prevent, so it is recorded here deliberately.
```

A gate that cannot be wrong is not a gate. Every probe that produces a surprising verdict must be
confirmed by reading the source before it is allowed to reopen an issue.

---

## 10. Adding a probe

```python
def probe_my_defect() -> tuple[str, str, Optional[str]]:
    """CRIT-nn: one sentence on what 'fixed' means."""
    src = (REPO_ROOT / "core/.../module.py").read_text(encoding="utf-8")
    if "good pattern" in src:
        return CONFIRMED, "why this proves the defect is absent", "measured value"
    return STILL_OPEN, "what the code actually does", "measured value"
```

Register it in `CHECKS` and, if it re-verifies a `RESOLVED` claim, in `RESOLVED_CLAIMS`. Then add
the finding to the selftest expectations.

Requirements:

- **Read-only.** No probe may mutate repository state.
- **Return a measurement.** `"evidence"` alone is not evidence; include the number.
- **Never raise.** A raising probe reports `UNKNOWN`, not a pass.
- **Accept both valid implementations** when a fix can reasonably land in more than one place.