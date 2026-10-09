# Separate Flaw Register — Compositional Engine

**Why this document exists.** The existing register cannot record a Rust finding. Verified 07-10-2026:

```
audit/registry.py:114   LOCATION_RE = re.compile(r"`([\w/\\.\-]+\.py)(?::(\d+)(?::\d+)?)?`")
audit/registry.py:300   if not (REPO_ROOT / target).exists():
audit/registry.py:78    SEVERITIES = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFORMATIONAL")
audit/registry.py:82    STATUSES   = ( ... )   # "shared with audit.continuity"
```

Three consequences:

1. **`.rs` locations are schema-invalid.** `crates/compositional/src/flash.rs:120` is rejected.
2. **Every location must exist under `REPO_ROOT`.** A separate tree location would need the new module
   to own its own root resolution.
3. **Severity and status vocabularies are the Python register's.** `CRIT-/HIGH-/MED-/LOW-` ID prefixes are
   already taken.

A **new module** is therefore required. It must be an independent package, not a branch of
`audit/registry.py`.

### Why this document exists — **INV-5**

🔴 This module **is** the implementation of **INV-5** — *"Own flaw register, `COMP-nn` namespace"*,
decided 07-10-2026 ([`engine_invariants.md`](engine_invariants.md) INV-5).

**INV-5 is not about tidiness, and the reason is measured.** ⚠️ The Python register holds **73 findings**
([`../audit/scientific_flaws.md`](../audit/scientific_flaws.md)) spanning every subsystem. If new-engine
defects were adjudicated in the same file:

| Consequence | Why it matters |
|---|---|
| A Rust finding competes with 73 legacy ones for adjudication attention | the new engine's defects get the *residual* attention |
| Its vocabulary is Python's — `CRIT-/HIGH-/MED-/LOW-` prefixes | already taken; a Rust finding cannot be named |
| Its schema rejects `.rs` locations | a finding **cannot be recorded at all** |
| Its continuity gate hard-imports the surrogate | the Rust engine cannot be re-measured without coupling to code it is supposed to be independent of |

⚠️ **The third row is the hard one.** This is not "the register is Python-flavoured" — it is **the register
cannot represent a Rust defect**. ✅ **Consequence for the engine:** a defect found in `crates/compositional/`
must be registered as `COMP-nn` via `audit_comp`, and **may not** be written into
`audit/scientific_flaws.md`.

🔴 **Standing rule — the register is machine-generated.** Never hand-edit `audit_comp/register.md`; add via
the CLI and validate. A hand-written row is a defect, exactly as in the Python register.

---

## 1. Module layout

```
audit_comp/
  __init__.py
  __main__.py          CLI:  python -m audit_comp --help
  registry.py          canonical writer + validator
  continuity.py        RESOLVED-claim re-measurement (shells out to cargo, never imports Python)
  register.md          THE REGISTER  (audit_comp/register.md)
  continuity_report.json
```

**Independence rules**

| Rule | Enforcement |
|---|---|
| `audit_comp/` must not import `audit.` | grep gate; `audit_comp/__init__.py` may not reference it |
| `audit/pipeline.py` and `audit/__main__.py` must not import `audit_comp` | grep gate — the Python gate must not silently start validating Rust findings |
| `audit_comp` has its own root resolution | `REPO_ROOT = Path(__file__).resolve().parent.parent` (same idiom, own module) |
| Its own CI job | separate workflow, separate required checks |

---

## 2. Record schema

Identical **shape** to the Python register — deliberately, so the discipline transfers — with a
Rust-appropriate location pattern and its own vocabularies.

```markdown
### COMP-01 — <title>

- **Severity:** <enum>
- **Category:** <enum>
- **Location:** `crates/compositional/src/<path>.rs:<line>`
- **Observed:** <what the code does>
- **Expected:** <what the physics/numerics require>
- **Impact:** <consequence>
- **Evidence:** <reproduction command + measured value + literature>
- **Status:** <enum>
- **Verification:** <the command that re-measures this claim>
```

### 2.1 Field semantics are stricter, not looser

The Python register's history (per `audit/registry.py`'s own docstring) includes records where a finding
was "resolved" in the register but not in code, and free-text severities. The compositional register
must not inherit that failure mode:

| Field | Rule |
|---|---|
| **Evidence** | Must contain a **command**, a **measured value**, and a **citation**. Prose alone is a validation error |
| **Verification** | **Mandatory.** The exact command that re-measures the claim. A finding without one cannot be marked `RESOLVED` |
| **Status** | `RESOLVED` is **rejected** unless `Verification` is present and the command reports `CONFIRMED` |
| **Location** | Must resolve on disk. Multi-line ranges permitted, e.g. `:120-135` |

### 2.2 Vocabularies

**Severity** — Rust-specific, replaces the Python set:

| Value | Meaning |
|---|---|
| `CRITICAL` | Physics is wrong, or the engine can produce a plausible-looking wrong answer |
| `HIGH` | A documented invariant is violated but results stay bounded |
| `MEDIUM` | Numerical degradation without wrong physics |
| `LOW` | Documentation, naming, style |
| `INFORMATIONAL` | Recorded design decision or accepted trade-off |

> Deliberately **not** reusing `CRITICAL/HIGH/MEDIUM/LOW/INFORMATIONAL` verbatim — the Python set is
> shared with a machine that owns those labels. Reuse risks a finding being filed against the wrong
> register.

**Category** — replaces the Python `MATHEMATICAL|PHYSICAL|NUMERICAL|SOFTWARE|PROVENANCE`:

| Value | Meaning |
|---|---|
| `THERMO` | EOS, PVT, phase equilibrium, flash |
| `PHYSICS` | Displacement, relative permeability, trapping, geomechanics |
| `NUMERICAL` | Discretisation, convergence order, linear algebra, timestep control |
| `CONSERVATION` | Mass, component closure, energy balance |
| `ROBUSTNESS` | Non-finite values, panic paths, error taxonomy coverage |
| `IO` | Format parsing, versioning, round-trip |
| `PERFORMANCE` | Allocation in hot loops, parallel scaling |
| `SOFTWARE` | API, trait design, ownership |
| `PROVENANCE` | Missing citation, unstated constant |

**Status** — **intentionally identical** to `audit/registry.py`'s `STATUSES`, so the two registers cannot
diverge semantically. Shared vocabulary, separate module. Any change to one must change the other in the
same commit.

---

## 3. ID namespace

| Engine | Prefixes |
|---|---|
| Python surrogate | `CRIT-nn`, `HIGH-nn`, `MED-nn`, `LOW-nn`, `INFO-nn` |
| **Compositional** | **`COMP-nn`** |

A single flat namespace per register. `COMP` is chosen because `COMP` is unused by the Python register
(verified: `CRIT`, `HIGH`, `MED`, `LOW`, `INFO`, `SCI`, `DRIFT`, `MISMATCH` are in use) and because it
cannot be confused with a severity tier.

**GitHub labels:** `comp-severity-<tier>`, `comp-category-<name>`, `comp-status-<name>` — namespaced so
the two registers never collide on a board.

---

## 4. CLI

Mirrors the Python CLI so the muscle memory transfers:

```bash
python -m audit_comp --register validate
python -m audit_comp --register new COMP-07 \
    --severity CRITICAL --category CONSERVATION \
    --location crates/compositional/src/flow/flux.rs:212 \
    --observed "residual returns to caller unaccumulated after Newton convergence" \
    --expected "per-component closure to 1e-12" \
    --impact "component mass balance silently drifts; recovery over-stated" \
    --evidence "cargo test comp::conservation::component_closure -- --nocapture -> 3.4e-10 (tol 1e-12)" \
    --verification "cargo test comp::conservation::component_closure"

python -m audit_comp --register issue COMP-07
python -m audit_comp.continuity check
python -m audit_comp.continuity check COMP-07
python -m audit_comp.continuity check-commit "<subject>" "<body>"
```

### 4.1 Continuity gate design

| Property | Requirement |
|---|---|
| **No Python-engine import** | Never imports `core.*`. Comparing against the surrogate would be coupling |
| **Re-measurement** | Runs the `Verification` command and parses its output. A `RESOLVED` claim is `CONFIRMED` only when the command passes |
| **Idempotent** | No side effects beyond writing the report |
| **1-commit-1-issue** | Rejects `Closes COMP-1, COMP-2` |
| **Spawn, don't import** | Uses `subprocess` on `cargo`, mirroring `audit/continuity.py`'s existing `subprocess` use |

### 4.2 The `Verification` field is the whole point

The Python register's documented history: seventeen findings were marked `RESOLVED` while the suite
reported `335 passed / 0 failed`; adversarial re-measurement found **8 requiring reopening, including 2
regressions**. That is the failure mode this design forecloses:

- `RESOLVED` **requires** a populated `Verification` command.
- The validator **rejects** `RESOLVED` records with an empty `Verification`, at write time — not at
  audit time.
- `audit_comp.continuity check` is a **release gate**, not an advisory report.

---

## 5. Migration and coexistence

| Concern | Resolution |
|---|---|
| Existing Python findings | **Stay where they are.** No migration, no rewriting of `audit/scientific_flaws.md` |
| Cross-references | Allowed **one way only** — a compositional finding may say "the Python engine has the same class of issue, see `audit/scientific_flaws.md`". The Python register must **not** reference compositional findings |
| Shared history | Both registers carry the same `RESOLVED`-requires-measurement discipline. The 05-10-2026 reversal is the cautionary precedent |
| Where the engines agree or disagree | Not a finding in either register. Input to the future coupling plan |

---

## 6. Implementation order

| # | Task | Gate |
|---|---|---|
| 1 | Create `audit_comp/` skeleton with the schema, vocabularies and ID namespace | `python -m audit_comp --register validate` runs on an empty register |
| 2 | Port the **writer** semantics from `audit/registry.py` — but as a **separate implementation** with a `.rs`-aware `LOCATION_RE` | A `COMP-01` with an `.rs` location validates |
| 3 | Reject `RESOLVED` without `Verification` | Validator test proves rejection |
| 4 | Write `audit_comp/continuity.py` — spawns `cargo`, never imports `core.*` | `check` on a seeded `RESOLVED` record passes, then fails when the command is broken |
| 5 | Add the independence grep gates | Gate fails when a forbidden import is introduced |
| 6 | Separate CI job, separate required checks | PR gate visible and independent |
| 7 | Backfill: register any flaw discovered while building M0–M1, with a measurement | `--register validate` clean |

> **Do not** extract a shared base class from `audit/registry.py` to avoid duplication. That would be a
> dependency between the two engines' tooling — the exact thing the separation decision forbids, one
> layer down.