# 🔄 Agent Handover — Rust 3D THMC Engine

> [!IMPORTANT]
> **This file exists for one reason: context rollover.** When a session is compacted, or a
> fresh agent takes over, this is what must survive. Everything on it is a **measured fact with
> a pointer to its authority** — never prose, never a summary of a summary.
>
> **Read it in full. It is ~9 KB / 160 lines on purpose** — the documents it points at total **1,087 KB**
> and cannot be re-read end-to-end after a rollover.

**As of 10-10-2026 · commit `37a0ffb` (engine work uncommitted at time of writing)**

---

## 0. Read in this order

| # | File | KB | Why |
|---|---|---|---|
| 1 | **this file** | 6 | survives rollover |
| 2 | [`compositional/README.md`](README.md) | 22 | scope + status |
| 3 | [`compositional/engine_invariants.md`](engine_invariants.md) **§0 only** | — | ⚠️ the file is split into 4; use its §0 index, never a linear read |
| 4 | [`compositional/build_plan.md`](build_plan.md) | 42 | current milestone + gates |
| 5 | [`compositional/spec_corrections_log.md`](spec_corrections_log.md) | **269** | ⚠️ **largest file in the repo.** Query it by `C-nnn`, never read it linearly |

🔴 **Do not read #3 or #5 end to end.** `engine_invariants.md` carries a §0 index precisely
because a linear read does not fit; `spec_corrections_log.md` at 269 KB will exhaust a context
window on its own.

---

## 1. 🔴 Current state — verify before trusting

| Fact | Value | Verified by |
|---|---|---|
| Rust engine code | 🔴 **NONE EXISTS** | `Get-ChildItem -Recurse -Include *.rs, Cargo.toml` |
| Rust toolchain | 🔴 **ABSENT** — `cargo`, `rustc`, `rustup` all `NOT FOUND` | re-checked 10-10-2026 |
| Current milestone | **M0 not started** — nothing implemented | [`build_plan.md`](build_plan.md) |
| Blocker B-1 | no toolchain, **deferred by owner decision** | [`compositional/README.md`](README.md) §B |
| Python engine | ✅ **shipped, untouched** | do not modify |
| Rust toolchain install | **deferred by owner decision** — do not install unprompted | — |

⚠️ **Nothing in `compositional/` or `thmc/` is a report of running software.** Every tolerance,
milestone and "verified" claim there describes **what the engine must do**.

---

## 2. 🔴 The seven invariants — never trade these away

| | |
|---|---|
| **INV-1** | **Fail loudly. Never fall back.** Typed error, stop. No fallback value, ever |
| **INV-2** | **Runs only when the user started it** — no automatic invocation |
| **INV-3** | **Simulation only** — no NPV, costs, prices |
| **INV-4** | **Output is training-ready** — labelled $(x,y)$, full fields, versioned |
| **INV-5** | **Own flaw register** — separate, `COMP-nn` namespace |
| **INV-6** | **Capability declaration ≠ failure** — `NOT_IMPLEMENTED` ≠ `FAILED` |
| **INV-7** | **Unconstrained by construction** — absurd input still gets full physical evaluation |

📌 **INV-1 and INV-7 are the two that get broken by accident.** A guard that clamps, saturates
or substitutes a default violates INV-1; a guard that *rejects* absurd input violates INV-7. 🔴
Every known break of this class is logged: **C-93** (clamp), **C-97** (clamp purged), **C-151**
($\Delta t$ cutback), **C-188 / §7s.6** (silent saturations).

---

## 3. 🔴 RETIRED — do not re-derive these

> **The single most valuable thing in this file.** Each of these was asserted by an agent,
> disproved by measurement, and **withdrawn**. 📌 **An agent that does not know this will
> confidently re-derive a wrong answer.**

| Retired claim | Superseded by | The measurement that killed it |
|---|---|---|
| **C-229** — *"the spec's $\alpha_0(r_w+L_p)$ is not the inversion of the tabulated ratio; it saturates"* | **C-230** | 🔴 **I misread $r_{we}$ as $r_{wo}$.** Table 1's column is $r_{we}/(r_w+L_p)$ — which **is** $\alpha_\theta$. Eq. 7 of the paper **is** $r_{we}=\alpha_\theta(r_w+L_p)$. **The reported "5.25× divergence" was an artefact of an inversion no source contains** |
| **C-98** — *"larger $\alpha_0$ means worse"* | **C-100**, then **C-232** | ⚠️ **Half right.** $S_H$ and $S_{wb}$ favour **more** phasing; $S_V$ favours **less**. 🔴 **The net optimum is parameter-dependent** (measured 45°/60°/90°) |
| **C-100** — *"larger $\alpha_0$ = better"* | **C-232** | ⚠️ **Half right, same reason.** Not a net rule |
| **C-92** — *"$S_v$ is non-monotonic"* | — | 🔴 **Wrong. $S_V$ is monotone.** |
| **C-171** | **C-176** | 🔴 **The submitted sign was the exact Mohr–Coulomb correspondence; my rule was wrong** |
| **C-203** | **C-205** | 🔴 **Submitted $g^2+gg''\ge0$ is CORRECT; my $g^2+2(g')^2$ was wrong** |
| **C-79b** (dynamic sparsity), **C-81** (Penéloux) | Rulings 8–11 | withdrawn |
| **C-211 / C-215** (μ) | Ruling 36 | 🔴 **Withdrawn before adoption** — $\mu$ had not been derived |
| **C-227** — *my transcription: $\alpha_0(120°)=0.618$* | **source** | 🔴 **The source reads $0.648$.** Two rows were missing entirely |
| **C-228** — *fitted coefficient $0.476$* | — | 🔴 **Does not survive a six-row refit** ($0.2843+0.4113\log_4N$, RMS 0.0298) |

### My recurring failure mode — the pattern behind all of them

> 🔴 **Asserting a physical property from geometry without deriving the convention first.**
> **Eight sign slips and five withdrawals.** ⚠️ Every correction came from **evaluating at a
> state whose value is known independently** — never from re-reading the geometry.

📌 **The discipline that works: evaluate at a state you can check by another route.**

---

## 4. Standing CI requirements — non-negotiable

- **No scalar factor typed by hand** — the `#[cfg(test)]` limit-case suite asserts it.
  🔴 Five factor-3/$\sqrt3$ inversions (C-161, C-164, C-205, C-214, C-217) were caught *only*
  this way, never by review.
- **Derive-then-assert** for every inequality
- **Two-sided independence** for equivalence tests — the two sides from *independent* routes
- **Code-block test pinning** — every submitted code block gets a test
- Every **saturating transform** and singularity guard paired with a `ValidityWarning` (§7s.6)
- Cumulative edits to guarded expressions
- Every `MeridianType` names the property its coefficients have — **verified by a test**
- 🔴 **No silent degradation.** A clamp that makes a failure disappear is worse than the failure

---

## 5. Verification commands

```bash
# the wiki gate — MANDATORY first action in every turn
Get-Content agent_wiki/README.md -Encoding UTF8 -TotalCount 30

# WIKI-SYNC GATE — a commit that changes code without changing agent_wiki/ FAILS
python .github/scripts/check_wiki_sync.py --base HEAD~1 --head HEAD
#   exit 0 = synced | 1 = source changed, no wiki change | 2 = immutable set touched
#   escape hatch (recorded in git history, never silent):  Docs-Skip: <reason>

# build the docs site (docs_dir = agent_wiki)
python -m mkdocs build --strict     # NOTE: 31 pre-existing warnings, see mkdocs.yml

# Python-side gates (NOT the Rust engine)
python -m audit.continuity check
python -m audit --register validate
python -m ruff check --select F821 .
```

⚠️ **`--strict` fails on 31 pre-existing link warnings** — all structural (links pointing
outside `docs_dir`). ✅ `strict: false` is set deliberately; the reason is recorded in `mkdocs.yml`.

🔴 **`audit/continuity.py::check_wiki` does NOT cover this section.** It reads three files only
(`audit/scientific_flaws.md`, `agent_wiki/README.md`, `development/common_pitfalls.md`), so it
never sees `compositional/` or `thmc/`. ✅ **That is the gap `check_wiki_sync.py` fills**, and it
runs in CI (`.github/workflows/wiki-sync.yml`) on every push and PR.

⚠️ **Python gates do not cover the Rust engine.** ✅ The Rust register is a *separate* spec
(`COMP-nn`, INV-5) and is **not yet machine-validated** — the Python `audit/` module cannot hold
a `.rs` finding.

---

## 6. 🔴 Rollover checklist — do this at every handoff

1. ☐ **Update §1** above and the `As of` line. ⚠️ If it is stale, say so — do not leave a stale
   claim looking current.
2. ☐ **Add any new retraction to §3.** 📌 A withdrawn claim is the *most* valuable thing to
   carry forward; an unrecorded one will be re-derived.
3. ☐ **Note what was measured, with the number.** ⚠️ Never carry a verdict without its
   measurement — that is how C-100 and C-229 happened.
4. ☐ **State the open blockers** and who owns them.
5. ☐ **Commit.** ✅ An uncommitted handover record is a handover record nobody receives.
6. ☐ **Confirm the wiki-sync gate passes** — it runs in CI, but a local
   `python .github/scripts/check_wiki_sync.py --base HEAD~1 --head HEAD` fails fast.

---

## 7. What is 🔴 NOT settled — do not act as if it were

| Open | Why it matters |
|---|---|
| **C-102** | No closed form for $\alpha_\theta$. Tables 1–5 are **finite-element output**; the paper says so. ✅ Use the six tabulated values **exactly**. Interpolation between them is **undefined** |
| **C-233** | The $0^\circ$ ambiguity: $L_p/4$ vs $0.250(r_w+L_p)$ — equal only if $r_w=0$. Table 3 favours branch 1, but that is an **inference**, not a statement by the authors |
| **C-234** | Eq. 9 is valid only for $0.30\le r_{wD}\le0.90$, 🔴 **but the $0^\circ$ case sits at $r_{wD}=0.178$**. 🔴 Must not be extrapolated **or clamped** — a clamp fabricates a skin |
| **C-237** | Two $s_V$ routes with **different functional forms** (Eq. 12, Eq. 15). ✅ Pick one, record it, **never blend** |
| **M7a, M7c–M7h** | 🔴 **No milestone bodies exist**, by owner decision. ⚠️ **Do not invent gates** |
| **Rust toolchain** | Deferred. 🔴 Do not install unprompted |

📌 **Literature L-1…L-9 is the only route to citable data.** ✅ L-1, L-7, L-8 are closed.
🔴 **No skill may close a `CONF-*` or `L-*` item** — see
[`agent_skills.md`](agent_skills.md) §2.