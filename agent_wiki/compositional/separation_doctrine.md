# Separation Doctrine — Phase 1 Only

> [!CAUTION]
> **Correction 07-10-2026.** This page was first written before the project owner restated the vision.
> It originally read as a near-permanent architectural law with CI gates that *fail the build*.
>
> **It is not that.** The owner clarified that separation is **"for development purpose to not
> distract"** — a **Phase 1 development discipline**, not an architecture decision. Coupling is
> intended, in a later phase, once the compositional engine is verified.
>
> **Concretely:** the gates in §3 enforce P1 discipline and are **retired by ADR at the start of P2**.
> Nothing in P1 may foreclose P2/P3 coupling. See [`vision_and_phases.md`](vision_and_phases.md) §2 for
> the phase model.

**Decision 07-10-2026:** during Phase 1, the compositional engine does not rely on the existing
codebase. Surrogate and compositional are **not coupled** in P1. Coupling is deferred to P3 and is
permitted only after the compositional engine is verified on all features and fronts.

This page makes P1 "separate" **enforceable** rather than aspirational. Every rule below has a
mechanical check — **valid for the duration of P1 only**.

### Which invariants this page enforces

| Invariant | Relationship to this page |
|---|---|
| **INV-1** fail loudly | ✅ **C-5** bans `unwrap`/`expect`/`panic!`; a crate that cannot express a failure has no way to fail loudly |
| **INV-2** user-initiated only | ⚪ **Not enforced here** — a caller-side property, belongs to the P2 routing layer ([`integration_plan.md`](integration_plan.md)) |
| **INV-3** simulation only | ⚪ **Not enforced here** — an output-schema property, enforced by [`output_schema.md`](output_schema.md) |
| **INV-4** training-ready | ⚪ **Partially** — C-7c guarantees a **single write path**, which is what makes the schema contract enforceable |
| **INV-5** own register | ✅ **This page's companion is [`register_spec.md`](register_spec.md)** — a separate module is required *because* the Python register cannot hold a `.rs` location |
| **INV-6** capability ≠ failure | ⚪ **Not enforced here** — carried by the status lattice, [`engine_numerics.md`](engine_numerics.md) 7q/7r |
| **INV-7** unconstrained | ⚪ **Not enforced here** — a property of the physics, [`engine_invariants.md`](engine_invariants.md) 7b |
| — internal coherence | ✅ **C-7**, added 09-10-2026 — 🔴 **permanent, never retired** (see C-7 below) |

⚠️ **C-1…C-4 retire at P2. C-7 does not.** Coupling the crate to *Python* is what P2 may relax; composing
the crate's *own* modules is correctness at every phase.

---

## 1. What "not coupled" means, precisely

| Prohibited now | Permitted now | Deferred |
|---|---|---|
| The compositional crate importing anything from `core/`, `ui/`, `evaluation/`, `utils/`, `validation/` | Transcribing **numeric reference values** (SPE 5 parameters, CMG expected outputs, Welge construction, MMS source terms) into the Rust side, with attribution | A defined interface between the two engines |
| Sharing `GeostatisticalParams`, `ReservoirData`, `EOSModelParameters` or any dataclass | Sharing **units and notation conventions** as documentation | Running either engine inside the other's process |
| A shared objective wrapper, shared profile schema, shared project file | Sharing **field-unit conventions** (STB, MSCF, psia, ft, bbl) | Optimisation loops spanning both engines |
| A shared flaw register or shared continuity gate | A shared **verification vocabulary** — the same V&V levels, the same discipline | Adjoint gradients spanning both |
| The compositional engine being "validated against the surrogate" | Validating the compositional engine against **independent analytic and CMG references** | Cross-engine differential testing |

> **The distinction that matters:** coupling is a **dependency**, not a shared reference. Comparing two
> independent implementations against the same published reference value is good practice. Importing
> one from the other is coupling.

---

## 2. Repository layout

```
crates/
  compositional/                     <- the new engine (Rust, standalone)
    Cargo.toml
    src/
      eos/            M1
      flash/          M2
      grid/           M0
      relperm/        M3
      flow/           M3-M4
      newton/         M0
      linear/         M0
      well/           M5
      io/             M5
      report/         M5
    tests/                          <- cargo test; NOT under tests/
  compositional-audit/               <- separate register tooling (Rust or Python, but standalone)
audit_comp/                          <- separate flaw register + continuity gate
agent_wiki/compositional/            <- this section
```

### 2.1 Placement rules

| Rule | Reason |
|---|---|
| `Cargo.toml` lives at `crates/compositional/`, **not** the repo root | A root `Cargo.toml` makes the Python repo a Cargo workspace member and invites `ruff`/`pytest` and Cargo tooling to fight over the same tree |
| Rust tests live at `crates/compositional/tests/` | `pyproject.toml` sets `testpaths = ["tests"]`. A directory named `tests/` at the repo root **would** be collected by pytest. Avoid that name. |
| `crates/` is not importable from Python | Nothing in `crates/` is on `sys.path`; Python cannot accidentally import it |
| No `__init__.py` anywhere under `crates/` | Prevents pytest collection and package shadowing |

---

## 3. Enforced checks — **Phase 1 scope**

Each check must fail CI when violated **during P1**. A rule with no check is a comment. **All of these
are retired by ADR when P2 begins** — they are a development aid, not an architecture.

### C-1 Forbidden-import gate

The compositional crate must not reference the Python codebase.

```bash
# Fails if any of the forbidden roots appear in Rust sources
grep -rnE '(engine_surrogate|core\.|data_models|evaluation\.|utils\.|validation\.|ui\.)' \
     crates/compositional/src --include='*.rs' && exit 1
```

Rust has no Python import mechanism, but **path references and FFI declarations are the risk**:
`extern "C"` bindings, `include!`, and hardcoded `../core/...` paths.

### C-2 No FFI to the Python side — **P1 only**

`unsafe extern "C"` is banned in `crates/compositional/src` for the whole of **P1 (M0–M6)**.
D8 §1.3 proposes `unsafe extern "C"` bindings — **deferred to P3**, where it becomes a legitimate
integration mechanism under a separate ADR with a stated ABI contract.

### C-3 Separate register module

`audit_comp/` is a distinct package with its own register file, ID namespace, and CLI.
**It must not be imported by `audit/`.** See [`register_spec.md`](register_spec.md).

### C-4 Separate continuity gate

`audit/continuity.py` **hard-imports** the surrogate:

```
audit/continuity.py:150-152   import inspect
                             from core.engine_surrogate import analytical_models as AM
audit/continuity.py:186-189   from core.engine_surrogate.surrogate_engine import SurrogateEngine
                             from core.data_models import (...)
```

A continuity gate for the compositional engine must therefore be **written from scratch**. It cannot
reuse that module, and it must not attempt to import the Python engine to compare results — that would
be coupling.

### C-5 Rust lints as a release gate

Mirrors the Python `ruff --select F821` gate. Recommended for M0:

| Lint | Purpose |
|---|---|
| `#![deny(unsafe_code)]` in `lib.rs` | Ban `unsafe` outright — makes C-2 a compile error, not a grep result |
| `clippy::unwrap_used`, `clippy::expect_used`, `clippy::panic` | D2 §1.1's "zero-panic architecture" as an enforced lint rather than prose |
| `clippy::float_cmp` | Guards exact float comparison, which is how conservation tests silently pass |
| `#![forbid(non_snake_case)]`, `#![deny(missing_docs)]` | API hygiene |
| `cargo clippy -- -D warnings` | Release gate |

### C-6 Conservation is a test, not a comment

The design set's `10^-12` component-mass-balance requirement (**D5 §3.1**) must be a Rust test that
**fails CI**. It must not be a documented aspiration.

### C-7 🔴 No orphan modules — the coherence gate (added 09-10-2026)

🔴 **Separation is a discipline about what the crate must not import. It says nothing about whether the
crate's own modules work _together_.** ⚠️ A crate can satisfy every check in §3, pass every per-module
unit test, and still be unable to produce **one run** — the disconnected-modules failure.

**Ruled: every `pub` module must be reachable from the driver, and reachable _by test_.**

| # | Check | Failure means |
|---|---|---|
| **C-7a** | No `pub` module is unreferenced by the driver or its test suite | a module exists that nothing exercises |
| **C-7b** | The end-to-end integration test completes a full spec → solve → output run | the subsystems do not compose |
| **C-7c** | One write path only — no module-local output writer | results exist that bypass IO and the schema |

⚠️ **C-7a is the load-bearing check.** 📌 **An uncalled module is not merely disconnected — it is dead code
that looks implemented**, and it will pass C-7b and every feature gate while contributing nothing.

⚠️ **This is an internal-coherence check, not an independence check.** It is **permanent**, and it survives
the P2 retirement of C-1…C-4: coupling the crate to *Python* is the thing P2 may relax, while composing
its *own* modules is correctness at every phase. **Never retire C-7.** Gate definition:
[`build_plan.md`](build_plan.md) M4.5.

---

## 4. Why `audit/continuity.py` cannot be reused — and what that costs

The Python continuity gate re-measures every `RESOLVED` claim by *executing* code:
`import numpy`, `from core.engine_surrogate.surrogate_engine import SurrogateEngine`, and reading
`surrogate_engine.py` / `pvt_state.py` as text.

For the compositional engine the equivalent gate must:

| Python gate does | Compositional gate must do |
|---|---|
| Import Python modules | Shell out to `cargo test <gate>` or run a `comp-verify` binary |
| Read `.py` as text for pattern checks | Read `.rs` as text (or better: instrument, not grep) |
| Re-measure numeric claims | Re-measure numeric claims by **re-running the simulation and comparing** |
| Share `STATUSES` with `audit/registry.py` | Share the **vocabulary** — deliberately, so the two registers cannot diverge semantically — but **not** the module |
| Enforce 1-commit-1-issue against `audit/scientific_flaws.md` | Enforce against the compositional register |

### 4.1 Correction to the existing documentation

`AGENTS.md` and `agent_wiki/README.md` describe the continuity gate as **"Autouse"** and
`"**Autouse**: `python -m audit.continuity check`…"**. Verified 07-10-2026: a search for
`autouse=True` across all Python files returns **zero** hits, and `audit/continuity.py` contains **no**
`pytest` or `fixture` reference. The gate is **CLI-invoked, not autouse**.

> **Consequence:** a `pytest` run does **not** currently re-verify `RESOLVED` claims. Any belief that
> green tests imply verified physics is unsupported. This is consistent with **HIGH-18** and is a
> documentation defect in `AGENTS.md` + `agent_wiki/README.md` worth registering.

---

## 5. Documentation separation

| Rule |
|---|
| No page in `agent_wiki/compositional/` may state what `core/engine_surrogate/` does, or vice versa |
| The compositional engine's physics claims cite `../thmc/` (the design documents) and the compositional register — **never** the Python finding register |
| Where the two engines disagree, that is a **future coupling-plan** input, not a defect in either |
| The CONF-* conflicts in `../thmc/conflict_and_gap_register.md` are **spec defects for the new engine**, not defects in the old one — see [`spec_defects.md`](spec_defects.md) |

---

## 6. Coupling — deferred, and deliberately undesigned now

Coupling is permitted only **after** the compositional engine passes its own gates. Designing the
interface now is how two engines end up sharing mutable state.

When that plan is written, the decisions it must make — recorded here so they are not made accidentally:

| Decision | Why it matters |
|---|---|
| **Cross-validation protocol** | Is the surrogate validated *against* the compositional engine, or are both against shared references? The latter is safer and was the choice made here. |
| **Profile schema** | A shared output contract is the most natural coupling point. Decide it late, from measured needs |
| **Objective ownership** | Per the **CONF-04** ruling, economics is off-core in both engines. A shared economic module would be the cleanest coupling point — and the cheapest to specify |
| **Fallback policy** | If the compositional engine fails or is too slow, does the surrogate silently take over? **A silent fallback is how invalid results ship.** Any coupling must fail loudly |
| **Which engine is authoritative for reporting** | Must be stated per output field, not globally |
| **Cost of dual maintenance** | Stated explicitly in the coupling ADR, not discovered later |

> **Recorded risk:** the `Economic_DCF_DTO` economic module (D6 §1.1) is the natural shared component.
> **CONF-58** and **CONF-59** — the four cost terms a reduced NPV expression drops, and the risk of
> booking recycled CO₂ as gas revenue — must be resolved **in the compositional engine's own economic
> module** before any coupling, so the bug is not replicated in two places.