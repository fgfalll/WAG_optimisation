# Agent Skills for the Rust 3D THMC Engine

> [!IMPORTANT]
> **A third-party skill is *procedural guidance*. It is never *data*.**
> No skill may close a `CONF-*` item, satisfy an `L-*` literature item, or supply a
> coefficient that enters a constitutive relation. See [§2](#2-the-governing-rule) for
> the measured cost of breaking this.

**Compiled 10-10-2026.** Eleven skills in `.agents/skills/`, installed via the `skills`
CLI (`npx skills`). 🔴 **None of them contains reservoir data.** They supply *method*
only — how to check an answer, how to design a benchmark, how to spot a unit error.

---

## 1. What is installed, and what each is actually good for

| Skill | Source | Use for | 🔴 Limitation |
|---|---|---|---|
| `verification-before-completion` | `obra/superpowers` — **extracted, see §4** | *"NO COMPLETION CLAIMS WITHOUT FRESH VERIFICATION EVIDENCE"* | ✅ **agrees** with `AGENTS.md`; taken **without** the framework |
| `systematic-debugging` | `obra/superpowers` — **extracted, see §4** | Hypothesis-driven; no fix without an isolated root cause | ✅ compatible |
| **`thmc-precision-checks`** | **this repo** | Unit table, dimensional traps, oracle ladder, evidence standard for **novel** relations | — |
| `numerical-verification` | `e-eight/scicomp-skills` | **The oracle ladder.** "A test asserts a property you can compute two independent ways." | none found |
| `nist-refprop` | `soljourner/claude-engineering-skills` | PVT / phase-equilibrium **validation oracle** (147+ pure fluids, VLE/LLE/VLLE) | commercial licence + `ctREFPROP`; 🔴 **not available in CI** |
| `dimensional-analysis` | `trailofbits/skills` | Unit-mismatch and precision-loss bug patterns | 🔴 **written for Solidity/DeFi** — token decimals, price oracles. **Use the patterns, ignore the examples** |
| `neqsim-capability-map` | `equinor/neqsim` | Architecture reference: a real compositional simulator's capabilities | Java; map only |
| `neqsim-eos-regression` | `equinor/neqsim` | $k_{ij}$ fitting, saturation-pressure matching, C7+ characterisation (CME/CVD/DL) | NeqSim workflow |
| `neqsim-phase-envelope` | `equinor/neqsim` | PT envelope generation, branch identity, Michelsen continuation | NeqSim API |
| `design-flash-benchmark` | `equinor/neqsim` | Flash benchmark **matrix design** | 🔴 design only — **supplies no expected values** |
| `analyze-gibbs-convergence` | `equinor/neqsim` | Gibbs-minimisation convergence, Jacobian conditioning, element-balance closure | chemical-equilibrium framing |
| `neqsim-input-validation` | `equinor/neqsim` | Input-validation patterns | NeqSim-specific |
| `agent-wiki` | this repo | The wiki gatekeeper (pre-existing) | — |
| `partial-differential-equations` | `neuralblitz/mito` | ⚠️ **623 bytes** — near-empty | 🔴 effectively a placeholder |

📌 **`neqsim` was the highest-value find.** The registry advertises one skill from that
repo, but the repository actually contains **106** under `.github/skills/`. ⚠️ A full
`git clone` fails on Windows with *filename too long* — NeqSim has pathologically long
Java filenames. ✅ **Use `git clone --filter=blob:none --no-checkout --depth 1`, then
`git show HEAD:<path>` to extract a single skill.**

---

## 2. 🔴 The governing rule — a skill is not a source

**Provenance for anything numeric, in order of strength:**

1. 🔴 **The primary source**, transcribed row by row, with row count recorded
2. ⚠️ **A secondary source**, labelled as such, with its own citation
3. 🔴 **Never** a remembered value, a submission's transcription, or a skill

**The measured cost of breaking it** — all three from the Karakas & Tariq work:

| Finding | What happened |
|---|---|
| **C-116** | Searched 129 PDFs, found nothing, then **quoted a four-row table from the submission** instead of marking it absent |
| **C-227** | With the real table in hand, transcribed **0.618** where the source reads **0.648**, and dropped two rows — the fit (RMS 0.0076) was good enough that the bad digit read as scatter |
| **C-229** | Misread the subscript $r_{we}$ as $r_{wo}$, **invented an inversion** no source contains, and reported a "measured 5.25× divergence" that was artefact |

> 📌 **None of those was fixed by better tooling.** All three were fixed by obtaining the
> paper and reading the equations. ✅ **Every table the engine needs — $\alpha_\theta$,
> $c_1/c_2$, $a_1..b_2$, $L_{p\min}/r_w$ — came from the source. No skill supplied one.**

**Standing rule added 10-10-2026:** ✅ **a correction that changes the _structure_ of an
equation — inverting it, bounding it, rescaling it — is admissible only after reading the
equation in the source, never inferred from a tabulated quantity plus a guess.**

---

## 3. 🔴 Registry gaps — reported, not papered over

The public registry returned **zero results** for every one of these:

> *geomechanics plasticity* · *Drucker-Prager* · *Mohr-Coulomb* · *solid mechanics
> plasticity* · *reactive transport* · *species transport* · *finite volume method* ·
> *conservation law discretization* · *flow in porous media* · *porous media* ·
> *method of manufactured solutions* · *upscaling permeability* · *CO₂ geological
> storage* · *phase equilibrium flash* · *well logging / petrophysics*

🔴 **Those are M7b, the geochemistry milestone, and the core spatial discretization.**
⚠️ **No skill covers them ⇒ they must be sourced from literature ([`literature_todo.md`](literature_todo.md))
or derived.** 🔴 **No milestone may be gated on a skill that does not exist.**

**Inverse lesson, equally important:** ✅ **for the domains the registry _does_ cover, it
is shallow.** Zero hits for *compressional PVT*, *relative permeability*, *SCAL*, or
*SPE benchmarks*. 🔴 **Install count is not relevance** — `*@arbor` returned fifteen
mirrors at 1.1K installs and is a dendrite-morphology tool.

---

## 3a. ✅ The index is a *small subset* — and the real gap is not skills at all

🔴 **skills.sh is not the whole market.** `add <owner/repo>` works on **any** GitHub repo,
whether or not it is indexed. A control-verified GitHub probe (controls first, to prove
the probe worked) shows collections the registry barely surfaces:

| Repo | ★ | Note |
|---|---|---|
| `obra/superpowers` | **297K** | agentic skills *framework* — methodology, not physics |
| `mattpocock/skills` | **283K** | large general collection |

📌 **But the highest-value target is not a skill.** It is a **reference implementation**:

### 🎯 `SINTEF-AppliedCompSci/MRST` — ★140, GPL-3.0, pushed 2026-10-09

**The MATLAB Reservoir Simulation Toolbox.** Verified topics:

`compositional` · `finite-volume` · `porous-media-flow` · `blackoil` · `co2-sequestration` ·
`multiscale` · `automatic-differentiation`

🔴 **This is the only actively-maintained, properly-licensed, permissively-visible
implementation found that covers the FVM *and* compositional *and* geomechanics gap that
§3 reports as unserved.** ✅ It is **actively developed** (last push 2026-10-09) and
**GPL-3.0**, so it is readable as a reference.

⚠️ **It is MATLAB, not Rust, and it is not a skill.** ✅ **Its role is the same as CMG GEM's
for the Python engine** — an **external oracle**. 📌 **Proposed use: a benchmark and
cross-check target for M2/M3/M7**, with the same standing caveat: ✅ **it can tell you
whether the Rust engine agrees; it cannot supply a coefficient into a constitutive relation.**

### 🔴 A false positive worth recording

`qinlingboy888/THMC_FVM_Phreeqc` — *"a finite volume method (FVM)-based coupled model designed to
simulate thermal (T), hydraulic (H), mechanical (M), and chemical (C) processes in fractured
reservoirs."*

🔴 **A perfect name match for the exact gap §3 reports as unserved — and it is unusable.**
Measured: **★9** · **no licence** · **created *and* last pushed 2024-08-14** (single day,
never touched since) · **no topics** · MATLAB.

> 📌 **This is the §7h.4 trap in its purest form.** The name matched the need *exactly*,
> which is what makes it dangerous. 🔴 **Had it been installed on the strength of its
> description, it would have become an uncitable "source" for the whole THMC formulation** —
> the precise failure mode of **C-116** (quoting a table that could not be sourced).
> ✅ **Provenance is checked before content is read.** A repository with no licence, one
> day of history, and no topics is not a source at any star count.

---

## 4. 🔴 `obra/superpowers` — vetted 10-10-2026, **framework REJECTED, content extracted**

**★297,010 · MIT · Jesse Vincent · HEAD `bb92a77` 2026-10-09 · 238 files · 15 skills.**
Cloned and read, not judged from the description. ✅ **Zero name collisions** with the eleven
skills already installed.

### 🔴 BLOCKER 1 — `using-superpowers` is an instruction-precedence hijack

The skill's verbatim core:

> `<EXTREMELY-IMPORTANT>` **"If you think there is even a 1% chance a skill might apply to what you
> are doing, you ABSOLUTELY MUST invoke the skill. … This is not negotiable. You cannot rationalize
> your way out of this."**
>
> **"Invoke relevant or requested skills BEFORE any response or action** — including clarifying
> questions, exploring the codebase, or checking files."

🔴 **This repo's `AGENTS.md` states the opposite, as an invariant:**

> *"the agent's **VERY FIRST tool call in EVERY turn MUST be** `view_file("agent_wiki/README.md")`
> … Bypassing this step to immediately search or edit source code is an **invariant violation**."*

⚠️ **These are mutually exclusive, and only one can hold.** superpowers claims the earlier slot
explicitly (*"Before entering plan mode: if you haven't already brainstormed, invoke the
brainstorming skill first"*). 📌 **In this repository, `AGENTS.md` wins — the wiki gatekeeper is the
project's own invariant, and superpowers is a generic framework that knows nothing about this
project.**

### 🔴 BLOCKER 2 — it is injected automatically, so it cannot be declined

`hooks/hooks.json` registers a **`SessionStart`** hook, matcher `startup|clear|compact`,
`"async": false`, executing `hooks/run-hook.cmd session-start`. That script reads
`skills/using-superpowers/SKILL.md` and emits it wrapped in
`<EXTREMELY_IMPORTANT>` as session context.

🔴 **So the instruction-precedence claim arrives at every session start whether or not anyone
opts in**, competing with the wiki gatekeeper for the first-action slot. ⚠️ A framework that must
be *defeated* to be safe is not one to install into a repo whose central invariant is
*which instruction fires first*.

### ⚠️ WARN 3 — `brainstorming` opens a network listener

`skills/brainstorming/scripts/start-server.sh` binds **a random high port**, default
`127.0.0.1` but with `--host 0.0.0.0` documented *"in remote/containerized environments"*;
**240-minute idle timeout**; `--open` **auto-launches a browser**; websocket protocol; writes
`.superpowers/brainstorm/` into the project.

🔴 **An undeclared long-lived listener plus a browser-launching path, in a repository whose entire
purpose is auditable verification.** Rejected on that basis alone.

### ✅ What was good — extracted without the framework

Two skills are pure Markdown, carry **no executable surface and no hooks**, and *reinforce*
rules already standing in this project:

| Skill | Why it is safe and useful |
|---|---|
| `verification-before-completion` | **"NO COMPLETION CLAIMS WITHOUT FRESH VERIFICATION EVIDENCE"** — ✅ already this project's rule (*"pytest passing is not evidence"*; every `Status:` line must cite a measurement). It **agrees** with `AGENTS.md`; superpowers' problem is `using-superpowers`, not this. |
| `systematic-debugging` | Hypothesis-driven, no fix without an isolated root cause. ✅ Compatible. |

📌 **Neither was taken from the registry — both were copied as single files**, with `hooks/`,
`using-superpowers`, and the server skill left behind.

⚠️ **Also present but not taken:** `writing-plans`, `writing-skills`, `executing-plans`,
`requesting-code-review`, `receiving-code-review`, `using-git-worktrees`,
`subagent-driven-development`, `finishing-a-development-branch`,
`dispatching-parallel-agents`, `diagnosing-superpowers`. ⚠️ `test-driven-development` is a
**conceptual** duplicate of a skill already active in this session, despite no path collision.

> ⚠️ **Star count is not evidence.** ★297,010 with 26,521 forks, created 2025-10-09, would place
> it among the most-starred repositories in GitHub's history. 📌 The same caution recorded for
> `*@arbor` applies here — ✅ **a popular methodology framework is not a safe dependency for a
> precision-critical repo until its instruction surface has been read.**

---

## 5. Rejected, with reasons — so the search is not repeated

| Skill | Reason |
|---|---|
| `tnav-reservoir-sim` | 🔴 **Self-declared *"educational emulation… results are approximate."*** For a precision engine this is an **anti-skill** — a source of correlations applied without provenance |
| `multi-phase-flows` | 🔴 Wrong domain — combustion/propulsion CFD (spray, cavitation, VOF), not porous media |
| `solver-numerics` | 🔴 Wrong domain — SPICE circuit simulation (MNA, trapezoidal for circuits) |
| `fenics-fem` | 🔴 Python FEniCS; this engine is Rust |
| `*@arbor` (15 mirrors, 1.1K) | 🔴 Dendrite morphology — **high install count is not relevance** |
| `physicsnemo` | ⚠️ Correct topic (FNO/GNN surrogates, INV-4) but the source repo has **2 GitHub stars** — too thin to lean on |
| `qinlingboy888/THMC_FVM_Phreeqc` | 🔴 **Perfect name match for the §3 gap, and unusable** — ★9, **no licence**, one day of history (2024-08-14), no topics. **The most dangerous false positive found.** See §3a |
| `yohanesnuwara/reservoir-geomechanics` | 🔴 **Student worked solutions to a university course.** Not a citable physics source, and using it as one would be an academic-integrity problem |
| `itasca-mcp` | ⚠️ Genuinely geomechanical, but it is an MCP bridge to **commercial ITASCA PFC/FLAC/3DEC** — unavailable here |
| `reservoir-geomechanics` (Zoback course) | 🔴 Same as above — solutions, not a source |

---

## 6. Going beyond published correlations — the evidence standard

🔴 **A novel relation is a *claim* like any other and owes the same evidence.** Before a
derived relation enters a solve path:

| # | Requirement | Test |
|---|---|---|
| 1 | **Dimensional closure** written out explicitly | overburden self-test: $\rho g\Delta z$, ✅ band 10.0–10.5 MPa/km |
| 2 | **Limiting cases** derived, not assumed | each is a `#[cfg(test)]` case |
| 3 | **No hand-typed scalar** where the limits constrain it | assert the factor via limits |
| 4 | **Derive-then-assert** for every inequality | oracle 2 — two independent routes |
| 5 | **Independent validation set** not used in fitting | held-out cases |
| 6 | **Provenance class recorded** | published / derived / assumed |
| 7 | **Failure mode characterised** | behaviour outside the validity region |

> 📌 **Criterion 3 is not optional.** Five factor-3/$\sqrt3$ inversions were typed by hand
> and caught **only** by limit-case tests — never by review. ✅ **The remedy already
> ruled: no scalar factor is typed by hand; the `#[cfg(test)]` limit-case suite asserts it.**

Full working detail — including the measured unit table, the temperature trap, and the
oracle ladder — is in `.agents/skills/thmc-precision-checks/SKILL.md` (repo root; outside
`docs_dir`, so it is not linkable from here).

---

## 7. 🔴 What a skill may never do

- 🔴 **Supply a coefficient** used in a constitutive relation
- 🔴 **Close a `CONF-*` item** or an `L-*` literature item
- 🔴 **Replace the §7h.4 three-part acceptance standard** ([`engine_spec_closures.md`](engine_spec_closures.md))
- 🔴 **Serve as a benchmark oracle** — `design-flash-benchmark` designs the *matrix*; ✅
  **it supplies no expected values**, which is exactly the correct division of labour

> 📌 **A skill can tell you _how_ to check a number. It cannot tell you _what_ the number
> is.** ✅ **Only a cited source can do that.**