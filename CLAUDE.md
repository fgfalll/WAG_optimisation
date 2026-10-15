# CLAUDE.md — CO₂-EOR Optimizer

This repository is a Python-based scientific CO₂-EOR simulation and optimization project.

## Mandatory Instructions

Before substantial development work:

1. Read `AGENTS.md`.
2. Read `agent_wiki/README.md` if it exists.
3. Read only the relevant Agent Wiki pages for the requested task.
4. Use the Wiki to identify the relevant source-of-truth implementation.
5. Verify HIGH and CRITICAL scientific behavior against the actual source code.
6. Run the smallest relevant tests before and after modifications.
7. Update the Agent Wiki when project knowledge changes.

`AGENTS.md` is the primary development policy.

`agent_wiki/` is the primary project-knowledge and architecture reference.

The source code remains the ultimate authority for actual implementation behavior.

## Scientific Integrity

This project is a scientific reservoir-engineering / CO₂-EOR simulator.

Do not introduce or silently preserve:

- unexplained hardcoded scientific values
- hidden calibration
- arbitrary correction factors
- recovery multipliers
- undocumented empirical fitting
- artificial result-producing fallbacks
- silent exception-based substitutions
- benchmark-specific tuning presented as general physics.

Never assume plausible results are physically valid.

Never claim scientific validation unless it was actually performed.

For scientific uncertainty, use:

`UNKNOWN`

or:

`AUDIT REQUIRED`

rather than guessing.

## Wiki Synchronization

Whenever a change affects:

- architecture
- execution flow
- physics
- equations
- parameters
- PVT
- CO₂ behavior
- reservoir behavior
- optimization
- validation
- dependencies
- assumptions
- fallbacks
- source-of-truth implementations

update the relevant `agent_wiki/` documentation.

Do not create duplicate documentation in `CLAUDE.md`.

Keep this file focused on Claude-specific steering and defer detailed project knowledge to `AGENTS.md` and `agent_wiki/`.

## Priority

Follow this hierarchy:

```text
AGENTS.md
    ↓
agent_wiki/
    ↓
actual source code
    ↓
tests and audit evidence
```

When documentation and source code disagree, verify the source and update the documentation rather than guessing.
