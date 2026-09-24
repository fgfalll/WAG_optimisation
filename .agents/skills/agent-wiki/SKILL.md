---
name: agent-wiki
description: Mandatory entry point and gatekeeper for ALL tasks in this repository. Consult this skill and agent_wiki/ before performing any code edits, searches, investigations, or refactoring in CO2 EOR Optimizer.
---

# Agent Wiki Mandatory Gatekeeper

## 🛑 Action Requirement Before Doing Any Work

Before running search tools (`grep_search`), terminal commands (`run_command`), or editing any source code:
1. **First Tool Call**: You MUST read [`agent_wiki/README.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/README.md) or the specific topic document in [`agent_wiki/`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki) relevant to the task.
2. **Confirm Active Engine**: Verify [`agent_wiki/architecture/source_of_truth_map.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/architecture/source_of_truth_map.md). Remember that 100% of simulation evaluations route strictly to `core/engine_surrogate/`. Do NOT modify `core/Phys_engine_full/`, `compositional_engine/`, or `unified_engine/` expecting them to affect optimization runs.
3. **Verify Physical Invariants & Traps**:
   - Check [`agent_wiki/development/common_pitfalls.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/development/common_pitfalls.md) (traps with unit conversions, NumPy 2.0, mass balance, Vogel IPR).
   - Check [`agent_wiki/development/safe_modification_rules.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/development/safe_modification_rules.md) for immutable invariants.
4. **Log Simulation Run Audits**: Whenever running or evaluating simulation runs, parameter sweeps, or benchmarks, create a date-stamped subfolder under [`agent_wiki/audit/simulation_run_audits/`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/simulation_run_audits/index.md) (`DD-MM-YYYY_<run_name>/`), save diagnostic artifacts, author `audit.md` (verdict, proposal, relevant files), and register it in `index.md`.

## Topic Navigation Map

| Topic | Relevant Wiki Document |
| :--- | :--- |
| **Wiki Overview & Reading Order** | [`agent_wiki/README.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/README.md) |
| **Active vs Dormant Engines** | [`agent_wiki/architecture/source_of_truth_map.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/architecture/source_of_truth_map.md) |
| **Simulation Run Audits** | [`agent_wiki/audit/simulation_run_audits/index.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/simulation_run_audits/index.md) |
| **Common Pitfalls & Gotchas** | [`agent_wiki/development/common_pitfalls.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/development/common_pitfalls.md) |
| **Safe Modification Rules** | [`agent_wiki/development/safe_modification_rules.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/development/safe_modification_rules.md) |
| **Risk Classification** | [`agent_wiki/development/change_safety_matrix.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/development/change_safety_matrix.md) |
| **Physics, EOR & Reservoir Models** | [`agent_wiki/physics/`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/physics/) |
| **Unit Definitions & Conversions** | [`agent_wiki/data/units.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/data/units.md) |
| **Known Limitations & Envelopes** | [`agent_wiki/validation/known_limitations.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/validation/known_limitations.md) |
| **Agent Skills Knowledge Base** | [`agent_wiki/development/agent_skills.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/development/agent_skills.md) |
