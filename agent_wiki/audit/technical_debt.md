# Technical Debt & Software Engineering Audit

> [!NOTE]
> This document catalogs **only active, open items**. For resolved flaws, historical post-mortems, and verification status, consult the [**Resolved Issues & Defect Resolution Archive**](resolved_issues.md).

## 1. Overview of Technical Debt

The codebase shows signs of rapid iterative prototyping, multiple architectural pivots (from 3D numerical grids to fast analytical surrogates for PhD research), and legacy preservation.

This document outlines the major software engineering defects, current test health, and identified runtime risks in UI modules.

---

## 2. Test Suite Status: FULLY PASSING (299 Passed, 23 Skipped, 0 Failed)

The automated test suite in `tests/` currently achieves a **100% pass rate**:
- **299 passed**, **23 skipped**, **0 failed** (executed in 129.06s).
- All prior unit fixture regressions (`economic_params is required`, `econ_params` unbound variable, NumPy 2.0 `np.trapz` removal, and ambiguous array truth evaluation) have been resolved.
- SFT-02 (`MagicMock` OPEX broadcasting) and SFT-03 (`OptimizationWidget.set_engine` alias) resolved, achieving 0 test failures.

---

## 3. Active Technical Debt Inventory

| Category | Component / Module | Scope & Impact | Risk Level | Proposed Architectural Refactoring |
| :--- | :--- | :--- | :---: | :--- |
| **Monolithic Modules** | `core/optimisation_engine.py` (~3,200 LOC) | Houses optimizer algorithms (GA, BO, PSO, DE), profile post-processing, and fallback handlers in a single monolithic orchestrator. | **MEDIUM** | Decompose into modular strategy patterns (`OptimizerBase`, `GeneticAlgorithmStrategy`, `BayesianStrategy`). |
| **Monolithic UI Views** | `ui/main_window.py`, `ui/optimization_widget.py` (~1,800 LOC each) | Tightly couples Qt GUI layout construction, worker thread management, and PDF/HTML report compilation. | **MEDIUM** | Extract report generation to `services/reporting/` and worker orchestration to dedicated presenters. |
| **Type Annotation Gaps** | Surrogate & Optimizer Boundaries | Extensive reliance on `Dict[str, Any]` and raw `**kwargs` dictionaries rather than typed dataclasses (`SimulationConfig`, `EconomicParameters`). | **LOW** | Enforce strict typing across `SurrogateEngineWrapper` and `OptimizationEngine` inputs. |
| **Dormant Package Isolation** | `compositional_engine/`, `unified_engine/`, `engine_simple/` | Dormant historical engines remain in tree (~10,000 LOC total) despite 100% active routing to `core/engine_surrogate/`. | **LOW** | Maintain strict deprecation barriers; prevent inadvertent imports into active modules. |
| **Deprecation Warning Noise** | Test Suite Execution | Scientific library warnings (e.g. future deprecations in SciPy and Pandas) emitted during test runs. | **LOW** | Audit and modernize library call patterns across analytical and reporting modules. |
