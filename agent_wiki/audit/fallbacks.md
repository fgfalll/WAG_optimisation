# Fallback & Exception-Swallowing Audit

> [!NOTE]
> This document catalogs **only active, open items**. For resolved flaws, historical post-mortems, and verification status, consult the [**Resolved Issues & Defect Resolution Archive**](resolved_issues.md).

## 1. Executive Summary

An AST audit across active Python modules originally identified **2,459 total fallbacks**, comprising:
- **1,710** dictionary `.get(key, default)` calls with implicit fallbacks
- **629** function parameter default values
- **96** `try...except` blocks that catch exceptions and return fallback values or constants
- **24** `try...except` blocks that silently pass (`pass`) without logging or re-raising
- **800** bounding/clipping calls (`np.clip`, `max`, `min`, `clamp`)

Following the strict eradication initiative:
- **All silent `except Exception: pass` blocks in the core mathematical and solver modules were completely eradicated (0 remaining)**.
- **Specific exception types are now caught** (e.g. `FloatingPointError`, `ZeroDivisionError`, `ValueError`, `RuntimeError`), and **state variables (Pressure, Saturation, Temperature) are explicitly logged**.
- **Class E artificial result-producing fallbacks were deleted** (reduced to 0 in active optimization paths). Unphysical chromosomes receive full mathematical failure penalties (`FAILURE_PENALTY`, $-10^{12}$) or `NaN` to naturally kill off unviable genetic lines.
- **Plotly and visualization libraries are enforced as hard mandatory requirements**, eradicating silent dummy mock plot swallowers.

---

## 2. Fallback Classification Taxonomy

Fallbacks in this repository fall into five distinct categories:

| Category | Description | Count | Severity | Risk Level | Status |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **Class A: Legitimate Numerical Safety** | Division-by-zero protection (`+ EPSILON`), preventing negative saturations | ~450 | LOW | Preserves physical meaning without distorting trends. | Active |
| **Class B: Legitimate Engineering Defaults** | Standard reservoir compressibility $4 \times 10^{-6}\text{ psi}^{-1}$ if unmeasured | ~600 | MEDIUM | Documented industry baseline; acceptable if flagged. | Active |

---

## 3. High-Priority Fallback Audit Status

All high-risk **Class C** (dependency mock swallowers), **Class D** (silent calculation exceptions), and **Class E** (artificial result-producing fallbacks) have been eliminated from active production paths. The active codebase retains exclusively:
- **Class A numerical safety guards** (e.g. `+ 1e-12` division protection and physical saturation clamps).
- **Class B engineering defaults** (documented default compressibility and standard water salinity).

Full historical remediation records and post-mortems for eradicated fallbacks are archived in [**`agent_wiki/audit/resolved_issues.md`**](resolved_issues.md).

