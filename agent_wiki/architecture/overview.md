# System Architecture Overview

## 1. Executive Architecture Summary

The **CO₂ EOR Optimizer** is a scientific application built in Python for optimizing Carbon Dioxide Enhanced Oil Recovery (CO₂-EOR) operations alongside Carbon Capture, Utilization, and Storage (CCUS).

It combines:
1. **PyQt6 Desktop GUI**: Interactive management of reservoir properties, well schedules, injection schemes, and optimization runs (`main.py`, `ui/`).
2. **Optimization Orchestrator**: Heuristic and metaheuristic algorithms (`core/optimisation_engine.py`):
   - Genetic Algorithm (PyGAD)
   - Bayesian Optimization (`bayesian-optimization`)
   - Particle Swarm Optimization (`pyswarms`)
   - Differential Evolution (`scipy.optimize.differential_evolution`)
3. **Simulation Engine**: Single, direct physics-informed simulation engine (`core/engine_surrogate`). All legacy engines (`compositional_engine`, `unified_engine`, `engine_simple`) and `EngineFactory` are deprecated and relocated to `deprecated/` to eliminate confusion.
4. **Economic & Environmental Objectives**: Engine-owned cash flow modeling (NPV) and carbon accounting, coupled with EPA Class VI geomechanical containment checks and environmental leakage penalties (`core/objectives/wrapper.py`).

---

## 2. High-Level Subsystem Diagram

```mermaid
graph TD
    A[main.py: App Entry Point] --> B[ui/main_window.py: PyQt6 Interface]
    B --> C[ui/optimization_widget.py]
    B --> D[ui/data_management_widget.py]
    
    C --> E[core/optimisation_engine.py: OptimizationEngine]
    
    E -->|DIRECT INSTANTIATION| G[core/engine_surrogate: Primary Physics-Informed Simulator]
    
    subgraph Deprecated Subsystems (deprecated/)
        H[deprecated/core/compositional_engine: 1D Finite Volume Solver]
        I[deprecated/core/unified_engine: Detailed 3D Grid & Solvers]
        K[deprecated/core/engine_simple: Material Balance Engine]
        F[deprecated/core/engine_factory.py: EngineFactory]
    end
    
    G --> N[core/engine_surrogate/analytical_models.py: Koval & PhD Hybrid Recovery]
    G --> O[core/engine_surrogate/profile_generator_fast.py: FastProfileGenerator & Composite IPR]
    G --> P[core/engine_surrogate/pvt_state.py: SolventExtendedPVTEngine & PR-EOS]
    G --> S_GEO[core/engine_surrogate/geomechanics_fault.py: GeomechanicsFaultModel]
    
    G --> Q[Results: RF, Cumulative Oil, Profiles, NPV, Storage Metrics]
    Q --> E
    E --> R[core/objectives/wrapper.py: Objective Evaluation]
    R --> S[Class VI Geomechanical Penalty & Fitness Evaluation]
```

---

## 3. Simulation Engine Reality

The codebase has been refactored from a multi-engine abstraction to a **clean, single-engine architecture**:

### A. `core/engine_surrogate` (ACTIVE - Intermediate Physics-Informed Simulator)
- **Status**: **100% Active Production Engine**.
- **Nature**: An **intermediate, physics-informed reduced-order reservoir simulator** that sits between full 3D compositional grid solvers and simple 0D tank models. It functions as a general reservoir simulator with specialized CO₂ EOR domain physics (solvent dissolution, swelling, viscosity reduction, viscous fingering, and geomechanics), NOT an empirical curve-fit or machine learning proxy.
- **Implementation**: [SurrogateEngineWrapper](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_engine.py) directly wraps [SurrogateEngine](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_engine.py).
- **Physical Approach**:
  - **Thermodynamic State & PVT**: `core/engine_surrogate/pvt_state.py` (`SolventExtendedPVTEngine`) decouples hydrodynamic saturation ($S_g$) from solvent composition ($x_{\text{CO2}}$). Pure CO₂ density is evaluated via analytical cubic Peng-Robinson EOS. Tracks oil swelling $S_F(P, x_{\text{CO2}})$, viscosity reduction $\mu_o(P, x_{\text{CO2}})$, mixture gas properties, and surface stage separation.
  - **Authentic Viscous Fingering**: Koval (1963) and Todd-Longstaff (1972) with monotonic mobility acceleration ($\partial f_g / \partial M > 0$) where effective Koval factor $K = H_k \cdot E_{\text{eff}}$.
  - **Geomechanics & Containment**: `core/engine_surrogate/geomechanics_fault.py` (`GeomechanicsFaultModel`) simulates in-situ stress paths ($\Delta \sigma_h = \gamma_h \Delta P$), Mohr-Coulomb fault slip tendency ($T_s$), caprock shear/tensile safety margins, and dynamic geological leakage.
  - **Deliverability & Inflow**: Composite Vogel-Darcy IPR clamped to physical reservoir limits.
  - **Pressure & Material Balance**: Dual-pressure Voidage Replacement Ratio (VRR) tracking local injection vs production volumes and deliverability-coupled implicit material balance ($dP = (q_{\text{net,IPR}} \cdot \Delta t) / (V_p \cdot c_t + J_{\text{eff}} \cdot \Delta t)$).
  - **Standardized 4-Stream Delivery**: Delivers comprehensive daily, monthly, and annual profiles for (1) Crude Oil, (2) Natural Gas, (3) Water / Formation Brine, and (4) Injection Agent (CO₂ & WAG water, with compressor capacity constraints).
- **Performance**: Extremely fast (~5–15 ms per scenario evaluation), enabling thousands of optimization iterations with full physical integrity.

### B. Legacy Engines & Factory (DEPRECATED - Moved to `deprecated/`)
- `deprecated/core/engine_factory.py`: Removed to eliminate confusing layers of indirection.
- `deprecated/core/compositional_engine/`: 1D finite-volume solver moved to `deprecated/`.
- `deprecated/core/unified_engine/`: 3D grid solver with scientific flaws (SCI-FLAW-04, SCI-FLAW-08) moved to `deprecated/`.
- `deprecated/core/engine_simple/`: 0D tank model moved to `deprecated/`.

---

## 4. Subsystem Coupling & Data Flow

| From Subsystem | To Subsystem | Transferred Data | Invariant Rule |
| :--- | :--- | :--- | :--- |
| `ui/optimization_widget.py` | `core/optimisation_engine.py` | `OptimizationConfig`, `ReservoirData` | Units must be Field (psi, STB, MSCF, ft). |
| `core/optimisation_engine.py` | `core/engine_surrogate/surrogate_engine.py` | `EORParameters`, `ReservoirData`, `OperationalParameters`, `EconomicParameters` | Directly instantiated `SurrogateEngineWrapper`. Never scale recovery by well count. |
| `core/engine_surrogate/` | `core/objectives/wrapper.py` | `SimulationResults` dict containing `npv`, `rf`, `storage_efficiency`, profiles | Wrapper is a dumb consumer; NPV and storage are computed in-engine. |
| `core/objectives/wrapper.py` | `core/optimisation_engine.py` | Multi-objective fitness score, constraint penalties | Unphysical candidates receive $-10^{12}$ or `NaN` to prevent survival. |
