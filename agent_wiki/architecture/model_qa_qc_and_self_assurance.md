# Subsurface Model Quality Assurance (QA/QC) & Dynamic Self-Assurance Engine

This document details the architectural specification and implementation plan for the **Subsurface Model Quality Assurance (QA/QC) & Dynamic Self-Assurance Engine** within the CO2 EOR Optimizer.

---

## 1. Context and Architectural Rationale

### 1.1 Decommissioning Aerospace Terminology
The legacy interface referred to model verification as "Pre-Flight Audit". In petroleum reservoir simulation, geological carbon storage (ISO 27914), and EPA Class VI Underground Injection Control (UIC) regulations, aerospace metaphors are inappropriate. All terminology is standardized to:
- **Model Quality Assurance (QA/QC) & Pre-Simulation Audit**
- **Subsurface Model Feasibility Gatekeeper**
- **Process-to-Simulator Dynamic Self-Assurance**

### 1.2 Motivation for Automated Self-Assurance (Self-QC)
Traditional simulation pre-checks only evaluate static parameter bounds (e.g. porosity in [0, 1]). However, subtle non-linear physical interactions often cause simulation or optimization failure:
- Injection bottomhole pressure exceeding the EPA Class VI fracture limit ($P_{\text{sandface}} \ge 0.90 \times P_{\text{frac}}$) under dynamic rate schedules, triggering simulator shut-in.
- Closed-loop carbon mass balance violations ($M_{\text{recycled}} > M_{\text{produced}}$ or unclosed continuity).
- Severe voidage replacement imbalance ($\text{VRR} \ll 0.8$ or $\text{VRR} \gg 1.2$) causing reservoir pressure to drop below bubble point ($P_b$) or surge to fracturing.
- Premature viscous fingering and gas channeling ($t_{\text{breakthrough}} < 6$ months) caused by unfavorable mobility contrast.
- Transient fault reactivation during injection cycles ($T_s(t) > 0.60$ or $\Delta CFS > 0$).

The Dynamic Self-Assurance Engine allows the user to click a single diagnostic action. The system executes a rapid baseline simulation using the active intermediate physics-informed surrogate simulator (`core/engine_surrogate`), analyses the resulting time series, predicts **what is physically sound versus what is anomalous**, calculates a **Model Assurance Health Score (0-100%)**, and provides actionable remediation guidance before optimization runs.

---

## 2. Multi-Domain Verification Architecture

The QA/QC system evaluates two coupled tiers:

```
+---------------------------------------------------------------------------------------------------+
|                           MODEL QUALITY ASSURANCE & SELF-CONTROL ENGINE                           |
+---------------------------------------------------------------------------------------------------+
                                                  |
                 +--------------------------------+-------------------------------+
                 |                                                                |
                 v                                                                v
   [TIER 1: STATIC INVARIANT GATE]                              [TIER 2: DYNAMIC SELF-ASSURANCE RUN]
   - OOIP & Pore Volume Conservation                            - Rapid Baseline Simulation Run (Surrogate)
   - PVT Thermodynamic Consistency (Pres vs MMP)                - Numerical Stability & Monotonicity
   - SCAL Corey Saturation Windows (Swc+Sorw < 1)               - Dynamic Closed-Loop Mass Balance (< 0.1%)
   - Well Deliverability & Drive Balance (P/I ratio)            - Voidage Replacement Ratio VRR(t) in [0.8, 1.2]
   - Caprock Seal Capacity & Initial Fault Slip                 - Injector Choking & BHP Sandface UIC Ceiling
                                                                - Viscous Fingering & Breakthrough Timing
                                                                - Transient Fault Reactivation Ts(t) & Delta CFS
                                                  |
                                                  v
                         +------------------------------------------------+
                         |       PREDICTIVE DIAGNOSTICS & HEALTH SCORING   |
                         |  - Composite Health Score (0 - 100%)           |
                         |  - Diagnosis: What is Wrong vs What is Good    |
                         |  - Prescriptive Engineering Recommendations    |
                         |  - One-Click Auto-Remediation Actions          |
                         +------------------------------------------------+
```

### 2.1 Tier 1: Static Parameter & Boundary Checks
1. **Volumetric & Grid Framework**:
   - Original Oil in Place ($OOIP = 7758 A h \phi (1 - S_{wi}) / B_{oi}$).
   - Discretization cell count and grid aspect ratios ($\Delta x, \Delta y, \Delta z$).
   - Dykstra-Parsons coefficient ($0.0 \le V_{DP} < 0.95$).
2. **PVT & Thermodynamics**:
   - Miscibility drive evaluation ($P_{\text{initial}} - MMP$).
   - Viscosity ratio ($\mu_o / \mu_{\text{CO2}}$) and gas formation volume factor ($B_g$).
3. **Corey Relative Permeability Endpoints**:
   - Mobile saturation span ($S_{wc} + S_{orw} < 1.0$).
   - Critical gas saturation ($S_{gc} \ge 0.0$).
   - Curvature exponents ($n_o, n_w, n_g \ge 1.0$).
4. **Wellbore Completion & Hydraulics**:
   - Drive balance: at least one injector and one producer when well count > 0.
   - Perforations contained within reservoir net pay interval.
5. **Geomechanics & Structural Containment**:
   - Caprock capillary entry pressure ($P_{ce}$) vs buoyant CO2 column ($H_{\text{max}} = P_{ce} / (\Delta \rho \cdot g)$).
   - Mohr-Coulomb slip tendency on all active faults ($T_s = \tau / \sigma_n' < \mu$).

### 2.2 Tier 2: Dynamic Self-Assurance Run & Anomaly Detection
When triggered, `ModelSelfAssuranceEngine` executes `SurrogateEngine.evaluate_scenario()` and performs automated anomaly detection:
1. **Numerical Stability**:
   - Verifies monotonicity of cumulative streams and absence of non-physical oscillations.
2. **Closed-Loop Mass Balance**:
   - Evaluates fluid mass conservation: $|\Delta M| / M_0 < 10^{-3}$.
   - Verifies closed carbon loop: $\text{Gross Injected} = \text{Purchased} + \text{Recycled} = \text{Stored} + \text{Produced} + \text{Leakage}$.
3. **Voidage Replacement Ratio (VRR) & Pressure Drift**:
   - Tracks dynamic average reservoir pressure $P(t)$.
   - Flags depletion below bubble point pressure ($P(t) < P_b$) or runaway pressure accumulation.
   - Target range: $0.8 \le \text{VRR}(t) \le 1.2$.
4. **Well Injectivity Choking**:
   - Checks if injector BHP hits the EPA Class VI sandface limit ($0.90 \times P_{\text{frac}}$).
   - If throttling occurs, flags lost sweep efficiency and calculates recommended rate derating.
5. **Viscous Fingering & Breakthrough Timing**:
   - Evaluates Koval factor $K = H_k \cdot E_{\text{eff}}$.
   - Predicts breakthrough time $t_{\text{bt}}$ (months); flags channeling ($t_{\text{bt}} < 6$ months) or stagnant displacement ($t_{\text{bt}} > 36$ months).
6. **Transient Fault Slip Evolution**:
   - Evaluates time-dependent slip tendency $T_s(t)$ and Coulomb stress transfer $\Delta CFS(t)$ during peak pore pressure periods.
7. **SPE Analog Recovery & Utilization Feasibility**:
   - Verifies incremental Recovery Factor ($RF \in [5\%, 35\%]$) and Net CO2 Utilization ($3-15 \text{ MSCF/STB}$) against empirical analogs.

---

## 3. UI/UX Architecture & Layout

### 3.1 Embedded Workstation View (View 5)
Located in `ui/workbench/components/visual_audit_gate_widget.py` (to be modernized as `ModelQualityAssuranceWidget`):
1. **Header Banner**:
   - Displays Model Assurance Health Score (0-100%).
   - Status badge: `[OPTIMIZATION READY]`, `[CONDITIONALLY ACCEPTABLE]`, or `[OPTIMIZATION BLOCKED]`.
   - Action buttons:
     - `Run Static QA/QC Audit`: Instant parameter validation.
     - `Execute Dynamic Self-Assurance Run`: Triggers diagnostic simulation.
     - `Apply Auto-Remediation`: Automatically fixes detected parameter contradictions.
     - `Approve & Confirm Model`: Locks validated state and signals readiness to optimization.
2. **Tabbed Inspection Workspace**:
   - **Tab 1: Multi-Domain Checklist (Table)**: Searchable matrix of all verification parameters across domains.
   - **Tab 2: Dynamic Self-Assurance Diagnostics (4-Panel Matplotlib Canvas)**:
     - Panel 1: Pressure Trajectory $P(t)$ vs Bubble Point $P_b$ and Class VI UIC Ceiling.
     - Panel 2: 4-Stream Production & Injection Profiles (Oil, Gas, Water Cut, Injected CO2).
     - Panel 3: Dynamic Voidage Replacement Ratio $\text{VRR}(t)$ and Mass Conservation Error.
     - Panel 4: Dynamic Fault Slip Tendency $T_s(t)$ on all active faults vs Byerlee Limit (0.60).
   - **Tab 3: Prescriptive Findings & Recommendations**:
     - Categorized findings detailing what is good, what is wrong, and recommended adjustments.

### 3.2 Dynamic Model Tree Integration
In `ui/workbench/components/model_tree_widget.py`, under `Surveillance & Model Quality Assurance`:
- `Model QA/QC Audit & Feasibility Gate (Table)`: Navigates to Tab 1.
- `Dynamic Self-Assurance & Health Diagnostic (Interactive Canvas)`: Navigates to Tab 2.
- `Multi-Domain Surveillance Workstation (Dashboard)`: Navigates to View 6 (`ModelEvaluationDashboard`).

---

## 4. Tracking Development: Data Management Tab

### 4.1 Current Status of Data Management Tab
The Data Management tab (`ui/data_management_widget.py`) is currently in active development and pending:
1. **QA/QC Modernization**:
   - Replace legacy "Pre-Flight Physical Audit" button with the unified `Model QA/QC Audit` action routing to the new self-assurance engine.
   - Remove emojis from audit status displays.
2. **UI/UX Updates**:
   - Align styling with standard clean Qt design (removing inconsistent stylesheet declarations).
   - Synchronize parameter changes with the Subsurface Workbench.
3. **Bugfixes & Robustness**:
   - Ensure grid permeability parsing safely supports scalar, 1D flattened, and 3D arrays (`PERMX.flat[0]`).
   - Clean handling of missing or optional PVT composition tables without unhandled exceptions.

---

## 5. Visual Artifacts & Screenshots

The following interface states have been captured and verified:

1. **Fault & Caprock Manager Dialog (Tab 1: Structural Fault System)**:
   - Location: `scratch/bench_fault_caprock_dialog_tab1.png`
   - Features: Multi-fault table with stability indicators, strike/dip/throw inputs, Mohr-Coulomb slip tendency calculations, and architectural presets.

2. **Fault & Caprock Manager Dialog (Tab 2: Inter-Fault Stress Transfer)**:
   - Location: `scratch/bench_fault_caprock_dialog_tab2.png`
   - Features: Multi-fault Coulomb stress transfer matrix ($\Delta CFS$) and 2-panel dislocation stress decay canvas.

3. **Fault & Caprock Manager Dialog (Tab 3: Caprock Confining Stratigraphy)**:
   - Location: `scratch/bench_fault_caprock_dialog_tab3.png`
   - Features: Confining layer table, lithology selection, and live EPA Class VI containment cards.

4. **Subsurface Workbench - Fault System Table View**:
   - Location: `scratch/bench_fault_system_table_view.png`
   - Features: Master fault inventory rendered full-screen in the middle data viewer with model tree navigation.

5. **Subsurface Workbench - Inter-Fault Stress Graph View**:
   - Location: `scratch/bench_fault_inter_stress_view.png`
   - Features: 2D spatial Coulomb stress perturbation halo and distance attenuation plot.

6. **Model Quality Assurance (QA/QC) Gate (Current Baseline View)**:
   - Location: `scratch/bench_current_qa_qc_gate.png`
   - Features: Diagnostic cards (OOIP, Miscibility, Wells, EPA Ceiling) and multi-domain checklist table.

7. **Data Management Tab (In Development)**:
   - Location: `scratch/bench_data_management_tab.png`
   - Features: Current data ingestion layout, pending QA update and UI/UX polish.

---

## 6. Implementation Phasing & Verification

### Phase 1: Terminology Cleanup
- Update all occurrences of "Pre-Flight" to "Model QA/QC" or "Model Assurance".
- Clean out emojis across all dialogs, labels, and banners.

### Phase 2: Core Diagnostic Engine
- Implement `core/diagnostics/model_self_assurance.py` (`ModelSelfAssuranceEngine`).
- Connect to `SurrogateEngine.evaluate_scenario()` for rapid test simulations.
- Implement automated anomaly detection algorithms for mass conservation, pressure drift, well choking, and fault reactivation.

### Phase 3: Modernized QA/QC Workstation Widget
- Upgrade `ui/workbench/components/visual_audit_gate_widget.py` with dynamic simulation runner, 4-panel diagnostic canvas, and auto-fix capabilities.

### Phase 4: Integration & Synchronization
- Update `ui/workbench/components/model_tree_widget.py` and `ui/workbench/subsurface_workbench_widget.py`.
- Ensure project save/load (`.tphd`) state persistence.

### Phase 5: Verification & Testing
- Validate with `pytest tests/test_project_save_load.py -v`.
- Execute automated self-assurance verification script.
