# Shared Earth Reservoir Workstation Architecture (v4.0)

> [!NOTE]
> This document is the definitive architectural specification for the **Integrated Shared Earth Model, Subsurface Visualizers, and Multi-Domain Diagnostic Workstation** in the CO₂ EOR Optimizer repository.

---

## 1. Architectural Paradigm: The 5-Pillar Shared Earth Model

In field-scale reservoir engineering and numerical simulation, the reservoir cannot be represented as isolated input textboxes. Modifying a parameter in one physical domain dynamically cascades through dependent physics across the model:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   SHARED EARTH RESERVOIR MODEL                                   │
├──────────────────────────┬──────────────────────────┬────────────────────────────────────────────┤
│ Static Grid & Geology    │ Fluids & SCAL            │ Well Architecture                          │
│ • Corner-point / Unstruct│ • Peng-Robinson EOS      │ • 3D Trajectories & Completions            │
│ • Facies, φ, k_x,k_y,k_z │ • Stone I 3-Phase RelPerm│ • Peaceman Anisotropic Index (WI_horiz)    │
│ • Geostatistics (SGS)    │ • Carlson Gas Hysteresis │ • Vogel-Darcy IPR & Lift Curves (VFP)      │
├──────────────────────────┴──────────────────────────┴────────────────────────────────────────────┤
│ Geomechanics, Caprock & Fault Integrity                                                          │
│ • 3D Fault Surface Mesh & Throw Offsets (SGR) | Poroelastic Stress Paths (σ_h, α)                 │
│ • Thermo-Elastic Cooling (ΔT) & Coulomb Failure Stress (ΔCFS) | Capillary Entry Pressure (P_ce)  │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Dynamic Flow Engine (Coupled 3D Surrogate Engine with Pseudo-Functions / Compositional Overlay)  │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### The 5 Coupled Pillars:
1. **Static Grid & Stratigraphy Framework**:
   - 3D grid container supporting structured IJK / corner-point grids initialized via Sequential Gaussian Simulation (SGS).
   - Directional absolute permeabilities ($k_x, k_y, k_z$), porosity ($\phi$), Net-to-Gross ($NTG$), and structural dip ($\theta_{dip}$).
   - Thickness-weighted average permeability: $k_{avg} = \sum k_i h_i / \sum h_i$.
   - Dykstra-Parsons heterogeneity index: $V_{DP} = (k_{50} - k_{84.1}) / k_{50}$, calibrating Koval heterogeneity factor $H_K = 10^{V_{DP} / (1 - V_{DP})}$.
2. **Fluid Thermodynamics & Multiphase SCAL**:
   - Multi-component Peng-Robinson EOS with volume translation and $k_{ij}$ binary interaction parameter matrix.
   - Rigorous 1D cell-to-cell slim-tube simulation solver determining dynamic Multiple-Contact Miscibility (MCM MMP) for impure gas streams ($\text{N}_2, \text{CH}_4, \text{H}_2\text{S}$) and heavy oils.
   - Modified Stone I 3-phase relative permeability model: $k_{ro} = \frac{S_o^*}{(1 - S_w^*)(1 - S_g^*)} \cdot k_{row}(S_w) \cdot k_{rog}(S_g)$.
   - Carlson (1981) / Land (1968) trapped gas saturation hysteresis during WAG water cycles: $S_{gt} = \frac{S_{gi}}{1 + C \cdot S_{gi}}$, where $C = \frac{1}{S_{gr}^{\text{max}}} - \frac{1}{1 - S_{wc}}$.
3. **Well Architecture, Completions & Inflow Hydraulics**:
   - True 3D well trajectories: Vertical, Horizontal, Deviated S-Curve, and Fishbone Multilaterals.
   - Peaceman (1983) anisotropic horizontal well index: $WI_{horiz} = \frac{2\pi \sqrt{k_y k_z} \cdot L_{lat}}{\ln(r_{o,h} / r_w) + s}$.
   - Composite Vogel-Darcy inflow performance relationship clamped to artificial lift capacity ($\le 1,000\text{ BOPD}$).
   - Well-driven pattern assignment (Huff-n-Puff for 1 well; WAG, SWAG, Continuous for $\ge 2$ wells with vertical completion overlap $\Omega_{overlap} \ge 20\%$).
4. **Geomechanics, Caprock & Fault Integrity**:
   - Near-wellbore thermo-poroelastic stress path accounting for cold $\text{CO}_2$ injection: $\Delta \sigma_h(t) = \gamma_h \Delta P(t) + \frac{E \alpha_{th}}{1 - \nu} \Delta T$ ($\Delta T = T_{inj} - T_{res} < 0$).
   - Caprock capillary entry pressure sealing assessment: $P_c = P_{\text{CO}_2} - P_w \ge P_{ce} = \frac{2 \gamma \cos \theta}{r_{throat}} \propto \sqrt{\frac{\phi_{cap}}{k_{cap}}}$.
   - Dynamic fault slip reactivation via Coulomb Failure Stress Change: $\Delta \text{CFS} = \Delta \tau - \mu_f (\Delta \sigma_n - \alpha \Delta P)$.
   - Shale Gouge Ratio (Yielding & Bretan 1997): $\text{SGR} = \frac{\sum (V_{shale} \Delta z)}{\text{Fault Throw}} \times 100\%$ ($\ge 30\%$ sealing).
5. **Dynamic Flow Engine & Material Balance**:
   - Dynamic voidage replacement tracking: $\text{VRR}(t) = \frac{B_w Q_{w,inj}(t) + B_{\text{CO}_2}(t) Q_{\text{CO}_2,inj}(t)}{B_o(t) Q_o(t) + B_w Q_w(t) + B_g(t) Q_{g,total}(t)}$.
   - Dynamic tank material balance ODE updating pore pressure $\bar{P}(t)$ and dynamic miscibility state $M_f(t)$: $\frac{d\bar{P}}{dt} = \frac{q_{inj,RB}(t) - q_{prod,RB}(t)}{V_p \cdot c_t + J_{eff} \cdot \Delta t}$.

---

## 2. Centralized In-Memory State Manager (`core/reservoir_state_manager.py`)

To eliminate parameter clobbering, race conditions, and synchronization lag, a unified state container is established:

```python
# Location: core/reservoir_state_manager.py
from PyQt6.QtCore import QObject, pyqtSignal

class ReservoirStateManager(QObject):
    """
    Centralized in-memory Shared Earth Model state container.
    Maintains dynamic coupling across Geology, PVT, Wells, Geomechanics, and Simulation.
    Emits granular Qt signals when any domain state changes.
    """
    state_modified = pyqtSignal(str) # Domain name
    geology_updated = pyqtSignal()
    pvt_updated = pyqtSignal()
    wells_updated = pyqtSignal()
    geomechanics_updated = pyqtSignal()
    simulation_ready = pyqtSignal(bool)
```

### Dynamic Coupling Invariants:
1. **Stratigraphy $\to$ Deliverability**: Changing layer thickness or permeability in the layer table immediately recalculates $k_{avg}$, $V_{DP}$, and all well indices ($WI, J$).
2. **PVT $\to$ Containment & Recovery**: Changing fluid composition or API immediately re-evaluates PR-EOS flash, updates $P_{MMP}$, recalculates fluid densities ($\Delta \rho$), and updates caprock capillary buoyant pressure ($\Delta P_{buoyancy}$).
3. **Wells $\to$ Pattern Conformance**: Moving a well in 3D space immediately recomputes inter-well transmissibility $T_{ij}$, vertical perforation overlap $\Omega_{overlap}$, and updates the well pattern scheme.
4. **Injection Pressure $\to$ Geomechanics**: Changing injection rate or target bottomhole pressure immediately recalculates sandface thermo-poroelastic stress ($\sigma_h(t)$), Coulomb stress change ($\Delta \text{CFS}$), and the Mohr-Coulomb failure metric ($F$-value).

---

## 3. Single Source of Truth (SSOT) Domain Separation

| Domain | Authoritative Owner | Non-Destructive Consumer |
| :--- | :--- | :--- |
| **Reservoir Geometry & Layer Heterogeneity** | `DataManagementWidget` (Reservoir Tab) | `SurrogateEngineWrapper`, `OptimizationEngine` |
| **Geomechanics, Faults & Capillary Containment** | `DataManagementWidget` (Geomechanics Section) | `GeomechanicsFaultModel`, `SurrogateEngineWrapper` |
| **3D Geostatistics & Variograms** | `DataManagementWidget` (Geostatistics Section) | `create_geostatistical_grid`, `SurrogateEngineWrapper` |
| **PVT, EOS Flash & Slim-Tube MMP** | `DataManagementWidget` (PVT Tab) | `SolventExtendedPVTEngine`, `SurrogateEngineWrapper` |
| **3-Phase Rel-Perm & Hysteresis** | `DataManagementWidget` (Rel-Perm Section) | `FastProfileGenerator`, `SurrogateEngineWrapper` |
| **Wells, 3D Trajectories & Completions** | `DataManagementWidget` (Wells Tab) | `FastProfileGenerator`, `SurrogateEngineWrapper` |
| **Pattern Injection Schemes** | `DataManagementWidget` (Wells Tab) | `FastProfileGenerator`, `SurrogateEngineWrapper` |
| **Surrogate Calibration Multipliers** | `DataManagementWidget` (Surrogate Tuning Tab) | `SurrogateEngineWrapper`, `OptimizationEngine` |
| **Macro Economics (Pricing, Base OPEX)** | `ConfigWidget` (Economics Tab) | `DataManagementWidget` (for screening NPV), `OptimizationEngine` |
| **Project Schedules & Operational Limits** | `ConfigWidget` (Operations Tab) | `DataManagementWidget` (for lifetime), `OptimizationEngine` |
| **Algorithm Search Hyperparameters** | `ConfigWidget` / `OptimizationWidget` | `OptimizationEngine` |

---

## 4. Subsurface Visualizers & Pre-Flight Confirmation Framework (Workstream 1.4)

1. **`GeostatisticsVisualizerWidget`** (`ui/widgets/geostatistics_visualizer_widget.py`):
   - **1D Semi-Variogram Fit Canvas**: Plots experimental spatial covariance lags ($\gamma(h)$) against theoretical Spherical, Exponential, and Gaussian curves with real-time sliders for Nugget ($c_0$), Sill ($c$), and Range ($a$).
   - **Fast 2D/3D SGS Realization Previewer**: Calls `core/geology/geostatistical_modeling.py` (GSTools) to generate a $50 \times 50 \times 10$ raster grid in $<200\text{ ms}$, rendering permeability/porosity heatmaps and histograms.
2. **`GeologyCrossSectionWidget`** (`ui/widgets/geology_cross_section_widget.py`):
   - **Interactive IJK Slicing**: Orthogonal cross-section canvas ($X$-$Z, Y$-$Z, X$-$Y$) with sliders confirming structural dip ($\theta_{dip}$), layer thickness variations, and $k_z / k_h$ anisotropy.
   - **Vertical Proportion Curves (VPC)**: Displays the vertical probability distribution of permeable sand ($k > 10\text{ mD}$) vs. impermeable shale barriers across each stratigraphic layer.
3. **`FaultGeometryVisualizerWidget`** (`ui/widgets/fault_geometry_visualizer_widget.py`):
   - **3D Fault Plane Mesh**: Renders 3D fault planes cutting through the reservoir bounding box, showing throw offset and layer juxtaposition.
   - **Shale Gouge Ratio (SGR) Surface Heatmap**: Color-codes fault seal integrity (Red: SGR < 20% leaking conduit; Yellow: 20–30% transitional; Green: > 30% ductile shale smear seal).
   - **Well-Fault Trajectory Overlay**: Displays 3D well trajectories alongside fault planes to verify wellbore-to-fault buffer distances and completion interval clearances.
4. **Unified Visual Data Confirmation Gate ("Shared Earth Check")** (`ui/dialogs/visual_audit_modal.py`):
   - Mandatory 3-tab visual confirmation modal before the engine executes base simulation or optimization:
     - *Tab 1: Heterogeneity & Variograms*
     - *Tab 2: Stratigraphy & Completions*
     - *Tab 3: Geomechanics & Containment*
   - Clicking **"Approve & Confirm Subsurface Model"** verifies, locks inputs, and dispatches the payload to `SurrogateEngineWrapper`.

---

## 5. Multi-Domain Diagnostic & Evaluation Workstation (Workstream 6)

The **`ModelEvaluationDashboard`** (`ui/widgets/model_evaluation_dashboard.py`) provides an exhaustive post-setup diagnostic and surveillance suite organized into five specialized tabs:

1. **Tab 1: Static QC & Geostatistical Surveillance**:
   - Experimental vs Theoretical Semi-Variograms along major, minor, and vertical azimuths.
   - Petrophysical CDF distributions and kernel density estimates for $\phi, k_h, k_v / k_h$, and $NTG$ per zone.
   - Interactive 3D cross-sectional fence diagrams showing facies continuity and layer pinch-outs.
2. **Tab 2: SCAL, Phase Behavior & Miscibility Surveillance**:
   - Complete PR-EOS $P$-$T$ Phase Envelopes (bubble/dew curves, cricondenbar, cricondentherm, critical point $(T_c, P_c)$) overlaying current reservoir state $(T_{res}, P_{res})$.
   - Slim-Tube Oil Recovery vs Pressure Break-Over Curve ($1.2\text{ PVI}$) determining dynamic MCM MMP.
   - Stone I 3-Phase Rel-Perm Ternary Surfaces with Carlson scanning loops.
3. **Tab 3: Dynamic Production & Material Balance Surveillance**:
   - Rate Transient Analysis (RTA) Log-Log Bourdet Derivative Diagnostic ($\Delta P / q$ vs $t_{mb}$, $d(\Delta P / q)/d\ln t_{mb}$).
   - Flow regime identification (linear, bilinear, radial, boundary-dominated).
   - Loss-ratio decline diagnostics: $D(t) = -\frac{1}{q}\frac{dq}{dt}$ and $b(t) = \frac{d}{dt}(1/D(t))$.
   - Volumetric Material Balance Error ($\text{MBErr} \le 0.1\%$) and Voidage Replacement Ratio ($\text{VRR}(t)$).
4. **Tab 4: Containment, Geomechanical Integrity & Risk Surveillance**:
   - Dynamic 3D Mohr-Coulomb failure circles evolution over project life.
   - $F$-value metric tracking distance to failure envelope: $F = \frac{\tau_{crit} - \tau_{max}}{\sigma_m' \sin \phi_f}$.
   - Effective stress path trajectories: $\sigma'_h(t), \sigma'_v(t)$ vs $P_{pore}(t)$ with tensile breakdown and fault slip thresholds.
   - 3D Fault Plane SGR and Coulomb Shear Failure ($\Delta \text{CFS}$) heatmaps.
5. **Tab 5: Pattern Sweep, Streamlines & Flooding Surveillance**:
   - 3D Streamline time-of-flight flow paths: $\tau(\mathbf{x}) = \int \frac{\phi}{v(\mathbf{s})} ds$.
   - Injector-Producer Allocation Factors (IPAF): $f_{ij} = Q_{ij} / \sum Q$.
   - Flooding Efficiency Quadrant Cross-Plots (Offset Oil Produced vs Fluid Injected) to identify channelized injectors.

---

## 6. Rendering Implementation Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ RENDERING ARCHITECTURE FOR SUBSURFACE ELEMENTS                                   │
├────────────────────────────────────────┬─────────────────────────────────────────┤
│ 3D Elements (OpenGL Accelerated)       │ 2D Elements (Publication Quality)       │
│ • Streamlines (GLLinePlotItem)         │ • Semi-Variograms (Matplotlib Canvas)   │
│ • Fault Surfaces (GLMeshItem)          │ • IJK Slices & VPC (Matplotlib Canvas)  │
│ • Well Trajectories (GLLinePlotItem)   │ • RTA Log-Log Bourdet (Matplotlib)      │
│ • Primary Engine: pyqtgraph.opengl     │ • Rel-Perm & Phase Envelopes (Matplotlib│
│ • Fallback Engine: Matplotlib mplot3d  │ • Mohr-Coulomb Circles (Matplotlib)     │
└────────────────────────────────────────┴─────────────────────────────────────────┘
```

1. **3D Elements (`pyqtgraph.opengl`)**:
   - Container: `GLViewWidget`.
   - Streamlines & Well Paths: `GLLinePlotItem` with coordinate arrays $(N, 3)$, batched for performance, with velocity floor $v_{\min} = 10^{-8}\text{ m/d}$ to prevent stagnation singularities.
   - Fault Surface Meshes: `GLMeshItem` with vertex colors mapped to Shale Gouge Ratio (SGR) values.
   - Zero-Dependency Fallback: Matplotlib `mplot3d` (`Axes3D`) automatically used if OpenGL context is unavailable.
2. **2D Elements (Matplotlib `FigureCanvasQTAgg`)**:
   - High-precision, publication-quality 2D scientific plotting for variograms, VPC distributions, 3-phase ternary diagrams, RTA Bourdet derivatives with logarithmic L-spacing smoothing ($L = 0.2-0.4$), and Mohr-Coulomb failure circles.

---

## 7. Master 6-Phase Implementation Roadmap

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Architecture Boundaries, Core Physics, State Manager & Schema v2.0      │
│ - Implement ReservoirStateManager (core/reservoir_state_manager.py)              │
│ - Reconcile Config vs Data Management SSOT boundaries                            │
│ - Update ReservoirData JSON schema (v2.0) with backward compatibility            │
│ - GATE 1: python -m pytest tests/test_project_save_load.py -v                    │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────▼─────────────────────────────────────────┐
│ Phase 2: EOS Thermodynamics, Slim-Tube MCM MMP & 3-Phase Rel-Perm                │
│ - Deploy PR-EOS multi-component flash with k_ij matrix                           │
│ - Deploy 1D cell-to-cell slim-tube simulation solver for dynamic MCM MMP         │
│ - Implement Stone I 3-phase rel-perm & Carlson gas hysteresis in profile engine  │
│ - GATE 2: python -m pytest tests/test_physics_validation.py -v                   │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────▼─────────────────────────────────────────┐
│ Phase 3: 3D Well Mechanics, Horizontal Peaceman WI & 3D Trajectory Renderer      │
│ - 3D inter-well transmissibility & vertical perforation overlap validation       │
│ - Enforce Composite Vogel-Darcy IPR deliverability clamping (q_o <= 1,000 BOPD)  │
│ - Implement anisotropic Peaceman Well Index (WI_horiz) for horizontal wells      │
│ - Build interactive 3D Well Trajectory & Pattern Renderer in Wells tab           │
│ - GATE 3: python -m pytest tests/test_project_save_load.py -v                    │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────▼─────────────────────────────────────────┐
│ Phase 4: Geostatistics & Geology Visualizers (Variograms, IJK Slicing, SGS)      │
│ - Implement GeostatisticsVisualizerWidget: 1D Variogram curve fit + 2D/3D SGS    │
│ - Implement GeologyCrossSectionWidget: IJK layer slice sliders + VPC curves      │
│ - Connect widgets to core/geology/geostatistical_modeling.py (GSTools)           │
│ - GATE 4: python -m pytest tests/test_project_save_load.py -v                    │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────▼─────────────────────────────────────────┐
│ Phase 5: 3D Fault Surface Visualizer, Containment & Pre-Flight Visual Audit Gate │
│ - Implement FaultGeometryVisualizerWidget: 3D fault mesh + SGR heatmap + wells   │
│ - Implement near-wellbore thermo-poroelastic stress path (sigma_h, delta_T)      │
│ - Build Shared Earth Visual Input Audit Modal (3-tab model confirmation gate)    │
│ - Embed interactive ISO 27914 Pre-Flight Verification Checklist modal            │
│ - GATE 5: python -m pytest tests/core/test_single_simulation.py -v               │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────▼─────────────────────────────────────────┐
│ Phase 6: Multi-Domain Diagnostic Dashboard & Master Engineering Sign-Off         │
│ - Build ModelEvaluationDashboard (ui/widgets/model_evaluation_dashboard.py)     │
│   • Tab 1: Static QC | Tab 2: SCAL/PVT | Tab 3: RTA/VRR | Tab 4: Geomechanics   │
│   • Tab 5: Pattern Sweep, 3D Streamline Time-of-Flight & IPAF Allocations        │
│ - Run full repository verification suite & synchronize Agent Wiki                │
│ - GATE 6: Master Engineering Sign-Off                                            │
└──────────────────────────────────────────────────────────────────────────────────┘
```
