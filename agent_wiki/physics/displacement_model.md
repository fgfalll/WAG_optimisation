# Displacement Model & Fractional Flow

## 1. Overview and Theory

Displacement in the CO₂ EOR optimizer is modeled through two classic theoretical frameworks depending on whether operating pressure is above or below MMP:
1. **Miscible Regime ($P \ge P_{MMP}$)**: Koval (1963) unstable miscible displacement theory.
2. **Immiscible Regime ($P < P_{MMP}$)**: Buckley-Leverett (1942) fractional flow theory with Welge (1952) tangent construction.

---

## 2. Miscible Displacement: Koval (1963) Formulation

Koval's theory models the fingering and channeling caused by unfavorable mobility ratios in heterogeneous porous media by defining an effective Koval factor $K$.

### Mathematical Formulations

#### 1. Effective Viscosity Ratio ($E_{eff}$)
Accounting for partial mixing between solvent and oil:
$$E_{eff} = (0.78 + 0.22 \cdot M_{eff}^{0.25})^4$$
Where $M_{eff} = \mu_o / \mu_{oe}$ is the effective mobility ratio from the Todd-Longstaff mixing rule.

#### 2. Effective Heterogeneity Factor ($H$)
$$H = \frac{1}{(1 - V_{DP} \cdot C_{trans})^2}$$
Where $V_{DP}$ is the Dykstra-Parsons coefficient and $C_{trans}$ is the calibration factor.

#### 3. Koval Factor ($K$)
$$K = H \cdot E_{eff}$$
Where $H = 1 / (1 - V_{DP})^2$ is the reservoir heterogeneity factor, and $E_{eff} = (0.78 + 0.22 \cdot M_{eff}^{0.25})^4$ is the effective mobility ratio.

#### 4. Koval Fractional Flow vs Saturation
In `FastProfileGenerator`, the fractional flow of solvent as a function of solvent saturation $S$ is:
$$f_g(S) = \frac{K \cdot S}{1 + S \cdot (K - 1)}$$

**Elimination of SCI-FLAW-01 & Monotonicity Verification**:
In an unstable displacement, an adverse mobility ratio ($M > 1$) must accelerate breakthrough and increase solvent channeling:
$$\frac{\partial f_g}{\partial K} = \frac{S (1 - S)}{[1 + S(K - 1)]^2} > 0 \quad \forall S \in (0, 1)$$
Since $\frac{\partial K}{\partial M} > 0$, it follows that:
$$\frac{\partial f_g}{\partial M} > 0$$
Higher mobility ratio $M$ strictly increases gas fractional flow and reduces breakthrough time ($t_{D,bt} = 1/K$). The prior inverted formulation in early drafts has been eradicated.

#### 5. Dimensionless Welge-Koval Effluent & Breakthrough
The solvent fractional flow $F_s$ at dimensionless time $t_D$ (pore volumes injected, PVI) is:
$$F_s = \begin{cases} 
0 & t_D < \frac{1}{K} \\
\frac{K - \sqrt{K / t_D}}{K - 1} & \frac{1}{K} \le t_D \le K \\
1 & t_D > K
\end{cases}$$
Breakthrough time occurs at:
$$t_{D,bt} = \frac{1}{K}$$
Displacement efficiency as a function of throughput $t_D$ is:
$$E_d(t_D) = \begin{cases}
t_D & t_D < \frac{1}{K} \\
\frac{2 \sqrt{K t_D} - 1 - t_D}{K - 1} & \frac{1}{K} \le t_D \le K \\
1.0 & t_D > K
\end{cases}$$
At breakthrough ($t_D = 1/K$):
$$E_d(t_{D,bt}) = \frac{1}{K}$$
- Implementation: [core/engine_surrogate/analytical_models.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py) and [profile_generator_fast.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/profile_generator_fast.py).


---

## 3. Immiscible Displacement: Buckley-Leverett (1942)

For pressures below MMP, multi-phase immiscible flow governs displacement.

### 1. Fractional Flow Equation ($f_g$)
$$f_g(S_g) = \frac{1}{1 + \left(\frac{k_{ro}(S_g)}{k_{rg}(S_g)}\right) \cdot \left(\frac{\mu_g}{\mu_o}\right)}$$
Where relative permeabilities $k_{ro}, k_{rg}$ follow Corey power-law models.

### 2. Welge Tangent Construction
The average gas saturation at breakthrough $\overline{S}_{g,bt}$ is determined by drawing a tangent line from initial gas saturation $S_{gc}$ to the fractional flow curve:
$$\left. \frac{\partial f_g}{\partial S_g} \right|_{S_{gf}} = \frac{f_g(S_{gf}) - f_g(S_{gc})}{S_{gf} - S_{gc}} = \frac{1}{\overline{S}_{g,bt} - S_{gc}}$$
Breakthrough recovery efficiency is:
$$E_{bt} = \frac{\overline{S}_{g,bt} - S_{gc}}{1 - S_{wi} - S_{gc}}$$
- Implementation: [core/engine_surrogate/analytical_models.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py#L242-L320).

---

## 4. Volumetric Sweep Efficiencies

Total field recovery combines displacement efficiency $E_d$ with macroscopic sweep efficiencies:
$$E_V = E_A \cdot E_I \cdot E_d$$

### A. Areal Sweep Efficiency ($E_A$) - Craig (1971)
Correlations based on mobility ratio $M$ and well flood pattern:
- **5-spot pattern**: $E_A = 0.517 - 0.072 \cdot \log_{10}(M)$
- **Line drive**: $E_A = 0.700 - 0.120 \cdot \log_{10}(M)$
- **Staggered line drive**: $E_A = 0.650 - 0.100 \cdot \log_{10}(M)$
- Capped between $[0.30, 0.95]$.
- Implementation: [core/engine_surrogate/surrogate_models.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_models.py#L148-L183).

### B. Vertical Sweep Efficiency ($E_I$) - Johnson (1956)
Simplified asymptotic relationship capturing vertical layering:
$$E_I \approx 1 - V_{DP}^{0.7}$$
- Capped between $[0.10, 1.00]$.
- Implementation: [core/engine_surrogate/surrogate_models.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_models.py#L185-L201).

---

## 5. Well Deliverability & Nodal Inflow Performance (IPR)

Field production rates must strictly respect Darcy subsurface deliverability rather than arbitrary profile scaling.

### A. Composite Vogel-Darcy Formulation
When reservoir pressure exceeds bubble point / MMP ($P_{\text{res}} \ge MMP$), flow into the producer follows a composite IPR:
1. **Above MMP ($P_{\text{wf}} \ge MMP$)**: Single-phase miscible Darcy flow:
   $$q_o(P_{\text{wf}}) = PI \cdot (P_{\text{res}} - P_{\text{wf}})$$
2. **Below MMP ($P_{\text{wf}} < MMP$)**: Two-phase gas evolution occurs near the sandface, following Vogel's quadratic gas-expansion curve:
   $$q_{\text{above}} = PI \cdot (P_{\text{res}} - MMP)$$
   $$q_{\text{below}} = \frac{PI \cdot MMP}{1.8} \left[ 1 - 0.2 \left(\frac{P_{\text{wf}}}{MMP}\right) - 0.8 \left(\frac{P_{\text{wf}}}{MMP}\right)^2 \right]$$
   $$q_{\text{ipr}}(P_{\text{wf}}) = q_{\text{above}} + q_{\text{below}}$$

### B. Total Field Deliverability Cap & Mass Conservation
Total instantaneous field production is bounded by active producer count:
$$q_{\text{field,max}} = n_{\text{producers}} \cdot q_{\text{ipr}}(P_{\text{wf}})$$
Peak rates in `FastProfileGenerator.generate_oil_profile` are strictly capped at $\min(q_{\text{peak}}, \, q_{\text{field,max}}, \, q_{\text{user\_max}})$, and cumulative profile integration is normalized to preserve $N_p = \text{OOIP} \times RF$.

### C. Near-Wellbore Flashing Derating
Operating producer BHP below MMP causes local gas breakout that reduces effective mixing and accelerates breakthrough. In `PhDHybridSurrogate`, the Todd-Longstaff parameter $\omega_{\text{tl}}$ is derated by:
$$\omega_{\text{tl,eff}} = \omega_{\text{tl}} \cdot \left[ 1.0 - 0.15 \left(1.0 - \frac{P_{\text{wf}}}{MMP}\right) \right]$$
