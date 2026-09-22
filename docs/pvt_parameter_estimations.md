# PVT Parameter Estimations: Mathematical Implementation

The Surrogate Engine UI provides a "Calculate PVT Properties" functionality designed to back-calculate rigorous fluid and rock thermodynamic properties from minimal screening data (e.g., API gravity, reservoir temperature, and initial pressure).

These calculations abandon simplistic heuristics in favor of the full, mathematically proper hierarchy of fluid property correlations used in industry-standard compositional simulators like CMG and ECLIPSE.

---

## 1. Dead Oil Viscosity ($\mu_{od}$)
The dead oil viscosity is calculated using the rigorous **Beggs and Robinson** empirical correlation for gas-free oil at reservoir temperature ($T_R$ in °F).

**Equations:**
$$ Z = 3.0324 - 0.02023 \cdot \text{API} $$
$$ X = 10^Z \cdot T_R^{-1.163} $$
$$ \mu_{od} = 10^X - 1 $$

---

## 2. Saturated (Live) Oil Viscosity ($\mu_{os}$)
The saturated (live) oil viscosity at the bubble point is computed using the **Chew and Connally** correlation. This mathematically depresses the dead oil viscosity based on the exact amount of dissolved solution gas ($R_s$ in scf/STB).

**Equations:**
$$ A = 10.715 \cdot (R_s + 100)^{-0.515} $$
$$ B = 5.44 \cdot (R_s + 150)^{-0.338} $$
$$ \mu_{os} = A \cdot \mu_{od}^B $$

---

## 3. Bubble Point Pressure Estimation ($P_b$)
The bubble point pressure ($P_b$) is explicitly solved using the **Vasquez and Beggs** correlation to determine the fluid phase state. 

The constants $C_1, C_2, C_3$ are selected precisely based on whether the API gravity is above or below $30^\circ$:
* **API $\le$ 30:** $C_1 = 0.0362$, $C_2 = 1.0937$, $C_3 = 25.7240$
* **API > 30:** $C_1 = 0.0178$, $C_2 = 1.1870$, $C_3 = 23.9310$

**Equation:**
$$ P_b = \left( \frac{R_s}{C_1 \cdot \gamma_g \cdot \exp \left( \frac{C_3 \cdot \text{API}}{T_R + 460} \right)} \right)^{\frac{1}{C_2}} $$

---

## 4. Undersaturated Viscosity Correction ($\mu_o$)
If the user-defined reservoir pressure ($P$) is strictly greater than the mathematically derived bubble point ($P_b$), the fluid is undersaturated. The code applies the **Vasquez and Beggs** compressibility correction to properly model the stiffening of the oil under pressure.

**Equations (for $P > P_b$):**
$$ m = 2.6 \cdot P^{1.187} \cdot \exp(-11.513 - 8.98 \times 10^{-5} \cdot P) $$
$$ \mu_o = \mu_{os} \cdot \left( \frac{P}{P_b} \right)^m $$

*(If $P \le P_b$, the fluid is left at the saturated state, meaning $\mu_o = \mu_{os}$.)*