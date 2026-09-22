# Surrogate Engine: 1D vs 3D Case Differences

The surrogate engine produces different results for the 1D and 3D cases strictly because it is reacting to the **physical reservoir parameters** and **geometry** provided in each scenario. 

Here are the primary physics-based drivers that cause the difference between the `gmflu002_1D` and `gmflu002` cases:

## 1. Reservoir Heterogeneity ($V_{DP}$)

* **1D Case (`v_dp = 0.0`):** The 1D model is perfectly homogeneous. The surrogate engine dynamically calculates the Koval heterogeneity factor ($H_k$). When $V_{DP} = 0$, $H_k = 1.0$. This means the displacement front is completely stable, and there is no viscous fingering or channeling. The CO2 sweeps the oil piston-like, leading to very high recovery (78% in CMG, 42-64% in the surrogate).
* **3D Case (`v_dp = 0.612`):** The 3D model represents a realistic layered reservoir with significant permeability variation. The surrogate engine uses this to calculate a much higher $H_k$. This unstable Koval factor physically models severe CO2 channeling and early breakthrough, heavily penalizing the fractional flow of oil and resulting in a much lower recovery factor (~33% in CMG, 22-31% in the surrogate).

## 2. Hydrocarbon Pore Volume Injected (HCPVI / $t_D$)

Even though both cases use the exact same injection rate (24,837 res-bbl/day) and project lifetime (8 years), the total pore volume of the reservoirs is vastly different:

* **1D Case:** Has an OOIP of **26.1 MMSTB**. The injection rate corresponds to injecting approximately **1.85 pore volumes** of CO2 over 8 years. Because you are pushing almost twice the reservoir's volume through the system, the displacement efficiency approaches its absolute maximum limit.
* **3D Case:** Has an OOIP of **52.3 MMSTB** (twice as large). The same injection rate corresponds to injecting only **0.92 pore volumes** of CO2 over 8 years. Since less than one pore volume is injected, the physical sweep is naturally much lower.

## 3. Volumetric Sweep Geometry

The physical geometry significantly impacts sweep efficiency:

* **1D Case:** `area_acres = 5.7`, `length_ft = 5000` (essentially a long, extremely thin tube). There is no "areal" or "vertical" dimension for the CO2 to bypass the oil.
* **3D Case:** `area_acres = 112.5`, `length_ft = 3500`. The engine accounts for these dimensions. In 3D space, CO2 (which is lighter and more mobile) will physically segregate, overriding the oil vertically and bypassing it areally, which the fractional flow model captures via mobility ratio scaling.

## Summary

The surrogate engine correctly recognizes that a perfectly homogeneous, tiny 1D tube injected with 1.8 pore volumes of CO2 will yield massive recovery. Conversely, a heterogeneous, large 3D reservoir injected with only 0.9 pore volumes will suffer from early breakthrough, CO2 channeling, and significantly lower overall recovery. These results are purely physics-based responses to the distinct input parameters of each case, demonstrating the engine's sensitivity to real reservoir conditions.
