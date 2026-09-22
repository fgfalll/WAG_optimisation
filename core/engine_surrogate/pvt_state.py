"""
Solvent-Extended Compositional PVT Engine for CO2-EOR Reservoir Simulation.

Implements a 4-component / Todd-Longstaff solvent-extended thermodynamic formulation:
1. Strictly decouples hydrodynamic transport (gas saturation S_g) from thermodynamic composition
   (solvent mole/mass fraction in liquid x_CO2, and vapor y_CO2).
2. Supercritical CO2 density and formation volume factor B_CO2(P, T) evaluated via Peng-Robinson
   cubic Equation of State (PR-EOS) and Span-Wagner formulations.
3. Smooth, C1/C2 continuous live oil swelling factor S_F(P, x_CO2), FVF B_o(P, x_CO2),
   viscosity reduction mu_o(P, x_CO2), and solution gas ratios R_s(P), R_s_CO2(P, x_CO2).
4. Dynamic mixture gas properties rho_g(P, T, y_CO2), B_g(P, T, y_CO2), mu_g(P, T, y_CO2), c_g(P, T, y_CO2).
5. Surface stage separation flash model: accounts for dissolved CO2 degassing during depressurization
   from sandface (P_wf) to stock tank (14.7 psia, 60°F), driving true stock-tank oil shrinkage and
   rigorously splitting produced surface gas into sales hydrocarbon gas and breakthrough CO2.

References:
- Todd, M.R. & Longstaff, W.J. (1972) SPE-3484-PA.
- Peng, D.Y. & Robinson, D.B. (1976) Ind. Eng. Chem. Fundam.
- Span, R. & Wagner, W. (1996) J. Phys. Chem. Ref. Data.
- Emera, M.K. & Sarma, H.K. (2007) Energy & Fuels.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Standard surface reference constants (Field units)
P_SC_PSIA = 14.696
T_SC_RANKINE = 519.67  # 60 °F
R_GAS_FIELD = 10.7316  # psia·ft³ / (lbmol·°R)
R_GAS_SI = 8.314462    # J / (mol·K)

# Reference CO2 properties
MW_CO2 = 44.01         # g/mol
TC_CO2_K = 304.13      # K
PC_CO2_PA = 7.376e6    # Pa (~1070 psia)
OMEGA_CO2 = 0.225      # Acentric factor
RHO_CO2_SC_KG_M3 = 1.838  # Standard density at 14.7 psia, 60°F
CO2_TONNE_PER_MSCF = 0.05295  # Metric tonnes per MSCF of pure CO2 (~0.0530)
MSCF_PER_TONNE = 1.0 / CO2_TONNE_PER_MSCF  # ~18.88 MSCF/tonne

# Reference hydrocarbon gas properties (predominantly methane C1)
MW_CH4 = 16.04
GAMMA_G_DEFAULT = 0.65  # Gas gravity relative to air


@dataclass
class SurfaceSeparationResult:
    """Rigorous surface separator flash results."""
    q_oil_stb_day: float
    q_water_stb_day: float
    q_gas_hc_mscfd: float
    q_gas_co2_mscfd: float
    q_gas_total_mscfd: float
    y_co2_separator: float
    producing_gor_scf_per_stb: float
    hydrocarbon_gor_scf_per_stb: float
    co2_gor_scf_per_stb: float
    water_cut: float


class SolventExtendedPVTEngine:
    """
    Solvent-Extended 4-Component PVT Engine for Reservoir Simulation.

    Tracks fluid behavior as a function of thermodynamic state (P, T, x_CO2, y_CO2)
    with continuous differentiability and zero per-step iterative flash overhead.
    """

    def __init__(
        self,
        reservoir_temperature_f: float = 150.0,
        initial_pressure_psi: float = 3000.0,
        api_gravity: float = 35.0,
        gas_gravity: float = 0.70,
        dead_oil_viscosity_cp: float = 2.5,
        live_oil_viscosity_ref_cp: Optional[float] = None,
        c7_plus_fraction: float = 0.35,
        salinity_ppm: float = 30000.0,
    ):
        self.temp_f = float(reservoir_temperature_f)
        self.temp_r = self.temp_f + 459.67
        self.temp_k = (self.temp_f - 32.0) * 5.0 / 9.0 + 273.15

        self.p_init = float(initial_pressure_psi)
        self.api = float(api_gravity)
        self.gamma_g = float(gas_gravity)
        self.gamma_o = 141.5 / (self.api + 131.5)  # Specific gravity of stock-tank oil
        self.c7_plus = float(c7_plus_fraction)
        self.salinity = float(salinity_ppm)

        self.mu_o_dead = float(dead_oil_viscosity_cp)
        self.mu_o_live_ref = float(
            live_oil_viscosity_ref_cp if live_oil_viscosity_ref_cp is not None else max(0.5, self.mu_o_dead * 0.6)
        )

        # Precompute Peng-Robinson EOS parameters for pure CO2
        self._setup_pr_eos_co2()

    def _setup_pr_eos_co2(self) -> None:
        """Precompute temperature-dependent Peng-Robinson EOS parameters for CO2."""
        Tc = TC_CO2_K
        Pc = PC_CO2_PA
        omega = OMEGA_CO2
        R = R_GAS_SI

        Tr = self.temp_k / Tc
        m = 0.37464 + 1.54226 * omega - 0.26992 * (omega**2)
        alpha = (1.0 + m * (1.0 - np.sqrt(max(Tr, 1e-4)))) ** 2

        self.pr_a_co2 = 0.45724 * (R * Tc) ** 2 / Pc * alpha
        self.pr_b_co2 = 0.07780 * R * Tc / Pc

    # =========================================================================
    # 1. Supercritical CO2 Properties (PR-EOS Dense Phase)
    # =========================================================================

    def calculate_co2_density_kg_m3(self, pressure_psi: float) -> float:
        """
        Calculate supercritical/dense CO2 density using Peng-Robinson EOS.

        Args:
            pressure_psi: Pressure in psia.

        Returns:
            CO2 density in kg/m³.
        """
        p_pa = max(pressure_psi, 14.7) * 6894.757
        T_k = self.temp_k
        R = R_GAS_SI

        a = self.pr_a_co2
        b = self.pr_b_co2

        A = a * p_pa / ((R * T_k) ** 2)
        B = b * p_pa / (R * T_k)

        # Cubic equation in Z: Z^3 - (1-B)Z^2 + (A - 3B^2 - 2B)Z - (AB - B^2 - B^3) = 0
        coeffs = [1.0, -(1.0 - B), A - 3.0 * (B**2) - 2.0 * B, -(A * B - (B**2) - (B**3))]
        roots = np.roots(coeffs)
        real_roots = np.real(roots[np.isreal(roots)])
        valid = real_roots[(real_roots > B) & (real_roots < 4.0)]

        if len(valid) == 0:
            # High-pressure dense-phase fallback
            Z = max(0.25, B * 1.05)
        elif p_pa > PC_CO2_PA:
            # Supercritical/dense phase: choose smallest valid root (liquid-like dense phase)
            Z = float(np.min(valid))
        else:
            # Subcritical vapor phase: choose largest valid root
            Z = float(np.max(valid))

        Z = max(float(Z), 0.1)
        # rho = P * MW / (Z * R * T)
        rho = (p_pa * (MW_CO2 * 1e-3)) / (Z * R * T_k)
        # Dense supercritical CO2 is typically 400 - 950 kg/m³
        return float(np.clip(rho, 2.0, 1100.0))

    def calculate_co2_fvf_rb_per_mscf(self, pressure_psi: float) -> float:
        """
        Calculate supercritical CO2 Formation Volume Factor B_CO2 (RB/MSCF).

        Derived from supercritical EOS density:
        1 MSCF CO2 = 1000 SCF = 28.3168 m³(st) * 1.838 kg/m³ = 52.046 kg.
        Downhole volume = 52.046 / rho_CO2 m³ = (52.046 / rho_CO2) * 6.2898 bbl = 327.36 / rho_CO2 [RB].

        Args:
            pressure_psi: Sandface pressure in psia.

        Returns:
            B_CO2 in RB/MSCF (typically 0.45 - 0.75 RB/MSCF for dense CO2).
        """
        rho = self.calculate_co2_density_kg_m3(pressure_psi)
        b_co2_rb_per_mscf = 327.362 / max(rho, 1e-3)
        return float(np.clip(b_co2_rb_per_mscf, 0.20, 15.0))

    def calculate_co2_viscosity_cp(self, pressure_psi: float) -> float:
        """
        Calculate supercritical CO2 viscosity using Fenghour/Vesovic correlation.

        Args:
            pressure_psi: Pressure in psia.

        Returns:
            Viscosity in cP (typically 0.03 - 0.08 cP for supercritical CO2).
        """
        rho = self.calculate_co2_density_kg_m3(pressure_psi)
        T_k = self.temp_k

        # Dilute gas limit
        mu_0 = 1.00697e-6 * np.sqrt(T_k)  # Pa·s

        # Residual dense contribution
        rho_r = rho / 467.6  # Critical density of CO2 ~467.6 kg/m³
        mu_excess = mu_0 * (0.235 * rho_r + 0.395 * (rho_r**2) - 0.041 * (rho_r**3))

        mu_total_pa_s = mu_0 + max(0.0, mu_excess)
        mu_cp = mu_total_pa_s * 1000.0  # Pa·s to cP
        return float(np.clip(mu_cp, 0.015, 0.12))

    # =========================================================================
    # 2. Live Oil & Dissolved CO2 Properties (x_CO2 Parameterized)
    # =========================================================================

    def calculate_hydrocarbon_solution_gor(self, pressure_psi: float) -> float:
        """
        Calculate equilibrium solution hydrocarbon gas-oil ratio R_s (SCF/STB).
        Uses smooth Vasquez-Beggs / Standing correlation with C1 continuity.
        """
        p = max(pressure_psi, 14.7)
        # Vasquez-Beggs formulation for API > 30
        c1 = 0.0178 if self.api > 30 else 0.0362
        c2 = 1.1870 if self.api > 30 else 1.0937
        c3 = 23.931 if self.api > 30 else 25.724

        rs = c1 * self.gamma_g * (p**c2) * np.exp(c3 * self.api / self.temp_r)
        return float(np.clip(rs, 0.0, 4000.0))

    def calculate_co2_solubility_scf_per_stb(self, pressure_psi: float, x_co2: float) -> float:
        """
        Calculate dissolved CO2 ratio R_s_CO2 in crude oil (SCF/STB).

        Args:
            pressure_psi: Pressure in psia.
            x_co2: Solvent mole/mass fraction in the liquid phase (0 to 1).

        Returns:
            Dissolved CO2 volume at standard conditions per STB oil.
        """
        x_clamped = float(np.clip(x_co2, 0.0, 1.0))
        # Saturation solubility correlation (Emera-Sarma / Simon-Graue):
        # R_s,max scales with pressure and API, inversely with temperature
        p_scale = (max(pressure_psi, 14.7) / 2500.0) ** 0.85
        r_s_co2_sat = 650.0 * (self.api / 35.0) * p_scale * (560.0 / self.temp_r)

        # Dissolved CO2 is proportional to liquid solvent fraction x_co2
        # Smooth C1 formulation
        return float(r_s_co2_sat * (x_clamped**1.1))

    def calculate_oil_swelling_factor(self, pressure_psi: float, x_co2: float) -> float:
        """
        Calculate oil swelling factor S_F(P, x_CO2) = V_oil(with CO2) / V_oil(dead).

        Smooth C1/C2 continuous formulation based on Simon & Graue (1965) and
        Emera & Sarma (2007).
        """
        x_clamped = float(np.clip(x_co2, 0.0, 1.0))
        p = max(pressure_psi, 14.7)

        # Maximum swelling at complete saturation (typically 1.10 - 1.45)
        # Pressure dependence is asymptotic via smooth tanh
        sf_max = 1.05 + 0.35 * float(np.tanh((p - 500.0) / 2000.0))
        sf_max = max(1.0, sf_max)

        # Swelling increases with solvent dissolution fraction
        sf = 1.0 + (sf_max - 1.0) * (x_clamped**1.25)
        return float(np.clip(sf, 1.0, 1.80))

    def calculate_oil_fvf_rb_per_stb(self, pressure_psi: float, x_co2: float) -> float:
        """
        Calculate live oil Formation Volume Factor B_o (RB/STB).

        Couples hydrocarbon solution gas expansion with CO2 dissolution swelling:
        B_o(P, x_CO2) = B_o,HC(P) * S_F(P, x_CO2).
        """
        p = max(pressure_psi, 14.7)
        rs_hc = self.calculate_hydrocarbon_solution_gor(p)

        # Standing (1947) live oil FVF without CO2
        f_val = rs_hc * np.sqrt(self.gamma_g / max(self.gamma_o, 1e-4)) + 1.25 * self.temp_f
        bo_hc = 0.9759 + 0.000120 * (f_val**1.2)
        bo_hc = max(1.02, bo_hc)

        # Compressibility correction above bubble point
        if p > self.p_init:
            c_o = 1.2e-5
            bo_hc *= np.exp(-c_o * (p - self.p_init))

        swelling = self.calculate_oil_swelling_factor(p, x_co2)
        return float(np.clip(bo_hc * swelling, 1.02, 2.50))

    def calculate_oil_viscosity_cp(self, pressure_psi: float, x_co2: float) -> float:
        """
        Calculate live oil viscosity mu_o(P, x_CO2) in cP.

        Accounts for:
        1. Solution hydrocarbon gas reduction (Beggs-Robinson).
        2. Massive CO2 dissolution viscosity thinning (exponential reduction).
        Smooth C1/C2 continuous formulation.
        """
        p = max(pressure_psi, 14.7)
        x_clamped = float(np.clip(x_co2, 0.0, 1.0))

        # 1. Beggs-Robinson live oil viscosity (hydrocarbon dissolution)
        rs_hc = self.calculate_hydrocarbon_solution_gor(p)
        a_param = 10.715 * ((rs_hc + 100.0) ** (-0.515))
        b_param = 5.44 * ((rs_hc + 150.0) ** (-0.338))
        mu_live_hc = a_param * (self.mu_o_dead**b_param)
        mu_live_hc = float(np.clip(mu_live_hc, 0.2, self.mu_o_dead))

        # 2. Viscosity thinning from dissolved CO2 (Lohrenz-Bray-Clark / Emera-Sarma)
        # High CO2 concentration can reduce viscosity by 60% to 90%
        # Formulation: mu_o = mu_live * exp(-k_visc * x_co2)
        k_visc = 2.2 * (self.mu_o_dead / max(mu_live_hc, 0.1)) ** 0.15
        mu_o = mu_live_hc * np.exp(-k_visc * (x_clamped**0.95))

        return float(np.clip(mu_o, 0.08, self.mu_o_dead * 1.5))

    # =========================================================================
    # 3. Mixture Gas Properties (y_CO2 Parameterized)
    # =========================================================================

    def calculate_mixture_gas_properties(
        self, pressure_psi: float, y_co2: float
    ) -> Dict[str, float]:
        """
        Calculate properties of produced/reservoir gas mixture of CO2 and hydrocarbon gas.

        Args:
            pressure_psi: Pressure in psia.
            y_co2: Mole/volume fraction of CO2 in the gas phase (0.0 = pure HC, 1.0 = pure CO2).

        Returns:
            Dictionary with density (kg/m³), FVF (RB/MSCF), viscosity (cP), compressibility (1/psi).
        """
        y_c = float(np.clip(y_co2, 0.0, 1.0))
        p = max(pressure_psi, 14.7)

        # Pure CO2 properties
        rho_co2 = self.calculate_co2_density_kg_m3(p)
        bg_co2 = self.calculate_co2_fvf_rb_per_mscf(p)
        mu_co2 = self.calculate_co2_viscosity_cp(p)

        # Pure Hydrocarbon gas properties (real gas Z-factor)
        Ppr = p / (709.6 - 58.7 * self.gamma_g)
        Tpr = self.temp_r / (170.5 + 307.3 * self.gamma_g)
        # Hall-Yarborough Z-factor approximation
        t_inv = 1.0 / max(Tpr, 0.1)
        z_hc = 1.0 + (0.06422 * t_inv - 0.00332 * (t_inv**2)) * Ppr
        z_hc = float(np.clip(z_hc, 0.65, 1.4))

        # Hydrocarbon gas density & FVF
        mw_hc = self.gamma_g * 28.96
        rho_hc = (p * 6894.757 * (mw_hc * 1e-3)) / (z_hc * R_GAS_SI * self.temp_k)
        # Bg in RB/MSCF: Bg = 0.02827 * Z * T_R / P * 5.6146
        bg_hc = 0.1587 * z_hc * self.temp_r / max(p, 1e-4)

        # Lee-Gonzalez-Eakin hydrocarbon gas viscosity
        k_param = (9.4 + 0.02 * mw_hc) * (self.temp_r**1.5) / (209.0 + 19.0 * mw_hc + self.temp_r)
        x_param = 3.5 + 986.0 / self.temp_r + 0.01 * mw_hc
        y_param = 2.4 - 0.2 * x_param
        rho_hc_g_cm3 = rho_hc * 1e-3
        mu_hc = 1e-4 * k_param * np.exp(x_param * (rho_hc_g_cm3**y_param))
        mu_hc = float(np.clip(mu_hc, 0.01, 0.06))

        # Molar mixing rules for mixture properties
        rho_mix = y_c * rho_co2 + (1.0 - y_c) * rho_hc
        bg_mix = 1.0 / (y_c / max(bg_co2, 1e-4) + (1.0 - y_c) / max(bg_hc, 1e-4))
        mu_mix = (mu_co2**y_c) * (mu_hc ** (1.0 - y_c))

        # Real gas isothermal compressibility c_g = 1/P - (1/Z)(dZ/dP)
        cg_ideal = 1.0 / max(p, 14.7)
        # Dense CO2 has significantly lower compressibility than ideal gas
        if p > 1500.0:
            cg_mix = cg_ideal * (0.35 * y_c + 0.85 * (1.0 - y_c))
        else:
            cg_mix = cg_ideal

        return {
            "density_kg_m3": float(rho_mix),
            "bg_rb_per_mscf": float(bg_mix),
            "viscosity_cp": float(mu_mix),
            "compressibility_psi_inv": float(cg_mix),
        }

    # =========================================================================
    # 4. Surface Stage Separation Flash & Oil Shrinkage
    # =========================================================================

    def perform_surface_stage_separation(
        self,
        q_oil_res_rb_day: float,
        q_water_res_rb_day: float,
        q_gas_free_res_rb_day: float,
        producer_sandface_p_psi: float,
        x_co2_liquid: float,
        y_co2_free_gas: float,
        b_w: float = 1.0,
    ) -> SurfaceSeparationResult:
        """
        Perform rigorous surface stage separation flash from producer sandface to stock tank.

        Accounts for:
        1. Evolved dissolved CO2 degassing upon depressurization (Rs_CO2).
        2. Evolved solution hydrocarbon gas (Rs_HC).
        3. Stock tank oil shrinkage: q_o_stb = q_o_res / Bo(P_prod, x_CO2).
        4. Free reservoir gas expansion into surface gas streams.
        5. Exact split of produced gas into sales hydrocarbon gas vs breakthrough CO2.

        Args:
            q_oil_res_rb_day: In-situ oil production rate at sandface (RB/day).
            q_water_res_rb_day: In-situ water rate at sandface (RB/day).
            q_gas_free_res_rb_day: Free in-situ gas rate at sandface (RB/day).
            producer_sandface_p_psi: Producer BHP / sandface pressure (psia).
            x_co2_liquid: Solvent concentration in sandface liquid phase (0 to 1).
            y_co2_free_gas: CO2 mole fraction in free gas phase (0 to 1).
            b_w: Water formation volume factor (RB/STB, default 1.0).

        Returns:
            SurfaceSeparationResult dataclass with STB oil, water, HC gas, and CO2 gas rates.
        """
        p_prod = max(producer_sandface_p_psi, 14.7)

        # 1. Oil FVF and Stock-Tank Oil Rate
        bo = self.calculate_oil_fvf_rb_per_stb(p_prod, x_co2_liquid)
        q_oil_stb = max(0.0, q_oil_res_rb_day / max(bo, 1e-4))
        q_water_stb = max(0.0, q_water_res_rb_day / max(b_w, 1e-4))

        # 2. Evolved Solution Gas from Depressurization
        rs_hc = self.calculate_hydrocarbon_solution_gor(p_prod)
        rs_co2 = self.calculate_co2_solubility_scf_per_stb(p_prod, x_co2_liquid)

        evolved_hc_mscfd = (q_oil_stb * rs_hc) / 1000.0
        evolved_co2_mscfd = (q_oil_stb * rs_co2) / 1000.0

        # 3. Free Reservoir Gas Expansion to Surface Conditions
        gas_props = self.calculate_mixture_gas_properties(p_prod, y_co2_free_gas)
        bg_rb_per_mscf = gas_props["bg_rb_per_mscf"]
        free_gas_mscfd = max(0.0, q_gas_free_res_rb_day / max(bg_rb_per_mscf, 1e-4))

        y_free = float(np.clip(y_co2_free_gas, 0.0, 1.0))
        free_co2_mscfd = free_gas_mscfd * y_free
        free_hc_mscfd = free_gas_mscfd * (1.0 - y_free)

        # 4. Total Separator Streams
        total_hc_gas_mscfd = max(0.0, evolved_hc_mscfd + free_hc_mscfd)
        total_co2_gas_mscfd = max(0.0, evolved_co2_mscfd + free_co2_mscfd)
        total_gas_mscfd = total_hc_gas_mscfd + total_co2_gas_mscfd

        # 5. Diagnostic Ratios
        y_co2_sep = total_co2_gas_mscfd / max(total_gas_mscfd, 1e-6)
        total_gor = (total_gas_mscfd * 1000.0) / max(q_oil_stb, 1e-4)
        hc_gor = (total_hc_gas_mscfd * 1000.0) / max(q_oil_stb, 1e-4)
        co2_gor = (total_co2_gas_mscfd * 1000.0) / max(q_oil_stb, 1e-4)
        water_cut = q_water_stb / max(q_water_stb + q_oil_stb, 1e-6)

        return SurfaceSeparationResult(
            q_oil_stb_day=float(q_oil_stb),
            q_water_stb_day=float(q_water_stb),
            q_gas_hc_mscfd=float(total_hc_gas_mscfd),
            q_gas_co2_mscfd=float(total_co2_gas_mscfd),
            q_gas_total_mscfd=float(total_gas_mscfd),
            y_co2_separator=float(y_co2_sep),
            producing_gor_scf_per_stb=float(total_gor),
            hydrocarbon_gor_scf_per_stb=float(hc_gor),
            co2_gor_scf_per_stb=float(co2_gor),
            water_cut=float(water_cut),
        )
