"""
Unified Subsurface Petrophysical & Geomechanical Property Distribution Engine.
Part of Workstream 1.4: Real-Time Subsurface Visualizers & Shared Earth Model.

Generates unified, physically correlated 3D scalar cubes (Permeability, Porosity,
Lithofacies, Pore Pressure, Young's Modulus, In-Situ Stresses, Slip Tendency)
honoring user-selected distribution methods (Dykstra-Parsons Layered, Petrofacies
Architecture, Geostatistical SGSIM, or Homogeneous).
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PetrophysicalCube:
    """
    Unified 3D multi-scalar property container for simulation and CAD visualization.
    All arrays have shape (nx, ny, nz) with Fortran/Cartesian ordering matching VTK/PyVista.
    """
    nx: int
    ny: int
    nz: int
    perm: np.ndarray           # Permeability in mD
    poro: np.ndarray           # Porosity in fraction (0 - 1)
    facies: np.ndarray         # Discrete facies (1=Clean Sand, 2=Carbonate/Silt, 3=Shale Baffle)
    pressure: np.ndarray       # Pore pressure in psia
    saturation: np.ndarray     # Oil saturation fraction So
    youngs_modulus: np.ndarray # Young's Modulus in GPa
    shmin: np.ndarray          # Minimum horizontal stress in psia
    slip_tendency: np.ndarray  # Mohr-Coulomb slip tendency Ts (0 - 1)
    sw: Optional[np.ndarray] = None            # Water saturation fraction Sw (0 - 1)
    sg: Optional[np.ndarray] = None            # Gas saturation fraction Sg (0 - 1)
    fluid_phase: Optional[np.ndarray] = None   # Discrete phase (1=Gas Cap, 2=Oil Leg, 3=Water Leg)
    miscibility_margin: Optional[np.ndarray] = None # Margin (P - MMP) in psia
    visco: Optional[np.ndarray] = None         # In-situ oil viscosity in cP
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        shape = (self.nx, self.ny, self.nz)
        if self.sw is None:
            self.sw = np.clip(1.0 - self.saturation, 0.0, 1.0)
        if self.sg is None:
            self.sg = np.zeros(shape, dtype=float)
        if self.fluid_phase is None:
            self.fluid_phase = np.where(self.saturation > 0.08, 2, 3)
        if self.miscibility_margin is None:
            self.miscibility_margin = self.pressure - 2150.0
        if self.visco is None:
            self.visco = np.full(shape, 2.5, dtype=float)

    def get_summary_stats(self) -> Dict[str, Dict[str, float]]:
        """Returns statistical summary (min, mean, max, p50) for key properties."""
        return {
            "permeability": {
                "min": float(np.min(self.perm)),
                "mean": float(np.mean(self.perm)),
                "max": float(np.max(self.perm)),
                "p50": float(np.median(self.perm)),
            },
            "porosity": {
                "min": float(np.min(self.poro)),
                "mean": float(np.mean(self.poro)),
                "max": float(np.max(self.poro)),
                "p50": float(np.median(self.poro)),
            },
            "saturation": {
                "min": float(np.min(self.saturation)),
                "mean": float(np.mean(self.saturation)),
                "max": float(np.max(self.saturation)),
                "p50": float(np.median(self.saturation)),
            },
            "sw": {
                "min": float(np.min(self.sw)),
                "mean": float(np.mean(self.sw)),
                "max": float(np.max(self.sw)),
                "p50": float(np.median(self.sw)),
            },
            "miscibility_margin": {
                "min": float(np.min(self.miscibility_margin)),
                "mean": float(np.mean(self.miscibility_margin)),
                "max": float(np.max(self.miscibility_margin)),
                "p50": float(np.median(self.miscibility_margin)),
            },
            "pressure": {
                "min": float(np.min(self.pressure)),
                "mean": float(np.mean(self.pressure)),
                "max": float(np.max(self.pressure)),
            },
            "youngs_modulus": {
                "min": float(np.min(self.youngs_modulus)),
                "mean": float(np.mean(self.youngs_modulus)),
                "max": float(np.max(self.youngs_modulus)),
            },
            "facies_proportions": {
                "sand_pct": float(np.mean(self.facies == 1) * 100.0),
                "silt_pct": float(np.mean(self.facies == 2) * 100.0),
                "shale_pct": float(np.mean(self.facies == 3) * 100.0),
            }
        }


def generate_petrophysical_cube(
    nx: int = 50,
    ny: int = 50,
    nz: int = 10,
    length_ft: float = 2000.0,
    width_ft: float = 2000.0,
    top_depth: float = 5000.0,
    thickness_ft: float = 50.0,
    distribution_method: str = "Layered (Dykstra-Parsons)",
    perm_base: float = 100.0,
    poro_base: float = 0.20,
    v_dp: float = 0.65,
    kv_kh: float = 0.10,
    facies_pattern: str = "Fluvial Channel Belt",
    sand_fraction: float = 0.65,
    silt_fraction: float = 0.25,
    shale_fraction: float = 0.10,
    poro_perm_model: str = "Kozeny-Carman",
    r2_correlation: float = 0.85,
    variogram_range_x: float = 800.0,
    variogram_range_y: float = 800.0,
    initial_pressure: float = 4000.0,
    overburden_grad: float = 1.00,
    stress_k0: float = 0.75,
    poissons_ratio: float = 0.25,
    youngs_modulus_base: float = 20.0,
    biot_coeff: float = 0.80,
    frac_grad: float = 0.85,
    random_seed: int = 42,
    # === FULL USER CONTROL PARAMETERS ===
    # Fluvial Channel Belt Object Parameters
    channel_azimuth_deg: float = 45.0,
    channel_sinuosity: float = 1.30,
    channel_wavelength_ft: float = 1500.0,
    channel_amplitude_ft: float = 350.0,
    channel_width_ft: float = 450.0,
    levee_width_ft: float = 250.0,
    aggradation_drift_ft: float = 30.0,
    num_channels: int = 1,
    # Barrier Island / Shoreface Object Parameters
    barrier_azimuth_deg: float = 90.0,
    barrier_width_ft: float = 800.0,
    lagoon_width_ft: float = 450.0,
    progradation_dip_deg: float = 2.0,
    # Carbonate Reef / Shoal Object Parameters
    reef_center_x: float = 1000.0,
    reef_center_y: float = 1000.0,
    reef_major_radius_ft: float = 650.0,
    reef_minor_radius_ft: float = 400.0,
    reef_azimuth_deg: float = 45.0,
    apron_width_ft: float = 300.0,
    # Explicit Per-Facies Petrophysical Controls (None = calculate from base)
    f1_perm: Optional[float] = None,
    f1_poro: Optional[float] = None,
    f2_perm: Optional[float] = None,
    f2_poro: Optional[float] = None,
    f3_perm: Optional[float] = None,
    f3_poro: Optional[float] = None,
    # Layered (Dykstra-Parsons) Trend Control
    layer_permeability_trend: str = "Fining Upward",
    # Variogram & SGSIM Control
    variogram_type: str = "Spherical",
    variogram_range_major: float = 1200.0,
    variogram_range_minor: float = 600.0,
    variogram_range_vert: float = 20.0,
    variogram_azimuth_deg: float = 45.0,
    nugget_effect: float = 0.05,
    sill_variance: float = 1.0,
    **kwargs
) -> PetrophysicalCube:
    """
    Synthesizes a unified 3D PetrophysicalCube honoring user parameters and geological models.
    Provides complete user control over all facies geometries, variograms, and rock properties.
    """
    np.random.seed(random_seed)
    shape = (nx, ny, nz)

    # Coordinate grids
    x_coords = np.linspace(0, length_ft, nx)
    y_coords = np.linspace(0, width_ft, ny)
    z_coords = np.linspace(top_depth, top_depth + thickness_ft, nz)
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")

    # Arrays initialization
    perm = np.zeros(shape, dtype=float)
    poro = np.zeros(shape, dtype=float)
    facies = np.ones(shape, dtype=int)  # 1=Sand, 2=Silt, 3=Shale

    method_lower = distribution_method.lower()

    # Determine per-facies baseline petrophysical values
    k_f1 = float(f1_perm) if f1_perm is not None and f1_perm > 0 else perm_base * 2.2
    phi_f1 = float(f1_poro) if f1_poro is not None and f1_poro > 0 else np.clip(poro_base * 1.25, 0.16, 0.38)

    k_f2 = float(f2_perm) if f2_perm is not None and f2_perm > 0 else perm_base * 0.35
    phi_f2 = float(f2_poro) if f2_poro is not None and f2_poro > 0 else np.clip(poro_base * 0.75, 0.08, 0.22)

    k_f3 = float(f3_perm) if f3_perm is not None and f3_perm > 0 else max(0.001, perm_base * 0.005)
    phi_f3 = float(f3_poro) if f3_poro is not None and f3_poro > 0 else np.clip(poro_base * 0.25, 0.01, 0.08)

    # =========================================================================
    # METHOD 1: FACIES-CONTROLLED (PETROFACIES ARCHITECTURE WITH OBJECT CONTROLS)
    # =========================================================================
    if "facies" in method_lower:
        pattern_str = facies_pattern.lower()

        if "channel" in pattern_str:
            # Full User Control Fluvial Meandering Channel Architecture
            rad_az = np.radians(channel_azimuth_deg)
            cos_az, sin_az = np.cos(rad_az), np.sin(rad_az)
            xc, yc = length_ft * 0.5, width_ft * 0.5

            # Rotated coordinate system aligned with channel flow direction
            X_rot = (X - xc) * cos_az + (Y - yc) * sin_az
            Y_rot = -(X - xc) * sin_az + (Y - yc) * cos_az

            # Meandering amplitude modulated by user sinuosity (1.0 = straight)
            sinuosity_factor = max(channel_sinuosity - 1.0, 0.0) / 0.30
            effective_amp = channel_amplitude_ft * sinuosity_factor
            wavelength = max(channel_wavelength_ft, 100.0)

            # Normalized depth for aggradation drift
            norm_z = (Z - top_depth) / max(thickness_ft, 1.0)

            # Minimum distance to any channel core
            min_dist = np.full(shape, 1e9, dtype=float)
            n_chan = max(int(num_channels), 1)
            spacing = (channel_width_ft + 2.0 * levee_width_ft + 300.0)

            for c_idx in range(n_chan):
                y_offset = (c_idx - (n_chan - 1) / 2.0) * spacing
                center_y = y_offset + effective_amp * np.sin(2.0 * np.pi * X_rot / wavelength) + aggradation_drift_ft * (norm_z - 0.5)
                dist = np.abs(Y_rot - center_y)
                min_dist = np.minimum(min_dist, dist)

            half_w = max(channel_width_ft * 0.5, 20.0)
            levee_w = max(levee_width_ft, 10.0)

            is_sand = min_dist <= half_w
            is_silt = (min_dist > half_w) & (min_dist <= half_w + levee_w)
            is_shale = min_dist > (half_w + levee_w)

            facies[is_sand] = 1
            facies[is_silt] = 2
            facies[is_shale] = 3

        elif "barrier" in pattern_str or "shoreface" in pattern_str:
            # Full User Control Barrier Island / Prograding Shoreface
            rad_az = np.radians(barrier_azimuth_deg)
            cos_az, sin_az = np.cos(rad_az), np.sin(rad_az)
            xc, yc = length_ft * 0.5, width_ft * 0.5

            X_rot = (X - xc) * cos_az + (Y - yc) * sin_az
            Y_rot = -(X - xc) * sin_az + (Y - yc) * cos_az

            # Clinoform progradation dip translation with depth
            dip_rad = np.radians(progradation_dip_deg)
            z_offset = (Z - top_depth) * np.tan(dip_rad)
            Y_eff = Y_rot + z_offset

            bw = max(barrier_width_ft, 50.0)
            lw = max(lagoon_width_ft, 50.0)

            # Core barrier sand body
            is_sand = (Y_eff >= -bw * 0.5) & (Y_eff <= bw * 0.5)
            # Lower shoreface / transition silt
            is_silt = (Y_eff > bw * 0.5) & (Y_eff <= bw * 0.5 + lw * 0.6)
            # Offshore basin shale or back-barrier lagoon
            is_shale = ~ (is_sand | is_silt)

            facies[is_sand] = 1
            facies[is_silt] = 2
            facies[is_shale] = 3

        else:
            # Carbonate Reef / Shoal Complex with Elliptical Geometry
            rad_az = np.radians(reef_azimuth_deg)
            cos_az, sin_az = np.cos(rad_az), np.sin(rad_az)
            dx = X - reef_center_x
            dy = Y - reef_center_y

            X_rot = dx * cos_az + dy * sin_az
            Y_rot = -dx * sin_az + dy * cos_az

            r_maj = max(reef_major_radius_ft, 50.0)
            r_min = max(reef_minor_radius_ft, 50.0)
            apron = max(apron_width_ft, 20.0)

            # Elliptical normalized radius
            r_norm = np.sqrt((X_rot / r_maj) ** 2 + (Y_rot / r_min) ** 2)

            is_sand = r_norm <= 1.0  # Reef core / grainstone shoal
            is_silt = (r_norm > 1.0) & (r_norm <= 1.0 + apron / r_maj)  # Debris apron / packstone
            is_shale = r_norm > 1.0 + apron / r_maj  # Basin mudstone

            facies[is_sand] = 1
            facies[is_silt] = 2
            facies[is_shale] = 3

        # Populate petrophysical properties honoring user per-facies values with geological noise
        noise = np.random.normal(0.0, 0.15, size=shape)

        mask1 = (facies == 1)
        perm[mask1] = np.clip(k_f1 * np.exp(noise[mask1]), 1.0, k_f1 * 5.0)
        poro[mask1] = np.clip(phi_f1 * (1.0 + noise[mask1] * 0.20), 0.12, 0.42)

        mask2 = (facies == 2)
        perm[mask2] = np.clip(k_f2 * np.exp(noise[mask2]), 0.1, k_f2 * 3.0)
        poro[mask2] = np.clip(phi_f2 * (1.0 + noise[mask2] * 0.25), 0.05, 0.24)

        mask3 = (facies == 3)
        perm[mask3] = np.clip(k_f3 * np.exp(noise[mask3] * 0.5), 0.0001, max(k_f3 * 4.0, 0.5))
        poro[mask3] = np.clip(phi_f3 * (1.0 + noise[mask3] * 0.30), 0.005, 0.10)

    # =========================================================================
    # METHOD 2: LAYERED (DYKSTRA-PARSONS WITH DIRECTIONAL TREND CONTROLS)
    # =========================================================================
    elif "layer" in method_lower or "dykstra" in method_lower:
        sigma_k = -np.log(max(1.0 - v_dp, 0.05))
        layer_factors = np.exp(np.random.normal(0.0, sigma_k * 0.45, size=nz))
        layer_factors /= np.mean(layer_factors)

        # Apply user vertical trend
        trend_str = layer_permeability_trend.lower()
        if "fining" in trend_str:
            trend_multiplier = np.linspace(1.8, 0.35, nz)
        elif "coarsening" in trend_str:
            trend_multiplier = np.linspace(0.35, 1.8, nz)
        elif "symmetric" in trend_str or "bar" in trend_str:
            trend_multiplier = 1.8 * np.sin(np.linspace(0.2, np.pi - 0.2, nz))
        else:
            trend_multiplier = np.ones(nz)

        layer_factors = layer_factors * trend_multiplier
        layer_factors /= np.mean(layer_factors)

        for k in range(nz):
            f_xy = np.random.normal(0.0, 0.15, size=(nx, ny))
            k_k = np.clip(perm_base * layer_factors[k] * np.exp(f_xy), 0.01, perm_base * 8.0)

            if "kozeny" in poro_perm_model.lower():
                phi_k = np.clip(poro_base * (k_k / max(perm_base, 1.0)) ** 0.22, 0.04, 0.40)
            else:
                phi_k = np.clip(poro_base * (1.0 + f_xy * 0.30) * (layer_factors[k] ** 0.20), 0.03, 0.40)

            perm[:, :, k] = k_k
            poro[:, :, k] = phi_k
            facies[:, :, k] = np.where(k_k >= perm_base * 0.8, 1, np.where(k_k >= perm_base * 0.1, 2, 3))

    # =========================================================================
    # METHOD 3: GEOSTATISTICAL SGSIM (VARIOGRAM-BASED SPECTRAL SIMULATION)
    # =========================================================================
    elif "geostat" in method_lower or "sgsim" in method_lower:
        rad_az = np.radians(variogram_azimuth_deg)
        cos_az, sin_az = np.cos(rad_az), np.sin(rad_az)

        r_maj = max(variogram_range_major, 50.0)
        r_min = max(variogram_range_minor, 50.0)
        r_vert = max(variogram_range_vert, 5.0)

        kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=length_ft / max(nx, 1))
        ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=width_ft / max(ny, 1))
        KX, KY = np.meshgrid(kx, ky, indexing="ij")

        # Rotated anisotropic wavenumber
        KX_rot = KX * cos_az + KY * sin_az
        KY_rot = -KX * sin_az + KY * cos_az
        K_dist = np.sqrt((KX_rot * r_maj) ** 2 + (KY_rot * r_min) ** 2)

        # Variogram model power spectrum
        v_type = variogram_type.lower()
        if "gaussian" in v_type:
            power_spectrum = np.exp(-(K_dist ** 2) / 4.0)
        elif "exponential" in v_type:
            power_spectrum = 1.0 / ((1.0 + K_dist ** 2) ** 1.5)
        else:  # Spherical default
            power_spectrum = np.exp(-K_dist)

        # Incorporate nugget effect (white noise baseline)
        nug = np.clip(nugget_effect, 0.0, 0.80)
        sill = max(sill_variance, 0.1)

        for k in range(nz):
            white_noise = np.random.normal(0.0, 1.0, size=(nx, ny))
            spectral_field = np.real(np.fft.ifft2(np.fft.fft2(white_noise) * np.sqrt(power_spectrum)))
            std_f = np.std(spectral_field)
            if std_f > 1e-6:
                spectral_field /= std_f

            # Combine structured field with nugget noise
            nugget_noise = np.random.normal(0.0, 1.0, size=(nx, ny))
            field = np.sqrt(sill * (1.0 - nug)) * spectral_field + np.sqrt(sill * nug) * nugget_noise

            # Vertical correlation through layers
            if k > 0:
                rho_z = np.exp(-(thickness_ft / max(nz, 1)) / r_vert)
                field = rho_z * prev_field + np.sqrt(1.0 - rho_z ** 2) * field
            prev_field = field

            k_k = np.clip(perm_base * np.exp(field * 0.70), 0.05, perm_base * 6.0)
            phi_k = np.clip(poro_base * (1.0 + field * 0.30), 0.03, 0.40)

            perm[:, :, k] = k_k
            poro[:, :, k] = phi_k
            facies[:, :, k] = np.where(k_k >= perm_base * 0.9, 1, np.where(k_k >= perm_base * 0.15, 2, 3))

    # =========================================================================
    # METHOD 4: HOMOGENEOUS (UNIFORM BASE VALUES)
    # =========================================================================
    else:
        perm.fill(perm_base)
        poro.fill(poro_base)
        facies.fill(1)

    # Dynamic phase saturations, fluid contacts & hydrostatic equilibrium
    oil_api = float(kwargs.get("oil_api", kwargs.get("api_gravity", 35.0)))
    gamma_o = 141.5 / max(oil_api + 131.5, 10.0)
    oil_grad = 0.4335 * gamma_o  # psi/ft
    water_grad = float(kwargs.get("water_gradient", 0.44))  # psi/ft
    gas_grad = float(kwargs.get("gas_gradient", 0.08))      # psi/ft
    mmp_val = float(kwargs.get("mmp_psia", kwargs.get("mmp", 2150.0)))
    dead_visc = float(kwargs.get("dead_oil_viscosity_cp", kwargs.get("oil_viscosity", 2.5)))

    # Fluid Contacts (TVD ft)
    woc_depth = float(kwargs.get("woc_depth", top_depth + thickness_ft * 0.75))
    has_gas_cap = bool(kwargs.get("has_gas_cap", False))
    goc_depth = float(kwargs.get("goc_depth", top_depth + thickness_ft * 0.20)) if has_gas_cap else None

    # Base irreducible water saturation modulated by permeability
    swc = np.clip(0.12 / np.sqrt(np.maximum(perm / 10.0, 0.01)), 0.08, 0.50)

    # Sigmoidal capillary transition zone across WOC
    h_trans = np.clip(8.0 / np.sqrt(np.maximum(perm / 50.0, 0.1)), 2.0, 25.0)
    woc_diff = Z - woc_depth
    sw_trans = swc + (1.0 - swc) / (1.0 + np.exp(-woc_diff / (h_trans * 0.4)))
    sw = np.clip(sw_trans, swc, 1.0)

    if has_gas_cap and goc_depth is not None:
        goc_diff = goc_depth - Z
        sg_zone = np.clip(1.0 - swc - 0.05, 0.0, 0.90) / (1.0 + np.exp(-goc_diff / 2.0))
        sg = np.clip(sg_zone, 0.0, 1.0 - sw)
        so = np.clip(1.0 - sw - sg, 0.0, 1.0)
        fluid_phase = np.where(Z < goc_depth, 1, np.where(Z < woc_depth, 2, 3))
    else:
        sg = np.zeros(shape, dtype=float)
        so = np.clip(1.0 - sw, 0.0, 1.0)
        fluid_phase = np.where(Z < woc_depth, 2, 3)

    saturation = so  # Primary oil saturation So

    # True hydrostatic pore pressure column coupled to fluid contacts
    p_ref = initial_pressure
    if has_gas_cap and goc_depth is not None and goc_depth > top_depth:
        p_goc = p_ref + gas_grad * (goc_depth - top_depth)
        p_woc = p_goc + oil_grad * (woc_depth - goc_depth)
        pressure = np.where(
            Z < goc_depth,
            p_ref + gas_grad * (Z - top_depth),
            np.where(
                Z < woc_depth,
                p_goc + oil_grad * (Z - goc_depth),
                p_woc + water_grad * (Z - woc_depth)
            )
        )
    else:
        p_woc = p_ref + oil_grad * (woc_depth - top_depth)
        pressure = np.where(
            Z <= woc_depth,
            p_ref + oil_grad * (Z - top_depth),
            p_woc + water_grad * (Z - woc_depth)
        )

    # Miscibility margin (P - MMP) in psia
    miscibility_margin = pressure - mmp_val

    # In-situ oil viscosity
    p_ratio = np.clip(pressure / max(p_ref, 100.0), 0.5, 2.0)
    visco = np.clip(dead_visc * (0.45 + 0.55 / p_ratio), 0.2, dead_visc * 2.5)

    # Depth-dependent in-situ total stresses
    sigma_v = overburden_grad * Z                         # Overburden stress (psi)
    sigma_v_eff = np.maximum(sigma_v - biot_coeff * pressure, 100.0) # Effective vertical stress
    shmin = (poissons_ratio / (1.0 - poissons_ratio)) * sigma_v_eff + biot_coeff * pressure # psi

    # Facies-modulated Young's Modulus (GPa)
    # Sandstone: ~20-28 GPa, Siltstone/Carbonate: ~30-45 GPa, Shale: ~8-15 GPa
    ym_map = {1: youngs_modulus_base, 2: youngs_modulus_base * 1.5, 3: youngs_modulus_base * 0.55}
    youngs_modulus = np.zeros(shape, dtype=float)
    for f_id, val in ym_map.items():
        youngs_modulus[facies == f_id] = val * (1.0 - poro[facies == f_id]) ** 2

    # Mohr-Coulomb slip tendency on critical 60° fault planes: Ts = tau / sigma_n'
    diff_stress = np.maximum(sigma_v - shmin, 50.0)
    tau = diff_stress * np.sin(np.radians(60.0)) * np.cos(np.radians(60.0))
    sigma_n_eff = np.maximum(shmin - pressure + diff_stress * (np.sin(np.radians(60.0)) ** 2), 50.0)
    slip_tendency = np.clip(tau / sigma_n_eff, 0.05, 0.95)

    return PetrophysicalCube(
        nx=nx, ny=ny, nz=nz,
        perm=perm, poro=poro, facies=facies,
        pressure=pressure, saturation=saturation,
        youngs_modulus=youngs_modulus, shmin=shmin,
        slip_tendency=slip_tendency,
        sw=sw, sg=sg, fluid_phase=fluid_phase,
        miscibility_margin=miscibility_margin, visco=visco,
        metadata={
            "distribution_method": distribution_method,
            "perm_base": perm_base,
            "poro_base": poro_base,
            "v_dp": v_dp,
            "facies_pattern": facies_pattern,
            "random_seed": random_seed,
            "woc_depth": woc_depth,
            "goc_depth": goc_depth,
            "oil_api": oil_api,
            "mmp_val": mmp_val
        }
    )
