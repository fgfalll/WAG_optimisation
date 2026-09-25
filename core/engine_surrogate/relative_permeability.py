"""
Stone I 3-Phase Relative Permeability & Carlson Gas Hysteresis Models
====================================================================

Standard Scientific Implementations for CO2 EOR:
- Modified Stone I (1970) Three-Phase Oil Relative Permeability Model
- Carlson (1981) / Land (1968) Trapped Gas Saturation Hysteresis Model
- Baker (1988) Saturation Weighting & Corey (1954) Normalization

References:
- Stone, H.L. (1970): 'Probability Model for Estimating Three-Phase Relative Permeability', JPT.
- Carlson, F.M. (1981): 'Simulation of Gas-Bleed and Water-Alternating-Gas Injected Reservoirs', SPE 10271.
- Land, C.S. (1968): 'Calculation of Imbibition Relative Permeability for Two- and Three-Phase Flow', SPEJ.
- Fayers, F.J. & Matthews, J.D. (1984): 'Evaluation of Normalized Stone's Methods for Estimating Three-Phase Relative Permeabilities', SPEJ.
"""

from typing import Dict, Tuple, Optional, Union
import numpy as np


def normalize_saturations(
    sw: Union[float, np.ndarray],
    sg: Union[float, np.ndarray],
    s_wc: float = 0.20,
    s_orw: float = 0.25,
    s_gc: float = 0.05,
    s_org: Optional[float] = None,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Computes normalized phase saturations for water, gas, and oil.

    Args:
        sw: Water saturation (0-1)
        sg: Gas saturation (0-1)
        s_wc: Connate / irreducible water saturation
        s_orw: Residual oil saturation to waterflooding
        s_gc: Critical / irreducible gas saturation
        s_org: Residual oil saturation to gasflooding (default s_orw)

    Returns:
        (sw_norm, sg_norm, so_norm) normalized saturations
    """
    if s_org is None:
        s_org = s_orw

    sw = np.clip(sw, 0.0, 1.0)
    sg = np.clip(sg, 0.0, 1.0)
    so = np.clip(1.0 - sw - sg, 0.0, 1.0)

    denom_w = max(1.0 - s_wc - s_orw, 1e-4)
    denom_g = max(1.0 - s_wc - s_gc, 1e-4)

    sw_norm = np.clip((sw - s_wc) / denom_w, 0.0, 1.0)
    sg_norm = np.clip((sg - s_gc) / denom_g, 0.0, 1.0)
    so_norm = np.clip((so - s_orw) / denom_w, 0.0, 1.0)

    return sw_norm, sg_norm, so_norm


def corey_two_phase_relperm(
    sw: Union[float, np.ndarray],
    sg: Union[float, np.ndarray],
    s_wc: float = 0.20,
    s_orw: float = 0.25,
    s_gc: float = 0.05,
    k_rw0: float = 0.30,
    k_ro0: float = 0.80,
    k_rg0: float = 0.30,
    n_w: float = 2.0,
    n_ow: float = 2.0,
    n_o: float = 2.0,
    n_g: float = 2.0,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Evaluates 2-phase Corey relative permeabilities for water-oil and gas-oil systems.

    Returns:
        dict with 'krw', 'krow', 'krg', 'krog'
    """
    sw_norm, sg_norm, _ = normalize_saturations(sw, sg, s_wc, s_orw, s_gc)

    # Water-Oil system (at Sg = 0)
    krw = np.where(sw < s_wc, 0.0, np.where(sw > 1.0 - s_orw, k_rw0, k_rw0 * (sw_norm ** n_w)))
    krow = np.where(sw < s_wc, k_ro0, np.where(sw > 1.0 - s_orw, 0.0, k_ro0 * ((1.0 - sw_norm) ** n_ow)))

    # Gas-Oil system (at Sw = Swc)
    krg = np.where(sg < s_gc, 0.0, np.where(sg > 1.0 - s_wc, k_rg0, k_rg0 * (sg_norm ** n_g)))
    krog = np.where(sg < s_gc, k_ro0, np.where(sg > 1.0 - s_wc, 0.0, k_ro0 * ((1.0 - sg_norm) ** n_o)))

    return {
        "krw": krw,
        "krow": krow,
        "krg": krg,
        "krog": krog,
    }


def stone_1_three_phase_relperm(
    sw: Union[float, np.ndarray],
    sg: Union[float, np.ndarray],
    s_wc: float = 0.20,
    s_orw: float = 0.25,
    s_gc: float = 0.05,
    k_rw0: float = 0.30,
    k_ro0: float = 0.80,
    k_rg0: float = 0.30,
    n_w: float = 2.0,
    n_ow: float = 2.0,
    n_o: float = 2.0,
    n_g: float = 2.0,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Computes 3-Phase Relative Permeabilities using the Modified Stone I Model.

    In Stone I (Fayers & Matthews 1984 form):
        k_ro = (S_o* / ((1 - S_w*) * (1 - S_g*))) * k_row(S_w) * k_rog(S_g) / k_ro0

    Returns:
        dict with 'krw', 'krg', 'kro', 'sw_norm', 'sg_norm', 'so_norm'
    """
    two_phase = corey_two_phase_relperm(
        sw=sw, sg=sg, s_wc=s_wc, s_orw=s_orw, s_gc=s_gc,
        k_rw0=k_rw0, k_ro0=k_ro0, k_rg0=k_rg0,
        n_w=n_w, n_ow=n_ow, n_o=n_o, n_g=n_g
    )

    sw_norm, sg_norm, so_norm = normalize_saturations(sw, sg, s_wc, s_orw, s_gc)

    # Stone I interpolation factor
    denom = np.maximum((1.0 - sw_norm) * (1.0 - sg_norm), 1e-6)
    factor = np.clip(so_norm / denom, 0.0, 1.0)

    # 3-Phase Oil Relative Permeability
    k_ro0_safe = max(k_ro0, 1e-4)
    kro_stone = factor * (two_phase["krow"] * two_phase["krog"]) / k_ro0_safe
    kro_stone = np.clip(kro_stone, 0.0, k_ro0)

    # For water and gas, phase permeability depends only on own saturation (Stone hypothesis)
    return {
        "krw": two_phase["krw"],
        "krg": two_phase["krg"],
        "kro": kro_stone,
        "sw_norm": sw_norm,
        "sg_norm": sg_norm,
        "so_norm": so_norm,
    }


def carlson_trapped_gas(
    s_gi: Union[float, np.ndarray],
    s_wc: float = 0.20,
    s_gr_max: float = 0.35,
) -> Union[float, np.ndarray]:
    """
    Calculates Trapped Gas Saturation (Sgt) using Carlson (1981) / Land (1968) Hysteresis.

    During WAG water injection cycles, imbibition traps disconnected gas bubbles:
        C = (1 / S_gr_max) - (1 / (1 - S_wc))
        S_gt = S_gi / (1 + C * S_gi)

    Args:
        s_gi: Initial / peak drainage gas saturation achieved prior to imbibition
        s_wc: Connate water saturation
        s_gr_max: Maximum residual / trapped gas saturation at maximum displacement

    Returns:
        S_gt: Trapped residual gas saturation
    """
    s_gi = np.clip(s_gi, 0.0, 1.0 - s_wc)
    s_gr_max = max(0.01, min(0.60, s_gr_max))

    # Land parameter C
    c_land = (1.0 / s_gr_max) - (1.0 / max(1.0 - s_wc, 0.01))
    c_land = max(0.0, c_land)

    s_gt = s_gi / (1.0 + c_land * s_gi)
    return np.clip(s_gt, 0.0, s_gr_max)


def carlson_imbibition_gas_relperm(
    sg: Union[float, np.ndarray],
    s_gi: float,
    s_wc: float = 0.20,
    s_gc: float = 0.05,
    s_gr_max: float = 0.35,
    k_rg0: float = 0.30,
    n_g: float = 2.0,
) -> Union[float, np.ndarray]:
    """
    Calculates the Carlson Imbibition Gas Relative Permeability Scanning Loop.

    As water re-invades during WAG water cycles, gas relative permeability drops
    along an imbibition branch that terminates at Sgt instead of Sgc.
    """
    s_gt = carlson_trapped_gas(s_gi, s_wc, s_gr_max)

    # Free gas saturation available for flow during imbibition
    denom_imbibition = max(s_gi - s_gt, 1e-4)
    sg_free_norm = np.clip((sg - s_gt) / denom_imbibition, 0.0, 1.0)

    # Drainage end-point at peak historical gas saturation
    two_phase_peak = corey_two_phase_relperm(
        sw=s_wc, sg=s_gi, s_wc=s_wc, s_gc=s_gc, k_rg0=k_rg0, n_g=n_g
    )
    krg_peak = two_phase_peak["krg"]

    # Imbibition scanning curve scaled from peak endpoint to Sgt
    krg_imb = np.where(sg <= s_gt, 0.0, krg_peak * (sg_free_norm ** n_g))
    return np.clip(krg_imb, 0.0, k_rg0)
