"""
3D Well Mechanics, Anisotropic Peaceman Well Index & Connectivity Models
========================================================================

Physics-grounded well deliverability, completions, and inter-well sweep mechanics
for the Surrogate Engine and Shared Earth Reservoir Model.

Key References:
- Peaceman, D.W. (1978): "Interpretation of Well-Block Pressures in Numerical
  Reservoir Simulation", SPE Journal, 18(3), 183-194. SPE-6893.
- Peaceman, D.W. (1983): "Interpretation of Well-Block Pressures in Numerical
  Reservoir Simulation with Nonsquare Grid Blocks and Anisotropic Permeability",
  SPE Journal, 23(3), 531-543. SPE-10520 / SPE-10194.
- Babu, D.K. & Odeh, A.S. (1989): "Productivity of a Horizontal Well",
  SPE Reservoir Engineering, 4(4), 417-421. SPE-18298.
- Vogel, J.V. (1968): "Inflow Performance Relationships for Solution-Gas Drive Wells",
  JPT, 20(1), 83-92. SPE-1476.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Field conversion constant: 2 * pi * 0.001127 = 0.00708 STB/d / (mD * ft * psi)
DARCY_FIELD_CONSTANT = 0.00708
DARCY_TRANSMISSIBILITY_CONSTANT = 0.001127


def calculate_peaceman_index_horizontal(
    ky_md: float,
    kz_md: float,
    length_lateral_ft: float,
    dy_ft: float,
    dz_ft: float,
    r_w_ft: float = 0.354,
    skin: float = 0.0,
    mu_cp: float = 1.0,
) -> float:
    """
    Calculate anisotropic horizontal well index (WI_horiz) per Peaceman (1983) SPE-10194.

    For a horizontal well drilled along the x-direction:
    Equivalent wellblock radius r_o,h:
        r_o,h = 0.28 * sqrt( (kz/ky)^0.5 * dy^2 + (ky/kz)^0.5 * dz^2 ) /
                       ( (kz/ky)^0.25 + (ky/kz)^0.25 )

    Horizontal Well Index:
        WI_horiz = (0.00708 * sqrt(ky * kz) * L_lat) / (mu * (ln(r_o,h / r_w) + S))

    Args:
        ky_md: Permeability perpendicular to lateral in horizontal plane (mD)
        kz_md: Vertical permeability (mD)
        length_lateral_ft: Completed lateral length (ft)
        dy_ft: Grid block width in y-direction (ft)
        dz_ft: Grid block height in z-direction (ft)
        r_w_ft: Wellbore radius (ft, default 0.354 ft = 8.5" hole)
        skin: Total skin factor (mechanical + completion)
        mu_cp: In-situ fluid viscosity (cP)

    Returns:
        Productivity or injectivity index in STB/d/psi (or RB/d/psi), clamped >= 1e-4.
    """
    if ky_md <= 0 or kz_md <= 0 or length_lateral_ft <= 0 or dy_ft <= 0 or dz_ft <= 0:
        return 1.0

    r_w = max(float(r_w_ft), 0.05)
    mu = max(float(mu_cp), 0.01)

    # Anisotropic permeability ratio
    k_ratio = float(np.clip(kz_md / ky_md, 1e-4, 1e4))
    
    # Equivalent wellblock radius in y-z plane (Peaceman 1983, Eq. 18)
    num = np.sqrt(np.sqrt(k_ratio) * (dy_ft**2) + (1.0 / np.sqrt(k_ratio)) * (dz_ft**2))
    denom_r = (k_ratio**0.25) + (1.0 / (k_ratio**0.25))
    r_oh = 0.28 * (num / max(denom_r, 1e-6))
    
    # Boundary and wellbore logarithmic ratio
    log_ratio = np.log(max(r_oh / r_w, 1.05))
    total_denom = max(log_ratio + float(skin), 0.1)

    # Effective geometric permeability in cross-section
    k_geom = np.sqrt(ky_md * kz_md)
    wi_horiz = (DARCY_FIELD_CONSTANT * k_geom * float(length_lateral_ft)) / (mu * total_denom)

    return float(max(wi_horiz, 1e-4))


def calculate_peaceman_index_vertical(
    kx_md: float,
    ky_md: float,
    h_perf_ft: float,
    dx_ft: float,
    dy_ft: float,
    r_w_ft: float = 0.354,
    skin: float = 0.0,
    mu_cp: float = 1.0,
) -> float:
    """
    Calculate anisotropic vertical well index (WI_vert) per Peaceman (1978, 1983).

    Equivalent wellblock radius r_o,v:
        r_o,v = 0.28 * sqrt( (ky/kx)^0.5 * dx^2 + (kx/ky)^0.5 * dy^2 ) /
                       ( (ky/kx)^0.25 + (kx/ky)^0.25 )

    Vertical Well Index:
        WI_vert = (0.00708 * sqrt(kx * ky) * h_perf) / (mu * (ln(r_o,v / r_w) + S))

    Returns:
        Well index in STB/d/psi (or RB/d/psi), clamped >= 1e-4.
    """
    if kx_md <= 0 or ky_md <= 0 or h_perf_ft <= 0 or dx_ft <= 0 or dy_ft <= 0:
        return 1.0

    r_w = max(float(r_w_ft), 0.05)
    mu = max(float(mu_cp), 0.01)

    k_ratio = float(np.clip(ky_md / kx_md, 1e-4, 1e4))
    num = np.sqrt(np.sqrt(k_ratio) * (dx_ft**2) + (1.0 / np.sqrt(k_ratio)) * (dy_ft**2))
    denom_r = (k_ratio**0.25) + (1.0 / (k_ratio**0.25))
    r_ov = 0.28 * (num / max(denom_r, 1e-6))

    log_ratio = np.log(max(r_ov / r_w, 1.05))
    total_denom = max(log_ratio + float(skin), 0.1)

    k_geom = np.sqrt(kx_md * ky_md)
    wi_vert = (DARCY_FIELD_CONSTANT * k_geom * float(h_perf_ft)) / (mu * total_denom)

    return float(max(wi_vert, 1e-4))


def calculate_vertical_perforation_overlap(
    perf1: List[Tuple[float, float]],
    perf2: List[Tuple[float, float]],
) -> Tuple[float, float]:
    """
    Calculate vertical perforation overlap length (ft) and overlap ratio Omega_overlap.

    Overlap ratio:
        Omega_overlap = h_overlap / min(h_perf1, h_perf2)

    A ratio Omega >= 0.20 (20%) is required for effective pattern flood communication.
    A ratio < 0.20 indicates vertical bypass or compartmentalization risk.

    Args:
        perf1: List of (top, bottom) depth tuples for well 1
        perf2: List of (top, bottom) depth tuples for well 2

    Returns:
        Tuple of (h_overlap_ft, omega_overlap_ratio)
    """
    if not perf1 or not perf2:
        return 0.0, 0.0

    # Total completed intervals for each well
    h1 = sum(max(0.0, abs(p[1] - p[0])) for p in perf1 if len(p) >= 2)
    h2 = sum(max(0.0, abs(p[1] - p[0])) for p in perf2 if len(p) >= 2)

    if h1 <= 0.0 or h2 <= 0.0:
        return 0.0, 0.0

    # Sum pairwise overlapping depth intervals
    total_overlap = 0.0
    for p1 in perf1:
        if len(p1) < 2:
            continue
        top1, bot1 = min(p1[0], p1[1]), max(p1[0], p1[1])
        for p2 in perf2:
            if len(p2) < 2:
                continue
            top2, bot2 = min(p2[0], p2[1]), max(p2[0], p2[1])

            overlap_top = max(top1, top2)
            overlap_bot = min(bot1, bot2)
            if overlap_bot > overlap_top:
                total_overlap += (overlap_bot - overlap_top)

    min_h = min(h1, h2)
    omega = min(1.0, max(0.0, total_overlap / min_h)) if min_h > 0 else 0.0
    return float(total_overlap), float(omega)


def calculate_interwell_transmissibility(
    x1_ft: float,
    y1_ft: float,
    x2_ft: float,
    y2_ft: float,
    h_overlap_ft: float,
    k_h_md: float,
    mu_cp: float = 1.0,
    r_w_ft: float = 0.354,
) -> float:
    """
    Calculate 3D geometric inter-well transmissibility in field units (RB/d/psi):

        T_ij = (0.001127 * k_h * h_overlap) / (mu * ln(D_ij / r_w))

    Args:
        x1_ft, y1_ft: Surface/midpoint coordinate of well 1
        x2_ft, y2_ft: Surface/midpoint coordinate of well 2
        h_overlap_ft: Vertical perforation overlap between well pair (ft)
        k_h_md: Horizontal permeability (mD)
        mu_cp: In-situ fluid viscosity (cP)
        r_w_ft: Wellbore radius (ft)

    Returns:
        Transmissibility in RB/d/psi (clamped >= 0.0)
    """
    if h_overlap_ft <= 0.0 or k_h_md <= 0.0:
        return 0.0

    dist = np.sqrt((x1_ft - x2_ft)**2 + (y1_ft - y2_ft)**2)
    dist = max(dist, 10.0)  # Avoid singularity for coincident coordinates

    r_w = max(r_w_ft, 0.05)
    mu = max(mu_cp, 0.01)

    log_dist = np.log(max(dist / r_w, 1.1))
    t_ij = (DARCY_TRANSMISSIBILITY_CONSTANT * float(k_h_md) * float(h_overlap_ft)) / (mu * log_dist)
    return float(max(t_ij, 0.0))


def generate_synthetic_well_trajectory(
    surface_x: float,
    surface_y: float,
    top_tvd: float,
    bottom_tvd: float,
    trajectory_type: str = "Vertical",
    lateral_length_ft: float = 1500.0,
    azimuth_deg: float = 0.0,
    num_points: int = 50,
) -> np.ndarray:
    """
    Generate realistic 3D trajectory coordinate array shape (N, 3) [X, Y, Z_TVD].

    - Vertical: Straight vertical path from top_tvd to bottom_tvd.
    - Horizontal: Vertical kick-off down to landing TVD, 90-degree curved build
      section, and straight lateral extension of length lateral_length_ft along azimuth.
    - Deviated: S-curve trajectory with directional kick and hold.
    """
    top_z = float(top_tvd)
    bot_z = float(bottom_tvd)
    traj = str(trajectory_type).lower()

    if "horiz" in traj:
        # Build section takes ~30% of vertical depth, lateral extends horizontally
        kop_z = top_z + (bot_z - top_z) * 0.70
        landing_z = bot_z
        n_vert = max(int(num_points * 0.35), 10)
        n_curve = max(int(num_points * 0.25), 10)
        n_lat = max(num_points - n_vert - n_curve, 10)

        # 1. Vertical section
        z_vert = np.linspace(top_z, kop_z, n_vert)
        x_vert = np.full(n_vert, surface_x)
        y_vert = np.full(n_vert, surface_y)

        # 2. Build curve
        theta = np.linspace(0, np.pi / 2, n_curve)
        rad_az = np.radians(azimuth_deg)
        build_radius = max(landing_z - kop_z, 50.0)
        
        curve_disp = build_radius * (1.0 - np.cos(theta))
        z_curve = kop_z + build_radius * np.sin(theta)
        x_curve = surface_x + curve_disp * np.cos(rad_az)
        y_curve = surface_y + curve_disp * np.sin(rad_az)

        # 3. Horizontal lateral
        lateral_dist = np.linspace(0, max(lateral_length_ft, 100.0), n_lat)
        end_curve_x = x_curve[-1]
        end_curve_y = y_curve[-1]
        x_lat = end_curve_x + lateral_dist * np.cos(rad_az)
        y_lat = end_curve_y + lateral_dist * np.sin(rad_az)
        z_lat = np.full(n_lat, landing_z)

        x_all = np.concatenate([x_vert, x_curve[1:], x_lat[1:]])
        y_all = np.concatenate([y_vert, y_curve[1:], y_lat[1:]])
        z_all = np.concatenate([z_vert, z_curve[1:], z_lat[1:]])

        return np.column_stack([x_all, y_all, z_all])

    elif "dev" in traj:
        # Deviated S-curve
        t = np.linspace(0, 1, num_points)
        z_all = top_z + (bot_z - top_z) * t
        rad_az = np.radians(azimuth_deg)
        disp = 300.0 * np.sin(np.pi * t)
        x_all = surface_x + disp * np.cos(rad_az)
        y_all = surface_y + disp * np.sin(rad_az)
        return np.column_stack([x_all, y_all, z_all])

    else:
        # Straight vertical
        z_all = np.linspace(top_z, bot_z, num_points)
        x_all = np.full(num_points, surface_x)
        y_all = np.full(num_points, surface_y)
        return np.column_stack([x_all, y_all, z_all])


def validate_well_network(
    wells: List[Any],
    reservoir_k_md: float = 100.0,
    kv_kh: float = 0.1,
    reservoir_h_ft: float = 100.0,
    dx_ft: float = 100.0,
    dy_ft: float = 100.0,
    dz_ft: float = 20.0,
    mu_oil_cp: float = 2.0,
    min_overlap_threshold: float = 0.20,
) -> Dict[str, Any]:
    """
    Validate well inventory, calculate anisotropic Peaceman indices, and check
    inter-well perforation overlaps and transmissibility.

    Returns:
        Dict with comprehensive diagnostic metrics and validation status.
    """
    kz_md = reservoir_k_md * kv_kh
    well_metrics = []
    injectors = []
    producers = []

    for w in wells:
        w_name = getattr(w, "name", "Well")
        w_meta = getattr(w, "metadata", {}) or {}
        w_type = str(w_meta.get("type", "")).lower()
        if not w_type:
            w_type = "injector" if "inj" in w_name.lower() or "injector" in str(w_meta.get("status", "")).lower() else "producer"
        is_inj = "inj" in w_type

        traj_type = str(w_meta.get("trajectory_type", "Vertical"))
        lat_len = float(w_meta.get("lateral_length", 1500.0) if "horiz" in traj_type.lower() else 0.0)
        skin = float(getattr(w, "skin_factor", 0.0) or w_meta.get("skin_factor", 0.0) or 0.0)
        rw = float(getattr(w, "wellbore_radius_ft", 0.354) or w_meta.get("wellbore_radius_ft", 0.354) or 0.354)

        # Perforations
        perfs = getattr(w, "perforations", []) or [
            (p.get("top", 0.0), p.get("bottom", 0.0)) for p in getattr(w, "perforation_properties", [])
        ]
        if not perfs:
            # Fall back to depths or default formation
            if hasattr(w, "depths") and len(w.depths) > 1:
                perfs = [(float(w.depths[0]), float(w.depths[-1]))]
            else:
                top = float(w_meta.get("TopDepth", 1000.0))
                bot = float(w_meta.get("BottomDepth", top + reservoir_h_ft))
                perfs = [(top, bot)]

        h_perf = sum(abs(p[1] - p[0]) for p in perfs if len(p) >= 2)

        # Compute Peaceman Index
        if "horiz" in traj_type.lower() or lat_len > 0:
            wi = calculate_peaceman_index_horizontal(
                ky_md=reservoir_k_md,
                kz_md=kz_md,
                length_lateral_ft=max(lat_len, 500.0),
                dy_ft=dy_ft,
                dz_ft=dz_ft,
                r_w_ft=rw,
                skin=skin,
                mu_cp=mu_oil_cp,
            )
        else:
            wi = calculate_peaceman_index_vertical(
                kx_md=reservoir_k_md,
                ky_md=reservoir_k_md,
                h_perf_ft=max(h_perf, 10.0),
                dx_ft=dx_ft,
                dy_ft=dy_ft,
                r_w_ft=rw,
                skin=skin,
                mu_cp=mu_oil_cp,
            )

        sx = float(w_meta.get("SurfaceX", w_meta.get("surface_x", 0.0)))
        sy = float(w_meta.get("SurfaceY", w_meta.get("surface_y", 0.0)))

        info = {
            "name": w_name,
            "role": "Injector" if is_inj else "Producer",
            "is_injector": is_inj,
            "trajectory": traj_type,
            "lateral_length_ft": lat_len,
            "perforations": perfs,
            "h_perf_ft": round(h_perf, 1),
            "peaceman_wi": round(wi, 2),
            "surface_x": sx,
            "surface_y": sy,
        }
        well_metrics.append(info)
        if is_inj:
            injectors.append(info)
        else:
            producers.append(info)

    # Operating pattern detection
    total_wells = len(wells)
    if total_wells == 0:
        pattern = "Zero-D Field Model (0 wells defined)"
    elif total_wells == 1:
        pattern = "Single-Well Huff-n-Puff (Cyclic CO2 EOR)"
    else:
        pattern = "Pattern Flood (Continuous / WAG / SWAG)"

    # Pairwise connectivity
    interwell_pairs = []
    warnings = []
    has_valid_pair = False

    for inj in injectors:
        for prod in producers:
            h_overlap, omega = calculate_vertical_perforation_overlap(
                inj["perforations"], prod["perforations"]
            )
            dist = np.sqrt((inj["surface_x"] - prod["surface_x"])**2 + (inj["surface_y"] - prod["surface_y"])**2)
            t_ij = calculate_interwell_transmissibility(
                inj["surface_x"], inj["surface_y"],
                prod["surface_x"], prod["surface_y"],
                h_overlap, reservoir_k_md, mu_oil_cp
            )

            is_valid = omega >= min_overlap_threshold
            if is_valid:
                has_valid_pair = True
                status_text = "VALID"
            elif omega > 0:
                status_text = f"WARNING: Low Overlap ({omega*100:.1f}% < 20%)"
                warnings.append(
                    f"Pair {inj['name']} -> {prod['name']}: Low perforation overlap ({omega*100:.1f}%). Ineffective vertical sweep."
                )
            else:
                status_text = "NO OVERLAP (0%)"
                warnings.append(
                    f"Pair {inj['name']} -> {prod['name']}: Zero perforation overlap. Wells are completed in separated layers."
                )

            interwell_pairs.append({
                "injector": inj["name"],
                "producer": prod["name"],
                "distance_ft": round(dist, 1),
                "overlap_ft": round(h_overlap, 1),
                "omega_ratio": round(omega, 3),
                "omega_pct": round(omega * 100.0, 1),
                "transmissibility": round(t_ij, 4),
                "status": status_text,
                "is_valid": is_valid,
            })

    if total_wells >= 2 and len(injectors) > 0 and len(producers) > 0 and not has_valid_pair:
        warnings.append(
            "CRITICAL: No injector-producer pair satisfies the 20% vertical perforation overlap criterion. Sweep efficiency severely compromised."
        )

    verdict = "PASSED" if (total_wells <= 1 or has_valid_pair or len(injectors) == 0) else "FLAGGED"

    return {
        "well_metrics": well_metrics,
        "interwell_pairs": interwell_pairs,
        "operating_pattern": pattern,
        "total_wells": total_wells,
        "n_injectors": len(injectors),
        "n_producers": len(producers),
        "has_valid_sweep_connection": has_valid_pair if (len(injectors) > 0 and len(producers) > 0) else True,
        "warnings": warnings,
        "overall_verdict": verdict,
    }
