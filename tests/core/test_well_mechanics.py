"""
Unit tests for 3D Well Mechanics, Anisotropic Peaceman WI & Sweep Validation
============================================================================
"""

import pytest
import numpy as np
from core.data_models import WellData
from core.engine_surrogate.well_mechanics import (
    calculate_peaceman_index_horizontal,
    calculate_peaceman_index_vertical,
    calculate_vertical_perforation_overlap,
    calculate_interwell_transmissibility,
    generate_synthetic_well_trajectory,
    validate_well_network,
)
from core.engine_surrogate.profile_generator_fast import FastProfileGenerator


class TestPeacemanWellIndices:
    """Validate isotropic and anisotropic Peaceman well productivity indices."""

    def test_horizontal_peaceman_index_anisotropic(self):
        """SPE-10194: Anisotropic horizontal well index with kv/kh = 0.1."""
        ky = 100.0  # mD
        kz = 10.0   # mD (kv/kh = 0.1)
        lat_len = 2000.0  # ft
        dy = 100.0  # ft
        dz = 20.0   # ft
        rw = 0.354  # ft
        skin = 0.0
        mu = 1.5    # cP

        wi_horiz = calculate_peaceman_index_horizontal(
            ky_md=ky,
            kz_md=kz,
            length_lateral_ft=lat_len,
            dy_ft=dy,
            dz_ft=dz,
            r_w_ft=rw,
            skin=skin,
            mu_cp=mu,
        )

        assert wi_horiz > 0.0
        # WI should scale proportionally with lateral length
        wi_double = calculate_peaceman_index_horizontal(
            ky_md=ky,
            kz_md=kz,
            length_lateral_ft=lat_len * 2.0,
            dy_ft=dy,
            dz_ft=dz,
            r_w_ft=rw,
            skin=skin,
            mu_cp=mu,
        )
        assert np.isclose(wi_double, wi_horiz * 2.0, rtol=1e-2)

    def test_vertical_peaceman_index(self):
        """Peaceman (1978): Vertical well index calculation."""
        kx = 100.0
        ky = 100.0
        h_perf = 50.0
        dx = 100.0
        dy = 100.0
        rw = 0.354
        wi_vert = calculate_peaceman_index_vertical(
            kx_md=kx, ky_md=ky, h_perf_ft=h_perf, dx_ft=dx, dy_ft=dy, r_w_ft=rw, skin=0.0, mu_cp=1.0
        )
        assert wi_vert > 0.0
        # Positive skin reduces WI
        wi_damaged = calculate_peaceman_index_vertical(
            kx_md=kx, ky_md=ky, h_perf_ft=h_perf, dx_ft=dx, dy_ft=dy, r_w_ft=rw, skin=5.0, mu_cp=1.0
        )
        assert wi_damaged < wi_vert


class TestPerforationOverlapAndTransmissibility:
    """Validate 3D vertical perforation overlap and inter-well transmissibility."""

    def test_perforation_overlap_scenarios(self):
        # Scenario 1: Identical intervals (100% overlap)
        h_overlap, omega = calculate_vertical_perforation_overlap([(1000.0, 1100.0)], [(1000.0, 1100.0)])
        assert h_overlap == pytest.approx(100.0)
        assert omega == pytest.approx(1.0)

        # Scenario 2: Partial overlap (30 ft out of 100 ft = 30%)
        h_overlap, omega = calculate_vertical_perforation_overlap([(1000.0, 1100.0)], [(1070.0, 1170.0)])
        assert h_overlap == pytest.approx(30.0)
        assert omega == pytest.approx(0.30)

        # Scenario 3: Poor overlap (< 20%)
        h_overlap, omega = calculate_vertical_perforation_overlap([(1000.0, 1100.0)], [(1090.0, 1190.0)])
        assert h_overlap == pytest.approx(10.0)
        assert omega == pytest.approx(0.10)
        assert omega < 0.20

        # Scenario 4: Disjoint intervals (0% overlap)
        h_overlap, omega = calculate_vertical_perforation_overlap([(1000.0, 1050.0)], [(1100.0, 1150.0)])
        assert h_overlap == 0.0
        assert omega == 0.0

    def test_interwell_transmissibility(self):
        t_ij = calculate_interwell_transmissibility(
            x1_ft=0.0, y1_ft=0.0,
            x2_ft=1000.0, y2_ft=0.0,
            h_overlap_ft=50.0,
            k_h_md=100.0,
            mu_cp=1.0,
        )
        assert t_ij > 0.0

        # Zero overlap yields zero transmissibility
        t_zero = calculate_interwell_transmissibility(
            x1_ft=0.0, y1_ft=0.0,
            x2_ft=1000.0, y2_ft=0.0,
            h_overlap_ft=0.0,
            k_h_md=100.0,
            mu_cp=1.0,
        )
        assert t_zero == 0.0


class TestSyntheticTrajectoryAndWellData:
    """Validate 3D trajectory synthesis and WellData routing."""

    def test_horizontal_trajectory_geometry(self):
        pts = generate_synthetic_well_trajectory(
            surface_x=500.0,
            surface_y=500.0,
            top_tvd=1000.0,
            bottom_tvd=3000.0,
            trajectory_type="Horizontal",
            lateral_length_ft=1500.0,
            azimuth_deg=0.0,
        )
        assert pts.shape[1] == 3
        # Wellhead at surface coordinate
        assert pts[0, 0] == pytest.approx(500.0)
        assert pts[0, 1] == pytest.approx(500.0)
        assert pts[0, 2] == pytest.approx(1000.0)
        # Lateral landing depth
        assert pts[-1, 2] == pytest.approx(3000.0)
        # Lateral length along x
        total_lat_disp = pts[-1, 0] - pts[0, 0]
        assert total_lat_disp > 1000.0

    def test_welldata_peaceman_routing(self):
        # Horizontal well
        w_horiz = WellData(
            name="Horiz-1",
            depths=np.array([1000.0, 3000.0]),
            properties={},
            units={},
            metadata={"trajectory_type": "Horizontal", "lateral_length": 2000.0},
        )
        wi_h = w_horiz.calculate_peaceman_index(k_mD=100.0, h_ft=50.0, dx_ft=100.0, dy_ft=100.0)
        assert wi_h > 0.0

        # Vertical well
        w_vert = WellData(
            name="Vert-1",
            depths=np.array([1000.0, 3000.0]),
            properties={},
            units={},
            metadata={"trajectory_type": "Vertical"},
        )
        wi_v = w_vert.calculate_peaceman_index(k_mD=100.0, h_ft=50.0, dx_ft=100.0, dy_ft=100.0)
        assert wi_v > 0.0


class TestWellNetworkValidationAndIPRClamping:
    """Validate well network validation and Composite Vogel-Darcy IPR deliverability clamping."""

    def test_well_network_validation(self):
        inj = WellData(
            name="Inj-1",
            depths=np.array([1000.0, 3000.0]),
            properties={},
            units={},
            metadata={"type": "injector", "SurfaceX": 200.0, "SurfaceY": 200.0},
            perforations=[[1500.0, 1600.0]],
        )
        prod = WellData(
            name="Prod-1",
            depths=np.array([1000.0, 3000.0]),
            properties={},
            units={},
            metadata={"type": "producer", "SurfaceX": 800.0, "SurfaceY": 800.0},
            perforations=[[1520.0, 1620.0]],
        )

        res = validate_well_network([inj, prod], reservoir_k_md=100.0)
        assert res["overall_verdict"] == "PASSED"
        assert res["n_injectors"] == 1
        assert res["n_producers"] == 1
        assert len(res["interwell_pairs"]) == 1
        pair = res["interwell_pairs"][0]
        # Overlap = 1600 - 1520 = 80 ft (80% of min(100, 100))
        assert pair["overlap_ft"] == pytest.approx(80.0)
        assert pair["omega_pct"] == pytest.approx(80.0)
        assert pair["is_valid"] is True

    def test_composite_vogel_darcy_deliverability_clamping(self):
        """Enforce single-well deliverability clamping (q_o <= 1,000 BOPD)."""
        gen = FastProfileGenerator()
        # High PI and large drawdown
        q_raw = gen.calculate_composite_ipr_deliverability(
            p_res=3500.0, p_wf=1500.0, mmp=2000.0, pi=20.0, max_deliverability_bopd=1000.0
        )
        # Theoretical unclamped rate would be > 20 * 1500 = 30,000 STB/d
        # Clamped rate must strictly not exceed 1,000 BOPD
        assert q_raw <= 1000.0
        assert q_raw == pytest.approx(1000.0)
