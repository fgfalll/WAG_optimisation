"""
Level 2: Physical Verification - Boundary Conditions: Wellbore Drawdown Limits.

Tests producer deliverability behavior as reservoir pressure approaches bottomhole flowing pressure:
    q_prod -> 0 as P_res -> P_wf.
"""

import numpy as np
import pytest
from core.engine_surrogate.profile_generator_fast import FastProfileGenerator


def test_producer_rate_drawdown_limit():
    """
    Verify Composite Vogel-Darcy IPR rate approaches 0 as reservoir pressure P_res -> bottomhole pressure P_wf.
    """
    generator = FastProfileGenerator()
    pi = 2.0  # STB/d/psi
    mmp = 2500.0

    # Test P_res = 1000 psi, P_wf = 1000 psi (zero drawdown)
    p_res = 1000.0
    p_wf = 1000.0
    
    # Using composite Vogel formula:
    # Above MMP: q_above = 0 (since Pres < MMP)
    # Below MMP: q = q_max * (1 - 0.2*(Pwf/Pres) - 0.8*(Pwf/Pres)^2)
    # When Pwf = Pres, Pwf/Pres = 1.0 => (1 - 0.2(1) - 0.8(1)) = 0.0 => q = 0.
    q_ipr = generator.calculate_composite_ipr_deliverability(p_res=p_res, p_wf=p_wf, pi=pi, mmp=mmp)
    assert np.isclose(q_ipr, 0.0, atol=1e-5), f"IPR rate was non-zero at zero drawdown: {q_ipr}"
