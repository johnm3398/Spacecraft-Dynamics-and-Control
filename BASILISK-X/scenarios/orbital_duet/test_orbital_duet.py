"""ORBITAL DUET - Focused configuration, physics, and smoke tests."""

from dataclasses import replace

import numpy as np

from config import CONFIG
from main import run
from mission_plan import first_order_set_drift_delta_v


def test_first_order_phasing_sign_convention() -> None:
    """A deputy commanded ahead must first lower its orbit with retrograde delta-v."""

    delta_v = first_order_set_drift_delta_v(0.0, 1_000.0, 1_000.0)
    assert np.isclose(delta_v, -1.0 / 3.0)


def test_fast_physical_burn_smoke() -> None:
    """Both spacecraft propagate and the deputy fires opposed physical thrusters."""

    config = replace(
        CONFIG,
        mission=replace(
            CONFIG.mission,
            detumble_duration_s=5.0,
            attitude_acquisition_s=240.0,
            navigation_acquisition_s=5.0,
            arrest_attitude_acquisition_s=60.0,
            final_verification_s=30.0,
        ),
        phasing=replace(
            CONFIG.phasing,
            target_along_track_separation_m=500.0,
            phasing_coast_duration_s=600.0,
            acquisition_position_tolerance_m=500.0,
        ),
        output=replace(CONFIG.output, save_plots=False, show_plots=False),
        vizard=replace(CONFIG.vizard, mode="off"),
    )
    result = run(config)

    assert result["spacecraft"].chief is not result["spacecraft"].deputy
    assert result["mission"].completed
    assert len(result["mission"].maneuvers) == 2
    assert result["mission"].maneuvers[0].signed_delta_v_m_s < 0.0
    assert result["mission"].maneuvers[1].signed_delta_v_m_s > 0.0

    prograde, retrograde = result["propulsion"].deputy.thrust_recorders
    assert np.max(prograde.thrustForce) > 0.0
    assert np.max(retrograde.thrustForce) > 0.0
    assert np.all(np.isfinite(result["data"]["relative_position_H_m"]))
