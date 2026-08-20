"""ORBITAL DUET - Spacecraft truth models.

Creates two independent six-degree-of-freedom Basilisk Spacecraft objects and
their physical post-deployment inertial initial states.  Environment, AOCS,
propulsion, mission sequencing, and analysis are owned elsewhere.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from Basilisk.simulation import spacecraft
from Basilisk.utilities import macros, orbitalMotion, simHelpers

from config import ScenarioConfig


@dataclass
class SpacecraftPair:
    """The independently integrated chief and deputy truth vehicles."""

    chief: Any
    deputy: Any
    chief_initial_r_N_m: np.ndarray
    chief_initial_v_N_m_s: np.ndarray
    deputy_initial_r_N_m: np.ndarray
    deputy_initial_v_N_m_s: np.ndarray


def _configure_bus(name: str, config: ScenarioConfig) -> Any:
    """Create one rigid spacecraft hub with explicit SI mass properties."""

    vehicle = spacecraft.Spacecraft()
    vehicle.ModelTag = name
    vehicle.hub.mHub = config.spacecraft.mass_kg
    vehicle.hub.r_BcB_B = np.asarray(config.spacecraft.center_of_mass_B_m).tolist()
    inertia = np.diag(config.spacecraft.principal_inertia_kg_m2).reshape(9).tolist()
    vehicle.hub.IHubPntBc_B = simHelpers.np2EigenMatrix3d(inertia)
    vehicle.hub.sigma_BNInit = np.asarray(config.spacecraft.initial_sigma_BN).tolist()
    vehicle.hub.omega_BN_BInit = np.asarray(
        config.spacecraft.initial_omega_BN_B_rad_s
    ).tolist()
    return vehicle


def _chief_initial_state(config: ScenarioConfig, earth: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert the selected chief orbit representation to inertial r/v."""

    if config.orbit.state_format == "cartesian":
        return (
            np.asarray(config.orbit.position_N_m, dtype=float),
            np.asarray(config.orbit.velocity_N_m_s, dtype=float),
        )

    elements = orbitalMotion.ClassicElements()
    elements.a = earth.radEquator + config.orbit.altitude_m
    elements.e = config.orbit.eccentricity
    elements.i = config.orbit.inclination_deg * macros.D2R
    elements.Omega = config.orbit.raan_deg * macros.D2R
    elements.omega = config.orbit.argument_of_periapsis_deg * macros.D2R
    elements.f = config.orbit.true_anomaly_deg * macros.D2R
    r_N, v_N = orbitalMotion.elem2rv(earth.mu, elements)
    return np.asarray(r_N, dtype=float), np.asarray(v_N, dtype=float)


def build_spacecraft_pair(simulation: Any, dynamics_task: str, earth: Any, config: ScenarioConfig) -> SpacecraftPair:
    """Create both vehicles and construct the deputy state from chief RTN offsets."""

    chief = _configure_bus("ORBITAL-DUET-CHIEF", config)
    deputy = _configure_bus("ORBITAL-DUET-DEPUTY", config)

    chief_r_N, chief_v_N = _chief_initial_state(config, earth)
    position_H = np.asarray(config.deployment.position_offset_H_m, dtype=float)
    velocity_H = np.asarray(config.deployment.delta_v_H_m_s, dtype=float)

    if config.deployment.apply_dispersions:
        generator = np.random.default_rng(config.deployment.random_seed)
        position_H += generator.normal(0.0, config.deployment.position_sigma_m, 3)
        velocity_H += generator.normal(0.0, config.deployment.delta_v_sigma_m_s, 3)

    # H = [R,T,N]: radial outward, along-track with chief velocity, orbit normal.
    deputy_r_N, deputy_v_N = orbitalMotion.hill2rv(
        chief_r_N, chief_v_N, position_H, velocity_H
    )
    deputy_r_N = np.asarray(deputy_r_N, dtype=float)
    deputy_v_N = np.asarray(deputy_v_N, dtype=float)

    chief.hub.r_CN_NInit = chief_r_N
    chief.hub.v_CN_NInit = chief_v_N
    deputy.hub.r_CN_NInit = deputy_r_N
    deputy.hub.v_CN_NInit = deputy_v_N

    simulation.AddModelToTask(dynamics_task, chief, 100)
    simulation.AddModelToTask(dynamics_task, deputy, 100)

    return SpacecraftPair(
        chief=chief,
        deputy=deputy,
        chief_initial_r_N_m=chief_r_N,
        chief_initial_v_N_m_s=chief_v_N,
        deputy_initial_r_N_m=deputy_r_N,
        deputy_initial_v_N_m_s=deputy_v_N,
    )
