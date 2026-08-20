"""ORBITAL DUET - Mission configuration.

This is the single place to edit mission, vehicle, environment, AOCS,
propulsion, visualization, and output assumptions.  SI units are used unless
the field name explicitly states otherwise.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


SCENARIO_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class MissionConfig:
    """Epoch, scheduler rates, and operational phase durations."""

    utc_epoch: str = "2025 MAR 20 12:00:00.000 (UTC)"
    dynamics_step_s: float = 2.0
    fsw_step_s: float = 0.5
    record_step_s: float = 10.0
    detumble_duration_s: float = 120.0
    attitude_acquisition_s: float = 240.0
    navigation_acquisition_s: float = 120.0
    maneuver_design_s: float = 10.0
    arrest_attitude_acquisition_s: float = 120.0
    final_verification_s: float = 600.0


@dataclass(frozen=True)
class OrbitConfig:
    """Initial chief orbit, expressed either as COEs or inertial Cartesian state."""

    state_format: str = "classical"  # "classical" or "cartesian"
    altitude_m: float = 500_000.0
    eccentricity: float = 0.001
    inclination_deg: float = 51.6
    raan_deg: float = 25.0
    argument_of_periapsis_deg: float = 15.0
    true_anomaly_deg: float = 5.0
    # Cartesian values are J2000-like inertial, Earth-centred [m] and [m/s].
    position_N_m: tuple[float, float, float] = (6_878_136.3, 0.0, 0.0)
    velocity_N_m_s: tuple[float, float, float] = (0.0, 4_700.0, 6_000.0)


@dataclass(frozen=True)
class DeploymentConfig:
    """Post-deployment deputy state relative to the chief RTN/Hill frame."""

    timing_s: float = 0.0
    position_offset_H_m: tuple[float, float, float] = (0.0, -20.0, 0.0)
    delta_v_H_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction_frame: str = "chief_RTN"
    apply_dispersions: bool = False
    position_sigma_m: float = 0.0
    delta_v_sigma_m_s: float = 0.0
    random_seed: int = 20250320


@dataclass(frozen=True)
class SpacecraftConfig:
    """Common bus properties for both independently propagated spacecraft."""

    mass_kg: float = 24.0
    principal_inertia_kg_m2: tuple[float, float, float] = (0.42, 0.38, 0.30)
    center_of_mass_B_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_sigma_BN: tuple[float, float, float] = (0.10, -0.15, 0.08)
    initial_omega_BN_B_rad_s: tuple[float, float, float] = (0.008, -0.006, 0.010)
    drag_area_m2: float = 0.12
    drag_coefficient: float = 2.2
    srp_area_m2: float = 0.10
    srp_reflection_coefficient: float = 1.3


@dataclass(frozen=True)
class PhasingConfig:
    """Differential-semi-major-axis acquisition target and guardrails."""

    target_along_track_separation_m: float = 10_000.0
    phasing_coast_duration_s: float = 7_200.0
    acquisition_position_tolerance_m: float = 2_000.0
    acquisition_drift_tolerance_m_s: float = 1.0
    earliest_maneuver_time_s: float = 300.0
    maximum_burn_duration_s: float = 90.0
    maximum_delta_v_m_s: float = 1.5


@dataclass(frozen=True)
class PropulsionConfig:
    """Opposed tangential thrusters mounted through the spacecraft centre of mass."""

    thrust_N: float = 0.50
    isp_s: float = 220.0
    cutoff_frequency_rad_s: float = 5.0
    location_B_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # The verified velocityPoint maneuver pattern maps +b3 to prograde through
    # a 90-degree reference-frame offset configured in aocs.py.
    prograde_direction_B: tuple[float, float, float] = (0.0, 0.0, 1.0)
    retrograde_direction_B: tuple[float, float, float] = (0.0, 0.0, -1.0)
    minimum_firing_s: float = 0.5
    model_propellant_depletion: bool = False


@dataclass(frozen=True)
class AocsConfig:
    """Representative sensor and actuator parameters, not vendor specifications."""

    attitude_navigation: str = "simple_nav_fallback"
    gyro_type: str = "mems"  # "mems" or "fog"
    imu_mems_noise_rad_s: float = 2.0e-5
    imu_fog_noise_rad_s: float = 2.0e-6
    imu_mems_bias_rad_s: tuple[float, float, float] = (2e-5, -1e-5, 1.5e-5)
    imu_fog_bias_rad_s: tuple[float, float, float] = (2e-6, -1e-6, 1.5e-6)
    imu_error_correlation_time_s: float = 3_600.0
    star_tracker_noise_rad: float = 20.0e-6
    tam_noise_T: float = 4.0e-9
    tam_bias_T: tuple[float, float, float] = (2e-9, -1e-9, 1e-9)
    tam_saturation_T: float = 70.0e-6
    tam_dcm_SB: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    css_noise: float = 0.01
    rw_model: str = "BCT_RWP015"
    # BCT_RWP015's 0.015 Nms limit is fixed by Basilisk's verified factory entry.
    rw_max_momentum_Nms: float = 0.015
    rw_initial_speed_rpm: float = 100.0
    mtq_max_dipole_Am2: float = 0.15
    controller_K: float = 0.01
    controller_P: float = 0.15
    controller_Ki: float = -1.0
    attitude_tolerance_deg: float = 2.0
    momentum_management_gain: float = 1.0e-4


@dataclass(frozen=True)
class EnvironmentConfig:
    """Fast/debug or practical high-fidelity LEO environment selection."""

    fidelity: str = "high"  # "fast" or "high"
    gravity_model: str = "GGM03S"
    gravity_file: str = "GGM03S.txt"  # resolved through Basilisk dataFetcher
    gravity_degree: int = 70
    gravity_order: int = 70
    atmosphere_model: str = "MSIS"
    enable_drag: bool = True
    enable_srp: bool = True
    enable_third_bodies: bool = True
    # Demonstration placeholders: replace with measured/predicted mission-date data.
    f107_daily_sfu: float = 110.0
    f107_81day_sfu: float = 110.0
    ap_daily: float = 8.0
    ap_history: tuple[float, ...] = (8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0)


@dataclass(frozen=True)
class VizardConfig:
    """Vizard output mode and Direct Communication settings."""

    mode: str = "playback"  # "off", "playback", or "live"
    live_acceleration_factor: float = 20.0
    direct_comm_address: str = "tcp://localhost:5556"


@dataclass(frozen=True)
class OutputConfig:
    """Recorder, diagnostics, plot, and output-directory settings."""

    save_plots: bool = True
    show_plots: bool = False
    print_diagnostics: bool = True
    plots_directory: Path = SCENARIO_DIR / "output" / "plots"
    vizard_directory: Path = SCENARIO_DIR / "output" / "vizard"
    vizard_filename: str = "orbital_duet.bin"


@dataclass(frozen=True)
class ScenarioConfig:
    """Complete editable ORBITAL DUET configuration."""

    mission: MissionConfig = field(default_factory=MissionConfig)
    orbit: OrbitConfig = field(default_factory=OrbitConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    spacecraft: SpacecraftConfig = field(default_factory=SpacecraftConfig)
    phasing: PhasingConfig = field(default_factory=PhasingConfig)
    propulsion: PropulsionConfig = field(default_factory=PropulsionConfig)
    aocs: AocsConfig = field(default_factory=AocsConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    vizard: VizardConfig = field(default_factory=VizardConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


CONFIG = ScenarioConfig()


def validate_configuration(config: ScenarioConfig) -> None:
    """Reject inconsistent inputs before Basilisk is initialized."""

    if config.environment.fidelity not in {"fast", "high"}:
        raise ValueError("Environment fidelity must be 'fast' or 'high'.")
    if config.vizard.mode not in {"off", "playback", "live"}:
        raise ValueError("Vizard mode must be 'off', 'playback', or 'live'.")
    if config.orbit.state_format not in {"classical", "cartesian"}:
        raise ValueError("Orbit state_format must be 'classical' or 'cartesian'.")
    if config.aocs.gyro_type not in {"mems", "fog"}:
        raise ValueError("GYRO type must be 'mems' or 'fog'.")
    if config.aocs.rw_model != "BCT_RWP015":
        raise ValueError("This readable scenario currently verifies only BCT_RWP015 wheels.")
    if not np.isclose(config.aocs.rw_max_momentum_Nms, 0.015):
        raise ValueError("BCT_RWP015 has a fixed 0.015 Nms momentum capacity in Basilisk.")
    if config.mission.dynamics_step_s <= 0.0 or config.mission.fsw_step_s <= 0.0:
        raise ValueError("Simulation task steps must be positive.")
    if config.environment.gravity_degree < 0:
        raise ValueError("Gravity degree cannot be negative.")
    if config.environment.gravity_file != "GGM03S.txt":
        raise ValueError(
            "This scenario currently verifies only Basilisk's registered GGM03S.txt."
        )
    if config.environment.gravity_order != config.environment.gravity_degree:
        raise ValueError(
            "Basilisk 2.11.1 exposes degree truncation only; configure gravity "
            "order equal to degree for the full triangular field."
        )
    if len(config.environment.ap_history) != 7:
        raise ValueError("MSIS Ap history must contain seven values.")
    if config.propulsion.thrust_N <= 0.0 or config.spacecraft.mass_kg <= 0.0:
        raise ValueError("Spacecraft mass and thruster force must be positive.")
    if config.deployment.timing_s != 0.0:
        raise ValueError(
            "ORBITAL DUET begins immediately after separation; deployment timing "
            "must remain 0 s until a carrier/separation effector is added."
        )


def print_configuration_summary(config: ScenarioConfig) -> None:
    """Print the effective mission settings before model construction."""

    print("\n" + "=" * 78)
    print("ORBITAL DUET - TWO-SPACECRAFT LEO PHASING DEMONSTRATOR")
    print("=" * 78)
    print(f"UTC epoch                    : {config.mission.utc_epoch}")
    print(f"Environment fidelity         : {config.environment.fidelity}")
    print(f"Dynamics / FSW steps         : {config.mission.dynamics_step_s:.2f} / "
          f"{config.mission.fsw_step_s:.2f} s")
    print(f"Initial orbit input          : {config.orbit.state_format}")
    print(f"Chief altitude               : {config.orbit.altitude_m / 1000.0:.1f} km")
    print(f"Initial deputy RTN offset    : {np.asarray(config.deployment.position_offset_H_m)} m")
    print(f"Target along-track geometry  : {config.phasing.target_along_track_separation_m:.1f} m")
    print(f"Planned phasing coast        : {config.phasing.phasing_coast_duration_s / 3600.0:.2f} h")
    print(f"Thruster force / Isp         : {config.propulsion.thrust_N:.3f} N / "
          f"{config.propulsion.isp_s:.1f} s")
    print(f"Gyro family                  : {config.aocs.gyro_type.upper()} demonstration defaults")
    print(f"Attitude navigation          : {config.aocs.attitude_navigation} (not GNSS)")
    print(f"Vizard mode                  : {config.vizard.mode}")
    print("=" * 78 + "\n")
