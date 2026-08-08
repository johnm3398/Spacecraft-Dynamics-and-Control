"""BASILISK-X nadir-pointing attitude-control scenario.

Propagate a six-degree-of-freedom spacecraft around a point-mass Earth and
command the spacecraft +b1 body axis to point toward nadir.

The scenario uses Basilisk's Hill-frame guidance, truth navigation, attitude
tracking-error calculation, nonlinear MRP feedback controller, and an ideal
external body-torque effector.

Visualization-off, live Direct Communication, and recorded-playback modes are
supported using the same BASILISK-X Vizard workflow as basic_earth_orbit.

This is intentionally an ideal-actuator baseline. Reaction wheels, actuator
saturation, disturbances, flexible dynamics, and environmental perturbations
are omitted so that the attitude-guidance and control behavior can be studied
in isolation.
"""

from pathlib import Path
from time import monotonic
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import attTrackingError, hillPoint, mrpFeedback
from Basilisk.simulation import extForceTorque, planetEphemeris, simpleNav, spacecraft
from Basilisk.utilities import (
    RigidBodyKinematics,
    SimulationBaseClass,
    macros,
    orbitalMotion,
    simIncludeGravBody,
    simHelpers,
)

from basiliskx.visualization.vizard_launcher import (
    is_vizard_running,
    launch_vizard,
    launch_vizard_playback,
)


# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Visualization modes:
#   "off"      - run at full speed without configuring Vizard
#   "live"     - stream while propagation is paced relative to wall-clock time
#   "playback" - run at full speed, save a recording, then open it in Vizard
VISUALIZATION_MODE = "playback"  # "off", "live", or "playback"

# Live mode advances this many seconds of simulation per wall-clock second.
LIVE_ACCELERATION_FACTOR = 10.0
VIZARD_ADDRESS = "tcp://localhost:5556"

# Attitude dynamics and control require a substantially faster update rate than
# the basic translational orbit example.
DYNAMICS_DT_S = 0.1
SIMULATION_MINUTES = 45.0

# -----------------------------------------------------------------------------
# Spacecraft physical properties
# -----------------------------------------------------------------------------

SPACECRAFT_MASS_KG = 750.0

# Principal moments of inertia [kg m^2].
INERTIA_XX_KG_M2 = 900.0
INERTIA_YY_KG_M2 = 800.0
INERTIA_ZZ_KG_M2 = 600.0

# -----------------------------------------------------------------------------
# Initial orbit
# -----------------------------------------------------------------------------

ALTITUDE_M = 550e3
ECCENTRICITY = 0.001
INCLINATION_DEG = 53.0
RAAN_DEG = 30.0
ARG_PERIAPSIS_DEG = 0.0
TRUE_ANOMALY_DEG = 0.0

# -----------------------------------------------------------------------------
# Initial rotational state
# -----------------------------------------------------------------------------

# Modified Rodrigues Parameters describing the initial spacecraft attitude.
INITIAL_SIGMA_BN = np.array(
    [
        0.10,
        0.20,
        -0.30,
    ]
)

# Initial body angular velocity relative to the inertial frame [rad/s].
INITIAL_OMEGA_BN_B_RAD_S = np.array(
    [
        0.001,
        -0.010,
        0.030,
    ]
)

# -----------------------------------------------------------------------------
# Attitude controller
# -----------------------------------------------------------------------------

# MRP feedback gains.
CONTROL_K = 3.5
CONTROL_P = 30.0

# A negative Ki disables integral feedback in Basilisk's mrpFeedback module.
CONTROL_KI = -1.0

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

SAVE_PLOTS = True
SHOW_PLOTS = True


# =============================================================================
# SCENARIO-LOCAL OUTPUT PATHS
# =============================================================================

SCENARIO_DIR = Path(__file__).resolve().parent

PLOTS_DIR = SCENARIO_DIR / "output" / "plots"
VIZARD_DIR = SCENARIO_DIR / "output" / "vizard"

VIZARD_FILE = VIZARD_DIR / "nadir_pointing.bin"


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================


def validate_configuration(
    *,
    visualization_mode: str,
    live_acceleration_factor: float,
    dynamics_dt_s: float,
    simulation_minutes: float,
    altitude_m: float,
    eccentricity: float,
    spacecraft_mass_kg: float,
) -> str:
    """Validate user configuration and return the normalized Vizard mode."""

    mode = visualization_mode.strip().lower()

    if mode not in {"off", "live", "playback"}:
        raise ValueError(
            "VISUALIZATION_MODE must be 'off', 'live', or 'playback'."
        )

    if live_acceleration_factor <= 0.0:
        raise ValueError(
            "LIVE_ACCELERATION_FACTOR must be greater than zero."
        )

    if dynamics_dt_s <= 0.0:
        raise ValueError(
            "DYNAMICS_DT_S must be greater than zero."
        )

    if simulation_minutes <= 0.0:
        raise ValueError(
            "SIMULATION_MINUTES must be greater than zero."
        )

    if altitude_m < 0.0:
        raise ValueError(
            "ALTITUDE_M must be non-negative."
        )

    if not 0.0 <= eccentricity < 1.0:
        raise ValueError(
            "ECCENTRICITY must describe an elliptical orbit "
            "(0 <= e < 1)."
        )

    if spacecraft_mass_kg <= 0.0:
        raise ValueError(
            "SPACECRAFT_MASS_KG must be greater than zero."
        )

    return mode


# =============================================================================
# SIMULATION CONSTRUCTION
# =============================================================================


def build_simulation(
    *,
    mode: str,
    dynamics_dt_s: float,
    simulation_minutes: float,
    altitude_m: float,
    eccentricity: float,
    inclination_deg: float,
    raan_deg: float,
    argument_of_periapsis_deg: float,
    true_anomaly_deg: float,
    spacecraft_mass_kg: float,
    initial_sigma_bn: np.ndarray,
    initial_omega_bn_b_rad_s: np.ndarray,
    control_k: float,
    control_p: float,
    control_ki: float,
) -> tuple[
    Any,
    str,
    Any,
    Any,
    Any,
    Any,
    Any,
    Any,
    Any,
    float,
    float,
    float,
    int,
    int,
]:
    """Build the 6-DOF spacecraft, guidance chain, and ideal torque controller."""

    task_name = "dynamicsTask"

    # -------------------------------------------------------------------------
    # Simulation process and task
    # -------------------------------------------------------------------------

    simulation = SimulationBaseClass.SimBaseClass()
    simulation.SetProgressBar(mode != "live")

    process = simulation.CreateNewProcess(
        "dynamicsProcess"
    )

    time_step_ns = macros.sec2nano(
        dynamics_dt_s
    )

    process.addTask(
        simulation.CreateNewTask(
            task_name,
            time_step_ns,
        )
    )

    # -------------------------------------------------------------------------
    # Spacecraft rigid body
    # -------------------------------------------------------------------------

    spacecraft_object = spacecraft.Spacecraft()
    spacecraft_object.ModelTag = "BASILISK-X-NADIR-SAT"

    spacecraft_object.hub.mHub = (
        spacecraft_mass_kg
    )

    inertia_flat = [
        INERTIA_XX_KG_M2,
        0.0,
        0.0,
        0.0,
        INERTIA_YY_KG_M2,
        0.0,
        0.0,
        0.0,
        INERTIA_ZZ_KG_M2,
    ]

    spacecraft_object.hub.IHubPntBc_B = (
        simHelpers.np2EigenMatrix3d(
            inertia_flat
        )
    )

    spacecraft_object.hub.r_BcB_B = [
        [0.0],
        [0.0],
        [0.0],
    ]

    simulation.AddModelToTask(
        task_name,
        spacecraft_object,
    )

    # -------------------------------------------------------------------------
    # Earth gravity
    # -------------------------------------------------------------------------

    gravity_factory = (
        simIncludeGravBody.gravBodyFactory()
    )

    earth = gravity_factory.createEarth()
    earth.isCentralBody = True

    gravity_factory.addBodiesTo(
        spacecraft_object
    )

    # -------------------------------------------------------------------------
    # Initial translational state
    # -------------------------------------------------------------------------

    initial_elements: Any = (
        orbitalMotion.ClassicElements()
    )

    initial_elements.a = (
        earth.radEquator + altitude_m
    )

    initial_elements.e = eccentricity

    initial_elements.i = (
        inclination_deg * macros.D2R
    )

    initial_elements.Omega = (
        raan_deg * macros.D2R
    )

    initial_elements.omega = (
        argument_of_periapsis_deg
        * macros.D2R
    )

    initial_elements.f = (
        true_anomaly_deg
        * macros.D2R
    )

    initial_position_m, initial_velocity_m_s = (
        orbitalMotion.elem2rv(
            earth.mu,
            initial_elements,
        )
    )

    spacecraft_object.hub.r_CN_NInit = (
        initial_position_m
    )

    spacecraft_object.hub.v_CN_NInit = (
        initial_velocity_m_s
    )

    # -------------------------------------------------------------------------
    # Initial rotational state
    # -------------------------------------------------------------------------

    spacecraft_object.hub.sigma_BNInit = (
        initial_sigma_bn.tolist()
    )

    spacecraft_object.hub.omega_BN_BInit = (
        initial_omega_bn_b_rad_s.tolist()
    )

    # -------------------------------------------------------------------------
    # Ideal body-torque actuator
    # -------------------------------------------------------------------------

    torque_effector = (
        extForceTorque.ExtForceTorque()
    )

    torque_effector.ModelTag = (
        "idealBodyTorque"
    )

    spacecraft_object.addDynamicEffector(
        torque_effector
    )

    simulation.AddModelToTask(
        task_name,
        torque_effector,
    )

    # -------------------------------------------------------------------------
    # Truth navigation
    # -------------------------------------------------------------------------

    navigation = simpleNav.SimpleNav()
    navigation.ModelTag = "SimpleNavigation"

    navigation.scStateInMsg.subscribeTo(
        spacecraft_object.scStateOutMsg
    )

    simulation.AddModelToTask(
        task_name,
        navigation,
    )

    # -------------------------------------------------------------------------
    # Hill-frame guidance
    # -------------------------------------------------------------------------

    guidance = hillPoint.hillPoint()
    guidance.ModelTag = "HillPointGuidance"

    guidance.transNavInMsg.subscribeTo(
        navigation.transOutMsg
    )

    # Leave the optional central-body ephemeris input unconnected. Earth is the
    # inertial origin in this scenario, so hillPoint's default zero position
    # and velocity already describe the required central-body state.

    simulation.AddModelToTask(
        task_name,
        guidance,
    )

    # -------------------------------------------------------------------------
    # Attitude tracking error
    # -------------------------------------------------------------------------

    tracking_error = (
        attTrackingError.attTrackingError()
    )

    tracking_error.ModelTag = (
        "NadirTrackingError"
    )

    tracking_error.attRefInMsg.subscribeTo(
        guidance.attRefOutMsg
    )

    tracking_error.attNavInMsg.subscribeTo(
        navigation.attOutMsg
    )

    # Rotate the Hill reference by 90 degrees about its second axis so that
    # spacecraft +b3, the body z-axis, points toward Earth.
    tracking_error.sigma_R0R = [
        0.0,
        np.tan(np.pi / 8.0),
        0.0,
]

    simulation.AddModelToTask(
        task_name,
        tracking_error,
    )

    # -------------------------------------------------------------------------
    # Nonlinear MRP attitude controller
    # -------------------------------------------------------------------------

    controller = mrpFeedback.mrpFeedback()
    controller.ModelTag = "MRPFeedback"

    controller.guidInMsg.subscribeTo(
        tracking_error.attGuidOutMsg
    )

    controller.K = control_k
    controller.P = control_p
    controller.Ki = control_ki

    simulation.AddModelToTask(
        task_name,
        controller,
    )

    # The controller requires the spacecraft inertia tensor.
    vehicle_config_payload = (
        messaging.VehicleConfigMsgPayload(
            ISCPntB_B=inertia_flat
        )
    )

    vehicle_config_message = (
        messaging.VehicleConfigMsg().write(
            vehicle_config_payload
        )
    )

    controller.vehConfigInMsg.subscribeTo(
        vehicle_config_message
    )

    # Apply the commanded control torque directly to the spacecraft.
    torque_effector.cmdTorqueInMsg.subscribeTo(
        controller.cmdTorqueOutMsg
    )

    # -------------------------------------------------------------------------
    # Simulation duration
    # -------------------------------------------------------------------------

    orbital_period_s = (
        2.0
        * np.pi
        * np.sqrt(
            initial_elements.a**3
            / earth.mu
        )
    )

    duration_ns = macros.min2nano(
        simulation_minutes
    )

    # -------------------------------------------------------------------------
    # Recorders
    # -------------------------------------------------------------------------

    spacecraft_recorder = (
        spacecraft_object.scStateOutMsg.recorder()
    )

    guidance_recorder = (
        tracking_error.attGuidOutMsg.recorder()
    )

    torque_recorder = (
        controller.cmdTorqueOutMsg.recorder()
    )

    navigation_attitude_recorder = (
        navigation.attOutMsg.recorder()
    )

    simulation.AddModelToTask(
        task_name,
        spacecraft_recorder,
    )

    simulation.AddModelToTask(
        task_name,
        guidance_recorder,
    )

    simulation.AddModelToTask(
        task_name,
        torque_recorder,
    )

    simulation.AddModelToTask(
        task_name,
        navigation_attitude_recorder,
    )

    return (
        simulation,
        task_name,
        spacecraft_object,
        spacecraft_recorder,
        guidance_recorder,
        torque_recorder,
        navigation_attitude_recorder,
        controller,
        vehicle_config_message,
        earth.mu,
        earth.radEquator,
        orbital_period_s,
        duration_ns,
        time_step_ns,
    )


# =============================================================================
# VIZARD CONFIGURATION AND LIVE EXECUTION
# =============================================================================


def configure_visualization(
    simulation: Any,
    task_name: str,
    spacecraft_object: Any,
    *,
    mode: str,
    live_acceleration_factor: float,
    vizard_address: str,
) -> tuple[Any | None, Any | None]:
    """Configure Vizard only when requested by the selected mode."""

    if mode == "off":
        return None, None

    from Basilisk.utilities import vizSupport

    if not vizSupport.vizFound:
        raise RuntimeError(
            "This Basilisk installation has no Vizard interface."
        )

    if mode == "playback":
        VIZARD_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        viz = vizSupport.enableUnityVisualization(
            simulation,
            task_name,
            spacecraft_object,
            saveFile=str(VIZARD_FILE),
        )

        return viz, None

    # Live-only dependency.
    from Basilisk.simulation import simSynch

    clock_sync = simSynch.ClockSynch()
    clock_sync.accelFactor = (
        live_acceleration_factor
    )

    simulation.AddModelToTask(
        task_name,
        clock_sync,
    )

    viz = vizSupport.enableUnityVisualization(
        simulation,
        task_name,
        spacecraft_object,
        liveStream=True,
    )

    viz.settings.keyboardLiveInput = "pq"

    viz.reqComProtocol = "tcp"
    viz.reqComAddress = "0.0.0.0"

    viz.reqPortNumber = (
        vizard_address
        .rsplit(":", maxsplit=1)[-1]
    )

    return viz, clock_sync


def execute_live_simulation(
    simulation: Any,
    viz: Any,
    clock_sync: Any,
    duration_ns: int,
    time_step_ns: int,
    vizard_process: Any,
) -> bool:
    """Run live propagation while supporting pause and clean termination."""

    current_stop_ns = 0
    paused = False
    last_key_time = monotonic() - 1.0

    print(
        "Live controls: "
        "p = pause/resume, "
        "q = quit"
    )

    try:
        while current_stop_ns < duration_ns:

            if not is_vizard_running(
                vizard_process
            ):
                print(
                    "Vizard exited; stopping "
                    "the live simulation."
                )
                return False

            if paused:
                viz.UpdateState(
                    current_stop_ns
                )

                clock_sync.Reset(0)

            else:
                current_stop_ns = min(
                    current_stop_ns
                    + time_step_ns,
                    duration_ns,
                )

                simulation.ConfigureStopTime(
                    current_stop_ns
                )

                simulation.ExecuteSimulation()

            key_input = (
                viz.userInputMsg
                .read()
                .keyboardInput
            )

            now = monotonic()

            if (
                key_input
                and now - last_key_time >= 1.0
            ):
                last_key_time = now

                if "q" in key_input:
                    print(
                        "Vizard requested "
                        "simulation shutdown."
                    )

                    viz.liveSettings.terminateVizard = True

                    viz.UpdateState(
                        current_stop_ns
                    )

                    return False

                if "p" in key_input:
                    paused = not paused

                    clock_sync.Reset(0)

                    print(
                        "Simulation paused."
                        if paused
                        else "Simulation resumed."
                    )

    except KeyboardInterrupt:
        print(
            "\nSimulation interrupted; "
            "closing Vizard."
        )

        viz.liveSettings.terminateVizard = True

        viz.UpdateState(
            current_stop_ns
        )

        return False

    return True


# =============================================================================
# POINTING ANALYSIS
# =============================================================================


def compute_nadir_pointing_error(
    r_eci_m: np.ndarray,
    sigma_bn: np.ndarray,
) -> np.ndarray:
    """Return the angle between spacecraft +b3 and the local nadir vector."""

    pointing_error_deg = np.empty(
        len(r_eci_m)
    )

    for index, (
        position,
        sigma,
    ) in enumerate(
        zip(
            r_eci_m,
            sigma_bn,
        )
    ):
        radial_hat_n = (
            position
            / np.linalg.norm(position)
        )

        nadir_hat_n = (
            -radial_hat_n
        )

        # MRP2C(sigma_BN) returns [BN]. Each row therefore contains a body
        # basis vector expressed in the inertial frame.
        dcm_bn = (
            RigidBodyKinematics.MRP2C(
                sigma
            )
        )

        b3_hat_n = np.asarray(
            dcm_bn[2]
        )

        cosine_error = np.clip(
            np.dot(b3_hat_n, nadir_hat_n),
            -1.0,
            1.0,
        )

        pointing_error_deg[index] = (
            np.degrees(
                np.arccos(
                    cosine_error
                )
            )
        )

    return pointing_error_deg


def compute_ground_track(
    time_s: np.ndarray,
    r_eci_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return geocentric latitude and relative Earth-fixed longitude."""

    angle = (
        planetEphemeris.OMEGA_EARTH
        * time_s
    )

    cosine = np.cos(angle)
    sine = np.sin(angle)

    x_fixed = (
        cosine * r_eci_m[:, 0]
        + sine * r_eci_m[:, 1]
    )

    y_fixed = (
        -sine * r_eci_m[:, 0]
        + cosine * r_eci_m[:, 1]
    )

    z_fixed = r_eci_m[:, 2]

    longitude_deg = np.degrees(
        np.arctan2(
            y_fixed,
            x_fixed,
        )
    )

    latitude_deg = np.degrees(
        np.arctan2(
            z_fixed,
            np.hypot(
                x_fixed,
                y_fixed,
            ),
        )
    )

    return (
        latitude_deg,
        longitude_deg,
    )


# =============================================================================
# PLOTTING
# =============================================================================


def create_plots(
    time_s: np.ndarray,
    sigma_br: np.ndarray,
    omega_br_b_rad_s: np.ndarray,
    torque_nm: np.ndarray,
    pointing_error_deg: np.ndarray,
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
) -> dict[str, Figure]:
    """Create attitude-control and mission-context plots."""

    figures: dict[str, Figure] = {}

    time_min = time_s / 60.0

    # -------------------------------------------------------------------------
    # Attitude tracking error
    # -------------------------------------------------------------------------

    attitude_error_norm = np.linalg.norm(
        sigma_br,
        axis=1,
    )

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    axis.semilogy(
        time_min,
        attitude_error_norm,
    )

    axis.set(
        xlabel="Simulation time [min]",
        ylabel=r"$||\sigma_{B/R}||$ [-]",
        title="Attitude Tracking Error",
    )

    axis.grid(True)

    figure.tight_layout()

    figures["attitude_error"] = figure

    # -------------------------------------------------------------------------
    # Rate tracking error
    # -------------------------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    for index in range(3):
        axis.plot(
            time_min,
            omega_br_b_rad_s[:, index],
            label=rf"$\omega_{{BR,{index + 1}}}$",
        )

    axis.set(
        xlabel="Simulation time [min]",
        ylabel="Rate error [rad/s]",
        title="Angular-Rate Tracking Error",
    )

    axis.grid(True)
    axis.legend()

    figure.tight_layout()

    figures["rate_error"] = figure

    # -------------------------------------------------------------------------
    # Control torque
    # -------------------------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    for index in range(3):
        axis.plot(
            time_min,
            torque_nm[:, index],
            label=rf"$L_{{r,{index + 1}}}$",
        )

    axis.set(
        xlabel="Simulation time [min]",
        ylabel="Control torque [N m]",
        title="Commanded Body Torque",
    )

    axis.grid(True)
    axis.legend()

    figure.tight_layout()

    figures["control_torque"] = figure

    # -------------------------------------------------------------------------
    # Physical nadir-pointing error
    # -------------------------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    axis.plot(
        time_min,
        pointing_error_deg,
    )

    axis.set(
        xlabel="Simulation time [min]",
        ylabel="Pointing error [deg]",
        title="+b3 Nadir-Pointing Error",
    )

    axis.grid(True)

    figure.tight_layout()

    figures["nadir_pointing_error"] = (
        figure
    )

    # -------------------------------------------------------------------------
    # Ground track for mission context
    # -------------------------------------------------------------------------

    longitude_plot = (
        longitude_deg.copy()
    )

    latitude_plot = (
        latitude_deg.copy()
    )

    crossings = (
        np.where(
            np.abs(
                np.diff(
                    longitude_plot
                )
            )
            > 180.0
        )[0]
        + 1
    )

    longitude_plot[
        crossings
    ] = np.nan

    latitude_plot[
        crossings
    ] = np.nan

    figure, axis = plt.subplots(
        figsize=(11, 5.5)
    )

    axis.plot(
        longitude_plot,
        latitude_plot,
        linewidth=1.2,
    )

    axis.scatter(
        longitude_deg[0],
        latitude_deg[0],
        s=50,
        label="Start",
    )

    axis.set(
        xlim=(-180.0, 180.0),
        ylim=(-90.0, 90.0),
        xlabel="Relative longitude [deg]",
        ylabel="Geocentric latitude [deg]",
        title="Ground Track During Pointing Scenario",
    )

    axis.grid(True)
    axis.legend()

    figure.tight_layout()

    figures["ground_track"] = figure

    return figures


def save_figures(
    figures: dict[str, Figure],
) -> None:
    """Save all scenario figures under the local output directory."""

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, figure in figures.items():
        output_path = (
            PLOTS_DIR / f"{name}.png"
        )

        figure.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )

        print(
            f"Saved plot: {output_path}"
        )


# =============================================================================
# REPORTING
# =============================================================================


def print_configuration_summary(
    *,
    mode: str,
    orbital_period_s: float,
    simulation_minutes: float,
) -> None:
    """Print the configured orbit, spacecraft, and controller parameters."""

    print("\n" + "=" * 68)
    print("BASILISK-X NADIR POINTING")
    print("=" * 68)

    print(
        f"Visualization mode    : {mode}"
    )

    print(
        f"Altitude              : "
        f"{ALTITUDE_M / 1000.0:.3f} km"
    )

    print(
        f"Inclination           : "
        f"{INCLINATION_DEG:.3f} deg"
    )

    print(
        f"Orbital period        : "
        f"{orbital_period_s / 60.0:.3f} min"
    )

    print(
        f"Simulation duration   : "
        f"{simulation_minutes:.3f} min"
    )

    print(
        f"Dynamics time step    : "
        f"{DYNAMICS_DT_S:.3f} s"
    )

    print(
        f"Spacecraft mass       : "
        f"{SPACECRAFT_MASS_KG:.1f} kg"
    )

    print(
        "Principal inertia     : "
        f"[{INERTIA_XX_KG_M2:.1f}, "
        f"{INERTIA_YY_KG_M2:.1f}, "
        f"{INERTIA_ZZ_KG_M2:.1f}] kg m^2"
    )

    print(
        f"MRP feedback K        : "
        f"{CONTROL_K:.3f}"
    )

    print(
        f"MRP feedback P        : "
        f"{CONTROL_P:.3f}"
    )

    print(
        "Controlled axis       : "
        "+b3 toward nadir"
    )

    if mode == "live":
        print(
            f"Live acceleration     : "
            f"{LIVE_ACCELERATION_FACTOR:.1f}x"
        )

        print(
            f"Vizard address        : "
            f"{VIZARD_ADDRESS}"
        )

    elif mode == "playback":
        print(
            f"Playback file         : "
            f"{VIZARD_FILE}"
        )

    print("=" * 68 + "\n")


# =============================================================================
# SCENARIO DRIVER
# =============================================================================


def run(
    *,
    visualization_mode: str = VISUALIZATION_MODE,
    live_acceleration_factor: float = LIVE_ACCELERATION_FACTOR,
    vizard_address: str = VIZARD_ADDRESS,
    dynamics_dt_s: float = DYNAMICS_DT_S,
    simulation_minutes: float = SIMULATION_MINUTES,
    altitude_m: float = ALTITUDE_M,
    eccentricity: float = ECCENTRICITY,
    inclination_deg: float = INCLINATION_DEG,
    raan_deg: float = RAAN_DEG,
    argument_of_periapsis_deg: float = ARG_PERIAPSIS_DEG,
    true_anomaly_deg: float = TRUE_ANOMALY_DEG,
    spacecraft_mass_kg: float = SPACECRAFT_MASS_KG,
    initial_sigma_bn: np.ndarray = INITIAL_SIGMA_BN,
    initial_omega_bn_b_rad_s: np.ndarray = INITIAL_OMEGA_BN_B_RAD_S,
    control_k: float = CONTROL_K,
    control_p: float = CONTROL_P,
    control_ki: float = CONTROL_KI,
    save_plots: bool = SAVE_PLOTS,
    show_plots: bool = SHOW_PLOTS,
) -> None:
    """Execute the nadir-pointing scenario."""

    mode = validate_configuration(
        visualization_mode=visualization_mode,
        live_acceleration_factor=live_acceleration_factor,
        dynamics_dt_s=dynamics_dt_s,
        simulation_minutes=simulation_minutes,
        altitude_m=altitude_m,
        eccentricity=eccentricity,
        spacecraft_mass_kg=spacecraft_mass_kg,
    )

    (
        simulation,
        task_name,
        spacecraft_object,
        spacecraft_recorder,
        guidance_recorder,
        torque_recorder,
        attitude_recorder,
        _controller,
        _vehicle_config_message,
        _mu_earth,
        _earth_radius_m,
        orbital_period_s,
        duration_ns,
        time_step_ns,
    ) = build_simulation(
        mode=mode,
        dynamics_dt_s=dynamics_dt_s,
        simulation_minutes=simulation_minutes,
        altitude_m=altitude_m,
        eccentricity=eccentricity,
        inclination_deg=inclination_deg,
        raan_deg=raan_deg,
        argument_of_periapsis_deg=argument_of_periapsis_deg,
        true_anomaly_deg=true_anomaly_deg,
        spacecraft_mass_kg=spacecraft_mass_kg,
        initial_sigma_bn=initial_sigma_bn,
        initial_omega_bn_b_rad_s=initial_omega_bn_b_rad_s,
        control_k=control_k,
        control_p=control_p,
        control_ki=control_ki,
    )

    # Keep the standalone vehicle-configuration message alive while the
    # controller remains subscribed to it.

    viz, clock_sync = (
        configure_visualization(
            simulation,
            task_name,
            spacecraft_object,
            mode=mode,
            live_acceleration_factor=live_acceleration_factor,
            vizard_address=vizard_address,
        )
    )

    print_configuration_summary(
        mode=mode,
        orbital_period_s=orbital_period_s,
        simulation_minutes=simulation_minutes,
    )

    # -------------------------------------------------------------------------
    # Launch live Vizard before initialization
    # -------------------------------------------------------------------------

    vizard_process = None

    if mode == "live":
        print(
            "Launching Vizard in "
            "Direct Communication mode..."
        )

        vizard_process = launch_vizard(
            address=vizard_address
        )

    # -------------------------------------------------------------------------
    # Initialize simulation
    # -------------------------------------------------------------------------

    try:
        simulation.InitializeSimulation()

    except KeyboardInterrupt:
        print(
            "\nSimulation interrupted "
            "during initialization."
        )

        if vizard_process is not None:
            vizard_process.terminate()

        return

    print(
        "Executing Basilisk simulation..."
    )

    # -------------------------------------------------------------------------
    # Execute
    # -------------------------------------------------------------------------

    if mode == "live":
        completed = execute_live_simulation(
            simulation,
            viz,
            clock_sync,
            duration_ns,
            time_step_ns,
            vizard_process,
        )

        if not completed:
            print(
                "Simulation terminated early."
            )
            return

    else:
        simulation.ConfigureStopTime(
            duration_ns
        )

        simulation.ExecuteSimulation()

    print(
        "Basilisk simulation complete."
    )

    # -------------------------------------------------------------------------
    # Playback mode
    # -------------------------------------------------------------------------

    if mode == "playback":

        if not VIZARD_FILE.is_file():
            raise FileNotFoundError(
                "Simulation completed, but no "
                "Vizard recording was created at "
                f"{VIZARD_FILE}"
            )

        print(
            "Launching Vizard recorded playback..."
        )

        launch_vizard_playback(
            VIZARD_FILE
        )

    # -------------------------------------------------------------------------
    # Retrieve logged data
    # -------------------------------------------------------------------------

    time_s = (
        np.asarray(
            guidance_recorder.times()
        )
        * macros.NANO2SEC
    )

    sigma_br = np.asarray(
        guidance_recorder.sigma_BR
    )

    omega_br_b_rad_s = np.asarray(
        guidance_recorder.omega_BR_B
    )

    torque_nm = np.asarray(
        torque_recorder.torqueRequestBody
    )

    sigma_bn = np.asarray(
        attitude_recorder.sigma_BN
    )

    r_eci_m = np.asarray(
        spacecraft_recorder.r_BN_N
    )

    # -------------------------------------------------------------------------
    # Mission analysis
    # -------------------------------------------------------------------------

    pointing_error_deg = (
        compute_nadir_pointing_error(
            r_eci_m,
            sigma_bn,
        )
    )

    (
        latitude_deg,
        longitude_deg,
    ) = compute_ground_track(
        time_s,
        r_eci_m,
    )

    # -------------------------------------------------------------------------
    # Results summary
    # -------------------------------------------------------------------------

    attitude_error_norm = np.linalg.norm(
        sigma_br,
        axis=1,
    )

    initial_pointing_error_deg = (
        pointing_error_deg[0]
    )

    final_pointing_error_deg = (
        pointing_error_deg[-1]
    )

    maximum_control_torque_nm = np.max(
        np.linalg.norm(
            torque_nm,
            axis=1,
        )
    )

    print("\n" + "=" * 68)
    print("SIMULATION RESULTS")
    print("=" * 68)

    print(
        f"Samples recorded      : "
        f"{len(time_s)}"
    )

    print(
        f"Initial pointing error: "
        f"{initial_pointing_error_deg:.3f} deg"
    )

    print(
        f"Final pointing error  : "
        f"{final_pointing_error_deg:.6f} deg"
    )

    print(
        f"Final MRP error norm  : "
        f"{attitude_error_norm[-1]:.9e}"
    )

    print(
        f"Peak control torque   : "
        f"{maximum_control_torque_nm:.6f} N m"
    )

    print("=" * 68 + "\n")

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------

    figures = create_plots(
        time_s,
        sigma_br,
        omega_br_b_rad_s,
        torque_nm,
        pointing_error_deg,
        latitude_deg,
        longitude_deg,
    )

    if save_plots:
        save_figures(
            figures
        )

    if show_plots:
        plt.show()

    else:
        plt.close(
            "all"
        )


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    run(
        visualization_mode=VISUALIZATION_MODE,
        simulation_minutes=SIMULATION_MINUTES,
        dynamics_dt_s=DYNAMICS_DT_S,
    )
