"""Illustrative BASILISK-X cooperative rendezvous near GEO.

Propagate an independently modelled target (chief) and servicer (deputy) in a
point-mass Earth gravity field. The translational scenario is described in the
target Hill frame and uses ideal impulsive velocity changes to move from a
passively safe relative orbit, through an inspection hold point, to a final
hold point near the target.

The servicer uses truth/SimpleNav navigation and Basilisk's
``locationPointing -> attTrackingError -> mrpFeedback`` chain to keep its +b1
boresight pointed toward the target. An ideal body-torque effector closes the
attitude-control loop. The target is disturbance-free, starts at zero angular
rate, and therefore remains inertially stabilised.

This is a low-fidelity engineering sandbox for learning and interview
preparation. It is not a flight-representative guidance, navigation, control,
collision-avoidance, capture, or contact-dynamics simulation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import attTrackingError, locationPointing, mrpFeedback
from Basilisk.simulation import extForceTorque, simpleNav, spacecraft
from Basilisk.utilities import (
    RigidBodyKinematics,
    SimulationBaseClass,
    macros,
    orbitalMotion,
    simHelpers,
    simIncludeGravBody,
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
#   "live"     - stream a paced simulation directly to Vizard
#   "playback" - run at full speed, save a recording, then open it in Vizard
VISUALIZATION_MODE = "live"  # "off", "live", or "playback"
LIVE_ACCELERATION_FACTOR = 100.0
VIZARD_ADDRESS = "tcp://localhost:5556"

# A one-second task rate resolves the ideal attitude-control transient while
# keeping the several-hour GEO rendezvous inexpensive to run.
DYNAMICS_DT_S = 1.0

# Circular, equatorial GEO target orbit. In this point-mass model the selected
# altitude produces approximately one sidereal-day orbital period.
GEO_ALTITUDE_M = 35_786e3
GEO_INCLINATION_DEG = 0.0
GEO_RAAN_DEG = 0.0
GEO_TRUE_ANOMALY_DEG = 0.0

# Hill coordinates are ordered [x, y, z]:
#   x - radial, positive away from Earth
#   y - along-track, positive in the target's direction of motion
#   z - orbit-normal, completing the right-handed frame
# The initial state describes a kilometre-scale bounded relative ellipse that
# remains well outside the target keep-out radius during the passive phase.
INITIAL_RELATIVE_POSITION_H_M = np.array([-500.0, -3000.0, 300.0])
INITIAL_RELATIVE_VELOCITY_H_M_S = np.array([0.0, 0.073, 0.0])

# Phase durations. Transfer velocities are computed from the linear
# Clohessy-Wiltshire state-transition equations and then applied to the full
# nonlinear Basilisk states as instantaneous velocity changes.
PASSIVE_SAFE_DURATION_HR = 2.0
INSPECTION_TRANSFER_DURATION_HR = 1.5
INSPECTION_HOLD_DURATION_HR = 0.5
TERMINAL_APPROACH_DURATION_HR = 1.0
FINAL_HOLD_DURATION_HR = 0.25

# Desired Hill-frame hold geometries. The inspection point is behind and above
# the target orbit plane; the final hold remains 50 m behind the target. No
# physical capture or contact is attempted.
INSPECTION_HOLD_POINT_H_M = np.array([0.0, -500.0, 100.0])
FINAL_HOLD_POINT_H_M = np.array([0.0, -50.0, 0.0])
HOLD_RELATIVE_VELOCITY_H_M_S = np.zeros(3)

# Guardrails for this illustrative maneuver sequence.
TARGET_KEEP_OUT_RADIUS_M = 10.0
MAX_SINGLE_IMPULSE_DELTA_V_M_S = 1.0

# Rigid-body properties. The target has no disturbance or control torque and
# begins at rest, which makes it an ideal inertially stabilised chief.
TARGET_MASS_KG = 2000.0
TARGET_PRINCIPAL_INERTIA_KG_M2 = np.array([2200.0, 2000.0, 1800.0])
TARGET_INITIAL_SIGMA_BN = np.zeros(3)
TARGET_INITIAL_OMEGA_BN_B_RAD_S = np.zeros(3)

SERVICER_MASS_KG = 750.0
SERVICER_PRINCIPAL_INERTIA_KG_M2 = np.array([900.0, 800.0, 600.0])
SERVICER_INITIAL_SIGMA_BN = np.array([0.10, 0.20, -0.30])
SERVICER_INITIAL_OMEGA_BN_B_RAD_S = np.array([0.001, -0.010, 0.030])

# The location-pointing guidance aligns this body-frame unit vector with the
# line of sight from the servicer to the target.
SERVICER_BORESIGHT_B = np.array([1.0, 0.0, 0.0])

# Nonlinear MRP feedback gains. A negative Ki disables integral feedback.
CONTROL_K = 3.5
CONTROL_P = 30.0
CONTROL_KI = -1.0

SAVE_PLOTS = True
SHOW_PLOTS = True


# =============================================================================
# SCENARIO-LOCAL OUTPUT PATHS
# =============================================================================

SCENARIO_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCENARIO_DIR / "output" / "plots"
VIZARD_DIR = SCENARIO_DIR / "output" / "vizard"
VIZARD_FILE = VIZARD_DIR / "cooperative_geo_rendezvous.bin"


# =============================================================================
# SMALL DATA CONTAINERS
# =============================================================================


@dataclass(frozen=True)
class ManeuverEvent:
    """One ideal impulse expressed in the target Hill frame."""

    label: str
    time_s: float
    relative_position_h_m: np.ndarray
    delta_v_h_m_s: np.ndarray


@dataclass(frozen=True)
class PhaseWindow:
    """Start and end time of one rendezvous phase."""

    label: str
    start_s: float
    end_s: float


@dataclass
class LiveControlState:
    """Keyboard state retained across all live rendezvous phases."""

    paused: bool = False
    last_key_time: float = field(default_factory=lambda: monotonic() - 1.0)


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================


def _validate_vector(name: str, value: np.ndarray) -> None:
    """Require a finite three-component configuration vector."""
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite three-component vector.")


def validate_configuration(
    *,
    visualization_mode: str,
    live_acceleration_factor: float,
    dynamics_dt_s: float,
    geo_altitude_m: float,
    initial_relative_position_h_m: np.ndarray,
    initial_relative_velocity_h_m_s: np.ndarray,
    inspection_hold_point_h_m: np.ndarray,
    final_hold_point_h_m: np.ndarray,
    phase_durations_hr: tuple[float, float, float, float, float],
    target_keep_out_radius_m: float,
    max_single_impulse_delta_v_m_s: float,
    servicer_boresight_b: np.ndarray,
) -> str:
    """Validate the editable settings and return the normalized Vizard mode."""
    mode = visualization_mode.strip().lower()
    if mode not in {"off", "live", "playback"}:
        raise ValueError("VISUALIZATION_MODE must be 'off', 'live', or 'playback'.")
    if live_acceleration_factor <= 0.0:
        raise ValueError("LIVE_ACCELERATION_FACTOR must be greater than zero.")
    if dynamics_dt_s <= 0.0:
        raise ValueError("DYNAMICS_DT_S must be greater than zero.")
    if geo_altitude_m <= 0.0:
        raise ValueError("GEO_ALTITUDE_M must be greater than zero.")
    if any(duration <= 0.0 for duration in phase_durations_hr):
        raise ValueError("Every rendezvous phase duration must be greater than zero.")
    if target_keep_out_radius_m <= 0.0:
        raise ValueError("TARGET_KEEP_OUT_RADIUS_M must be greater than zero.")
    if max_single_impulse_delta_v_m_s <= 0.0:
        raise ValueError(
            "MAX_SINGLE_IMPULSE_DELTA_V_M_S must be greater than zero."
        )

    _validate_vector("INITIAL_RELATIVE_POSITION_H_M", initial_relative_position_h_m)
    _validate_vector(
        "INITIAL_RELATIVE_VELOCITY_H_M_S", initial_relative_velocity_h_m_s
    )
    _validate_vector("INSPECTION_HOLD_POINT_H_M", inspection_hold_point_h_m)
    _validate_vector("FINAL_HOLD_POINT_H_M", final_hold_point_h_m)
    _validate_vector("SERVICER_BORESIGHT_B", servicer_boresight_b)

    if np.linalg.norm(initial_relative_position_h_m) <= target_keep_out_radius_m:
        raise ValueError("The initial servicer position is inside the keep-out radius.")
    if np.linalg.norm(inspection_hold_point_h_m) <= target_keep_out_radius_m:
        raise ValueError("The inspection hold point is inside the keep-out radius.")
    if np.linalg.norm(final_hold_point_h_m) <= target_keep_out_radius_m:
        raise ValueError("The final hold point is inside the keep-out radius.")
    if np.linalg.norm(servicer_boresight_b) <= np.finfo(float).eps:
        raise ValueError("SERVICER_BORESIGHT_B must have non-zero length.")

    return mode


# =============================================================================
# SIMULATION CONSTRUCTION
# =============================================================================


def build_simulation(
    *,
    mode: str,
    dynamics_dt_s: float,
    geo_altitude_m: float,
    geo_inclination_deg: float,
    geo_raan_deg: float,
    geo_true_anomaly_deg: float,
    initial_relative_position_h_m: np.ndarray,
    initial_relative_velocity_h_m_s: np.ndarray,
    servicer_boresight_b: np.ndarray,
    control_k: float,
    control_p: float,
    control_ki: float,
) -> tuple[Any, str, Any, Any, Any, Any, Any, float, float, float, int]:
    """Build two 6-DOF spacecraft and the servicer target-pointing loop."""
    task_name = "dynamicsTask"
    simulation = SimulationBaseClass.SimBaseClass()
    simulation.SetProgressBar(mode != "live")

    process = simulation.CreateNewProcess("dynamicsProcess")
    time_step_ns = macros.sec2nano(dynamics_dt_s)
    process.addTask(simulation.CreateNewTask(task_name, time_step_ns))

    # -------------------------------------------------------------------------
    # Independently propagated target and servicer rigid bodies
    # -------------------------------------------------------------------------

    target = spacecraft.Spacecraft()
    target.ModelTag = "GEO-TARGET"
    target.hub.mHub = TARGET_MASS_KG
    target_inertia = np.diag(TARGET_PRINCIPAL_INERTIA_KG_M2).reshape(9).tolist()
    target.hub.IHubPntBc_B = simHelpers.np2EigenMatrix3d(target_inertia)
    target.hub.sigma_BNInit = TARGET_INITIAL_SIGMA_BN.tolist()
    target.hub.omega_BN_BInit = TARGET_INITIAL_OMEGA_BN_B_RAD_S.tolist()

    servicer = spacecraft.Spacecraft()
    servicer.ModelTag = "GEO-SERVICER"
    servicer.hub.mHub = SERVICER_MASS_KG
    servicer_inertia = (
        np.diag(SERVICER_PRINCIPAL_INERTIA_KG_M2).reshape(9).tolist()
    )
    servicer.hub.IHubPntBc_B = simHelpers.np2EigenMatrix3d(servicer_inertia)
    servicer.hub.sigma_BNInit = SERVICER_INITIAL_SIGMA_BN.tolist()
    servicer.hub.omega_BN_BInit = SERVICER_INITIAL_OMEGA_BN_B_RAD_S.tolist()

    simulation.AddModelToTask(task_name, target)
    simulation.AddModelToTask(task_name, servicer)

    # A point-mass Earth acts on each Spacecraft module independently.
    gravity_factory = simIncludeGravBody.gravBodyFactory()
    earth = gravity_factory.createEarth()
    earth.isCentralBody = True
    gravity_factory.addBodiesTo(target)
    gravity_factory.addBodiesTo(servicer)

    # The target defines the chief orbit from which the Hill frame is formed.
    target_elements: Any = orbitalMotion.ClassicElements()
    target_elements.a = earth.radEquator + geo_altitude_m
    target_elements.e = 0.0
    target_elements.i = geo_inclination_deg * macros.D2R
    target_elements.Omega = geo_raan_deg * macros.D2R
    target_elements.omega = 0.0
    target_elements.f = geo_true_anomaly_deg * macros.D2R

    target_position_n_m, target_velocity_n_m_s = orbitalMotion.elem2rv(
        earth.mu, target_elements
    )
    target.hub.r_CN_NInit = target_position_n_m
    target.hub.v_CN_NInit = target_velocity_n_m_s

    # hill2rv maps the user-friendly initial deputy state into the inertial
    # Cartesian state required by the second Basilisk spacecraft propagator.
    servicer_position_n_m, servicer_velocity_n_m_s = orbitalMotion.hill2rv(
        target_position_n_m,
        target_velocity_n_m_s,
        np.asarray(initial_relative_position_h_m, dtype=float),
        np.asarray(initial_relative_velocity_h_m_s, dtype=float),
    )
    servicer.hub.r_CN_NInit = servicer_position_n_m
    servicer.hub.v_CN_NInit = servicer_velocity_n_m_s

    # -------------------------------------------------------------------------
    # Truth navigation and servicer target-pointing attitude control
    # -------------------------------------------------------------------------

    target_navigation = simpleNav.SimpleNav()
    target_navigation.ModelTag = "TargetTruthNavigation"
    target_navigation.scStateInMsg.subscribeTo(target.scStateOutMsg)
    simulation.AddModelToTask(task_name, target_navigation)

    servicer_navigation = simpleNav.SimpleNav()
    servicer_navigation.ModelTag = "ServicerTruthNavigation"
    servicer_navigation.scStateInMsg.subscribeTo(servicer.scStateOutMsg)
    simulation.AddModelToTask(task_name, servicer_navigation)

    pointing_guidance = locationPointing.locationPointing()
    pointing_guidance.ModelTag = "ServicerTargetPointing"
    pointing_guidance.pHat_B = (
        np.asarray(servicer_boresight_b, dtype=float)
        / np.linalg.norm(servicer_boresight_b)
    ).tolist()
    pointing_guidance.useBoresightRateDamping = 1
    pointing_guidance.scTargetInMsg.subscribeTo(target_navigation.transOutMsg)
    pointing_guidance.scTransInMsg.subscribeTo(servicer_navigation.transOutMsg)
    pointing_guidance.scAttInMsg.subscribeTo(servicer_navigation.attOutMsg)
    simulation.AddModelToTask(task_name, pointing_guidance)

    tracking_error = attTrackingError.attTrackingError()
    tracking_error.ModelTag = "ServicerTargetTrackingError"
    tracking_error.attRefInMsg.subscribeTo(pointing_guidance.attRefOutMsg)
    tracking_error.attNavInMsg.subscribeTo(servicer_navigation.attOutMsg)
    simulation.AddModelToTask(task_name, tracking_error)

    controller = mrpFeedback.mrpFeedback()
    controller.ModelTag = "ServicerMRPFeedback"
    controller.guidInMsg.subscribeTo(tracking_error.attGuidOutMsg)
    controller.K = control_k
    controller.P = control_p
    controller.Ki = control_ki
    simulation.AddModelToTask(task_name, controller)

    # Keep this standalone configuration message alive for the full run.
    vehicle_config_payload = messaging.VehicleConfigMsgPayload(
        ISCPntB_B=servicer_inertia
    )
    vehicle_config_message = messaging.VehicleConfigMsg().write(
        vehicle_config_payload
    )
    controller.vehConfigInMsg.subscribeTo(vehicle_config_message)

    ideal_torque = extForceTorque.ExtForceTorque()
    ideal_torque.ModelTag = "ServicerIdealBodyTorque"
    ideal_torque.cmdTorqueInMsg.subscribeTo(controller.cmdTorqueOutMsg)
    servicer.addDynamicEffector(ideal_torque)
    simulation.AddModelToTask(task_name, ideal_torque)

    # Record truth states for Hill-frame and physical pointing analysis.
    target_recorder = target.scStateOutMsg.recorder()
    servicer_recorder = servicer.scStateOutMsg.recorder()
    simulation.AddModelToTask(task_name, target_recorder)
    simulation.AddModelToTask(task_name, servicer_recorder)

    mean_motion_rad_s = np.sqrt(earth.mu / target_elements.a**3)
    orbital_period_s = 2.0 * np.pi / mean_motion_rad_s

    return (
        simulation,
        task_name,
        target,
        servicer,
        target_recorder,
        servicer_recorder,
        vehicle_config_message,
        earth.radEquator,
        mean_motion_rad_s,
        orbital_period_s,
        time_step_ns,
    )


# =============================================================================
# RELATIVE-MOTION AND RENDEZVOUS UTILITIES
# =============================================================================


def get_relative_state_hill(
    target_position_state: Any,
    target_velocity_state: Any,
    servicer_position_state: Any,
    servicer_velocity_state: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read current states and return target and relative Hill-frame states."""
    target_position_n_m = np.asarray(
        simHelpers.EigenVector3d2np(target_position_state.getState()), dtype=float
    )
    target_velocity_n_m_s = np.asarray(
        simHelpers.EigenVector3d2np(target_velocity_state.getState()), dtype=float
    )
    servicer_position_n_m = np.asarray(
        simHelpers.EigenVector3d2np(servicer_position_state.getState()), dtype=float
    )
    servicer_velocity_n_m_s = np.asarray(
        simHelpers.EigenVector3d2np(servicer_velocity_state.getState()), dtype=float
    )

    relative_position_h_m, relative_velocity_h_m_s = orbitalMotion.rv2hill(
        target_position_n_m,
        target_velocity_n_m_s,
        servicer_position_n_m,
        servicer_velocity_n_m_s,
    )
    return (
        target_position_n_m,
        target_velocity_n_m_s,
        np.asarray(relative_position_h_m, dtype=float),
        np.asarray(relative_velocity_h_m_s, dtype=float),
    )


def cw_targeting_velocity(
    initial_position_h_m: np.ndarray,
    target_position_h_m: np.ndarray,
    transfer_duration_s: float,
    mean_motion_rad_s: float,
) -> np.ndarray:
    """Return the initial Hill velocity that targets a final Hill position.

    This uses the linear Clohessy-Wiltshire solution about the circular target
    orbit. Basilisk still propagates the resulting inertial state with the full
    nonlinear two-body equations, so a small terminal position error is normal.
    """
    nt = mean_motion_rad_s * transfer_duration_s
    cosine = np.cos(nt)
    sine = np.sin(nt)
    n = mean_motion_rad_s

    phi_rr = np.array(
        [
            [4.0 - 3.0 * cosine, 0.0, 0.0],
            [6.0 * (sine - nt), 1.0, 0.0],
            [0.0, 0.0, cosine],
        ],
        dtype=float,
    )
    phi_rv = np.array(
        [
            [sine / n, 2.0 * (1.0 - cosine) / n, 0.0],
            [-2.0 * (1.0 - cosine) / n, (4.0 * sine - 3.0 * nt) / n, 0.0],
            [0.0, 0.0, sine / n],
        ],
        dtype=float,
    )

    right_hand_side = np.asarray(target_position_h_m, dtype=float) - (
        phi_rr @ np.asarray(initial_position_h_m, dtype=float)
    )
    try:
        return np.linalg.solve(phi_rv, right_hand_side)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "The selected transfer duration makes the CW targeting matrix singular; "
            "choose a different duration."
        ) from error


def apply_relative_velocity_impulse(
    *,
    label: str,
    time_s: float,
    commanded_relative_velocity_h_m_s: np.ndarray,
    target_position_state: Any,
    target_velocity_state: Any,
    servicer_position_state: Any,
    servicer_velocity_state: Any,
    max_single_impulse_delta_v_m_s: float,
) -> ManeuverEvent:
    """Instantaneously replace deputy velocity while preserving its position."""
    (
        target_position_n_m,
        target_velocity_n_m_s,
        relative_position_h_m,
        current_relative_velocity_h_m_s,
    ) = get_relative_state_hill(
        target_position_state,
        target_velocity_state,
        servicer_position_state,
        servicer_velocity_state,
    )

    commanded_velocity = np.asarray(commanded_relative_velocity_h_m_s, dtype=float)
    delta_v_h_m_s = commanded_velocity - current_relative_velocity_h_m_s
    delta_v_magnitude_m_s = float(np.linalg.norm(delta_v_h_m_s))
    if delta_v_magnitude_m_s > max_single_impulse_delta_v_m_s:
        raise RuntimeError(
            f"{label} requires {delta_v_magnitude_m_s:.3f} m/s, exceeding the "
            f"configured {max_single_impulse_delta_v_m_s:.3f} m/s guardrail."
        )

    # hill2rv includes both frame rotation and target translation, yielding the
    # inertial servicer velocity corresponding to the commanded Hill velocity.
    _, commanded_servicer_velocity_n_m_s = orbitalMotion.hill2rv(
        target_position_n_m,
        target_velocity_n_m_s,
        relative_position_h_m,
        commanded_velocity,
    )
    servicer_velocity_state.setState(commanded_servicer_velocity_n_m_s)

    return ManeuverEvent(
        label=label,
        time_s=time_s,
        relative_position_h_m=relative_position_h_m.copy(),
        delta_v_h_m_s=delta_v_h_m_s.copy(),
    )


# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================


def configure_visualization(
    simulation: Any,
    task_name: str,
    target: Any,
    servicer: Any,
    *,
    mode: str,
    live_acceleration_factor: float,
    vizard_address: str,
) -> tuple[Any | None, Any | None]:
    """Configure BASILISK-X playback or live visualization for both vehicles."""
    if mode == "off":
        return None, None

    from Basilisk.utilities import vizSupport

    if not vizSupport.vizFound:
        raise RuntimeError("This Basilisk installation has no Vizard interface.")

    spacecraft_list = [target, servicer]
    if mode == "playback":
        VIZARD_DIR.mkdir(parents=True, exist_ok=True)
        viz = vizSupport.enableUnityVisualization(
            simulation,
            task_name,
            spacecraft_list,
            saveFile=str(VIZARD_FILE),
        )
    else:
        from Basilisk.simulation import simSynch

        clock_sync = simSynch.ClockSynch()
        clock_sync.accelFactor = live_acceleration_factor
        simulation.AddModelToTask(task_name, clock_sync)

        viz = vizSupport.enableUnityVisualization(
            simulation,
            task_name,
            spacecraft_list,
            liveStream=True,
        )
        viz.settings.keyboardLiveInput = "pq"
        viz.reqComProtocol = "tcp"
        viz.reqComAddress = "0.0.0.0"
        viz.reqPortNumber = vizard_address.rsplit(":", maxsplit=1)[-1]

    # The target is the first spacecraft, so mode 2 draws relative trajectories
    # about the chief. Labels make the two vehicles unambiguous in playback.
    viz.settings.showSpacecraftLabels = 1
    viz.settings.trueTrajectoryLinesOn = 2
    viz.settings.orbitLinesOn = 2
    viz.settings.mainCameraTarget = target.ModelTag

    if mode == "playback":
        return viz, None
    return viz, clock_sync


# =============================================================================
# PHASED SIMULATION EXECUTION
# =============================================================================


def advance_simulation_phase(
    *,
    simulation: Any,
    phase_label: str,
    current_time_ns: int,
    duration_s: float,
    time_step_ns: int,
    mode: str,
    viz: Any,
    clock_sync: Any,
    vizard_process: Any,
    live_controls: LiveControlState,
) -> tuple[int, bool]:
    """Advance one phase, with keyboard and process checks in live mode."""
    end_time_ns = current_time_ns + macros.sec2nano(duration_s)
    print(f"  Phase: {phase_label} ({duration_s / 3600.0:.2f} hr)")

    if mode != "live":
        simulation.ConfigureStopTime(end_time_ns)
        simulation.ExecuteSimulation()
        return end_time_ns, True

    try:
        while current_time_ns < end_time_ns:
            if vizard_process is None or not is_vizard_running(vizard_process):
                print("Vizard exited; stopping the live simulation.")
                return current_time_ns, False

            if live_controls.paused:
                viz.UpdateState(current_time_ns)
                clock_sync.Reset(0)
            else:
                current_time_ns = min(
                    current_time_ns + time_step_ns,
                    end_time_ns,
                )
                simulation.ConfigureStopTime(current_time_ns)
                simulation.ExecuteSimulation()

            if not is_vizard_running(vizard_process):
                print("Vizard exited; stopping the live simulation.")
                return current_time_ns, False

            key_input = viz.userInputMsg.read().keyboardInput
            now = monotonic()
            if key_input and now - live_controls.last_key_time >= 1.0:
                live_controls.last_key_time = now
                if "q" in key_input:
                    print("Vizard requested simulation shutdown.")
                    viz.liveSettings.terminateVizard = True
                    viz.UpdateState(current_time_ns)
                    return current_time_ns, False
                if "p" in key_input:
                    live_controls.paused = not live_controls.paused
                    clock_sync.Reset(0)
                    print(
                        "Simulation paused."
                        if live_controls.paused
                        else "Simulation resumed."
                    )
    except KeyboardInterrupt:
        print("\nSimulation interrupted; closing Vizard.")
        viz.liveSettings.terminateVizard = True
        viz.UpdateState(current_time_ns)
        return current_time_ns, False

    return end_time_ns, True


def execute_rendezvous_phases(
    *,
    simulation: Any,
    target: Any,
    servicer: Any,
    mean_motion_rad_s: float,
    time_step_ns: int,
    mode: str,
    viz: Any,
    clock_sync: Any,
    vizard_process: Any,
    passive_safe_duration_hr: float,
    inspection_transfer_duration_hr: float,
    inspection_hold_duration_hr: float,
    terminal_approach_duration_hr: float,
    final_hold_duration_hr: float,
    inspection_hold_point_h_m: np.ndarray,
    final_hold_point_h_m: np.ndarray,
    hold_relative_velocity_h_m_s: np.ndarray,
    max_single_impulse_delta_v_m_s: float,
) -> tuple[list[ManeuverEvent], list[PhaseWindow], bool]:
    """Execute the passive, inspection, terminal, and final-hold sequence."""
    target_position_state = target.dynManager.getStateObject(
        target.hub.nameOfHubPosition
    )
    target_velocity_state = target.dynManager.getStateObject(
        target.hub.nameOfHubVelocity
    )
    servicer_position_state = servicer.dynManager.getStateObject(
        servicer.hub.nameOfHubPosition
    )
    servicer_velocity_state = servicer.dynManager.getStateObject(
        servicer.hub.nameOfHubVelocity
    )

    current_time_ns = 0
    maneuvers: list[ManeuverEvent] = []
    phase_windows: list[PhaseWindow] = []
    live_controls = LiveControlState()

    def run_phase(label: str, duration_hr: float) -> bool:
        nonlocal current_time_ns
        start_s = current_time_ns * macros.NANO2SEC
        current_time_ns, completed = advance_simulation_phase(
            simulation=simulation,
            phase_label=label,
            current_time_ns=current_time_ns,
            duration_s=duration_hr * 3600.0,
            time_step_ns=time_step_ns,
            mode=mode,
            viz=viz,
            clock_sync=clock_sync,
            vizard_process=vizard_process,
            live_controls=live_controls,
        )
        phase_windows.append(
            PhaseWindow(
                label=label,
                start_s=start_s,
                end_s=current_time_ns * macros.NANO2SEC,
            )
        )
        return completed

    # Phase 1: natural motion only. This demonstrates the initial passive-safe
    # geometry before any active rendezvous command is applied.
    if not run_phase("Passive-safe relative orbit", passive_safe_duration_hr):
        return maneuvers, phase_windows, False

    # DV1 targets the inspection point at the end of the transfer interval.
    _, _, relative_position_h_m, _ = get_relative_state_hill(
        target_position_state,
        target_velocity_state,
        servicer_position_state,
        servicer_velocity_state,
    )
    inspection_transfer_s = inspection_transfer_duration_hr * 3600.0
    inspection_departure_velocity_h_m_s = cw_targeting_velocity(
        relative_position_h_m,
        inspection_hold_point_h_m,
        inspection_transfer_s,
        mean_motion_rad_s,
    )
    maneuvers.append(
        apply_relative_velocity_impulse(
            label="DV1 inspection transfer injection",
            time_s=current_time_ns * macros.NANO2SEC,
            commanded_relative_velocity_h_m_s=(
                inspection_departure_velocity_h_m_s
            ),
            target_position_state=target_position_state,
            target_velocity_state=target_velocity_state,
            servicer_position_state=servicer_position_state,
            servicer_velocity_state=servicer_velocity_state,
            max_single_impulse_delta_v_m_s=max_single_impulse_delta_v_m_s,
        )
    )
    if not run_phase("Transfer to inspection hold", inspection_transfer_duration_hr):
        return maneuvers, phase_windows, False

    # DV2 removes the arrival velocity. A small cross-track displacement means
    # this is only approximately stationary under natural relative dynamics.
    maneuvers.append(
        apply_relative_velocity_impulse(
            label="DV2 inspection arrival braking",
            time_s=current_time_ns * macros.NANO2SEC,
            commanded_relative_velocity_h_m_s=hold_relative_velocity_h_m_s,
            target_position_state=target_position_state,
            target_velocity_state=target_velocity_state,
            servicer_position_state=servicer_position_state,
            servicer_velocity_state=servicer_velocity_state,
            max_single_impulse_delta_v_m_s=max_single_impulse_delta_v_m_s,
        )
    )
    if not run_phase("Inspection hold", inspection_hold_duration_hr):
        return maneuvers, phase_windows, False

    # DV3 begins the final approach from the actual post-hold state rather than
    # assuming the servicer stayed perfectly fixed at the commanded point.
    _, _, relative_position_h_m, _ = get_relative_state_hill(
        target_position_state,
        target_velocity_state,
        servicer_position_state,
        servicer_velocity_state,
    )
    terminal_approach_s = terminal_approach_duration_hr * 3600.0
    terminal_departure_velocity_h_m_s = cw_targeting_velocity(
        relative_position_h_m,
        final_hold_point_h_m,
        terminal_approach_s,
        mean_motion_rad_s,
    )
    maneuvers.append(
        apply_relative_velocity_impulse(
            label="DV3 terminal approach injection",
            time_s=current_time_ns * macros.NANO2SEC,
            commanded_relative_velocity_h_m_s=terminal_departure_velocity_h_m_s,
            target_position_state=target_position_state,
            target_velocity_state=target_velocity_state,
            servicer_position_state=servicer_position_state,
            servicer_velocity_state=servicer_velocity_state,
            max_single_impulse_delta_v_m_s=max_single_impulse_delta_v_m_s,
        )
    )
    if not run_phase("Terminal approach", terminal_approach_duration_hr):
        return maneuvers, phase_windows, False

    # DV4 brakes at the final 50 m hold point. A pure along-track offset with
    # zero Hill velocity is an equilibrium of the linear circular-orbit model.
    maneuvers.append(
        apply_relative_velocity_impulse(
            label="DV4 final hold braking",
            time_s=current_time_ns * macros.NANO2SEC,
            commanded_relative_velocity_h_m_s=hold_relative_velocity_h_m_s,
            target_position_state=target_position_state,
            target_velocity_state=target_velocity_state,
            servicer_position_state=servicer_position_state,
            servicer_velocity_state=servicer_velocity_state,
            max_single_impulse_delta_v_m_s=max_single_impulse_delta_v_m_s,
        )
    )
    completed = run_phase("Final hold verification", final_hold_duration_hr)
    return maneuvers, phase_windows, completed


# =============================================================================
# DATA EXTRACTION AND RENDEZVOUS ANALYSIS
# =============================================================================


def extract_rendezvous_data(
    target_recorder: Any,
    servicer_recorder: Any,
    servicer_boresight_b: np.ndarray,
) -> dict[str, np.ndarray]:
    """Convert truth recordings into Hill states and rendezvous metrics."""
    time_s = np.asarray(servicer_recorder.times(), dtype=float) * macros.NANO2SEC
    target_position_n_m = np.asarray(target_recorder.r_BN_N, dtype=float)
    target_velocity_n_m_s = np.asarray(target_recorder.v_BN_N, dtype=float)
    servicer_position_n_m = np.asarray(servicer_recorder.r_BN_N, dtype=float)
    servicer_velocity_n_m_s = np.asarray(servicer_recorder.v_BN_N, dtype=float)
    servicer_sigma_bn = np.asarray(servicer_recorder.sigma_BN, dtype=float)

    sample_count = len(time_s)
    recorded_lengths = {
        len(target_position_n_m),
        len(target_velocity_n_m_s),
        len(servicer_position_n_m),
        len(servicer_velocity_n_m_s),
        len(servicer_sigma_bn),
    }
    if recorded_lengths != {sample_count}:
        raise RuntimeError(
            "Target and servicer recorders produced mismatched histories."
        )

    relative_position_h_m = np.empty((sample_count, 3), dtype=float)
    relative_velocity_h_m_s = np.empty((sample_count, 3), dtype=float)
    pointing_error_deg = np.empty(sample_count, dtype=float)
    boresight_hat_b = np.asarray(servicer_boresight_b, dtype=float)
    boresight_hat_b /= np.linalg.norm(boresight_hat_b)

    for index in range(sample_count):
        relative_position, relative_velocity = orbitalMotion.rv2hill(
            target_position_n_m[index],
            target_velocity_n_m_s[index],
            servicer_position_n_m[index],
            servicer_velocity_n_m_s[index],
        )
        relative_position_h_m[index] = relative_position
        relative_velocity_h_m_s[index] = relative_velocity

        line_of_sight_n = target_position_n_m[index] - servicer_position_n_m[index]
        line_of_sight_n /= np.linalg.norm(line_of_sight_n)
        dcm_bn = np.asarray(
            RigidBodyKinematics.MRP2C(servicer_sigma_bn[index]), dtype=float
        )
        boresight_hat_n = boresight_hat_b @ dcm_bn
        cosine_error = np.clip(np.dot(boresight_hat_n, line_of_sight_n), -1.0, 1.0)
        pointing_error_deg[index] = np.degrees(np.arccos(cosine_error))

    relative_range_m = np.linalg.norm(relative_position_h_m, axis=1)
    closing_rate_m_s = -np.einsum(
        "ij,ij->i", relative_position_h_m, relative_velocity_h_m_s
    ) / np.maximum(relative_range_m, np.finfo(float).eps)

    return {
        "time_s": time_s,
        "relative_position_h_m": relative_position_h_m,
        "relative_velocity_h_m_s": relative_velocity_h_m_s,
        "relative_range_m": relative_range_m,
        "closing_rate_m_s": closing_rate_m_s,
        "pointing_error_deg": pointing_error_deg,
    }


# =============================================================================
# PLOTTING
# =============================================================================


def _shade_phases(axis: Any, phase_windows: list[PhaseWindow]) -> None:
    """Add restrained phase bands to a time-history axis."""
    for index, phase in enumerate(phase_windows):
        start_hr = phase.start_s / 3600.0
        end_hr = phase.end_s / 3600.0
        axis.axvspan(
            start_hr,
            end_hr,
            color="tab:blue" if index % 2 == 0 else "tab:orange",
            alpha=0.045,
        )
        axis.text(
            0.5 * (start_hr + end_hr),
            0.97,
            f"P{index + 1}",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
        )


def create_plots(
    data: dict[str, np.ndarray],
    maneuvers: list[ManeuverEvent],
    phase_windows: list[PhaseWindow],
    inspection_hold_point_h_m: np.ndarray,
    final_hold_point_h_m: np.ndarray,
) -> dict[str, Figure]:
    """Create the required relative-motion and target-pointing plots."""
    figures: dict[str, Figure] = {}
    time_s = data["time_s"]
    time_hr = time_s / 3600.0
    relative_position_h_m = data["relative_position_h_m"]

    # -------------------------------------------------------------------------
    # Three-dimensional Hill-frame relative trajectory
    # -------------------------------------------------------------------------

    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    phase_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for index, phase in enumerate(phase_windows):
        mask = (time_s >= phase.start_s) & (time_s <= phase.end_s)
        if np.any(mask):
            segment = relative_position_h_m[mask]
            axis.plot(
                segment[:, 0],
                segment[:, 1],
                segment[:, 2],
                color=phase_colors[index % len(phase_colors)],
                linewidth=1.6,
                label=f"P{index + 1}: {phase.label}",
            )

    axis.scatter(0.0, 0.0, 0.0, marker="*", s=150, color="black", label="Target")
    axis.scatter(
        *relative_position_h_m[0],
        marker="o",
        s=65,
        color="tab:green",
        label="Initial point",
    )
    axis.scatter(
        *inspection_hold_point_h_m,
        marker="D",
        s=70,
        color="tab:cyan",
        label="Inspection hold command",
    )
    axis.scatter(
        *final_hold_point_h_m,
        marker="D",
        s=70,
        color="tab:pink",
        label="Final hold command",
    )

    for index, maneuver in enumerate(maneuvers, start=1):
        axis.scatter(
            *maneuver.relative_position_h_m,
            marker="^",
            s=55,
            color="tab:orange",
        )
        axis.text(*maneuver.relative_position_h_m, f" DV{index}", fontsize=8)

    axis.scatter(
        *relative_position_h_m[-1],
        marker="x",
        s=80,
        color="red",
        label="Terminal simulated point",
    )
    axis.set(
        xlabel="Radial x [m]",
        ylabel="Along-track y [m]",
        zlabel="Cross-track z [m]",
        title="Servicer Trajectory in the Target Hill Frame",
    )
    axis.view_init(elev=24.0, azim=-58.0)
    axis.grid(True)
    axis.legend(fontsize=8, loc="best")
    figure.tight_layout()
    figures["hill_relative_trajectory"] = figure

    # -------------------------------------------------------------------------
    # Relative range
    # -------------------------------------------------------------------------

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(time_hr, data["relative_range_m"], color="tab:blue")
    _shade_phases(axis, phase_windows)
    axis.set(
        xlabel="Simulation time [hr]",
        ylabel="Relative range [m]",
        title="Servicer-to-Target Range",
    )
    axis.grid(True)
    figure.tight_layout()
    figures["relative_range"] = figure

    # -------------------------------------------------------------------------
    # Closing rate: positive values mean the range is decreasing
    # -------------------------------------------------------------------------

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(time_hr, data["closing_rate_m_s"], color="tab:orange")
    axis.axhline(0.0, color="black", linewidth=0.8)
    _shade_phases(axis, phase_windows)
    axis.set(
        xlabel="Simulation time [hr]",
        ylabel="Closing rate [m/s]",
        title="Relative Closing Rate (Positive Means Closing)",
    )
    axis.grid(True)
    figure.tight_layout()
    figures["closing_rate"] = figure

    # -------------------------------------------------------------------------
    # Physical angle between +b1 and the target line of sight
    # -------------------------------------------------------------------------

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.semilogy(
        time_hr,
        np.maximum(data["pointing_error_deg"], 1e-8),
        color="tab:green",
    )
    _shade_phases(axis, phase_windows)
    axis.set(
        xlabel="Simulation time [hr]",
        ylabel="Target-pointing error [deg]",
        title="Servicer +b1 Target-Pointing Error",
    )
    axis.grid(True, which="both")
    figure.tight_layout()
    figures["target_pointing_error"] = figure

    return figures


def save_figures(figures: dict[str, Figure]) -> None:
    """Save all analysis figures under the scenario-local output folder."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        output_path = PLOTS_DIR / f"{name}.png"
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot: {output_path}")


# =============================================================================
# REPORTING
# =============================================================================


def print_configuration_summary(
    *,
    mode: str,
    geo_altitude_m: float,
    orbital_period_s: float,
    dynamics_dt_s: float,
    phase_durations_hr: tuple[float, float, float, float, float],
    initial_relative_position_h_m: np.ndarray,
    initial_relative_velocity_h_m_s: np.ndarray,
    inspection_hold_point_h_m: np.ndarray,
    final_hold_point_h_m: np.ndarray,
    live_acceleration_factor: float,
    vizard_address: str,
) -> None:
    """Print the effective orbit, relative geometry, phases, and Vizard mode."""
    print("\n" + "=" * 76)
    print("BASILISK-X COOPERATIVE GEO RENDEZVOUS")
    print("=" * 76)
    print(f"Visualization mode      : {mode}")
    print(f"Target GEO altitude     : {geo_altitude_m / 1000.0:.3f} km")
    print(f"Target orbital period   : {orbital_period_s / 3600.0:.4f} hr")
    print(f"Dynamics time step      : {dynamics_dt_s:.3f} s")
    print(f"Initial Hill position   : {initial_relative_position_h_m} m")
    print(f"Initial Hill velocity   : {initial_relative_velocity_h_m_s} m/s")
    print(f"Inspection hold command : {inspection_hold_point_h_m} m")
    print(f"Final hold command      : {final_hold_point_h_m} m")
    phase_duration_text = ", ".join(
        f"{value:.2f}" for value in phase_durations_hr
    )
    print(f"Phase durations [hr]    : {phase_duration_text}")
    print(f"Total duration          : {sum(phase_durations_hr):.2f} hr")
    print("Servicer boresight      : +b1 toward target")
    if mode == "live":
        print(f"Live acceleration       : {live_acceleration_factor:.1f}x")
        print(f"Vizard address          : {vizard_address}")
    elif mode == "playback":
        print(f"Playback file           : {VIZARD_FILE}")
    print("=" * 76 + "\n")


def print_results_summary(
    *,
    data: dict[str, np.ndarray],
    maneuvers: list[ManeuverEvent],
    final_hold_point_h_m: np.ndarray,
    target_keep_out_radius_m: float,
) -> None:
    """Report maneuver cost, terminal state, safety distance, and pointing."""
    relative_position_h_m = data["relative_position_h_m"]
    relative_velocity_h_m_s = data["relative_velocity_h_m_s"]
    relative_range_m = data["relative_range_m"]
    pointing_error_deg = data["pointing_error_deg"]

    print("\n" + "=" * 76)
    print("RENDEZVOUS RESULTS")
    print("=" * 76)
    for maneuver in maneuvers:
        magnitude = np.linalg.norm(maneuver.delta_v_h_m_s)
        print(
            f"{maneuver.label:<34}: {magnitude:.4f} m/s "
            f"at t={maneuver.time_s / 3600.0:.3f} hr"
        )
    total_delta_v_m_s = sum(
        float(np.linalg.norm(maneuver.delta_v_h_m_s)) for maneuver in maneuvers
    )
    final_position_error_m = np.linalg.norm(
        relative_position_h_m[-1] - final_hold_point_h_m
    )
    print(f"Total ideal delta-v                : {total_delta_v_m_s:.4f} m/s")
    print(f"Minimum simulated range            : {np.min(relative_range_m):.3f} m")
    print(f"Configured keep-out radius         : {target_keep_out_radius_m:.3f} m")
    print(f"Final Hill position [m]            : {relative_position_h_m[-1]}")
    print(f"Final Hill velocity [m/s]          : {relative_velocity_h_m_s[-1]}")
    print(f"Final hold-point position error     : {final_position_error_m:.3f} m")
    print(f"Final +b1 target-pointing error     : {pointing_error_deg[-1]:.6f} deg")
    if np.min(relative_range_m) <= target_keep_out_radius_m:
        print("WARNING: The simulated trajectory entered the keep-out radius.")
    print("=" * 76 + "\n")


# =============================================================================
# SCENARIO DRIVER
# =============================================================================


def run(
    *,
    visualization_mode: str = VISUALIZATION_MODE,
    live_acceleration_factor: float = LIVE_ACCELERATION_FACTOR,
    vizard_address: str = VIZARD_ADDRESS,
    dynamics_dt_s: float = DYNAMICS_DT_S,
    geo_altitude_m: float = GEO_ALTITUDE_M,
    geo_inclination_deg: float = GEO_INCLINATION_DEG,
    geo_raan_deg: float = GEO_RAAN_DEG,
    geo_true_anomaly_deg: float = GEO_TRUE_ANOMALY_DEG,
    initial_relative_position_h_m: np.ndarray = INITIAL_RELATIVE_POSITION_H_M,
    initial_relative_velocity_h_m_s: np.ndarray = INITIAL_RELATIVE_VELOCITY_H_M_S,
    passive_safe_duration_hr: float = PASSIVE_SAFE_DURATION_HR,
    inspection_transfer_duration_hr: float = INSPECTION_TRANSFER_DURATION_HR,
    inspection_hold_duration_hr: float = INSPECTION_HOLD_DURATION_HR,
    terminal_approach_duration_hr: float = TERMINAL_APPROACH_DURATION_HR,
    final_hold_duration_hr: float = FINAL_HOLD_DURATION_HR,
    inspection_hold_point_h_m: np.ndarray = INSPECTION_HOLD_POINT_H_M,
    final_hold_point_h_m: np.ndarray = FINAL_HOLD_POINT_H_M,
    hold_relative_velocity_h_m_s: np.ndarray = HOLD_RELATIVE_VELOCITY_H_M_S,
    target_keep_out_radius_m: float = TARGET_KEEP_OUT_RADIUS_M,
    max_single_impulse_delta_v_m_s: float = MAX_SINGLE_IMPULSE_DELTA_V_M_S,
    servicer_boresight_b: np.ndarray = SERVICER_BORESIGHT_B,
    control_k: float = CONTROL_K,
    control_p: float = CONTROL_P,
    control_ki: float = CONTROL_KI,
    save_plots: bool = SAVE_PLOTS,
    show_plots: bool = SHOW_PLOTS,
) -> None:
    """Run the complete illustrative cooperative GEO rendezvous.

    The top-level constants provide readable standalone defaults. Keyword
    arguments allow focused experiments without editing the scenario itself.
    """
    phase_durations_hr = (
        passive_safe_duration_hr,
        inspection_transfer_duration_hr,
        inspection_hold_duration_hr,
        terminal_approach_duration_hr,
        final_hold_duration_hr,
    )
    mode = validate_configuration(
        visualization_mode=visualization_mode,
        live_acceleration_factor=live_acceleration_factor,
        dynamics_dt_s=dynamics_dt_s,
        geo_altitude_m=geo_altitude_m,
        initial_relative_position_h_m=initial_relative_position_h_m,
        initial_relative_velocity_h_m_s=initial_relative_velocity_h_m_s,
        inspection_hold_point_h_m=inspection_hold_point_h_m,
        final_hold_point_h_m=final_hold_point_h_m,
        phase_durations_hr=phase_durations_hr,
        target_keep_out_radius_m=target_keep_out_radius_m,
        max_single_impulse_delta_v_m_s=max_single_impulse_delta_v_m_s,
        servicer_boresight_b=servicer_boresight_b,
    )

    (
        simulation,
        task_name,
        target,
        servicer,
        target_recorder,
        servicer_recorder,
        _vehicle_config_message,
        _earth_radius_m,
        mean_motion_rad_s,
        orbital_period_s,
        time_step_ns,
    ) = build_simulation(
        mode=mode,
        dynamics_dt_s=dynamics_dt_s,
        geo_altitude_m=geo_altitude_m,
        geo_inclination_deg=geo_inclination_deg,
        geo_raan_deg=geo_raan_deg,
        geo_true_anomaly_deg=geo_true_anomaly_deg,
        initial_relative_position_h_m=initial_relative_position_h_m,
        initial_relative_velocity_h_m_s=initial_relative_velocity_h_m_s,
        servicer_boresight_b=servicer_boresight_b,
        control_k=control_k,
        control_p=control_p,
        control_ki=control_ki,
    )

    viz, clock_sync = configure_visualization(
        simulation,
        task_name,
        target,
        servicer,
        mode=mode,
        live_acceleration_factor=live_acceleration_factor,
        vizard_address=vizard_address,
    )

    print_configuration_summary(
        mode=mode,
        geo_altitude_m=geo_altitude_m,
        orbital_period_s=orbital_period_s,
        dynamics_dt_s=dynamics_dt_s,
        phase_durations_hr=phase_durations_hr,
        initial_relative_position_h_m=initial_relative_position_h_m,
        initial_relative_velocity_h_m_s=initial_relative_velocity_h_m_s,
        inspection_hold_point_h_m=inspection_hold_point_h_m,
        final_hold_point_h_m=final_hold_point_h_m,
        live_acceleration_factor=live_acceleration_factor,
        vizard_address=vizard_address,
    )

    vizard_process = None
    if mode == "live":
        print("Launching Vizard in Direct Communication mode...")
        vizard_process = launch_vizard(address=vizard_address)

    try:
        simulation.InitializeSimulation()
    except KeyboardInterrupt:
        print("\nSimulation interrupted during initialization.")
        if vizard_process is not None:
            vizard_process.terminate()
        return

    print("Executing phased Basilisk rendezvous...")
    if mode == "live":
        print("Live controls: p = pause/resume, q = quit")

    maneuvers, phase_windows, completed = execute_rendezvous_phases(
        simulation=simulation,
        target=target,
        servicer=servicer,
        mean_motion_rad_s=mean_motion_rad_s,
        time_step_ns=time_step_ns,
        mode=mode,
        viz=viz,
        clock_sync=clock_sync,
        vizard_process=vizard_process,
        passive_safe_duration_hr=passive_safe_duration_hr,
        inspection_transfer_duration_hr=inspection_transfer_duration_hr,
        inspection_hold_duration_hr=inspection_hold_duration_hr,
        terminal_approach_duration_hr=terminal_approach_duration_hr,
        final_hold_duration_hr=final_hold_duration_hr,
        inspection_hold_point_h_m=inspection_hold_point_h_m,
        final_hold_point_h_m=final_hold_point_h_m,
        hold_relative_velocity_h_m_s=hold_relative_velocity_h_m_s,
        max_single_impulse_delta_v_m_s=max_single_impulse_delta_v_m_s,
    )
    if not completed:
        print("Simulation terminated early.")
        return
    print("Basilisk rendezvous simulation complete.")

    if mode == "playback":
        if not VIZARD_FILE.is_file():
            raise FileNotFoundError(
                "Simulation completed, but no Vizard recording was created at "
                f"{VIZARD_FILE}"
            )
        print("Launching Vizard recorded playback...")
        launch_vizard_playback(VIZARD_FILE)

    data = extract_rendezvous_data(
        target_recorder,
        servicer_recorder,
        servicer_boresight_b,
    )
    print_results_summary(
        data=data,
        maneuvers=maneuvers,
        final_hold_point_h_m=final_hold_point_h_m,
        target_keep_out_radius_m=target_keep_out_radius_m,
    )

    figures = create_plots(
        data,
        maneuvers,
        phase_windows,
        inspection_hold_point_h_m,
        final_hold_point_h_m,
    )
    if save_plots:
        save_figures(figures)
    if show_plots:
        plt.show()
    else:
        plt.close("all")


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    # Change the constants above for normal use, or provide focused overrides
    # here while experimenting with a particular rendezvous phase.
    run(
        visualization_mode=VISUALIZATION_MODE,
        dynamics_dt_s=DYNAMICS_DT_S,
        passive_safe_duration_hr=PASSIVE_SAFE_DURATION_HR,
        inspection_transfer_duration_hr=INSPECTION_TRANSFER_DURATION_HR,
        inspection_hold_duration_hr=INSPECTION_HOLD_DURATION_HR,
        terminal_approach_duration_hr=TERMINAL_APPROACH_DURATION_HR,
        final_hold_duration_hr=FINAL_HOLD_DURATION_HR,
    )
