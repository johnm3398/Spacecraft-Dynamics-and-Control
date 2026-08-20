"""ORBITAL DUET - Engineering data reduction, validation, and plots.

Records truth and environmental messages, reconstructs deputy motion in the
chief RTN frame, computes orbital/LOS metrics, compares first-order phasing to
nonlinear propagation, and saves concise engineering-review figures.
"""

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from Basilisk.utilities import macros, orbitalMotion

from aocs import AocsPair
from config import ScenarioConfig
from environment import EnvironmentHandles
from mission_plan import MissionResult
from spacecraft_model import SpacecraftPair


@dataclass
class ScenarioRecorders:
    """Truth and environmental recorders created before initialization."""

    chief_state: Any
    deputy_state: Any
    chief_density: Any
    deputy_density: Any
    chief_drag: Any | None
    deputy_drag: Any | None


def configure_recorders(
    simulation: Any,
    dynamics_task: str,
    pair: SpacecraftPair,
    environment: EnvironmentHandles,
    config: ScenarioConfig,
) -> ScenarioRecorders:
    """Create message recorders and variable loggers at the configured rate."""

    sampling_ns = macros.sec2nano(config.mission.record_step_s)
    chief_state = pair.chief.scStateOutMsg.recorder(sampling_ns)
    deputy_state = pair.deputy.scStateOutMsg.recorder(sampling_ns)
    chief_density = environment.vehicles["chief"].density_msg.recorder(sampling_ns)
    deputy_density = environment.vehicles["deputy"].density_msg.recorder(sampling_ns)
    for recorder in (chief_state, deputy_state, chief_density, deputy_density):
        simulation.AddModelToTask(dynamics_task, recorder, 5)

    chief_drag = None
    deputy_drag = None
    if environment.vehicles["chief"].drag is not None:
        chief_drag = environment.vehicles["chief"].drag.logger(
            "forceExternal_B", sampling_ns
        )
        simulation.AddModelToTask(dynamics_task, chief_drag, 5)
    if environment.vehicles["deputy"].drag is not None:
        deputy_drag = environment.vehicles["deputy"].drag.logger(
            "forceExternal_B", sampling_ns
        )
        simulation.AddModelToTask(dynamics_task, deputy_drag, 5)
    return ScenarioRecorders(
        chief_state,
        deputy_state,
        chief_density,
        deputy_density,
        chief_drag,
        deputy_drag,
    )


def _signed_phase_angle(r_chief: np.ndarray, v_chief: np.ndarray, r_deputy: np.ndarray) -> float:
    """Return deputy-minus-chief phase about the chief orbit normal [rad]."""

    normal = np.cross(r_chief, v_chief)
    normal /= np.linalg.norm(normal)
    numerator = np.dot(normal, np.cross(r_chief, r_deputy))
    denominator = np.dot(r_chief, r_deputy)
    return float(np.arctan2(numerator, denominator))


def extract_results(
    recorders: ScenarioRecorders,
    environment: EnvironmentHandles,
) -> dict[str, np.ndarray]:
    """Convert Basilisk histories into inertial, orbital, LOS, and RTN metrics."""

    time_s = recorders.chief_state.times() * macros.NANO2SEC
    chief_r = np.asarray(recorders.chief_state.r_BN_N)
    chief_v = np.asarray(recorders.chief_state.v_BN_N)
    deputy_r = np.asarray(recorders.deputy_state.r_BN_N)
    deputy_v = np.asarray(recorders.deputy_state.v_BN_N)
    count = min(len(time_s), len(deputy_r))
    time_s, chief_r, chief_v, deputy_r, deputy_v = (
        value[:count] for value in (time_s, chief_r, chief_v, deputy_r, deputy_v)
    )

    relative_position_H = np.zeros((count, 3))
    relative_velocity_H = np.zeros((count, 3))
    phase_angle_rad = np.zeros(count)
    chief_elements = np.zeros((count, 6))
    deputy_elements = np.zeros((count, 6))
    for index in range(count):
        relative_position_H[index], relative_velocity_H[index] = orbitalMotion.rv2hill(
            chief_r[index], chief_v[index], deputy_r[index], deputy_v[index]
        )
        phase_angle_rad[index] = _signed_phase_angle(
            chief_r[index], chief_v[index], deputy_r[index]
        )
        for output, r_N, v_N in (
            (chief_elements, chief_r[index], chief_v[index]),
            (deputy_elements, deputy_r[index], deputy_v[index]),
        ):
            elements = orbitalMotion.rv2elem(environment.earth.mu, r_N, v_N)
            output[index] = [
                elements.a,
                elements.e,
                elements.i,
                elements.Omega,
                elements.omega,
                elements.f,
            ]

    relative_r_N = deputy_r - chief_r
    relative_v_N = deputy_v - chief_v
    los_range_m = np.linalg.norm(relative_r_N, axis=1)
    los_range_rate_m_s = np.sum(relative_r_N * relative_v_N, axis=1) / np.maximum(
        los_range_m, np.finfo(float).eps
    )

    result = {
        "time_s": time_s,
        "chief_r_N_m": chief_r,
        "chief_v_N_m_s": chief_v,
        "deputy_r_N_m": deputy_r,
        "deputy_v_N_m_s": deputy_v,
        "relative_position_H_m": relative_position_H,
        "relative_velocity_H_m_s": relative_velocity_H,
        "los_range_m": los_range_m,
        "los_range_rate_m_s": los_range_rate_m_s,
        "phase_angle_deg": phase_angle_rad * macros.R2D,
        "chief_elements": chief_elements,
        "deputy_elements": deputy_elements,
        "differential_semimajor_axis_m": deputy_elements[:, 0] - chief_elements[:, 0],
        "chief_altitude_m": np.linalg.norm(chief_r, axis=1) - environment.earth.radEquator,
        "deputy_altitude_m": np.linalg.norm(deputy_r, axis=1) - environment.earth.radEquator,
        "chief_density_kg_m3": np.asarray(recorders.chief_density.neutralDensity)[:count],
        "deputy_density_kg_m3": np.asarray(recorders.deputy_density.neutralDensity)[:count],
        "earth_mu_m3_s2": np.asarray(environment.earth.mu),
    }
    if recorders.chief_drag is not None:
        result["chief_drag_force_B_N"] = np.asarray(
            recorders.chief_drag.forceExternal_B
        )[:count]
    if recorders.deputy_drag is not None:
        result["deputy_drag_force_B_N"] = np.asarray(
            recorders.deputy_drag.forceExternal_B
        )[:count]
    return result


def _analytical_along_track(
    time_s: np.ndarray,
    mission: MissionResult,
) -> np.ndarray:
    """Construct the piecewise first-order along-track phasing expectation."""

    initial_y = mission.initial_relative_position_H_m[1]
    if not mission.maneuvers:
        return np.full_like(time_s, initial_y)
    set_time = mission.maneuvers[0].command_time_s
    arrest_time = mission.maneuvers[-1].command_time_s
    drift_time = np.clip(time_s - set_time, 0.0, max(arrest_time - set_time, 0.0))
    return initial_y + mission.planned_drift_rate_m_s * drift_time


def print_results_summary(
    data: dict[str, np.ndarray],
    mission: MissionResult,
    aocs: AocsPair,
    config: ScenarioConfig,
) -> None:
    """Report mission success metrics and verify major physical message chains."""

    final_position = data["relative_position_H_m"][-1]
    final_velocity = data["relative_velocity_H_m_s"][-1]
    along_track_error = (
        final_position[1] - config.phasing.target_along_track_separation_m
    )
    chief_a = data["chief_elements"][-1, 0]
    mean_motion = np.sqrt(float(data["earth_mu_m3_s2"]) / chief_a**3)
    # Differential semi-major axis governs secular phase drift.  Instantaneous
    # RTN velocity also contains the intentionally retained bounded epicycle.
    mean_along_track_drift_m_s = (
        -1.5 * mean_motion * data["differential_semimajor_axis_m"][-1]
    )
    acquired = (
        abs(along_track_error) <= config.phasing.acquisition_position_tolerance_m
        and abs(mean_along_track_drift_m_s)
        <= config.phasing.acquisition_drift_tolerance_m_s
    )
    analytical = _analytical_along_track(data["time_s"], mission)
    phase_rms = np.sqrt(
        np.mean((data["relative_position_H_m"][:, 1] - analytical) ** 2)
    )

    print("=" * 78)
    print("ORBITAL DUET RESULTS")
    print("=" * 78)
    for maneuver in mission.maneuvers:
        print(
            f"{maneuver.label:<12}: {maneuver.signed_delta_v_m_s:+.4f} m/s, "
            f"{maneuver.duration_s:.2f} s, {maneuver.direction}"
        )
    print(f"Final RTN position [m]       : {final_position}")
    print(f"Final RTN velocity [m/s]     : {final_velocity}")
    print(f"Final LOS range / rate       : {data['los_range_m'][-1]:.2f} m / "
          f"{data['los_range_rate_m_s'][-1]:+.4f} m/s")
    print(f"Final differential a         : {data['differential_semimajor_axis_m'][-1]:+.3f} m")
    print(f"Estimated secular T drift    : {mean_along_track_drift_m_s:+.4f} m/s")
    print(f"Instantaneous RTN velocity   : {final_velocity} m/s (includes epicycle)")
    print(f"Along-track target error     : {along_track_error:+.2f} m")
    print(f"Analytic/nonlinear y RMS     : {phase_rms:.2f} m")
    print(f"Acquisition condition        : {'SATISFIED' if acquired else 'NOT SATISFIED'}")
    print(f"Chief sensor samples         : IMU={len(aocs.chief.recorders['imu'].times())}, "
          f"ST={len(aocs.chief.recorders['star_tracker'].times())}, "
          f"TAM={len(aocs.chief.recorders['tam'].times())}")
    print("Navigation declaration       : SimpleNav truth-like/emulated; not GNSS")
    print("=" * 78 + "\n")


def _mark_maneuvers(axis: Any, mission: MissionResult) -> None:
    """Mark commanded burn epochs on a plot time axis expressed in hours."""

    for maneuver in mission.maneuvers:
        axis.axvline(
            maneuver.command_time_s / 3600.0,
            color="tab:red",
            linestyle="--",
            alpha=0.65,
            label=maneuver.label,
        )


def create_plots(
    data: dict[str, np.ndarray],
    mission: MissionResult,
    aocs: AocsPair,
) -> dict[str, Figure]:
    """Create compact orbit, relative-motion, environment, maneuver, and AOCS plots."""

    plt.close("all")
    time_h = data["time_s"] / 3600.0
    figures: dict[str, Figure] = {}

    fig, axis = plt.subplots(figsize=(8, 7))
    axis.plot(data["chief_r_N_m"][:, 0] / 1000, data["chief_r_N_m"][:, 1] / 1000, label="Chief")
    axis.plot(data["deputy_r_N_m"][:, 0] / 1000, data["deputy_r_N_m"][:, 1] / 1000, label="Deputy")
    axis.set(xlabel="Inertial x [km]", ylabel="Inertial y [km]", title="Earth-centred inertial trajectories")
    axis.axis("equal")
    axis.grid(True)
    axis.legend()
    figures["inertial_trajectories"] = fig

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    labels = ("Radial R", "Along-track T", "Cross-track N")
    for index, axis in enumerate(axes):
        axis.plot(time_h, data["relative_position_H_m"][:, index], label=labels[index])
        _mark_maneuvers(axis, mission)
        axis.set_ylabel("Position [m]")
        axis.grid(True)
    axes[-1].set_xlabel("Mission elapsed time [h]")
    axes[0].set_title("Deputy relative position in chief RTN/Hill")
    figures["relative_rtn_position"] = fig

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(time_h, data["los_range_m"])
    axes[0].set_ylabel("LOS range [m]")
    axes[1].plot(time_h, data["los_range_rate_m_s"])
    axes[1].set_ylabel("Range rate [m/s]")
    axes[2].plot(time_h, data["differential_semimajor_axis_m"])
    axes[2].set_ylabel("Deputy-chief a [m]")
    axes[2].set_xlabel("Mission elapsed time [h]")
    for axis in axes:
        _mark_maneuvers(axis, mission)
        axis.grid(True)
    figures["relative_orbit_metrics"] = fig

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(time_h, data["relative_position_H_m"][:, 1], label="Nonlinear Basilisk")
    axis.plot(time_h, _analytical_along_track(data["time_s"], mission), "--", label="First-order estimate")
    _mark_maneuvers(axis, mission)
    axis.set(xlabel="Mission elapsed time [h]", ylabel="Along-track separation [m]", title="Analytical versus nonlinear phasing")
    axis.grid(True)
    axis.legend()
    figures["phasing_comparison"] = fig

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].semilogy(time_h, np.maximum(data["chief_density_kg_m3"], 1e-30), label="Chief")
    axes[0].semilogy(time_h, np.maximum(data["deputy_density_kg_m3"], 1e-30), label="Deputy")
    axes[0].set_ylabel("Density [kg/m³]")
    axes[0].legend()
    if "deputy_drag_force_B_N" in data:
        axes[1].plot(time_h, np.linalg.norm(data["deputy_drag_force_B_N"], axis=1))
    axes[1].set(xlabel="Mission elapsed time [h]", ylabel="Deputy drag force [N]")
    for axis in axes:
        axis.grid(True)
    figures["environment_drag"] = fig

    att_rec = aocs.deputy.recorders["attitude_error"]
    rw_rec = aocs.deputy.recorders["wheel_speeds"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    att_time_h = att_rec.times() * macros.NANO2SEC / 3600.0
    axes[0].plot(att_time_h, np.linalg.norm(np.asarray(att_rec.sigma_BR), axis=1))
    axes[0].set_ylabel(r"$||\sigma_{B/R}||$")
    rw_time_h = rw_rec.times() * macros.NANO2SEC / 3600.0
    axes[1].plot(rw_time_h, np.asarray(rw_rec.wheelSpeeds) / macros.rpm2radsec)
    axes[1].set(xlabel="Mission elapsed time [h]", ylabel="Wheel speed [rpm]")
    for axis in axes:
        axis.grid(True)
    figures["deputy_aocs"] = fig
    return figures


def save_plots(figures: dict[str, Figure], config: ScenarioConfig) -> None:
    """Save every engineering-review figure under the scenario output tree."""

    config.output.plots_directory.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        path = config.output.plots_directory / f"{name}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        print(f"Saved plot: {path}")
