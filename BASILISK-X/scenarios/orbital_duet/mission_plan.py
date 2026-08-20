"""ORBITAL DUET - Operational mission timeline and phasing logic.

Sequences deployment, acquisition, finite set-drift and arrest burns, coast,
and final verification.  First-order circular-orbit equations size and
sanity-check the burns; nonlinear Basilisk truth propagation determines the
mission outcome.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable

import numpy as np

from Basilisk.utilities import macros, orbitalMotion

from config import ScenarioConfig
from propulsion import PropulsionPair, command_burn
from spacecraft_model import SpacecraftPair


class MissionPhase(Enum):
    """Readable operational states; this is intentionally not a generic framework."""

    DEPLOY = auto()
    DETUMBLE = auto()
    ATTITUDE_ACQUISITION = auto()
    INITIAL_COAST_NAVIGATION_ACQUISITION = auto()
    RELATIVE_STATE_ASSESSMENT = auto()
    PHASING_MANEUVER_DESIGN = auto()
    SET_DRIFT_ATTITUDE_ACQUISITION = auto()
    SET_DRIFT_BURN = auto()
    PHASING_COAST = auto()
    RELATIVE_STATE_MONITORING = auto()
    ARREST_ATTITUDE_ACQUISITION = auto()
    ARREST_BURN = auto()
    FINAL_ORBIT_VERIFICATION = auto()
    ACQUISITION_COMPLETE = auto()


@dataclass(frozen=True)
class PhaseRecord:
    """One executed timeline interval."""

    phase: MissionPhase
    start_s: float
    end_s: float


@dataclass(frozen=True)
class ManeuverRecord:
    """One finite physical burn command and its first-order expectation."""

    label: str
    command_time_s: float
    direction: str
    signed_delta_v_m_s: float
    duration_s: float
    estimated_achieved_delta_v_m_s: float


@dataclass
class MissionResult:
    """Timeline and maneuver information retained for analysis."""

    phases: list[PhaseRecord]
    maneuvers: list[ManeuverRecord]
    initial_relative_position_H_m: np.ndarray
    planned_drift_rate_m_s: float
    completed: bool


AdvanceFunction = Callable[[float, MissionPhase], bool]


def current_relative_state_H(pair: SpacecraftPair) -> tuple[np.ndarray, np.ndarray]:
    """Read truth messages and return deputy relative state in chief RTN/Hill."""

    chief_state = pair.chief.scStateOutMsg.read()
    deputy_state = pair.deputy.scStateOutMsg.read()
    position_H, velocity_H = orbitalMotion.rv2hill(
        np.asarray(chief_state.r_BN_N),
        np.asarray(chief_state.v_BN_N),
        np.asarray(deputy_state.r_BN_N),
        np.asarray(deputy_state.v_BN_N),
    )
    return np.asarray(position_H), np.asarray(velocity_H)


def first_order_set_drift_delta_v(
    current_along_track_m: float,
    target_along_track_m: float,
    coast_duration_s: float,
) -> float:
    r"""Estimate tangential set-drift delta-v for a near-circular chief.

    For small tangential impulses, ``delta_a ~= 2*delta_v/n`` and
    ``delta_y_dot ~= -3*delta_v``.  Positive along-track separation therefore
    requires a negative (retrograde) deputy burn: lowering the orbit increases
    mean motion.  The returned sign is positive prograde.
    """

    required_displacement = target_along_track_m - current_along_track_m
    return -required_displacement / (3.0 * coast_duration_s)


def _recorded_advance(
    phase: MissionPhase,
    duration_s: float,
    advance: AdvanceFunction,
    phases: list[PhaseRecord],
    current_time_s: float,
) -> tuple[float, bool]:
    """Advance one named phase and retain its actual requested interval."""

    print(f"  {phase.name.replace('_', ' '):<42} t={current_time_s:8.1f} s")
    completed = advance(duration_s, phase)
    end_s = current_time_s + duration_s
    phases.append(PhaseRecord(phase=phase, start_s=current_time_s, end_s=end_s))
    return end_s, completed


def execute_mission(
    simulation: Any,
    pair: SpacecraftPair,
    propulsion: PropulsionPair,
    config: ScenarioConfig,
    advance: AdvanceFunction,
) -> MissionResult:
    """Execute the explicit ORBITAL DUET timeline using physical deputy thrusters."""

    phases: list[PhaseRecord] = []
    maneuvers: list[ManeuverRecord] = []
    time_s = 0.0
    print("MISSION TIMELINE")

    preliminary_phases = (
        (MissionPhase.DEPLOY, config.mission.dynamics_step_s),
        (MissionPhase.DETUMBLE, config.mission.detumble_duration_s),
        (MissionPhase.ATTITUDE_ACQUISITION, config.mission.attitude_acquisition_s),
        (
            MissionPhase.INITIAL_COAST_NAVIGATION_ACQUISITION,
            config.mission.navigation_acquisition_s,
        ),
        (MissionPhase.RELATIVE_STATE_ASSESSMENT, config.mission.fsw_step_s),
        (MissionPhase.PHASING_MANEUVER_DESIGN, config.mission.maneuver_design_s),
    )
    for phase, duration in preliminary_phases:
        time_s, completed = _recorded_advance(phase, duration, advance, phases, time_s)
        if not completed:
            return MissionResult(phases, maneuvers, np.zeros(3), 0.0, False)

    acquisition_duration_s = max(
        config.mission.fsw_step_s,
        config.phasing.earliest_maneuver_time_s - time_s,
    )
    time_s, completed = _recorded_advance(
        MissionPhase.SET_DRIFT_ATTITUDE_ACQUISITION,
        acquisition_duration_s,
        advance,
        phases,
        time_s,
    )
    if not completed:
        return MissionResult(phases, maneuvers, np.zeros(3), 0.0, False)

    relative_position_H, _ = current_relative_state_H(pair)
    set_drift_delta_v = first_order_set_drift_delta_v(
        relative_position_H[1],
        config.phasing.target_along_track_separation_m,
        config.phasing.phasing_coast_duration_s,
    )
    if abs(set_drift_delta_v) > config.phasing.maximum_delta_v_m_s:
        raise RuntimeError(
            f"First-order set-drift burn is {set_drift_delta_v:.3f} m/s, beyond "
            f"the {config.phasing.maximum_delta_v_m_s:.3f} m/s guardrail."
        )
    burn_duration_s = (
        abs(set_drift_delta_v)
        * config.spacecraft.mass_kg
        / config.propulsion.thrust_N
    )
    burn_duration_s = max(burn_duration_s, config.propulsion.minimum_firing_s)
    direction = "prograde" if set_drift_delta_v > 0.0 else "retrograde"
    command_time_ns = int(simulation.TotalSim.CurrentNanos)
    command_burn(
        propulsion.deputy,
        direction,
        burn_duration_s,
        command_time_ns,
        config,
    )
    maneuvers.append(
        ManeuverRecord(
            label="SET-DRIFT",
            command_time_s=time_s,
            direction=direction,
            signed_delta_v_m_s=set_drift_delta_v,
            duration_s=burn_duration_s,
            estimated_achieved_delta_v_m_s=(
                np.sign(set_drift_delta_v)
                * config.propulsion.thrust_N
                * burn_duration_s
                / config.spacecraft.mass_kg
            ),
        )
    )
    time_s, completed = _recorded_advance(
        MissionPhase.SET_DRIFT_BURN,
        burn_duration_s + config.mission.dynamics_step_s,
        advance,
        phases,
        time_s,
    )
    if not completed:
        return MissionResult(phases, maneuvers, relative_position_H, 0.0, False)

    time_s, completed = _recorded_advance(
        MissionPhase.PHASING_COAST,
        config.phasing.phasing_coast_duration_s,
        advance,
        phases,
        time_s,
    )
    if not completed:
        return MissionResult(phases, maneuvers, relative_position_H, -3 * set_drift_delta_v, False)

    for phase, duration in (
        (MissionPhase.RELATIVE_STATE_MONITORING, config.mission.fsw_step_s),
        (
            MissionPhase.ARREST_ATTITUDE_ACQUISITION,
            config.mission.arrest_attitude_acquisition_s,
        ),
    ):
        time_s, completed = _recorded_advance(phase, duration, advance, phases, time_s)
        if not completed:
            return MissionResult(phases, maneuvers, relative_position_H, -3 * set_drift_delta_v, False)

    arrest_delta_v = -set_drift_delta_v
    arrest_direction = "prograde" if arrest_delta_v > 0.0 else "retrograde"
    arrest_duration_s = (
        abs(arrest_delta_v) * config.spacecraft.mass_kg / config.propulsion.thrust_N
    )
    arrest_duration_s = max(arrest_duration_s, config.propulsion.minimum_firing_s)
    command_time_ns = int(simulation.TotalSim.CurrentNanos)
    command_burn(
        propulsion.deputy,
        arrest_direction,
        arrest_duration_s,
        command_time_ns,
        config,
    )
    maneuvers.append(
        ManeuverRecord(
            label="ARREST",
            command_time_s=time_s,
            direction=arrest_direction,
            signed_delta_v_m_s=arrest_delta_v,
            duration_s=arrest_duration_s,
            estimated_achieved_delta_v_m_s=(
                np.sign(arrest_delta_v)
                * config.propulsion.thrust_N
                * arrest_duration_s
                / config.spacecraft.mass_kg
            ),
        )
    )
    time_s, completed = _recorded_advance(
        MissionPhase.ARREST_BURN,
        arrest_duration_s + config.mission.dynamics_step_s,
        advance,
        phases,
        time_s,
    )
    if not completed:
        return MissionResult(phases, maneuvers, relative_position_H, -3 * set_drift_delta_v, False)

    time_s, completed = _recorded_advance(
        MissionPhase.FINAL_ORBIT_VERIFICATION,
        config.mission.final_verification_s,
        advance,
        phases,
        time_s,
    )
    phases.append(
        PhaseRecord(
            MissionPhase.ACQUISITION_COMPLETE,
            start_s=time_s,
            end_s=time_s,
        )
    )
    print(f"  {'ACQUISITION COMPLETE':<42} t={time_s:8.1f} s\n")
    return MissionResult(
        phases=phases,
        maneuvers=maneuvers,
        initial_relative_position_H_m=relative_position_H,
        planned_drift_rate_m_s=-3.0 * set_drift_delta_v,
        completed=completed,
    )
