"""ORBITAL DUET - Physical finite-thrust propulsion.

Attaches opposed prograde/retrograde Basilisk dynamic thrusters to each
spacecraft and owns their on-time command messages.  Mission logic may request
burns, but it never edits translational states directly.
"""

from dataclasses import dataclass
from typing import Any

from Basilisk.architecture import messaging
from Basilisk.simulation import thrusterDynamicEffector
from Basilisk.utilities import simIncludeThruster

from config import ScenarioConfig
from spacecraft_model import SpacecraftPair


@dataclass
class PropulsionSystem:
    """One two-thruster cluster and its persistent command publisher."""

    label: str
    effector: Any
    command_message: Any
    command_payload: Any
    thrust_recorders: list[Any]


@dataclass
class PropulsionPair:
    """Propulsion systems for both spacecraft."""

    chief: PropulsionSystem
    deputy: PropulsionSystem


def _build_system(
    simulation: Any,
    dynamics_task: str,
    vehicle: Any,
    label: str,
    config: ScenarioConfig,
) -> PropulsionSystem:
    """Create opposed finite-thrust devices through the bus centre of mass."""

    effector = thrusterDynamicEffector.ThrusterDynamicEffector()
    effector.ModelTag = f"{vehicle.ModelTag}-PhasingThrusters"

    factory = simIncludeThruster.thrusterFactory()
    for direction in (
        config.propulsion.prograde_direction_B,
        config.propulsion.retrograde_direction_B,
    ):
        factory.create(
            "Blank_Thruster",
            list(config.propulsion.location_B_m),
            list(direction),
            cutoffFrequency=config.propulsion.cutoff_frequency_rad_s,
            MaxThrust=config.propulsion.thrust_N,
            steadyIsp=config.propulsion.isp_s,
        )
    factory.addToSpacecraft(effector.ModelTag, effector, vehicle)
    simulation.AddModelToTask(dynamics_task, effector, 220)

    payload = messaging.THRArrayOnTimeCmdMsgPayload(OnTimeRequest=[0.0, 0.0])
    command = messaging.THRArrayOnTimeCmdMsg().write(payload)
    effector.cmdsInMsg.subscribeTo(command)

    recorders: list[Any] = []
    for output in effector.thrusterOutMsgs:
        recorder = output.recorder()
        simulation.AddModelToTask(dynamics_task, recorder, 5)
        recorders.append(recorder)

    return PropulsionSystem(
        label=label,
        effector=effector,
        command_message=command,
        command_payload=payload,
        thrust_recorders=recorders,
    )


def build_propulsion(
    simulation: Any,
    dynamics_task: str,
    pair: SpacecraftPair,
    config: ScenarioConfig,
) -> PropulsionPair:
    """Attach a physical phasing cluster to both independent spacecraft."""

    return PropulsionPair(
        chief=_build_system(simulation, dynamics_task, pair.chief, "chief", config),
        deputy=_build_system(simulation, dynamics_task, pair.deputy, "deputy", config),
    )


def command_burn(
    system: PropulsionSystem,
    direction: str,
    duration_s: float,
    command_time_ns: int,
    config: ScenarioConfig,
) -> None:
    """Publish a finite on-time command through the real thruster message path."""

    if direction not in {"prograde", "retrograde"}:
        raise ValueError("Burn direction must be 'prograde' or 'retrograde'.")
    if duration_s < config.propulsion.minimum_firing_s:
        raise ValueError("Requested burn is shorter than the configured minimum firing.")
    if duration_s > config.phasing.maximum_burn_duration_s:
        raise ValueError("Requested burn exceeds the configured duration guardrail.")

    requests = [0.0, 0.0]
    requests[0 if direction == "prograde" else 1] = float(duration_s)
    system.command_payload.OnTimeRequest = requests
    system.command_message.write(system.command_payload, time=command_time_ns)


def command_all_off(system: PropulsionSystem, command_time_ns: int) -> None:
    """Explicitly clear any persistent command state after a live ad-hoc burn."""

    system.command_payload.OnTimeRequest = [0.0, 0.0]
    system.command_message.write(system.command_payload, time=command_time_ns)
