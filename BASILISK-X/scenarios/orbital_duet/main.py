"""ORBITAL DUET - Integrated two-spacecraft LEO phasing mission.

This is the single executable entry point.  It deliberately exposes the
Basilisk assembly order while subsystem details remain in focused modules.
Run from this directory with ``python main.py``.
"""

from typing import Any

import matplotlib.pyplot as plt

from Basilisk.utilities import SimulationBaseClass, macros

from analysis import (
    configure_recorders,
    create_plots,
    extract_results,
    print_results_summary,
    save_plots,
)
from aocs import build_aocs
from config import CONFIG, ScenarioConfig, print_configuration_summary, validate_configuration
from environment import (
    attach_environment_to_spacecraft,
    build_environment,
    print_environment_summary,
)
from mission_plan import execute_mission
from propulsion import build_propulsion
from spacecraft_model import build_spacecraft_pair
from visualization import (
    SimulationExecutor,
    configure_visualization,
    launch_live_if_requested,
    launch_playback_if_requested,
)


def create_simulation(config: ScenarioConfig) -> tuple[Any, str, str]:
    """Create one Basilisk process with explicit dynamics and FSW task rates."""

    simulation = SimulationBaseClass.SimBaseClass()
    # Mission phases already print explicit progress; repeated per-phase bars obscure it.
    simulation.SetProgressBar(False)
    process = simulation.CreateNewProcess("orbitalDuetProcess")

    dynamics_task = "orbitalDuetDynamicsTask"
    fsw_task = "orbitalDuetFswTask"
    # At coincident ticks the higher-priority FSW task publishes commands before
    # the dynamics task consumes them; asynchronous rates retain prior messages.
    process.addTask(
        simulation.CreateNewTask(fsw_task, macros.sec2nano(config.mission.fsw_step_s)),
        20,
    )
    process.addTask(
        simulation.CreateNewTask(
            dynamics_task, macros.sec2nano(config.mission.dynamics_step_s)
        ),
        10,
    )
    return simulation, dynamics_task, fsw_task


def run(config: ScenarioConfig = CONFIG) -> dict[str, Any]:
    """Assemble, execute, analyse, and return the complete mission result."""

    validate_configuration(config)
    print_configuration_summary(config)

    simulation, dynamics_task, fsw_task = create_simulation(config)

    # The construction order mirrors the physical architecture rather than
    # hiding it behind a scenario framework.
    environment = build_environment(simulation, dynamics_task, config)
    spacecraft_pair = build_spacecraft_pair(
        simulation, dynamics_task, environment.earth, config
    )
    attach_environment_to_spacecraft(
        simulation, dynamics_task, environment, spacecraft_pair, config
    )
    aocs_pair = build_aocs(
        simulation,
        dynamics_task,
        fsw_task,
        spacecraft_pair,
        environment,
        config,
    )
    propulsion_pair = build_propulsion(
        simulation, dynamics_task, spacecraft_pair, config
    )
    recorders = configure_recorders(
        simulation, dynamics_task, spacecraft_pair, environment, config
    )
    visualization = configure_visualization(
        simulation,
        dynamics_task,
        spacecraft_pair,
        aocs_pair,
        propulsion_pair,
        config,
    )

    print_environment_summary(environment, config)
    launch_live_if_requested(visualization, config)

    # Initialization calls each module's SelfInit/CrossInit/Reset lifecycle
    # after every publisher/subscriber connection above has been established.
    simulation.InitializeSimulation()
    executor = SimulationExecutor(
        simulation=simulation,
        visualization=visualization,
        propulsion=propulsion_pair,
        config=config,
    )

    mission_result = execute_mission(
        simulation,
        spacecraft_pair,
        propulsion_pair,
        config,
        executor.advance,
    )
    if not mission_result.completed:
        raise RuntimeError("ORBITAL DUET ended before acquisition verification.")

    data = extract_results(recorders, environment)
    print_results_summary(data, mission_result, aocs_pair, config)
    figures = create_plots(data, mission_result, aocs_pair)
    if config.output.save_plots:
        save_plots(figures, config)
    if config.output.show_plots:
        plt.show()
    else:
        plt.close("all")

    launch_playback_if_requested(visualization)
    if environment.spice is not None:
        environment.gravity_factory.unloadSpiceKernels()

    return {
        "configuration": config,
        "environment": environment,
        "spacecraft": spacecraft_pair,
        "aocs": aocs_pair,
        "propulsion": propulsion_pair,
        "mission": mission_result,
        "data": data,
        "figures": figures,
    }


if __name__ == "__main__":
    run()
