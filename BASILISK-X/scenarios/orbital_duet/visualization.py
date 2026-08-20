"""ORBITAL DUET - Vizard configuration and execution control.

Implements exactly off, playback, and live modes using the existing
basiliskx.visualization launcher.  Interactive commands stay outside mission
truth dynamics and use the deputy's physical thruster command message.
"""

from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from Basilisk.utilities import macros

from basiliskx.visualization.vizard_launcher import (
    is_vizard_running,
    launch_vizard,
    launch_vizard_playback,
)
from aocs import AocsPair
from config import ScenarioConfig
from mission_plan import MissionPhase
from propulsion import PropulsionPair, command_burn
from spacecraft_model import SpacecraftPair


@dataclass
class VisualizationRuntime:
    """Configured Vizard interface and child process handles."""

    mode: str
    viz: Any | None = None
    clock_sync: Any | None = None
    process: Any | None = None
    playback_file: Any | None = None


@dataclass
class SimulationExecutor:
    """Advance Basilisk normally or one step at a time for live interaction."""

    simulation: Any
    visualization: VisualizationRuntime
    propulsion: PropulsionPair
    config: ScenarioConfig
    paused: bool = False
    quit_requested: bool = False
    last_key_time: float = field(default_factory=lambda: monotonic() - 1.0)

    def advance(self, duration_s: float, phase: MissionPhase) -> bool:
        """Advance one mission interval; live mode polls Vizard between steps."""

        target_ns = int(self.simulation.TotalSim.CurrentNanos) + macros.sec2nano(duration_s)
        if self.visualization.mode != "live":
            self.simulation.ConfigureStopTime(target_ns)
            self.simulation.ExecuteSimulation()
            return True

        step_ns = macros.sec2nano(self.config.mission.dynamics_step_s)
        while int(self.simulation.TotalSim.CurrentNanos) < target_ns:
            if not is_vizard_running(self.visualization.process):
                print("Vizard exited; ending live mission execution.")
                return False
            self._read_live_input()
            if self.quit_requested:
                return False
            if self.paused:
                self.visualization.viz.UpdateState(int(self.simulation.TotalSim.CurrentNanos))
                self.visualization.clock_sync.Reset(int(self.simulation.TotalSim.CurrentNanos))
                continue
            next_ns = min(target_ns, int(self.simulation.TotalSim.CurrentNanos) + step_ns)
            self.simulation.ConfigureStopTime(next_ns)
            self.simulation.ExecuteSimulation()
        return True

    def _read_live_input(self) -> None:
        """Map p/q/b keys to pause, quit, and a physical ad-hoc deputy burn."""

        inputs = self.visualization.viz.userInputMsg.read().keyboardInput
        now = monotonic()
        if now - self.last_key_time < 0.75:
            return
        if "p" in inputs:
            self.paused = not self.paused
            print("Live simulation paused." if self.paused else "Live simulation resumed.")
            self.last_key_time = now
        if "q" in inputs:
            self.quit_requested = True
            self.visualization.viz.liveSettings.terminateVizard = True
            self.visualization.viz.UpdateState(int(self.simulation.TotalSim.CurrentNanos))
            self.last_key_time = now
        if "b" in inputs:
            burn_s = max(2.0, self.config.propulsion.minimum_firing_s)
            command_burn(
                self.propulsion.deputy,
                "prograde",
                burn_s,
                int(self.simulation.TotalSim.CurrentNanos),
                self.config,
            )
            print(f"Ad-hoc deputy prograde burn commanded for {burn_s:.1f} s.")
            self.last_key_time = now


def configure_visualization(
    simulation: Any,
    dynamics_task: str,
    pair: SpacecraftPair,
    aocs: AocsPair,
    propulsion: PropulsionPair,
    config: ScenarioConfig,
) -> VisualizationRuntime:
    """Configure no Vizard modules, a playback recorder, or live streaming."""

    mode = config.vizard.mode
    if mode == "off":
        return VisualizationRuntime(mode=mode)

    from Basilisk.utilities import vizSupport

    if not vizSupport.vizFound:
        raise RuntimeError("This Basilisk installation has no Vizard interface.")

    spacecraft_list = [pair.chief, pair.deputy]
    common = {
        "rwEffectorList": [aocs.chief.wheels, aocs.deputy.wheels],
        "thrEffectorList": [[propulsion.chief.effector], [propulsion.deputy.effector]],
        "cssList": [[aocs.chief.css], [aocs.deputy.css]],
    }
    if mode == "playback":
        config.output.vizard_directory.mkdir(parents=True, exist_ok=True)
        playback_file = config.output.vizard_directory / config.output.vizard_filename
        viz = vizSupport.enableUnityVisualization(
            simulation,
            dynamics_task,
            spacecraft_list,
            saveFile=str(playback_file),
            **common,
        )
        runtime = VisualizationRuntime(mode=mode, viz=viz, playback_file=playback_file)
    else:
        from Basilisk.simulation import simSynch

        clock_sync = simSynch.ClockSynch()
        clock_sync.accelFactor = config.vizard.live_acceleration_factor
        simulation.AddModelToTask(dynamics_task, clock_sync, 1)
        viz = vizSupport.enableUnityVisualization(
            simulation,
            dynamics_task,
            spacecraft_list,
            liveStream=True,
            **common,
        )
        viz.settings.keyboardLiveInput = "pqb"
        viz.reqComProtocol = "tcp"
        viz.reqComAddress = "0.0.0.0"
        viz.reqPortNumber = config.vizard.direct_comm_address.rsplit(":", 1)[-1]
        runtime = VisualizationRuntime(mode=mode, viz=viz, clock_sync=clock_sync)

    viz.settings.showSpacecraftLabels = 1
    viz.settings.trueTrajectoryLinesOn = 2
    viz.settings.orbitLinesOn = 2
    viz.settings.mainCameraTarget = pair.chief.ModelTag
    return runtime


def launch_live_if_requested(runtime: VisualizationRuntime, config: ScenarioConfig) -> None:
    """Launch the shared BASILISK-X Vizard client for Direct Communication."""

    if runtime.mode == "live":
        print("Live controls: p = pause/resume, b = deputy prograde burn, q = quit")
        runtime.process = launch_vizard(address=config.vizard.direct_comm_address)


def launch_playback_if_requested(runtime: VisualizationRuntime) -> None:
    """Open a successfully generated playback file after propagation."""

    if runtime.mode != "playback":
        return
    if runtime.playback_file is None or not runtime.playback_file.is_file():
        raise FileNotFoundError(f"Vizard recording was not created: {runtime.playback_file}")
    launch_vizard_playback(runtime.playback_file)
