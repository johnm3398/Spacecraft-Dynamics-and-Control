"""Basic BASILISK-X two-body Earth-orbit scenario.

Propagate one spacecraft around a point-mass Earth from classical orbital
elements. The scenario supports visualization-off, live Direct Communication,
and recorded-playback modes and produces local mission-analysis plots.

Edit the user-configuration constants for standalone execution, or import this
module and pass keyword overrides to :func:`run`. This is intentionally a
minimal baseline with no perturbations, attitude control, or maneuvers.
"""

from pathlib import Path
from time import monotonic
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from Basilisk.simulation import planetEphemeris, spacecraft
from Basilisk.utilities import (
    SimulationBaseClass,
    macros,
    orbitalMotion,
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
#   "live"     - stream while propagation is paced relative to wall-clock time
#   "playback" - run at full speed, save a recording, then open it in Vizard
VISUALIZATION_MODE = "playback"  # "off", "live", or "playback"

# Live mode advances this many seconds of simulation per wall-clock second.
LIVE_ACCELERATION_FACTOR = 50.0
VIZARD_ADDRESS = "tcp://localhost:5556"

# Basilisk evaluates the dynamics task at this fixed interval.
DYNAMICS_DT_S = 1.0
SIMULATION_ORBITS = 3.0

# Initial classical orbital elements. Angular inputs are specified in degrees
# here and converted to radians when the Basilisk initial state is built.
# ALTITUDE_M defines the semi-major axis as Earth radius + ALTITUDE_M.
ALTITUDE_M = 550e3
ECCENTRICITY = 0.001
INCLINATION_DEG = 53.0
RAAN_DEG = 30.0
ARG_PERIAPSIS_DEG = 0.0
TRUE_ANOMALY_DEG = 0.0

# Save figures as PNG files and/or display Matplotlib windows after propagation.
SAVE_PLOTS = True
SHOW_PLOTS = True


# =============================================================================
# SCENARIO-LOCAL OUTPUT PATHS
# =============================================================================

# Resolve paths from this file rather than the terminal's working directory.
SCENARIO_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCENARIO_DIR / "output" / "plots"
VIZARD_DIR = SCENARIO_DIR / "output" / "vizard"
VIZARD_FILE = VIZARD_DIR / "basic_earth_orbit.bin"


# =============================================================================
# CONFIGURATION VALIDATION AND SIMULATION CONSTRUCTION
# =============================================================================


def validate_configuration(
    *,
    visualization_mode: str,
    live_acceleration_factor: float,
    dynamics_dt_s: float,
    simulation_orbits: float,
    altitude_m: float,
    eccentricity: float,
) -> str:
    """Validate user settings and return the normalized visualization mode."""
    mode = visualization_mode.strip().lower()
    if mode not in {"off", "live", "playback"}:
        raise ValueError("VISUALIZATION_MODE must be 'off', 'live', or 'playback'.")
    if dynamics_dt_s <= 0.0:
        raise ValueError("DYNAMICS_DT_S must be greater than zero.")
    if simulation_orbits <= 0.0:
        raise ValueError("SIMULATION_ORBITS must be greater than zero.")
    if live_acceleration_factor <= 0.0:
        raise ValueError("LIVE_ACCELERATION_FACTOR must be greater than zero.")
    if altitude_m < 0.0:
        raise ValueError("ALTITUDE_M must be non-negative.")
    if not 0.0 <= eccentricity < 1.0:
        raise ValueError("ECCENTRICITY must describe an elliptical orbit (0 <= e < 1).")
    return mode


def build_simulation(
    *,
    mode: str,
    dynamics_dt_s: float,
    simulation_orbits: float,
    altitude_m: float,
    eccentricity: float,
    inclination_deg: float,
    raan_deg: float,
    argument_of_periapsis_deg: float,
    true_anomaly_deg: float,
) -> tuple[
    Any, str, Any, Any, float, float, float, int, int
]:
    """Build the spacecraft, two-body gravity model, and state recorder."""
    task_name = "dynamicsTask"

    # SimBaseClass is the top-level Basilisk simulation container. A process
    # owns one or more tasks, and each task schedules models at a fixed rate.
    simulation = SimulationBaseClass.SimBaseClass()
    simulation.SetProgressBar(mode != "live")

    process = simulation.CreateNewProcess("dynamicsProcess")
    time_step_ns = macros.sec2nano(dynamics_dt_s)
    process.addTask(simulation.CreateNewTask(task_name, time_step_ns))

    # Spacecraft supplies the translational state and integrates the equations
    # of motion once it is added to the dynamics task.
    spacecraft_object = spacecraft.Spacecraft()
    spacecraft_object.ModelTag = "BASILISK-X-SAT"
    simulation.AddModelToTask(task_name, spacecraft_object)

    # gravBodyFactory creates Earth's point-mass gravity body. Marking Earth as
    # central keeps this basic orbit Earth-centered, then addBodiesTo connects
    # the gravity model to the spacecraft dynamics.
    gravity_factory = simIncludeGravBody.gravBodyFactory()
    earth = gravity_factory.createEarth()
    earth.isCentralBody = True
    gravity_factory.addBodiesTo(spacecraft_object)

    # Define the orbit in familiar classical elements, then convert those
    # elements into the Cartesian position and velocity Basilisk propagates.
    # ClassicElements fields are typed as optional by Basilisk's generated stubs.
    initial_elements: Any = orbitalMotion.ClassicElements()
    initial_elements.a = earth.radEquator + altitude_m
    initial_elements.e = eccentricity
    initial_elements.i = inclination_deg * macros.D2R
    initial_elements.Omega = raan_deg * macros.D2R
    initial_elements.omega = argument_of_periapsis_deg * macros.D2R
    initial_elements.f = true_anomaly_deg * macros.D2R

    initial_position_m, initial_velocity_m_s = orbitalMotion.elem2rv(
        earth.mu, initial_elements
    )
    spacecraft_object.hub.r_CN_NInit = initial_position_m
    spacecraft_object.hub.v_CN_NInit = initial_velocity_m_s

    # Kepler's third law gives the period used to express the requested run
    # length in nanoseconds, Basilisk's internal scheduler time unit.
    orbital_period_s = 2.0 * np.pi * np.sqrt(initial_elements.a**3 / earth.mu)
    duration_ns = macros.sec2nano(simulation_orbits * orbital_period_s)

    # A recorder subscribes to the spacecraft state output message. Its logged
    # position and velocity samples are read after propagation for analysis.
    state_recorder = spacecraft_object.scStateOutMsg.recorder()
    simulation.AddModelToTask(task_name, state_recorder)

    return (
        simulation,
        task_name,
        spacecraft_object,
        state_recorder,
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
    """Configure only the visualization components needed by the chosen mode."""
    # Off mode deliberately avoids even importing Basilisk's Vizard helpers.
    if mode == "off":
        return None, None

    # Keep Vizard and live-stream-only dependencies out of non-visual runs.
    from Basilisk.utilities import vizSupport

    if not vizSupport.vizFound:
        raise RuntimeError("This Basilisk installation has no Vizard interface.")

    # Playback records visualization messages during a full-speed simulation.
    # Vizard is launched only after run() confirms that this file was written.
    if mode == "playback":
        VIZARD_DIR.mkdir(parents=True, exist_ok=True)
        viz = vizSupport.enableUnityVisualization(
            simulation,
            task_name,
            spacecraft_object,
            saveFile=str(VIZARD_FILE),
        )
        return viz, None

    # Basilisk 2.11.1's official streaming example uses simSynch.ClockSynch.
    from Basilisk.simulation import simSynch

    # ClockSynch limits live propagation relative to wall-clock time so Vizard
    # can render the stream instead of receiving the orbit instantaneously.
    clock_sync = simSynch.ClockSynch()
    clock_sync.accelFactor = live_acceleration_factor
    simulation.AddModelToTask(task_name, clock_sync)

    viz = vizSupport.enableUnityVisualization(
        simulation, task_name, spacecraft_object, liveStream=True
    )

    # Ask Vizard to return the p/q keys and expose Basilisk as the Direct
    # Communication server on the port in the client address.
    viz.settings.keyboardLiveInput = "pq"
    viz.reqComProtocol = "tcp"
    viz.reqComAddress = "0.0.0.0"
    viz.reqPortNumber = vizard_address.rsplit(":", maxsplit=1)[-1]
    return viz, clock_sync


def execute_live_simulation(
    simulation: Any,
    viz: Any,
    clock_sync: Any,
    duration_ns: int,
    time_step_ns: int,
    vizard_process: Any,
) -> bool:
    """Advance one dynamics step at a time and process Vizard keyboard input."""
    current_stop_ns = 0
    paused = False
    last_key_time = monotonic() - 1.0
    print("Live controls: p = pause/resume, q = quit")

    # Execute one scheduled interval at a time so keyboard input can be checked
    # between updates. A single long ExecuteSimulation() call would block this.
    try:
        while current_stop_ns < duration_ns:
            if not is_vizard_running(vizard_process):
                print("Vizard exited; stopping the live simulation.")
                return False

            if paused:
                # This mirrors the official stream example: update only Vizard
                # and reset wall-clock synchronization while dynamics are paused.
                viz.UpdateState(current_stop_ns)
                clock_sync.Reset(0)
            else:
                current_stop_ns = min(
                    current_stop_ns + time_step_ns, duration_ns
                )
                simulation.ConfigureStopTime(current_stop_ns)
                simulation.ExecuteSimulation()

            if not is_vizard_running(vizard_process):
                print("Vizard exited; stopping the live simulation.")
                return False

            key_input = viz.userInputMsg.read().keyboardInput
            now = monotonic()

            # Debounce repeated key reports before applying pause or quit.
            if key_input and now - last_key_time >= 1.0:
                last_key_time = now
                if "q" in key_input:
                    print("Vizard requested simulation shutdown.")
                    viz.liveSettings.terminateVizard = True
                    viz.UpdateState(current_stop_ns)
                    return False
                if "p" in key_input:
                    paused = not paused
                    clock_sync.Reset(0)
                    print("Simulation paused." if paused else "Simulation resumed.")
    except KeyboardInterrupt:
        print("\nSimulation interrupted; closing Vizard.")
        viz.liveSettings.terminateVizard = True
        viz.UpdateState(current_stop_ns)
        return False

    return True


# =============================================================================
# MISSION-ANALYSIS UTILITIES
# =============================================================================


def compute_ground_track(
    time_s: np.ndarray, r_eci_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return geocentric latitude and relative Earth-fixed longitude in degrees.

    Earth rotates at a constant rate and the frames align at simulation start,
    so absolute longitude is not tied to a real UTC epoch.
    """
    # Rotate each inertial position about Earth's spin axis. Vectorizing the
    # transformation avoids constructing one rotation matrix per sample.
    angle = planetEphemeris.OMEGA_EARTH * time_s
    cosine = np.cos(angle)
    sine = np.sin(angle)
    x_fixed = cosine * r_eci_m[:, 0] + sine * r_eci_m[:, 1]
    y_fixed = -sine * r_eci_m[:, 0] + cosine * r_eci_m[:, 1]
    z_fixed = r_eci_m[:, 2]

    longitude_deg = np.degrees(np.arctan2(y_fixed, x_fixed))
    latitude_deg = np.degrees(
        np.arctan2(z_fixed, np.hypot(x_fixed, y_fixed))
    )
    return latitude_deg, longitude_deg


def compute_orbital_elements(
    mu_m3_s2: float, r_eci_m: np.ndarray, v_eci_m_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover semi-major axis, eccentricity, and inclination histories."""
    count = len(r_eci_m)
    semi_major_axis_m = np.empty(count)
    eccentricity = np.empty(count)
    inclination_deg = np.empty(count)

    # rv2elem converts each recorded Cartesian state back to osculating
    # classical elements, making numerical drift easy to inspect.
    for index, (position, velocity) in enumerate(zip(r_eci_m, v_eci_m_s)):
        elements: Any = orbitalMotion.rv2elem(mu_m3_s2, position, velocity)
        if elements is None:
            raise RuntimeError("Basilisk could not recover the orbital elements.")
        semi_major_axis_m[index] = elements.a
        eccentricity[index] = elements.e
        inclination_deg[index] = np.degrees(elements.i)

    return semi_major_axis_m, eccentricity, inclination_deg


# =============================================================================
# PLOTTING AND FILE OUTPUT
# =============================================================================


def create_plots(
    time_s: np.ndarray,
    r_eci_m: np.ndarray,
    earth_radius_m: float,
    altitude_m: np.ndarray,
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    semi_major_axis_m: np.ndarray,
    eccentricity: np.ndarray,
    inclination_deg: np.ndarray,
) -> dict[str, Figure]:
    """Create the ground-track, orbit-history, and inertial-trajectory plots."""
    figures: dict[str, Figure] = {}

    # Break the line at the dateline instead of drawing across the map.
    longitude_plot = longitude_deg.copy()
    latitude_plot = latitude_deg.copy()
    crossings = np.where(np.abs(np.diff(longitude_plot)) > 180.0)[0] + 1
    longitude_plot[crossings] = np.nan
    latitude_plot[crossings] = np.nan

    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(longitude_plot, latitude_plot, linewidth=1.2, label="Ground track")
    axis.scatter(longitude_deg[0], latitude_deg[0], s=50, label="Start")
    axis.scatter(longitude_deg[-1], latitude_deg[-1], marker="x", s=50, label="End")
    axis.set(
        xlim=(-180.0, 180.0),
        ylim=(-90.0, 90.0),
        xlabel="Relative longitude [deg]",
        ylabel="Geocentric latitude [deg]",
        title="Spacecraft Ground Track (arbitrary t=0 longitude)",
    )
    axis.set_xticks(np.arange(-180.0, 181.0, 30.0))
    axis.set_yticks(np.arange(-90.0, 91.0, 15.0))
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    figures["ground_track"] = figure

    # Altitude history is measured from Earth's equatorial reference radius.
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(time_s / 60.0, altitude_m / 1000.0)
    axis.set(
        xlabel="Simulation time [min]",
        ylabel="Altitude [km]",
        title="Spacecraft Altitude History",
    )
    axis.grid(True)
    figure.tight_layout()
    figures["altitude"] = figure

    # Plot the three recovered element histories on a shared time axis.
    figure, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    time_min = time_s / 60.0
    axes[0].plot(time_min, semi_major_axis_m / 1000.0)
    axes[0].set(ylabel="a [km]", title="Osculating Orbital Elements")
    axes[1].plot(time_min, eccentricity)
    axes[1].set_ylabel("e [-]")
    axes[2].plot(time_min, inclination_deg)
    axes[2].set(xlabel="Simulation time [min]", ylabel="i [deg]")
    for axis in axes:
        axis.grid(True)
    figure.tight_layout()
    figures["orbital_elements"] = figure

    # Convert to kilometers and build a wireframe reference sphere for a
    # geometrically scaled view of the inertial trajectory.
    r_km = r_eci_m / 1000.0
    earth_radius_km = earth_radius_m / 1000.0
    longitude = np.linspace(0.0, 2.0 * np.pi, 40)
    colatitude = np.linspace(0.0, np.pi, 20)
    x_earth = earth_radius_km * np.outer(np.cos(longitude), np.sin(colatitude))
    y_earth = earth_radius_km * np.outer(np.sin(longitude), np.sin(colatitude))
    z_earth = earth_radius_km * np.outer(np.ones_like(longitude), np.cos(colatitude))

    figure = plt.figure(figsize=(8, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(r_km[:, 0], r_km[:, 1], r_km[:, 2], label="Trajectory")
    axis.scatter(*r_km[0], s=50, label="Initial position")
    axis.plot_wireframe(x_earth, y_earth, z_earth, linewidth=0.4, alpha=0.4)
    limit_km = 1.1 * np.max(np.linalg.norm(r_km, axis=1))
    axis.set(
        xlim=(-limit_km, limit_km),
        ylim=(-limit_km, limit_km),
        zlim=(-limit_km, limit_km),
        xlabel="ECI X [km]",
        ylabel="ECI Y [km]",
        zlabel="ECI Z [km]",
        title="Spacecraft Inertial Trajectory",
    )
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.legend()
    figure.tight_layout()
    figures["inertial_trajectory"] = figure

    return figures


def save_figures(figures: dict[str, Figure]) -> None:
    """Save all figures under the scenario's output directory."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        output_path = PLOTS_DIR / f"{name}.png"
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot: {output_path}")


# =============================================================================
# REPORTING AND SCENARIO DRIVER
# =============================================================================


def print_configuration_summary(
    *,
    mode: str,
    earth_radius_m: float,
    orbital_period_s: float,
    altitude_m: float,
    eccentricity: float,
    inclination_deg: float,
    raan_deg: float,
    argument_of_periapsis_deg: float,
    true_anomaly_deg: float,
    simulation_orbits: float,
    dynamics_dt_s: float,
    live_acceleration_factor: float,
    vizard_address: str,
) -> None:
    """Print the configured initial orbit and run duration."""
    print("\n" + "=" * 68)
    print("BASILISK-X BASIC EARTH ORBIT")
    print("=" * 68)
    print(f"Visualization mode    : {mode}")
    print(f"Reference altitude    : {altitude_m / 1000.0:.3f} km")
    print(f"Semi-major axis       : {(earth_radius_m + altitude_m) / 1000.0:.3f} km")
    print(f"Eccentricity          : {eccentricity:.6f}")
    print(f"Inclination           : {inclination_deg:.3f} deg")
    print(f"RAAN                  : {raan_deg:.3f} deg")
    print(f"Argument of periapsis : {argument_of_periapsis_deg:.3f} deg")
    print(f"True anomaly          : {true_anomaly_deg:.3f} deg")
    print(f"Orbital period        : {orbital_period_s / 60.0:.3f} min")
    duration_min = simulation_orbits * orbital_period_s / 60.0
    print(f"Simulation duration   : {duration_min:.3f} min")
    print(f"Dynamics time step    : {dynamics_dt_s:.3f} s")
    if mode == "live":
        print(f"Live acceleration     : {live_acceleration_factor:.1f}x")
        print(f"Vizard address        : {vizard_address}")
    elif mode == "playback":
        print(f"Playback file         : {VIZARD_FILE}")
    print("=" * 68 + "\n")


def run(
    *,
    visualization_mode: str = VISUALIZATION_MODE,
    live_acceleration_factor: float = LIVE_ACCELERATION_FACTOR,
    vizard_address: str = VIZARD_ADDRESS,
    dynamics_dt_s: float = DYNAMICS_DT_S,
    simulation_orbits: float = SIMULATION_ORBITS,
    altitude_m: float = ALTITUDE_M,
    eccentricity: float = ECCENTRICITY,
    inclination_deg: float = INCLINATION_DEG,
    raan_deg: float = RAAN_DEG,
    argument_of_periapsis_deg: float = ARG_PERIAPSIS_DEG,
    true_anomaly_deg: float = TRUE_ANOMALY_DEG,
    save_plots: bool = SAVE_PLOTS,
    show_plots: bool = SHOW_PLOTS,
) -> None:
    """Execute the scenario using top-level defaults unless overridden.

    Args:
        visualization_mode: Use ``"off"``, ``"live"``, or ``"playback"``.
        live_acceleration_factor: Simulated-time acceleration in live mode.
        vizard_address: Vizard Direct Communication client address.
        dynamics_dt_s: Dynamics task time step in seconds.
        simulation_orbits: Number of orbital periods to simulate.
        altitude_m: Offset above Earth's equatorial radius defining semi-major axis.
        eccentricity: Initial classical eccentricity.
        inclination_deg: Initial inclination in degrees.
        raan_deg: Initial right ascension of the ascending node in degrees.
        argument_of_periapsis_deg: Initial argument of periapsis in degrees.
        true_anomaly_deg: Initial true anomaly in degrees.
        save_plots: Save analysis figures under the scenario output directory.
        show_plots: Display analysis figures after simulation.
    """
    # Normalize and reject invalid user inputs before constructing Basilisk
    # objects. Keyword arguments override the defaults at the top of this file.
    mode = validate_configuration(
        visualization_mode=visualization_mode,
        live_acceleration_factor=live_acceleration_factor,
        dynamics_dt_s=dynamics_dt_s,
        simulation_orbits=simulation_orbits,
        altitude_m=altitude_m,
        eccentricity=eccentricity,
    )

    # Build the dynamics task and retain the objects needed later for Vizard,
    # state extraction, and mission analysis.
    (
        simulation,
        task_name,
        spacecraft_object,
        state_recorder,
        mu_earth,
        earth_radius_m,
        orbital_period_s,
        duration_ns,
        time_step_ns,
    ) = build_simulation(
        mode=mode,
        dynamics_dt_s=dynamics_dt_s,
        simulation_orbits=simulation_orbits,
        altitude_m=altitude_m,
        eccentricity=eccentricity,
        inclination_deg=inclination_deg,
        raan_deg=raan_deg,
        argument_of_periapsis_deg=argument_of_periapsis_deg,
        true_anomaly_deg=true_anomaly_deg,
    )

    # Visualization is configured before initialization because the Vizard
    # interface is itself a model scheduled on the Basilisk dynamics task.
    viz, clock_sync = configure_visualization(
        simulation,
        task_name,
        spacecraft_object,
        mode=mode,
        live_acceleration_factor=live_acceleration_factor,
        vizard_address=vizard_address,
    )

    # Echo the effective settings, including any run() keyword overrides.
    print_configuration_summary(
        mode=mode,
        earth_radius_m=earth_radius_m,
        orbital_period_s=orbital_period_s,
        altitude_m=altitude_m,
        eccentricity=eccentricity,
        inclination_deg=inclination_deg,
        raan_deg=raan_deg,
        argument_of_periapsis_deg=argument_of_periapsis_deg,
        true_anomaly_deg=true_anomaly_deg,
        simulation_orbits=simulation_orbits,
        dynamics_dt_s=dynamics_dt_s,
        live_acceleration_factor=live_acceleration_factor,
        vizard_address=vizard_address,
    )

    # In live mode, start the client before initialization attempts the Direct
    # Communication handshake. Off and playback modes launch nothing here.
    vizard_process = None
    if mode == "live":
        print("Launching Vizard in Direct Communication mode...")
        vizard_process = launch_vizard(address=vizard_address)

    # Initialization runs each Basilisk model's startup and reset routines.
    try:
        simulation.InitializeSimulation()
    except KeyboardInterrupt:
        print("\nSimulation interrupted during initialization.")
        if vizard_process is not None:
            vizard_process.terminate()
        return

    print("Executing Basilisk simulation...")

    # Live mode advances incrementally for interaction. Off and playback modes
    # use one unrestricted call and therefore complete as quickly as possible.
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
            print("Simulation terminated early.")
            return
    else:
        simulation.ConfigureStopTime(duration_ns)
        simulation.ExecuteSimulation()
    print("Basilisk simulation complete.")

    # Playback is opened only after successful propagation and file creation.
    if mode == "playback":
        if not VIZARD_FILE.is_file():
            raise FileNotFoundError(
                "Simulation completed, but no Vizard recording was created at "
                f"{VIZARD_FILE}"
            )
        print("Launching Vizard recorded playback...")
        launch_vizard_playback(VIZARD_FILE)

    # Convert the recorder's nanosecond time stamps and message fields into
    # NumPy arrays for the analysis functions below.
    time_s = np.asarray(state_recorder.times()) * macros.NANO2SEC
    r_eci_m = np.asarray(state_recorder.r_BN_N)
    v_eci_m_s = np.asarray(state_recorder.v_BN_N)

    # Derive altitude, osculating elements, and the approximate ground track
    # from the propagated Cartesian state history.
    altitude_history_m = np.linalg.norm(r_eci_m, axis=1) - earth_radius_m
    (
        semi_major_axis_m,
        eccentricity_history,
        inclination_history_deg,
    ) = compute_orbital_elements(mu_earth, r_eci_m, v_eci_m_s)
    latitude_deg, longitude_deg = compute_ground_track(time_s, r_eci_m)

    # Report simple conservation and altitude checks for this two-body case.
    print("\n" + "=" * 68)
    print("SIMULATION RESULTS")
    print("=" * 68)
    print(f"Samples recorded      : {len(time_s)}")
    print(f"Minimum altitude      : {np.min(altitude_history_m) / 1000.0:.3f} km")
    print(f"Maximum altitude      : {np.max(altitude_history_m) / 1000.0:.3f} km")
    print(f"Final semi-major axis : {semi_major_axis_m[-1] / 1000.0:.6f} km")
    print(f"Final eccentricity    : {eccentricity_history[-1]:.9f}")
    print(f"Final inclination     : {inclination_history_deg[-1]:.6f} deg")
    print("=" * 68 + "\n")

    # Generate the complete mission-analysis figure set, then independently
    # honor the save and display controls.
    figures = create_plots(
        time_s,
        r_eci_m,
        earth_radius_m,
        altitude_history_m,
        latitude_deg,
        longitude_deg,
        semi_major_axis_m,
        eccentricity_history,
        inclination_history_deg,
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
    # For one-off overrides, replace this with, for example:
    # run(visualization_mode="off", simulation_orbits=1.0)
    run(
        visualization_mode=VISUALIZATION_MODE,
        simulation_orbits=SIMULATION_ORBITS,
        dynamics_dt_s=DYNAMICS_DT_S,
    )
