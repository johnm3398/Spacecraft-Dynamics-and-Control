"""Utilities for launching Vizard from BASILISK-X."""

from pathlib import Path
import subprocess


DEFAULT_VIZARD_EXECUTABLE = Path(
    "/Applications/Vizard.app/Contents/MacOS/Vizard"
)

DEFAULT_VIZARD_ADDRESS = "tcp://localhost:5556"


def is_vizard_running(process: subprocess.Popen) -> bool:
    """Return whether a launched Vizard process is still running.

    ``Popen.poll()`` returns ``None`` while the child process is active and
    its exit code after it has stopped. This helper gives callers a readable
    way to monitor Vizard without needing to know that subprocess detail.
    """
    return process.poll() is None


def launch_vizard(
    address: str = DEFAULT_VIZARD_ADDRESS,
    executable: Path = DEFAULT_VIZARD_EXECUTABLE,
) -> subprocess.Popen:
    """
    Launch Vizard in Direct Communication mode.

    Starts the Vizard desktop application as a separate process and instructs
    it to connect to a running Basilisk live-stream interface using the
    supplied ZeroMQ address.

    This function is intended for simulations where Basilisk and Vizard run
    concurrently and simulation states are streamed directly to Vizard.

    Parameters
    ----------
    address : str, optional
        ZeroMQ address exposed by the Basilisk live-stream interface.
        The default is ``DEFAULT_VIZARD_ADDRESS``.

    executable : pathlib.Path, optional
        Path to the Vizard executable.
        The default is ``DEFAULT_VIZARD_EXECUTABLE``.

    Returns
    -------
    subprocess.Popen
        Handle to the running Vizard process. The process object can be used
        to inspect, wait for, or terminate Vizard.

    Raises
    ------
    FileNotFoundError
        If the configured Vizard executable does not exist.

    Notes
    -----
    This function only launches the Vizard client. The corresponding Basilisk
    simulation must separately configure and expose a compatible live-stream
    interface at the specified address.
    """

    executable = Path(executable)

    if not executable.exists():
        raise FileNotFoundError(
            "Vizard executable was not found at:\n"
            f"{executable}\n\n"
            "Install Vizard.app in /Applications or provide "
            "a different executable path."
        )

    process = subprocess.Popen(
        [
            str(executable),
            "--args",
            "-directComm",
            address,
        ]
    )

    print(
        f"Vizard launched with PID {process.pid}\n"
        f"Connecting to {address}"
    )

    return process


def launch_vizard_playback(
    file_path: str | Path,
    executable: Path = DEFAULT_VIZARD_EXECUTABLE,
) -> subprocess.Popen:
    """
    Launch Vizard and load a recorded Basilisk simulation.

    Starts the Vizard desktop application as a separate process and instructs
    it to open an existing Basilisk Vizard playback file.

    This function is intended for offline visualization workflows where the
    Basilisk simulation is completed first and its visualization data are
    written to a ``.bin`` file for later playback.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the recorded Basilisk Vizard playback file.

    executable : pathlib.Path, optional
        Path to the Vizard executable.
        The default is ``DEFAULT_VIZARD_EXECUTABLE``.

    Returns
    -------
    subprocess.Popen
        Handle to the running Vizard process. The process object can be used
        to inspect, wait for, or terminate Vizard.

    Raises
    ------
    FileNotFoundError
        If the configured Vizard executable does not exist or if the
        requested playback file cannot be found.

    Notes
    -----
    This function does not generate the playback file. The Basilisk scenario
    must first record visualization data using ``vizSupport``, for example::

        vizSupport.enableUnityVisualization(
            simulation,
            task_name,
            spacecraft_object,
            saveFile=str(vizard_file),
        )

    The resulting playback file can then be opened with this function after
    the simulation has completed.
    """

    executable = Path(executable)
    file_path = Path(file_path).resolve()

    if not executable.exists():
        raise FileNotFoundError(
            "Vizard executable was not found at:\n"
            f"{executable}\n\n"
            "Install Vizard.app in /Applications or provide "
            "a different executable path."
        )

    if not file_path.exists():
        raise FileNotFoundError(
            "Vizard playback file was not found at:\n"
            f"{file_path}"
        )

    process = subprocess.Popen(
        [
            str(executable),
            "--args",
            "-loadFile",
            str(file_path),
        ]
    )

    print(
        f"Vizard launched with PID {process.pid}\n"
        f"Playback file: {file_path}"
    )

    return process
