"""Utilities for launching Vizard from BASILISK-X."""

from pathlib import Path
import subprocess


DEFAULT_VIZARD_EXECUTABLE = Path(
    "/Applications/Vizard.app/Contents/MacOS/Vizard"
)

DEFAULT_VIZARD_ADDRESS = "tcp://localhost:5556"


def launch_vizard(
    address: str = DEFAULT_VIZARD_ADDRESS,
    executable: Path = DEFAULT_VIZARD_EXECUTABLE,
) -> subprocess.Popen:
    """
    Launch Vizard in Direct Communication mode.

    Parameters
    ----------
    address
        ZeroMQ address exposed by the Basilisk live-stream interface.

    executable
        Path to the Vizard executable.

    Returns
    -------
    subprocess.Popen
        Handle to the running Vizard process.

    Raises
    ------
    FileNotFoundError
        If Vizard is not installed at the configured location.
    """

    executable = Path(executable)

    if not executable.exists():
        raise FileNotFoundError(
            "Vizard executable was not found at:\n"
            f"{executable}\n\n"
            "Install Vizard.app in /Applications or provide a different executable path."
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