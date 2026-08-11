"""Utilities for launching Vizard from BASILISK-X."""

import os
import platform
import shutil
import subprocess
from pathlib import Path


MACOS_VIZARD_EXECUTABLE = Path(
    "/Applications/Vizard.app/Contents/MacOS/Vizard"
)
DEFAULT_VIZARD_ADDRESS = "tcp://localhost:5556"
VIZARD_EXECUTABLE_ENV = "BASILISKX_VIZARD_EXECUTABLE"


def find_vizard_executable(
    executable: str | Path | None = None,
) -> Path:
    """Find the Vizard executable for the current platform.

    Parameters
    ----------
    executable : str or pathlib.Path or None, optional
        Explicit executable path. If omitted, the environment, common install
        locations, and the system ``PATH`` are searched in that order. An
        invalid explicit path or environment override raises immediately.

    Returns
    -------
    pathlib.Path
        Resolved path to the Vizard executable.

    Raises
    ------
    FileNotFoundError
        If no Vizard executable can be found.
    """
    system = platform.system()
    guidance = (
        f"Set {VIZARD_EXECUTABLE_ENV} to the full executable path.\n"
        "Example macOS path: "
        "/Applications/Vizard.app/Contents/MacOS/Vizard\n"
        r"Example Windows path: C:\Program Files\Vizard\Vizard.exe"
    )

    configured_path = None
    configured_source = ""
    if executable is not None:
        configured_path = executable
        configured_source = "the explicit executable argument"
    else:
        environment_path = os.environ.get(VIZARD_EXECUTABLE_ENV)
        if environment_path:
            configured_path = environment_path
            configured_source = VIZARD_EXECUTABLE_ENV

    if configured_path is not None:
        candidate = Path(configured_path).expanduser()
        if candidate.is_file():
            if system == "Windows" or os.access(candidate, os.X_OK):
                return candidate.resolve()
            reason = "exists but is not executable"
        else:
            path_from_system = shutil.which(str(configured_path))
            if path_from_system:
                return Path(path_from_system).resolve()
            reason = "does not exist or is not a file"

        raise FileNotFoundError(
            f"Vizard executable was not found (detected OS: {system}).\n"
            f"The path configured by {configured_source} {reason}:\n"
            f"{candidate}\n{guidance}"
        )

    candidates: list[Path] = []

    if system == "Darwin":
        candidates.extend(
            [
                MACOS_VIZARD_EXECUTABLE,
                Path.home()
                / "Applications/Vizard.app/Contents/MacOS/Vizard",
            ]
        )
    elif system == "Windows":
        program_files = Path(
            os.environ.get("ProgramFiles", r"C:\Program Files")
        )
        program_files_x86 = Path(
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        )
        candidates.extend(
            [
                program_files / "Vizard/Vizard.exe",
                program_files / "AVS/Vizard/Vizard.exe",
                program_files_x86 / "Vizard/Vizard.exe",
                program_files_x86 / "AVS/Vizard/Vizard.exe",
            ]
        )
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            candidates.append(
                Path(local_app_data) / "Programs/Vizard/Vizard.exe"
            )
    elif system == "Linux":
        candidates.extend(
            [
                Path("/usr/local/bin/Vizard"),
                Path("/opt/Vizard/Vizard"),
            ]
        )

    non_executable_candidates: list[Path] = []
    for candidate in candidates:
        if candidate.is_file():
            if system == "Windows" or os.access(candidate, os.X_OK):
                return candidate.resolve()
            non_executable_candidates.append(candidate)

    executable_names = (
        ("Vizard.exe", "Vizard")
        if system == "Windows"
        else ("Vizard", "vizard")
    )
    for executable_name in executable_names:
        path_from_system = shutil.which(executable_name)
        if path_from_system:
            return Path(path_from_system).resolve()

    detail = ""
    if non_executable_candidates:
        candidate_list = "\n".join(
            str(candidate) for candidate in non_executable_candidates
        )
        detail = (
            "\nThese files exist but are not executable:\n"
            f"{candidate_list}"
        )
    raise FileNotFoundError(
        f"Vizard executable was not found (detected OS: {system})."
        f"{detail}\n{guidance}"
    )


def is_vizard_running(process: subprocess.Popen | None) -> bool:
    """Return whether a launched Vizard process is still running."""
    return process is not None and process.poll() is None


def terminate_process(
    process: subprocess.Popen | None,
    timeout: float = 5.0,
) -> int | None:
    """Terminate a child process and wait for it to exit.

    Parameters
    ----------
    process : subprocess.Popen or None
        Child process to stop.
    timeout : float, optional
        Seconds to wait before forcefully killing an unresponsive process.

    Returns
    -------
    int or None
        Process exit code, or ``None`` when no process was supplied.

    Raises
    ------
    ValueError
        If ``timeout`` is negative.
    """
    if timeout < 0:
        raise ValueError("timeout must be greater than or equal to zero")
    if process is None:
        return None
    return_code = process.poll()
    if return_code is not None:
        return return_code

    try:
        process.terminate()
    except ProcessLookupError:
        return process.wait()

    try:
        return process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        pass

    try:
        process.kill()
    except ProcessLookupError:
        pass

    while True:
        try:
            return process.wait()
        except KeyboardInterrupt:
            continue


def terminate_vizard(
    process: subprocess.Popen | None,
    timeout: float = 5.0,
) -> int | None:
    """Terminate Vizard using the shared child-process cleanup behavior."""
    return terminate_process(process, timeout)


def wait_for_vizard(
    process: subprocess.Popen,
    terminate_timeout: float = 5.0,
) -> int | None:
    """Wait for Vizard while supporting safe terminal interruption.

    Parameters
    ----------
    process : subprocess.Popen
        Vizard process returned by a launcher function.
    terminate_timeout : float, optional
        Seconds to wait after Ctrl+C before forcefully killing Vizard.

    Returns
    -------
    int or None
        Vizard process exit code.
    """
    print(
        "Waiting for Vizard. Close or quit it normally, or press Ctrl+C "
        "here to terminate it safely."
    )
    while True:
        try:
            return_code = process.wait(timeout=0.25)
            break
        except subprocess.TimeoutExpired:
            continue
        except KeyboardInterrupt:
            print("\nTerminal interrupt received; terminating Vizard...")
            return_code = terminate_vizard(process, terminate_timeout)
            break

    print(f"Vizard exited with code {return_code}.")
    return return_code


def launch_vizard(
    address: str = DEFAULT_VIZARD_ADDRESS,
    executable: str | Path | None = None,
) -> subprocess.Popen:
    """Launch Vizard in Direct Communication mode.

    Parameters
    ----------
    address : str, optional
        ZeroMQ address exposed by the Basilisk live-stream interface.
    executable : str or pathlib.Path or None, optional
        Explicit Vizard executable path. Automatic discovery is used when
        omitted.

    Returns
    -------
    subprocess.Popen
        Handle to the running Vizard process.
    """
    executable_path = find_vizard_executable(executable)
    command = [str(executable_path)]
    if platform.system() == "Darwin":
        command.append("--args")
    command.extend(["-directComm", address])
    process = subprocess.Popen(command)

    print(
        f"Vizard executable: {executable_path}\n"
        f"Vizard launched with PID {process.pid}\n"
        f"Connecting to {address}"
    )
    return process


def launch_vizard_playback(
    file_path: str | Path,
    executable: str | Path | None = None,
) -> subprocess.Popen:
    """Launch Vizard with a recorded Basilisk simulation.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the recorded Basilisk Vizard playback file.
    executable : str or pathlib.Path or None, optional
        Explicit Vizard executable path. Automatic discovery is used when
        omitted.

    Returns
    -------
    subprocess.Popen
        Handle to the running Vizard process.

    Raises
    ------
    FileNotFoundError
        If the playback file or Vizard executable cannot be found.
    """
    playback_path = Path(file_path).expanduser().resolve()
    if not playback_path.is_file():
        raise FileNotFoundError(
            "Vizard playback file was not found at:\n"
            f"{playback_path}"
        )

    executable_path = find_vizard_executable(executable)
    command = [str(executable_path)]
    if platform.system() == "Darwin":
        command.append("--args")
    command.extend(["-loadFile", str(playback_path)])
    process = subprocess.Popen(command)

    print(
        f"Vizard executable: {executable_path}\n"
        f"Vizard launched with PID {process.pid}\n"
        f"Playback file: {playback_path}"
    )
    return process
