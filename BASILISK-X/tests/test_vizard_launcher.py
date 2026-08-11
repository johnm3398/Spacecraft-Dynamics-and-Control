"""Unit tests for the cross-platform Vizard launcher.

What
----
These tests cover executable discovery, OS-specific command construction,
live and playback launches, process-state checks, and shutdown behavior.

Why
---
Launcher defects often appear only on another operating system or while a
process is exiting. Testing every decision branch prevents a local Vizard
installation from hiding a bad override, command, or cleanup race.

How
---
Temporary files model installed executables and recordings. ``monkeypatch``
selects an operating system or environment, while ``Mock`` replaces
``subprocess.Popen`` processes. No test in this module starts real Vizard.
Each test follows Arrange, Act, Assert: establish one boundary condition,
perform one public operation, and verify its result and side effects.
"""

import stat
import subprocess
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from basiliskx.visualization import vizard_launcher


def _make_executable(path: Path) -> Path:
    """Create a portable stand-in for an executable file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture(autouse=True)
def _clear_vizard_override(monkeypatch):
    """Keep the developer's real override out of isolated unit tests."""
    monkeypatch.delenv(
        vizard_launcher.VIZARD_EXECUTABLE_ENV,
        raising=False,
    )


# Executable discovery -----------------------------------------------------


def test_find_vizard_executable_prefers_explicit_path(
    monkeypatch,
    tmp_path,
):
    explicit_path = _make_executable(tmp_path / "explicit" / "Vizard.exe")
    environment_path = _make_executable(
        tmp_path / "environment" / "Vizard.exe"
    )
    monkeypatch.setattr(vizard_launcher.platform, "system", lambda: "Windows")
    monkeypatch.setenv(
        vizard_launcher.VIZARD_EXECUTABLE_ENV,
        str(environment_path),
    )

    result = vizard_launcher.find_vizard_executable(explicit_path)

    assert result == explicit_path.resolve()


def test_invalid_explicit_path_does_not_fall_back_to_environment(
    monkeypatch,
    tmp_path,
):
    missing_path = tmp_path / "missing" / "Vizard.exe"
    environment_path = _make_executable(tmp_path / "Vizard.exe")
    monkeypatch.setattr(vizard_launcher.platform, "system", lambda: "Windows")
    monkeypatch.setattr(vizard_launcher.shutil, "which", Mock(return_value=None))
    monkeypatch.setenv(
        vizard_launcher.VIZARD_EXECUTABLE_ENV,
        str(environment_path),
    )

    with pytest.raises(FileNotFoundError) as error:
        vizard_launcher.find_vizard_executable(missing_path)

    assert "explicit executable argument" in str(error.value)
    assert str(missing_path) in str(error.value)


def test_find_vizard_executable_uses_environment(monkeypatch, tmp_path):
    environment_path = _make_executable(tmp_path / "Vizard.exe")
    monkeypatch.setattr(vizard_launcher.platform, "system", lambda: "Windows")
    monkeypatch.setenv(
        vizard_launcher.VIZARD_EXECUTABLE_ENV,
        str(environment_path),
    )

    result = vizard_launcher.find_vizard_executable()

    assert result == environment_path.resolve()


def test_invalid_environment_path_does_not_fall_back(monkeypatch, tmp_path):
    missing_path = tmp_path / "missing" / "Vizard.exe"
    which = Mock(return_value=None)
    monkeypatch.setattr(vizard_launcher.platform, "system", lambda: "Windows")
    monkeypatch.setattr(vizard_launcher.shutil, "which", which)
    monkeypatch.setenv(
        vizard_launcher.VIZARD_EXECUTABLE_ENV,
        str(missing_path),
    )

    with pytest.raises(FileNotFoundError) as error:
        vizard_launcher.find_vizard_executable()

    assert vizard_launcher.VIZARD_EXECUTABLE_ENV in str(error.value)
    assert call("Vizard.exe") not in which.call_args_list


def test_find_vizard_executable_checks_macos_application_location(
    monkeypatch,
    tmp_path,
):
    application_path = _make_executable(
        tmp_path / "Vizard.app" / "Contents" / "MacOS" / "Vizard"
    )
    monkeypatch.setattr(vizard_launcher.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        vizard_launcher,
        "MACOS_VIZARD_EXECUTABLE",
        application_path,
    )

    result = vizard_launcher.find_vizard_executable()

    assert result == application_path.resolve()


def test_find_vizard_executable_checks_windows_program_files(
    monkeypatch,
    tmp_path,
):
    program_files = tmp_path / "Program Files"
    executable_path = _make_executable(
        program_files / "Vizard" / "Vizard.exe"
    )
    monkeypatch.setattr(vizard_launcher.platform, "system", lambda: "Windows")
    monkeypatch.setenv("ProgramFiles", str(program_files))
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path / "Program Files x86"))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)

    result = vizard_launcher.find_vizard_executable()

    assert result == executable_path.resolve()


def test_find_vizard_executable_falls_back_to_path(monkeypatch, tmp_path):
    executable_path = _make_executable(tmp_path / "bin" / "Vizard")
    which = Mock(
        side_effect=lambda name: (
            str(executable_path) if name == "Vizard" else None
        )
    )
    monkeypatch.setattr(vizard_launcher.platform, "system", lambda: "FreeBSD")
    monkeypatch.setattr(vizard_launcher.shutil, "which", which)

    result = vizard_launcher.find_vizard_executable()

    assert result == executable_path.resolve()
    which.assert_called_once_with("Vizard")


def test_non_executable_posix_override_has_clear_error(monkeypatch, tmp_path):
    non_executable = tmp_path / "Vizard"
    non_executable.touch()
    monkeypatch.setattr(vizard_launcher.platform, "system", lambda: "Linux")
    monkeypatch.setattr(vizard_launcher.os, "access", lambda path, mode: False)
    monkeypatch.setattr(vizard_launcher.shutil, "which", Mock(return_value=None))

    with pytest.raises(FileNotFoundError) as error:
        vizard_launcher.find_vizard_executable(non_executable)

    assert "exists but is not executable" in str(error.value)


def test_non_executable_common_candidate_is_reported(monkeypatch, tmp_path):
    non_executable = tmp_path / "Vizard"
    non_executable.touch()
    monkeypatch.setattr(vizard_launcher.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        vizard_launcher,
        "MACOS_VIZARD_EXECUTABLE",
        non_executable,
    )
    monkeypatch.setattr(vizard_launcher.os, "access", lambda path, mode: False)
    monkeypatch.setattr(vizard_launcher.shutil, "which", Mock(return_value=None))

    with pytest.raises(FileNotFoundError) as error:
        vizard_launcher.find_vizard_executable()

    assert "files exist but are not executable" in str(error.value)
    assert str(non_executable) in str(error.value)


def test_missing_executable_error_explains_all_recovery_options(monkeypatch):
    monkeypatch.setattr(vizard_launcher.platform, "system", lambda: "Plan9")
    monkeypatch.setattr(vizard_launcher.shutil, "which", Mock(return_value=None))

    with pytest.raises(FileNotFoundError) as error:
        vizard_launcher.find_vizard_executable()

    message = str(error.value)
    assert "detected OS: Plan9" in message
    assert vizard_launcher.VIZARD_EXECUTABLE_ENV in message
    assert "/Applications/Vizard.app/Contents/MacOS/Vizard" in message
    assert r"C:\Program Files\Vizard\Vizard.exe" in message


# Command and launch behavior ---------------------------------------------


def test_launch_vizard_builds_macos_command(monkeypatch):
    executable = Path("/Applications/Vizard.app/Contents/MacOS/Vizard")
    process = Mock(pid=1234)
    popen = Mock(return_value=process)
    monkeypatch.setattr(
        vizard_launcher,
        "find_vizard_executable",
        Mock(return_value=executable),
    )
    monkeypatch.setattr(vizard_launcher.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(vizard_launcher.subprocess, "Popen", popen)

    result = vizard_launcher.launch_vizard()

    assert result is process
    popen.assert_called_once_with(
        [
            str(executable),
            "--args",
            "-directComm",
            vizard_launcher.DEFAULT_VIZARD_ADDRESS,
        ]
    )


def test_launch_vizard_uses_discovery_without_starting_vizard(
    monkeypatch,
    capsys,
):
    executable_argument = Path("configured-vizard.exe")
    executable_path = Path("Vizard.exe")
    process = Mock(pid=1234)
    finder = Mock(return_value=executable_path)
    popen = Mock(return_value=process)
    monkeypatch.setattr(vizard_launcher, "find_vizard_executable", finder)
    monkeypatch.setattr(vizard_launcher.platform, "system", lambda: "Windows")
    monkeypatch.setattr(vizard_launcher.subprocess, "Popen", popen)

    result = vizard_launcher.launch_vizard(
        address="tcp://localhost:6000",
        executable=executable_argument,
    )

    assert result is process
    finder.assert_called_once_with(executable_argument)
    popen.assert_called_once_with(
        ["Vizard.exe", "-directComm", "tcp://localhost:6000"]
    )
    output = capsys.readouterr().out
    assert "Vizard.exe" in output
    assert "1234" in output
    assert "tcp://localhost:6000" in output


def test_launch_vizard_playback_validates_file_before_discovery(
    monkeypatch,
    tmp_path,
):
    finder = Mock()
    monkeypatch.setattr(vizard_launcher, "find_vizard_executable", finder)
    missing_recording = tmp_path / "missing.bin"

    with pytest.raises(FileNotFoundError) as error:
        vizard_launcher.launch_vizard_playback(missing_recording)

    assert str(missing_recording.resolve()) in str(error.value)
    finder.assert_not_called()


def test_launch_vizard_playback_builds_and_launches_command(
    monkeypatch,
    tmp_path,
    capsys,
):
    recording = tmp_path / "recording with spaces.bin"
    recording.touch()
    executable_argument = Path("configured-vizard.exe")
    executable_path = Path("Vizard.exe")
    process = Mock(pid=4321)
    finder = Mock(return_value=executable_path)
    popen = Mock(return_value=process)
    monkeypatch.setattr(vizard_launcher, "find_vizard_executable", finder)
    monkeypatch.setattr(vizard_launcher.platform, "system", lambda: "Windows")
    monkeypatch.setattr(vizard_launcher.subprocess, "Popen", popen)

    result = vizard_launcher.launch_vizard_playback(
        recording,
        executable=executable_argument,
    )

    assert result is process
    finder.assert_called_once_with(executable_argument)
    popen.assert_called_once_with(
        ["Vizard.exe", "-loadFile", str(recording.resolve())]
    )
    output = capsys.readouterr().out
    assert "Vizard.exe" in output
    assert "4321" in output
    assert str(recording.resolve()) in output


# Process lifecycle --------------------------------------------------------


def test_is_vizard_running():
    running_process = Mock()
    running_process.poll.return_value = None
    stopped_process = Mock()
    stopped_process.poll.return_value = 0

    assert vizard_launcher.is_vizard_running(running_process)
    assert not vizard_launcher.is_vizard_running(stopped_process)
    assert not vizard_launcher.is_vizard_running(None)


def test_terminate_vizard_accepts_no_process():
    assert vizard_launcher.terminate_vizard(None) is None


def test_terminate_vizard_rejects_negative_timeout():
    process = Mock()

    with pytest.raises(ValueError, match="greater than or equal to zero"):
        vizard_launcher.terminate_vizard(process, timeout=-1.0)

    process.poll.assert_not_called()


def test_terminate_vizard_returns_existing_exit_code():
    process = Mock()
    process.poll.return_value = 7

    result = vizard_launcher.terminate_vizard(process)

    assert result == 7
    process.terminate.assert_not_called()
    process.wait.assert_not_called()


def test_terminate_vizard_waits_for_normal_shutdown():
    process = Mock()
    process.poll.return_value = None
    process.wait.return_value = 0

    result = vizard_launcher.terminate_vizard(process, timeout=2.0)

    assert result == 0
    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=2.0)
    process.kill.assert_not_called()


def test_terminate_vizard_kills_process_after_timeout():
    process = Mock()
    process.poll.return_value = None
    process.wait.side_effect = [
        subprocess.TimeoutExpired("Vizard", 0.1),
        -9,
    ]

    result = vizard_launcher.terminate_vizard(process, timeout=0.1)

    assert result == -9
    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert process.wait.call_args_list == [call(timeout=0.1), call()]


def test_terminate_vizard_handles_exit_race():
    process = Mock()
    process.poll.return_value = None
    process.terminate.side_effect = ProcessLookupError
    process.wait.return_value = 0

    result = vizard_launcher.terminate_vizard(process)

    assert result == 0
    process.wait.assert_called_once_with()
    process.kill.assert_not_called()


def test_wait_for_vizard_returns_after_gui_exit(capsys):
    process = Mock()
    process.wait.return_value = 0

    result = vizard_launcher.wait_for_vizard(process)

    assert result == 0
    process.terminate.assert_not_called()
    process.wait.assert_called_once_with(timeout=0.25)
    assert "Vizard exited with code 0" in capsys.readouterr().out


def test_wait_for_vizard_escalates_repeated_keyboard_interrupt(
    capsys,
):
    process = Mock()
    process.poll.return_value = None
    process.wait.side_effect = [KeyboardInterrupt, KeyboardInterrupt, -9]

    result = vizard_launcher.wait_for_vizard(process)

    assert result == -9
    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert process.wait.call_args_list == [
        call(timeout=0.25),
        call(timeout=5.0),
        call(),
    ]
    assert "terminating Vizard" in capsys.readouterr().out


if __name__ == "__main__":
    vizard = vizard_launcher.launch_vizard()
    vizard_launcher.wait_for_vizard(vizard)
