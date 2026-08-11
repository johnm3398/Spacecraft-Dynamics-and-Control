"""Opt-in integration test for a real Basilisk-to-Vizard live stream.

What
----
The fast tests supervise mocked Vizard and Basilisk processes. The opt-in test
launches the installed Vizard application, runs Basilisk's live-stream example
in a child Python process, and then shuts both processes down.

Why
---
Unit tests can prove command construction and cleanup decisions, but only a
real integration test can confirm that the installed Basilisk and Vizard
versions communicate successfully on the current machine. Supervising both
processes prevents Basilisk's connection wait from trapping the terminal after
the Vizard GUI has closed.

How
---
Normal ``pytest`` runs skip this test because it opens a GUI. Opt in by setting
``BASILISKX_RUN_VIZARD_INTEGRATION_TEST=1`` before running pytest, or execute
this file directly. In PowerShell, set the variable with
``$env:BASILISKX_RUN_VIZARD_INTEGRATION_TEST = "1"``. A ``finally`` block
polling lets the test notice either process exiting. ``finally`` cleanup covers
success, failure, and Ctrl+C, so neither process is left behind.
"""

import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock, call

import pytest

from basiliskx.visualization.vizard_launcher import (
    is_vizard_running,
    launch_vizard,
    terminate_process,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASILISK_STREAM_EXAMPLE = (
    PROJECT_ROOT / "examples" / "scenarioBasicOrbitStream.py"
)
RUN_INTEGRATION_ENV = "BASILISKX_RUN_VIZARD_INTEGRATION_TEST"
PROCESS_POLL_INTERVAL = 0.25


def _wait_for_simulation(
    simulation_process: subprocess.Popen,
    vizard_process: subprocess.Popen,
    poll_interval: float = PROCESS_POLL_INTERVAL,
) -> int | None:
    """Wait for Basilisk while returning promptly when Vizard exits."""
    if poll_interval <= 0:
        raise ValueError("poll_interval must be greater than zero")

    print(
        "Live-stream test running. Close or quit Vizard normally, or press "
        "Ctrl+C here to stop both processes safely."
    )
    while True:
        if not is_vizard_running(vizard_process):
            print("Vizard exited; stopping the waiting Basilisk simulation.")
            return None
        try:
            return simulation_process.wait(timeout=poll_interval)
        except subprocess.TimeoutExpired:
            continue


def run_auto_vizard_stream() -> bool:
    """Run the real live-stream integration check with guaranteed cleanup."""
    if not BASILISK_STREAM_EXAMPLE.is_file():
        raise FileNotFoundError(
            "Could not find Basilisk streaming example:\n"
            f"{BASILISK_STREAM_EXAMPLE}"
        )

    print("Starting BASILISK-X visualization integration test...")
    vizard_process = None
    simulation_process = None
    simulation_return_code = None
    try:
        vizard_process = launch_vizard(
            address="tcp://localhost:5556"
        )
        print(f"Vizard started with PID {vizard_process.pid}")
        print("Starting Basilisk simulation...")
        simulation_process = subprocess.Popen(
            [sys.executable, str(BASILISK_STREAM_EXAMPLE)],
        )
        simulation_return_code = _wait_for_simulation(
            simulation_process,
            vizard_process,
        )
        if simulation_return_code not in (None, 0):
            raise subprocess.CalledProcessError(
                simulation_return_code,
                simulation_process.args,
            )
    finally:
        if simulation_process is not None:
            return_code = terminate_process(simulation_process)
            print(f"Basilisk cleanup completed with code {return_code}.")
        if vizard_process is not None:
            return_code = terminate_process(vizard_process)
            print(f"Vizard cleanup completed with code {return_code}.")
        print("BASILISK-X visualization integration test finished.")

    return simulation_return_code is not None


def test_wait_for_simulation_returns_when_basilisk_finishes():
    simulation_process = Mock()
    simulation_process.wait.return_value = 0
    vizard_process = Mock()
    vizard_process.poll.return_value = None

    result = _wait_for_simulation(
        simulation_process,
        vizard_process,
    )

    assert result == 0
    simulation_process.wait.assert_called_once_with(timeout=0.25)


def test_wait_for_simulation_returns_when_vizard_exits():
    simulation_process = Mock()
    vizard_process = Mock()
    vizard_process.poll.return_value = 0

    result = _wait_for_simulation(
        simulation_process,
        vizard_process,
    )

    assert result is None
    simulation_process.wait.assert_not_called()


def test_wait_for_simulation_detects_vizard_exit_after_poll_timeout():
    simulation_process = Mock()
    simulation_process.wait.side_effect = subprocess.TimeoutExpired(
        "Basilisk",
        0.25,
    )
    vizard_process = Mock()
    vizard_process.poll.side_effect = [None, 0]

    result = _wait_for_simulation(
        simulation_process,
        vizard_process,
    )

    assert result is None
    assert simulation_process.wait.call_args_list == [call(timeout=0.25)]


def test_wait_for_simulation_rejects_nonpositive_poll_interval():
    with pytest.raises(ValueError, match="greater than zero"):
        _wait_for_simulation(Mock(), Mock(), poll_interval=0)


def test_run_auto_vizard_stream_cleans_up_after_vizard_exit(
    monkeypatch,
    tmp_path,
):
    example = tmp_path / "scenarioBasicOrbitStream.py"
    example.touch()
    vizard_process = Mock(pid=100)
    simulation_process = Mock(pid=200)
    launcher = Mock(return_value=vizard_process)
    popen = Mock(return_value=simulation_process)
    waiter = Mock(return_value=None)
    cleanup = Mock(side_effect=[-15, 0])
    monkeypatch.setattr("test_auto_vizard_stream.BASILISK_STREAM_EXAMPLE", example)
    monkeypatch.setattr("test_auto_vizard_stream.launch_vizard", launcher)
    monkeypatch.setattr("test_auto_vizard_stream.subprocess.Popen", popen)
    monkeypatch.setattr("test_auto_vizard_stream._wait_for_simulation", waiter)
    monkeypatch.setattr("test_auto_vizard_stream.terminate_process", cleanup)

    completed = run_auto_vizard_stream()

    assert not completed
    assert cleanup.call_args_list == [
        call(simulation_process),
        call(vizard_process),
    ]


def test_run_auto_vizard_stream_cleans_up_after_keyboard_interrupt(
    monkeypatch,
    tmp_path,
):
    example = tmp_path / "scenarioBasicOrbitStream.py"
    example.touch()
    vizard_process = Mock(pid=100)
    simulation_process = Mock(pid=200)
    cleanup = Mock(side_effect=[-15, 0])
    monkeypatch.setattr("test_auto_vizard_stream.BASILISK_STREAM_EXAMPLE", example)
    monkeypatch.setattr(
        "test_auto_vizard_stream.launch_vizard",
        Mock(return_value=vizard_process),
    )
    monkeypatch.setattr(
        "test_auto_vizard_stream.subprocess.Popen",
        Mock(return_value=simulation_process),
    )
    monkeypatch.setattr(
        "test_auto_vizard_stream._wait_for_simulation",
        Mock(side_effect=KeyboardInterrupt),
    )
    monkeypatch.setattr("test_auto_vizard_stream.terminate_process", cleanup)

    with pytest.raises(KeyboardInterrupt):
        run_auto_vizard_stream()

    assert cleanup.call_args_list == [
        call(simulation_process),
        call(vizard_process),
    ]


@pytest.mark.skipif(
    os.environ.get(RUN_INTEGRATION_ENV) != "1",
    reason=f"set {RUN_INTEGRATION_ENV}=1 to run the GUI integration test",
)
def test_auto_vizard_stream() -> None:
    """Exercise the explicitly enabled real-GUI integration path."""
    assert run_auto_vizard_stream(), (
        "Vizard exited before the Basilisk simulation completed"
    )


def main() -> None:
    """Run the integration check directly and report terminal interruption."""
    try:
        completed = run_auto_vizard_stream()
    except KeyboardInterrupt:
        print("\nIntegration test interrupted by user; cleanup completed.")
    else:
        if not completed:
            print("Integration test stopped because Vizard was closed.")


if __name__ == "__main__":
    main()
