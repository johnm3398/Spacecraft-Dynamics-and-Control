"""End-to-end BASILISK-X Vizard auto-launch test."""

from pathlib import Path
import subprocess
import sys

from basiliskx.visualization.vizard_launcher import launch_vizard


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASILISK_STREAM_EXAMPLE = (
    PROJECT_ROOT
    / "examples"
    / "scenarioBasicOrbitStream.py"
)


def main() -> None:
    """Run the Basilisk live-stream example with automatic Vizard startup."""

    if not BASILISK_STREAM_EXAMPLE.exists():
        raise FileNotFoundError(
            "Could not find Basilisk streaming example:\n"
            f"{BASILISK_STREAM_EXAMPLE}"
        )

    print("Starting BASILISK-X visualization test...")

    # ---------------------------------------------------------
    # 1. Launch Vizard automatically
    # ---------------------------------------------------------

    vizard_process = launch_vizard(
        address="tcp://localhost:5556"
    )

    print(
        f"Vizard started with PID {vizard_process.pid}"
    )

    # ---------------------------------------------------------
    # 2. Start the Basilisk live-stream scenario
    # ---------------------------------------------------------

    print("Starting Basilisk simulation...")

    try:
        subprocess.run(
            [
                sys.executable,
                str(BASILISK_STREAM_EXAMPLE),
            ],
            check=True,
        )

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")

    finally:
        print("BASILISK-X visualization test finished.")


if __name__ == "__main__":
    main()