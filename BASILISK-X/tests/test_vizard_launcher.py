from basiliskx.visualization.vizard_launcher import launch_vizard


if __name__ == "__main__":
    vizard = launch_vizard()

    input("Press Enter to terminate Vizard...")

    vizard.terminate()