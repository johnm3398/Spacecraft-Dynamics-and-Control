# BASILISK-X

BASILISK-X is a personal learning and experimentation workspace built on the [Basilisk astrodynamics simulation framework](https://github.com/AVSLab/basilisk), developed by the AVS Lab at the University of Colorado Boulder.

The **“X” stands for “Experimental.”** This workspace is intended as a sandbox for experimenting with spacecraft dynamics, GNC, astrodynamics, mission simulation, and software architecture while learning how spacecraft simulation frameworks are structured and used in practice.

## Purpose

The main goals of BASILISK-X are to:

* learn how modular spacecraft simulation software is structured;
* experiment with spacecraft dynamics, orbit propagation, attitude guidance and control, relative motion, propulsion, and mission analysis;
* apply concepts learned through the University of Colorado Boulder [Spacecraft Dynamics and Control Specialization](https://www.coursera.org/specializations/spacecraft-dynamics-control);
* gain experience working with message-based simulation architectures, dynamics modules, flight-software algorithms, numerical simulation, testing, and visualization;
* develop small scenarios and reusable utilities as practical engineering exercises.

The scenarios in this folder are therefore intended primarily for **learning, experimentation, and engineering study**.

## Attribution

**BASILISK-X is not an independent spacecraft dynamics or simulation framework.**

The underlying simulation engine, spacecraft dynamics models, flight-software modules, task and message architecture, numerical simulation infrastructure, utilities, and Vizard interfaces used by these scenarios are provided by **Basilisk**, developed and maintained by the [AVS Laboratory at the University of Colorado Boulder](https://github.com/AVSLab/basilisk).

Basilisk is installed as a dependency and is not reproduced or claimed as original work in this repository.

The work contained in BASILISK-X primarily consists of my own:

* scenario configuration and integration;
* experiments built using Basilisk modules;
* engineering analysis and post-processing;
* plotting and visualization workflows;
* reusable helper utilities;
* implementations used to reinforce spacecraft dynamics, GNC, and software-engineering concepts.

Where Basilisk examples, modules, interfaces, or architectural patterns are used, the original Basilisk project remains the authoritative source.

## Installation

Basilisk is installed from its official Python distribution:

```bash
python -m pip install "bsk[all,examples]==2.11.1"
```

The local BASILISK-X utilities can then be installed in editable mode:

```bash
python -m pip install -e .
```

This keeps the upstream Basilisk framework separate from the experiments and utilities developed in this workspace.

## Project Structure

The workspace is organized broadly as:

```text
BASILISK-X/
├── examples/       # Example code from AVS lab demonstrating Basilisk's use cases
├── scenarios/      # My scenarios that I am coding up to better understand the framework
├── src/            # Reusable BASILISK-X utilities and extensions 
├── tests/          # Unit and integration tests
├── pyproject.toml
└── requirements.txt
```

Reusable functionality that may be useful across several scenarios is placed under `src/`, while mission-specific configuration and analysis remain within individual scenario folders.

## Areas of Experimentation

Current and planned work includes:

* orbit propagation and astrodynamics;
* spacecraft attitude dynamics;
* attitude guidance and control;
* quaternion and MRP-based control experiments;
* relative orbital motion;
* rendezvous and proximity operations;
* spacecraft phasing;
* finite-thrust propulsion;
* propellant and mass depletion;
* navigation and state estimation;
* mission logic and autonomy;
* Monte Carlo and sensitivity studies;
* Vizard-based 3D visualization.

The intention is to progressively increase model fidelity while retaining a clear understanding of the underlying physics and software architecture.

## References

* [AVS Lab Basilisk GitHub Repository](https://github.com/AVSLab/basilisk)
* [Basilisk Documentation](https://avslab.github.io/basilisk/)
* [Spacecraft Dynamics and Control Specialization, University of Colorado Boulder](https://www.coursera.org/specializations/spacecraft-dynamics-control)

All credit for the Basilisk simulation framework and its associated core modules belongs to the AVS Lab and the contributors to the Basilisk project.
