# Spacecraft Dynamics and Control

This repository is a comprehensive collection of tools, codes, and resources developed to study and analyze the dynamics and control of spacecraft. It is structured to support learning, testing, and research in the field of spacecraft attitude and motion kinematics, as well as their applications in control systems.

## Motivation

The study of spacecraft dynamics and control is crucial for advancing space exploration and satellite technology. This repository was created to:

- Facilitate a deeper understanding of rigid body kinematics and dynamics.
- Provide a modular library (**AttitudeKinematicsLib**) of reusable functions and utilities for attitude representation and transformations (e.g., DCMs, Euler Angles, Principal Rotation Vectors, CRPs, MRPs).
- Serve as a testing ground for validating mathematical formulations and numerical implementations of attitude kinematics and dynamics.
- Support coursework, simulations, and projects related to spacecraft control systems.

## Repository Structure

- **1- Kinematics: Describing the Motions of Spacecraft**  
  Contains notebooks and materials for studying the foundational principles of spacecraft kinematics, including rigid body motion and static attitude determination.

- **2- Kinetics: Studying Spacecraft Motion**  
  A separate module dedicated to the study of forces and torques acting on spacecraft, and their impact on motion.

- **AttitudeKinematicsLib**  
  A custom Python library housing core functions for handling attitude representations and kinematic transformations:
  - **CRP.py**: Functions related to Classical Rodrigues Parameters.
  - **DCM_utils.py**: Utility functions for Direction Cosine Matrices (DCM), including validation and matrix operations.
  - **EulerAngles.py**: Euler angle transformations and related utilities.
  - **EulerRodriguesParameters.py**: Functions for working with Euler Parameters (quaternions).
  - **MRP.py**: Functions for Modified Rodrigues Parameters.
  - **PRV.py**: Principal Rotation Vectors and Axis-Angle transformations.
  - **Notebook Env for Testing**: A controlled environment with scripts and notebooks for testing the implemented functions, comparing with reference implementations (e.g., AVS Lab codes), and validating numerical accuracy.

- **Codes from AVS Lab**  
  Reference implementations and legacy codes from the AVS Lab for validation and benchmarking.

- **Project Data**  
  Contains datasets and additional resources required for simulations and analysis.

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/spacecraft-dynamics-and-control.git
   cd spacecraft-dynamics-and-control
