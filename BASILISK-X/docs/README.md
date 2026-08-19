> **Draft status — AI-generated:** This page was drafted by OpenAI Codex on 2026-08-19. It has not yet been technically vetted by the BASILISK-X repository owner. The repository owner will review, correct, and maintain it. Verify APIs, units, frames, assumptions, and version compatibility before engineering use.

# Basilisk Engineering Field Manual

This directory is the canonical Markdown source for the BASILISK-X **Basilisk Engineering Field Manual**. It is a practical learning and engineering reference built around the code in this repository, not a replacement for the official Basilisk documentation.

This first-edition draft spans the full learning path from simulation architecture through mission analysis, GNC, multi-spacecraft systems, uncertainty, optical navigation, MuJoCo, and BSK-RL. It is intentionally detailed, but it remains a reviewable draft: follow the source links, challenge assumptions, and record corrections as each chapter is vetted.

## Start here

1. [Quick Start](QUICK_START.md) — the shortest path from an engineering question to a working Basilisk simulation.
2. [Scope, versions, and source provenance](00_scope_versions_and_source_provenance.md) — what these notes are based on and which compatibility limitations are already known.
3. [Architecture, execution, and lifecycle](01_architecture_execution_and_lifecycle.md) — how `SimulationBaseClass`, processes, tasks, models, events, and initialization fit together.
4. [Messages, time ordering, frames, and units](02_messages_time_ordering_frames_and_units.md) — the interfaces where many subtle simulation errors originate.
5. [Simulation workflow, fidelity, and validation](03_simulation_workflow_fidelity_and_validation.md) — how to choose the smallest model that can answer the engineering question.
6. Use the complete [learning roadmap](18_learning_roadmap.md) to choose an exercise-driven route through the remaining chapters.

## Complete manual map

### Foundations

- [00 — Scope, versions, and source provenance](00_scope_versions_and_source_provenance.md)
- [01 — Architecture, execution, and lifecycle](01_architecture_execution_and_lifecycle.md)
- [02 — Messages, time, ordering, frames, and units](02_messages_time_ordering_frames_and_units.md)
- [03 — Simulation workflow, fidelity, and validation](03_simulation_workflow_fidelity_and_validation.md)

### Physical modelling and mission analysis

- [04 — Orbits, environment, and mission analysis](04_orbits_environment_and_mission_analysis.md)
- [05 — Spacecraft dynamics, effectors, and multibody systems](05_spacecraft_dynamics_effectors_and_multibody.md)
- [06 — Attitude guidance, navigation, and control](06_attitude_guidance_navigation_and_control.md)
- [07 — Actuators, propulsion, and resources](07_actuators_propulsion_and_resources.md)
- [08 — Sensors, estimation, access, and communications](08_sensors_estimation_access_and_communications.md)

### Missions, fleets, and uncertainty

- [09 — Multi-spacecraft simulation, relative motion, and RPO](09_multi_spacecraft_relative_motion_and_rpo.md)
- [10 — Modes, events, faults, and deterministic mission autonomy](10_modes_events_faults_and_mission_autonomy.md)
- [11 — Monte Carlo, uncertainty, and statistics](11_monte_carlo_uncertainty_and_statistics.md)

### Specialist architectures

- [12 — Optical navigation](12_optical_navigation.md)
- [13 — MuJoCo, robotics, contact, and deployables](13_mujoco_robotics_contact_and_deployables.md)
- [14 — BSK-RL and decision autonomy](14_bsk_rl_and_decision_autonomy.md)

### Reuse, review, and progression

- [15 — Reusable patterns and minimal skeletons](15_reusable_patterns_and_minimal_skeletons.md)
- [16 — Pitfalls, debugging, and validation](16_pitfalls_debugging_and_validation.md)
- [17 — BASILISK-X architecture and extension policy](17_basiliskx_architecture_and_extension_policy.md)
- [18 — Personal learning roadmap](18_learning_roadmap.md)

## Quick-reference pages

- [Reference index](reference/README.md)
- [Example tree and asset map](reference/example_tree_and_asset_map.md)
- [Example capability index](reference/example_capability_index.md)
- [Module and message glossary](reference/module_and_message_glossary.md)
- [Frame, unit, and initialization checklists](reference/frame_unit_and_initialization_checklists.md)

## Status and authority

These pages have three possible kinds of statements:

- **Repository observation:** behavior or architecture directly observed in this local source tree.
- **Upstream statement:** behavior stated in official AVS Lab Basilisk documentation or source.
- **Engineering recommendation:** an interpretation or recommended practice for BASILISK-X.

When those categories could be confused, the page should label them explicitly. Local examples are evidence of an implementation pattern, not automatic proof that the pattern is preferred for production work.

The authority order for resolving disagreements is:

1. the installed Basilisk source for the exact version being executed;
2. official AVS Lab documentation for that same version;
3. version-matched upstream examples and tests;
4. this manual;
5. unverified copied or legacy examples.

## Current version basis

The current project dependency is:

```text
bsk[all,examples]==2.11.1
```

The copied `examples/` directory does not record its upstream Git tag or commit and contains some APIs from a newer development line. Read [the provenance chapter](00_scope_versions_and_source_provenance.md) before assuming that every copied example runs unchanged against 2.11.1.

## Directory structure

```text
docs/
├── README.md
├── QUICK_START.md
├── 00_scope_versions_and_source_provenance.md
├── 01_architecture_execution_and_lifecycle.md
├── 02_messages_time_ordering_frames_and_units.md
├── 03_simulation_workflow_fidelity_and_validation.md
├── 04_orbits_environment_and_mission_analysis.md
├── 05_spacecraft_dynamics_effectors_and_multibody.md
├── 06_attitude_guidance_navigation_and_control.md
├── 07_actuators_propulsion_and_resources.md
├── 08_sensors_estimation_access_and_communications.md
├── 09_multi_spacecraft_relative_motion_and_rpo.md
├── 10_modes_events_faults_and_mission_autonomy.md
├── 11_monte_carlo_uncertainty_and_statistics.md
├── 12_optical_navigation.md
├── 13_mujoco_robotics_contact_and_deployables.md
├── 14_bsk_rl_and_decision_autonomy.md
├── 15_reusable_patterns_and_minimal_skeletons.md
├── 16_pitfalls_debugging_and_validation.md
├── 17_basiliskx_architecture_and_extension_policy.md
├── 18_learning_roadmap.md
└── reference/
    ├── README.md
    ├── example_tree_and_asset_map.md
    ├── example_capability_index.md
    ├── module_and_message_glossary.md
    └── frame_unit_and_initialization_checklists.md
```

## Documentation rules

Future contributions should:

- cite exact examples, modules, classes, functions, and message types where useful;
- state the Basilisk version against which an API or behavior was verified;
- distinguish truth dynamics, sensor simulation, navigation, FSW, actuation, and analysis;
- state reference frames, attitude conventions, units, timestamps, and fidelity assumptions;
- prefer compact patterns over copying complete example scripts;
- identify idealized shortcuts such as direct state changes, prescribed attitude, perfect navigation, or ideal force/torque;
- include a validation method and engineering metric, not only plotting instructions;
- mark unresolved claims rather than filling gaps by inference.

## Official references

- [AVS Lab Basilisk repository](https://github.com/AVSLab/basilisk)
- [Basilisk documentation](https://avslab.github.io/basilisk/)
- [Basilisk integrated examples](https://avslab.github.io/basilisk/examples/index.html)
- [AVS Lab BSK-RL documentation](https://avslab.github.io/bsk_rl/)

This manual is an AI-assisted working draft that will be technically vetted and revised by the BASILISK-X repository owner.
