> **Draft status — AI-generated:** This page was drafted by OpenAI Codex on 2026-08-19. It has not yet been technically vetted by the BASILISK-X repository owner. The repository owner will review, correct, and maintain it. Verify APIs, units, frames, assumptions, and version compatibility before engineering use.

# Simulation Workflow, Fidelity, and Validation

This chapter is a decision guide for turning an aerospace question into a Basilisk simulation that is no more complicated than the question requires. It is based on the Basilisk 2.11.1 examples copied into this repository and on the three current BASILISK-X scenarios. The examples demonstrate APIs and architectural patterns; they do not, by themselves, certify a mission model.

## The central idea: simulate an argument, not a spacecraft

A simulation should support a specific engineering claim. Start with the claim and work backward to the minimum truth model, measurements, logic, and evidence needed to defend it.

```text
engineering question
        |
        v
decision and quantitative success criteria
        |
        v
truth states -> environment -> dynamics and effectors
        |                         |
        +------> sensors ---------+
                   |
                   v
             navigation estimate
                   |
                   v
       guidance -> tracking -> control -> actuator commands
                   |
                   v
       recorders -> derived metrics -> verification evidence
                   |
                   v
       fidelity and uncertainty sufficient for the decision?
```

The left side is the simulated physical world. The right side is what flight software is allowed to know and command. Keeping that distinction visible prevents one of the most common modeling errors: allowing guidance or autonomy to use truth that would not be available onboard.

## The engineering-question worksheet

Before constructing `SimBaseClass`, write down the following items. If an item cannot be answered, the model is not ready to grow in fidelity.

| Decision item | Question to answer | Required artifact |
|---|---|---|
| Engineering question | What decision will this run change? | One falsifiable sentence |
| Success criteria | Which numerical thresholds define success or failure? | Metrics, limits, and evaluation interval |
| Truth | Which physical states must exist to compute those metrics? | State inventory with frames and units |
| Environment | Which forces, torques, geometry, or ephemerides materially affect the result? | Environment inventory and omissions |
| Dynamics | Which degrees of freedom and effectors must exchange momentum, mass, or energy? | Dynamics topology |
| Sensing | What would be measured, at what rate, with what errors? | Sensor and measurement model |
| Navigation | Which estimated states and covariances are available to FSW? | Navigation message contract |
| FSW | Which guidance, tracking, control, allocation, and mode logic are needed? | Message-flow diagram and rates |
| Outputs | Which messages or states must be retained? | Recorder plan and sampling interval |
| Validation | What analytic result, invariant, limiting case, or independent model can challenge the result? | V&V matrix |

## Step 1: formulate the question and metric

“Propagate a spacecraft” is a task, not an engineering question. Better questions are:

- Does a two-body integrator conserve specific orbital energy to the required tolerance over three periods?
- Can an ideal torque law acquire nadir pointing within 300 seconds without exceeding 2 N m?
- Does a planned relative transfer remain outside a 10 m keep-out sphere under the assumed initial-state error?
- Is battery state of charge positive through the worst eclipse sequence?

The metric determines what must be modeled. A pointing-settling question needs attitude, rate, inertia, a reference, and a torque path; it does not automatically need atmospheric drag, high-degree gravity, thermal dynamics, or a camera renderer.

## Step 2: define the truth state

Truth is the state of the simulated physical world. In Basilisk it can include:

- spacecraft hub position, velocity, attitude, and angular rate;
- reaction-wheel speeds, fuel mass, slosh coordinates, hinge angles, or other `StateEffector` states;
- applied forces and torques from `DynamicEffector` modules;
- celestial ephemerides and body orientation;
- environmental quantities such as eclipse, magnetic field, atmosphere, power, or thermal state.

Use the smallest state set capable of producing the success metric. For a point-mass orbit check, translation alone is sufficient. For wheel saturation, the wheel momentum states are essential. For propellant usage, instantaneous velocity resets are insufficient because no mass flow exists.

Useful starting points are [scenarioBasicOrbit.py](../examples/scenarioBasicOrbit.py), [scenarioFormationBasic.py](../examples/scenarioFormationBasic.py), [scenarioAttitudeFeedbackRW.py](../examples/scenarioAttitudeFeedbackRW.py), and [scenarioFuelSlosh.py](../examples/scenarioFuelSlosh.py).

## Step 3: choose the environment

Add an environmental effect only when its magnitude, timescale, or geometry can change the decision.

| Question | Minimum useful environment | Add next only if needed |
|---|---|---|
| Short two-body propagation | Central point-mass gravity | Nonspherical gravity, third bodies, drag, SRP |
| Long LEO orbit-plane or lifetime drift | Earth gravity with required harmonics; epoch/orientation where tesseral terms matter | Atmosphere, Sun/Moon, SRP |
| Interplanetary geometry | SPICE ephemerides and relevant gravitating bodies | High-order small-body gravity, SRP, relativity if justified |
| Ground access | Body orientation, epoch, planet-fixed site geometry | Terrain, refraction, link budget |
| Eclipse and power | Sun ephemeris, occulting body, attitude-dependent collection | Detailed cell/thermal degradation |
| Magnetic control | Magnetic field model and body attitude | Space weather or higher-fidelity field data |

[scenarioBasicOrbit.py](../examples/scenarioBasicOrbit.py) contrasts point-mass and spherical-harmonic cases. [scenarioOrbitMultiBody.py](../examples/scenarioOrbitMultiBody.py), [scenarioSpiceSpacecraft.py](../examples/scenarioSpiceSpacecraft.py), [scenarioMagneticFieldWMM.py](../examples/scenarioMagneticFieldWMM.py), [scenarioDragSensitivity.py](../examples/scenarioDragSensitivity.py), and [scenarioPowerDemo.py](../examples/scenarioPowerDemo.py) show separate environment increments.

Do not combine all of them into a default “high-fidelity spacecraft.” That creates expensive runs and makes discrepancies harder to diagnose.

## Step 4: choose dynamics and actuator fidelity

Basilisk distinguishes the integrated spacecraft state from effectors coupled to it:

- A `StateEffector` contributes its own integrated state and exchanges momentum or mass with the hub. Reaction wheels, tanks, slosh particles, and hinged bodies are typical examples.
- A `DynamicEffector` applies forces or torques without adding generalized coordinates to the spacecraft state. External force/torque and dynamic thruster models are typical examples.

Choose the actuator representation from the metric:

| Metric | Adequate first model | Inadequate when |
|---|---|---|
| Transfer geometry and ideal delta-v | Instantaneous velocity change | Burn duration, pointing, execution error, or propellant matters |
| Control-law sign and ideal settling | `ExtForceTorque` commanded body torque | Allocation, saturation, momentum, power, or failures matter |
| Wheel momentum and saturation | Reaction-wheel state effector plus motor-torque allocation | Structural flexibility or detailed motor electronics matter |
| Finite burn | Thruster dynamic effector | Feed-system transients or plume/contact physics matter |
| Fuel use and mass properties | Tank state effector connected to thrusters | Slosh or moving center of mass matters |

Compare the ideal impulse in [scenarioOrbitManeuver.py](../examples/scenarioOrbitManeuver.py) with the finite-thrust treatment in [scenarioOrbitManeuverTH.py](../examples/scenarioOrbitManeuverTH.py). Compare ideal torque in [scenarioAttitudeFeedback.py](../examples/scenarioAttitudeFeedback.py) with reaction wheels in [scenarioAttitudeFeedbackRW.py](../examples/scenarioAttitudeFeedbackRW.py).

## Step 5: separate sensors, navigation, and truth

A useful conceptual chain is:

```text
truth state -> sensor physics/noise -> measurement -> estimator -> navigation message
                                                        |
                                                        v
                                                     FSW input
```

`SimpleNav` is convenient for initial architecture work. With error models disabled, it behaves like truth-derived navigation. It is not evidence that a real navigation system can achieve the same performance.

For a control-only baseline, truth navigation is often exactly the right minimum fidelity. For navigation performance, autonomous safety, or observability questions, explicitly record both truth and estimated states and evaluate their difference. Sensor and estimator examples include [scenarioCSS.py](../examples/scenarioCSS.py), [scenarioCSSFilters.py](../examples/scenarioCSSFilters.py), [scenarioTAM.py](../examples/scenarioTAM.py), [scenarioSmallBodyNav.py](../examples/scenarioSmallBodyNav.py), and [scenarioSmallBodyNavUKF.py](../examples/scenarioSmallBodyNavUKF.py).

## Step 6: construct the FSW chain

Keep the functional stages distinct even when they run in one task:

```text
navigation -> guidance reference -> tracking error -> control demand
           -> control allocation -> device command -> dynamic/state effector
```

For attitude control, a recurring chain is:

```text
SimpleNav.transOutMsg --> hillPoint/locationPointing --> AttRefMsg
SimpleNav.attOutMsg  -------------------------------> attTrackingError
attTrackingError.attGuidOutMsg ---------------------> mrpFeedback
mrpFeedback.cmdTorqueOutMsg ------------------------> ideal torque
                                      or
                                      +-------------> rwMotorTorque
                                                      -> wheel command
                                                      -> wheel state effector
```

The current [nadir-pointing scenario](../scenarios/nadir_pointing/nadir_pointing.py) implements the ideal-torque form. [scenarioAttitudeFeedbackRW.py](../examples/scenarioAttitudeFeedbackRW.py) adds allocation and reaction-wheel states. `BskSim` adds gateway messages, task enable/disable logic, and events when several flight modes must share command interfaces; see [BSK_Fsw.py](../examples/BskSim/models/BSK_Fsw.py).

## Step 7: select processes, tasks, rates, and order

A task is a rate group. A model should not receive a fast rate merely because another model needs it.

- Put tightly coupled numerical dynamics at a rate that resolves the fastest modeled physical mode.
- Run sensors at their acquisition rate.
- Run estimation and control at rates appropriate to bandwidth and data availability.
- Run mission logic more slowly unless a safety response requires otherwise.
- Use explicit model priorities when same-tick data freshness matters.
- Document whether a consumer reads the producer’s current sample or previous sample.

A one-task script is appropriate while learning a single chain. Multi-rate processes become valuable when timing is part of the question. [BSK_Dynamics.py](../examples/BskSim/models/BSK_Dynamics.py), [BSK_Fsw.py](../examples/BskSim/models/BSK_Fsw.py), and [BSK_MultiSatMasters.py](../examples/MultiSatBskSim/BSK_MultiSatMasters.py) show increasingly explicit scheduling.

## Step 8: initialize in a reproducible order

The robust construction sequence is:

1. Create `SimBaseClass`, processes, and tasks.
2. Instantiate models and assign unique `ModelTag` values.
3. Configure physical parameters and initial states.
4. Attach gravity bodies and effectors to each spacecraft.
5. Create one-time configuration messages and retain their Python objects.
6. Subscribe every required input message.
7. Add models to tasks with intentional priorities.
8. Add recorders and visualization before initialization.
9. Call `InitializeSimulation()` once the graph is complete.
10. Configure stop time and execute.

Changing a registered integrated state after initialization is a special operation, not normal configuration. The maneuver examples use it deliberately to represent an ideal impulse. It should be labeled as such.

## Step 9: record evidence, not everything

Work backward from the metrics. For each metric record:

- the state or message used to calculate it;
- its frame and units;
- the producer and whether it is truth, measurement, estimate, reference, command, or actuator response;
- a sample rate sufficient to capture extrema and transients.

Recorders are scheduled models and consume memory. A six-hour mission does not require every diagnostic at the dynamics integration rate. The official examples often use `simHelpers.samplingTime` to cap plotting histories.

Typical evidence sets are:

| Study | Minimum evidence |
|---|---|
| Orbit propagation | Time, inertial position/velocity, energy/angular momentum or orbital elements |
| Attitude acquisition | Reference, navigation attitude/rate, tracking error, command, actuator response |
| Rendezvous | Chief/deputy truth, estimated relative state, commands, achieved delta-v/thrust, constraints |
| Navigation | Truth, measurement, estimate, covariance, innovations/residuals |
| Power | Generation/load messages, eclipse state, storage state and constraint crossings |

## Fidelity ladders

Fidelity should rise one rung at a time, with a comparison at each transition.

### Orbit propagation

1. Point-mass central gravity.
2. Nonspherical gravity to the degree/order justified by duration and altitude.
3. Epoch, rotating body orientation, and third-body ephemerides.
4. Drag and SRP with simple geometry.
5. Attitude-dependent area, stochastic density, or detailed geometry.

Templates: [scenarioBasicOrbit.py](../examples/scenarioBasicOrbit.py), [scenarioOrbitMultiBody.py](../examples/scenarioOrbitMultiBody.py), [scenarioDragSensitivity.py](../examples/scenarioDragSensitivity.py), and [scenarioStochasticDragSpacecraft.py](../examples/scenarioStochasticDragSpacecraft.py). The last file is source-reading material in this checkout: it imports `Basilisk.simulation.igbmNoiseStateEffector`, which is not available in the pinned Basilisk 2.11.1 installation.

### Attitude guidance and control

1. Prescribed or inertial reference; rigid hub; ideal torque.
2. Orbit- or target-relative guidance.
3. Reaction-wheel or thruster allocation and saturation.
4. Environmental disturbance torques and momentum management.
5. Flexible appendages, sensor errors, failures, and mode logic.

Templates: [scenarioAttitudeGuidance.py](../examples/scenarioAttitudeGuidance.py), [scenarioAttitudeFeedback.py](../examples/scenarioAttitudeFeedback.py), [scenarioAttitudeFeedbackRW.py](../examples/scenarioAttitudeFeedbackRW.py), [scenarioMomentumDumping.py](../examples/scenarioMomentumDumping.py), and [scenarioFlexiblePanel.py](../examples/scenarioFlexiblePanel.py).

### Relative motion and rendezvous

1. Analytic Hill/CW geometry.
2. Two nonlinear point-mass spacecraft with truth-relative analysis.
3. Ideal impulsive targeting and constraint monitoring.
4. Finite thrusters, attitude coupling, allocation, and propellant.
5. Relative sensors, estimation, dispersions, closed-loop guidance, and abort logic.
6. Contact or articulated servicing dynamics if contact is part of the question.

Templates: [scenarioFormationBasic.py](../examples/scenarioFormationBasic.py), [scenarioRendezVous.py](../examples/scenarioRendezVous.py), [scenarioFormationReconfig.py](../examples/scenarioFormationReconfig.py), [scenarioDragRendezvous.py](../examples/scenarioDragRendezvous.py), and [scenarioSimpleDocking.py](../examples/mujoco/scenarioSimpleDocking.py).

### Navigation and optical navigation

1. Truth-derived navigation to validate the downstream FSW chain.
2. Analytic measurement generation and sensor noise.
3. Estimator and covariance consistency.
4. Synthetic image generation plus image processing.
5. Faults, missed detections, lighting/shape uncertainty, and Monte Carlo.

Templates: [scenarioSmallBodyNav.py](../examples/scenarioSmallBodyNav.py), [scenarioSmallBodyNavUKF.py](../examples/scenarioSmallBodyNavUKF.py), and the [OpNav scenario set](../examples/OpNavScenarios/scenariosOpNav/).

## Verification and validation strategy

Verification asks whether the model was implemented correctly. Validation asks whether it represents the real phenomenon well enough for the decision.

### Verification ladder

1. **Configuration checks:** finite values, vector shapes, positive masses, valid orbit domains, connected required messages.
2. **Unit tests:** frame transforms, targeting equations, controllers, and post-processing against hand calculations.
3. **Limiting cases:** zero perturbation, zero command, symmetric geometry, or disabled noise.
4. **Conservation checks:** energy and angular momentum for closed two-body motion; momentum accounting for internal effectors.
5. **Time-step convergence:** repeat with smaller task/integration steps and compare mission metrics, not only trajectories.
6. **Independent formulation:** compare Cartesian propagation with element trends, nonlinear relative motion with CW over its valid range, or a Basilisk result with another trusted implementation.
7. **Message and timing audit:** inspect execution order and verify sample age at each consumer.
8. **Regression tests:** freeze toleranced engineering metrics, not binary plot images.

### Validation questions

- Are the dominant real-world effects present?
- Are omitted effects smaller than the decision margin?
- Are parameters traceable to a source and uncertainty?
- Does a sensor model represent the measurement delivered to the estimator, rather than truth with cosmetic noise?
- Does the actuator model enforce the limits relevant to the claim?
- Have results been compared with test data, flight data, literature, or a separately derived model where available?

### Monte Carlo comes last, not first

Do not disperse an unverified deterministic model. First define a deterministic pass/fail metric, identify uncertain inputs and correlations, verify reproducible seeds, and then use the Monte Carlo infrastructure. Starting points are [scenarioMonteCarloAttRW.py](../examples/scenarioMonteCarloAttRW.py) and [MonteCarloExamples](../examples/MonteCarloExamples/).

## Applying the workflow to current BASILISK-X scenarios

| Scenario | Defensible current claim | What it must not yet claim | Best next validation |
|---|---|---|---|
| [basic_earth_orbit.py](../scenarios/basic_earth_orbit/basic_earth_orbit.py) | Point-mass Cartesian propagation and simple derived orbit/ground-track analysis | Absolute Earth-fixed longitude or operational orbit prediction | Energy/angular-momentum and step-size convergence tests |
| [nadir_pointing.py](../scenarios/nadir_pointing/nadir_pointing.py) | Ideal-torque rigid-body nadir acquisition using a Basilisk guidance/control chain | Reaction-wheel sizing, jitter, saturation, or sensor-limited pointing | Analytic sign/frame checks and deterministic settling/torque tests |
| [cooperative_geo_rendezvous.py](../scenarios/cooperative_geo_rendezvous/cooperative_geo_rendezvous.py) | CW-targeted ideal impulses applied to two nonlinear point-mass trajectories, plus truth target pointing | Flight-representative RPO, cooperation, collision avoidance, or propellant performance | CW/nonlinear comparison, constraint tests, and independent delta-v reconstruction |

## Stop conditions before adding fidelity

Do not add another module until all of the following are true:

- the engineering decision and success metric are explicit;
- the current model has a known validation reference;
- the current result is insensitive to further time-step reduction at the required tolerance;
- truth, measurement, estimate, reference, command, and response are not conflated;
- the proposed effect is plausibly large enough to change the decision;
- the additional parameters have credible values and uncertainties;
- the new fidelity can be turned off for an A/B comparison.

That discipline is the practical meaning of “minimum useful fidelity.”
