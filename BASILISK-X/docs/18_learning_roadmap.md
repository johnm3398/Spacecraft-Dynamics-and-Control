> **Draft status — AI-generated:** This page was drafted by OpenAI Codex on 2026-08-19. It has not yet been technically vetted by the BASILISK-X repository owner. The repository owner will review, correct, and maintain it. Verify APIs, units, frames, assumptions, and version compatibility before engineering use.

# Personal Basilisk Learning Roadmap

This roadmap is ordered by transferable engineering intuition, not by example count. Advance when the exit criteria are met; elapsed calendar time is irrelevant. Every exercise should remain small, deterministic, and accompanied by a numerical check.

```text
execution architecture
       ↓
orbit/environment/numerics
       ↓
6-DOF spacecraft dynamics
       ↓
guidance/tracking/control
       ↓
physical actuators/resources
       ↓
sensors/navigation
       ↓
access and mission operations
       ↓
multiple spacecraft/relative motion/RPO
       ↓
reusable modes and fleet architectures
       ↓
Monte Carlo and validation
       ↓
OpNav ───── optional MuJoCo branch
       ↓
decision autonomy / BSK-RL
```

## How to study an example

For every selected source, produce a one-page engineering trace:

1. What question is this example actually answering?
2. What are truth states, estimates, references, commands, and resource states?
3. Which process/task runs every model, at what period and priority?
4. Draw every material message from producer to consumer.
5. Identify every idealization and direct truth-state manipulation.
6. State the frame and units of every logged vector used in a metric.
7. Reproduce one numerical result independently.
8. Change one assumption and predict the result before running.

Do not begin by copying the whole file. Reconstruct its minimal causal chain.

## Stage 0 — Reproducible workspace and provenance

**Study**

- [`README.md`](../README.md), [`requirements.txt`](../requirements.txt), and [`pyproject.toml`](../pyproject.toml).
- [`00_scope_versions_and_source_provenance.md`](00_scope_versions_and_source_provenance.md).
- [`QUICK_START.md`](QUICK_START.md).

**Why it matters:** the local dependency is Basilisk 2.11.1, while parts of the copied example tree contain development-era APIs. A simulation that cannot identify its source version, configuration, and assets is not a reproducible engineering result.

**Build exercise:** create no new scenario yet. Record the interpreter, Basilisk version/path, platform, dependency lock, random seed policy, and one command that runs an existing headless scenario. Capture expected numerical outputs, not only plots.

**Exit criteria**

- You can explain which code is upstream example material and which is BASILISK-X.
- You can detect an API/version mismatch before editing a scenario to “make it work.”
- You know which generated products should be ephemeral and which are controlled references.

**Premature jump:** adding frameworks or new physics while the current environment and example provenance are ambiguous.

## Stage 1 — Scheduler, lifecycle, and messages

**Study**

- [`scenarioBasicOrbit.py`](../examples/scenarioBasicOrbit.py).
- [`scenarioAttitudePointing.py`](../examples/scenarioAttitudePointing.py).
- [`scenarioAttitudePointingPy.py`](../examples/scenarioAttitudePointingPy.py) for a custom `SysModel`.
- The official [process/task](https://avslab.github.io/basilisk/Learn/bskPrinciples/bskPrinciples-1.html) and [messaging](https://avslab.github.io/basilisk/Learn/bskPrinciples/bskPrinciples-3.html) introductions.

**Why it matters:** every later result depends on task rates, execution order, message ownership, initialization, and recorder placement.

**Build exercise:** use an existing simple scenario to make an execution-order/message-flow diagram. Add no physics. Predict which producer value a slower consumer reads at coincident and noncoincident updates, then verify from timestamps.

**Exit criteria**

- You can build `SimBaseClass → Process → Task → Models` from memory.
- You can state why `consumer.input.subscribeTo(producer.output)` is the correct direction.
- You understand `InitializeSimulation()`, absolute stop time, repeated execution, and recorder scheduling.
- You can diagnose an unlinked, unwritten, or one-cycle-stale message.

**Premature jump:** adopting BskSim classes before you can trace an equivalent standalone simulation.

## Stage 2 — Orbit propagation, environment, and numerical error

**Study**

- [`scenarioBasicOrbit.py`](../examples/scenarioBasicOrbit.py).
- [`scenarioOrbitMultiBody.py`](../examples/scenarioOrbitMultiBody.py).
- [`scenarioOrbitConsistencyVerification.py`](../examples/scenarioOrbitConsistencyVerification.py).
- [`scenarioDragDeorbit.py`](../examples/scenarioDragDeorbit.py).
- [`scenarioIntegrators.py`](../examples/scenarioIntegrators.py) and [`scenarioVariableTimeStepIntegrators.py`](../examples/scenarioVariableTimeStepIntegrators.py).

**Why it matters:** physical-model fidelity and numerical integration error are different. You need a baseline before attributing drift to perturbations.

**Build exercise:** propagate one orbit through point-mass, J2, third-body, and drag configurations. For each step, predict which orbital element or conserved quantity should change. Run a step-size/integrator convergence study separately.

**Exit criteria**

- Two-body energy and angular momentum errors are quantified.
- You can explain central body, inertial origin, SPICE ephemeris, body orientation, and harmonic degree/order.
- Every added perturbation is tied to a mission metric and validation expectation.

**Premature jump:** enabling high-degree gravity, many third bodies, and a dense atmosphere because they sound “high fidelity,” without showing that they affect the answer.

## Stage 3 — Rigid spacecraft 6-DOF and effectors

**Study**

- [`scenarioAttitudePointing.py`](../examples/scenarioAttitudePointing.py) for hub attitude state.
- [`scenarioAttitudeGG.py`](../examples/scenarioAttitudeGG.py) for a physical disturbance torque.
- [`scenarioHingedRigidBody.py`](../examples/scenarioHingedRigidBody.py) and [`scenarioFuelSlosh.py`](../examples/scenarioFuelSlosh.py) for state effectors.
- [`scenarioConstrainedDynamics.py`](../examples/scenarioConstrainedDynamics.py) for coupled bodies.

**Why it matters:** `Spacecraft` is a coupled translational/rotational dynamic object. State effectors, dynamic effectors, moving mass properties, and conservation checks determine whether a model is physically closed.

**Build exercise:** run an unforced asymmetric rigid spacecraft with nonzero angular rate, then add one known disturbance or one state effector. Check angular momentum/energy or the appropriate balance before and after.

**Exit criteria**

- You can distinguish hub/body reference point/center of mass.
- You can explain and correctly attach a `StateEffector` versus `DynamicEffector`.
- You can identify when an effector also requires scheduler registration.
- You validate physics without relying on Vizard appearance.

**Premature jump:** using MuJoCo for a conventional rigid bus or simple hinge before understanding standard Basilisk effectors.

## Stage 4 — Attitude reference, tracking, and ideal control

**Study**

- [`scenarioAttitudeGuidance.py`](../examples/scenarioAttitudeGuidance.py).
- [`scenarioAttitudeFeedback2T.py`](../examples/scenarioAttitudeFeedback2T.py).
- [`scenarioSweepingSpacecraft.py`](../examples/scenarioSweepingSpacecraft.py).
- [`06_attitude_guidance_navigation_and_control.md`](06_attitude_guidance_navigation_and_control.md).

**Why it matters:** it establishes the reusable `truth → navigation → AttRef → AttGuid → CmdTorque` chain and the meaning of `B`, `N`, and `R` frames.

**Build exercise:** implement no new controller. Use existing modules to compare inertial hold and Hill pointing with identical truth/control. Vary FSW rate, add a known disturbance torque, and measure settling/steady-state error.

**Exit criteria**

- You can interpret `sigma_BN`, `[BN]`, `omega_BN_B`, `sigma_BR`, and `omega_BR_B` without guessing.
- You can derive the expected reference axes for inertial/Hill pointing.
- You can explain the roles of guidance, `attTrackingError`, and `mrpFeedback` separately.
- Control performance is reported with time-domain metrics and task/message timing.

**Premature jump:** tuning gains by appearance or converting everything to quaternions before declaring convention and interface direction.

## Stage 5 — Physical actuators, momentum, propulsion, and resources

**Study**

- [`scenarioAttitudeFeedbackRW.py`](../examples/scenarioAttitudeFeedbackRW.py).
- [`scenarioAttitudeFeedback2T_TH.py`](../examples/scenarioAttitudeFeedback2T_TH.py) and [`scenarioAttitudeFeedback2T_stateEffTH.py`](../examples/scenarioAttitudeFeedback2T_stateEffTH.py).
- [`scenarioOrbitManeuverTH.py`](../examples/scenarioOrbitManeuverTH.py).
- [`scenarioMomentumDumping.py`](../examples/scenarioMomentumDumping.py).
- [`scenarioPowerDemo.py`](../examples/scenarioPowerDemo.py) and [`scenarioAttitudeFeedbackRWPower.py`](../examples/scenarioAttitudeFeedbackRWPower.py).
- [`07_actuators_propulsion_and_resources.md`](07_actuators_propulsion_and_resources.md).

**Why it matters:** ideal force/torque proves a control concept, not actuator or resource feasibility.

**Build exercise:** replace an ideal attitude torque with a three- or four-wheel chain. Apply the same maneuver, then record requested/realized torque, wheel speed/momentum, saturation margin, power, and attitude error. Add unloading only after momentum buildup is demonstrated.

**Exit criteria**

- You can trace control request → allocation → device command → physical effector → truth.
- Wheel/thruster array ordering and configuration messages are verified.
- You understand the local version's `ThrusterDynamicEffector` versus `ThrusterStateEffector` integrator contracts.
- A finite burn reports achieved impulse/terminal error and explicitly states whether fuel depletion is modelled.

**Premature jump:** claiming propulsion performance from a direct velocity reset or a thruster with no connected tank.

## Stage 6 — Sensors, stochastic models, and navigation estimation

**Study**

- [`scenarioCSS.py`](../examples/scenarioCSS.py).
- [`scenarioCSSFilters.py`](../examples/scenarioCSSFilters.py).
- [`scenarioTAM.py`](../examples/scenarioTAM.py) and [`scenarioTAMcomparison.py`](../examples/scenarioTAMcomparison.py).
- [`scenarioGaussMarkovRandomWalk.py`](../examples/scenarioGaussMarkovRandomWalk.py).
- [`scenarioSmallBodyNavUKF.py`](../examples/scenarioSmallBodyNavUKF.py) after mastering the simpler filters.
- [`08_sensors_estimation_access_and_communications.md`](08_sensors_estimation_access_and_communications.md).

**Why it matters:** truth, sensor output, calibrated measurement, estimate, and FSW input are different artifacts.

**Build exercise:** configure a CSS constellation with bias/noise/eclipse, run one sunline filter, and compare estimated sun direction with truth. Plot angular error, covariance, post-fit residuals, observation count, and outage recovery.

**Exit criteria**

- You can label `SimpleNav` as truth-like, error-emulating, or estimator output based on its configuration and producer chain.
- Sensor geometry, bias, noise, saturation, sample rate, and environmental inputs are explicit.
- Error and covariance are compared in the same frame/state ordering.
- Repeated stochastic runs use controlled seeds.

**Premature jump:** calling noise-corrupted truth an estimator, or tuning filter covariance only until plots look smooth.

## Stage 7 — Access, payloads, data, and mission operations

**Study**

- [`scenarioGroundDownlink.py`](../examples/scenarioGroundDownlink.py).
- [`scenarioGroundLocationImaging.py`](../examples/scenarioGroundLocationImaging.py).
- [`scenarioGroundMapping.py`](../examples/scenarioGroundMapping.py).
- [`scenarioDataDemo.py`](../examples/scenarioDataDemo.py).
- [`scenarioSensorThermal.py`](../examples/scenarioSensorThermal.py).

**Why it matters:** mission success is produced by geometry, pointing, payload operation, storage, downlink, power, and thermal state—not by orbital propagation alone.

**Build exercise:** simulate one target image and one ground-station return. Gate imaging on access and pointing, gate downlink on contact, and report delivered data, storage peak, energy minimum, missed opportunities, and target geometry.

**Exit criteria**

- You distinguish an `AccessMsg` from an RF link.
- Resource status is connected to mission decisions when required, not merely plotted afterward.
- Signed data rates, capacity units, and event/timing semantics are verified.

**Premature jump:** adding an RL scheduler before a deterministic rule can complete and score the mission.

## Stage 8 — Independent multiple spacecraft and relative frames

**Study**

- [`scenarioFormationBasic.py`](../examples/scenarioFormationBasic.py).
- [`scenarioSpacecraftLocation.py`](../examples/scenarioSpacecraftLocation.py).
- [`scenarioSatelliteConstellation.py`](../examples/scenarioSatelliteConstellation.py).
- [`scenarioTwoChargedSC.py`](../examples/scenarioTwoChargedSC.py) for genuinely coupled interspacecraft dynamics.

**Why it matters:** several `Spacecraft` objects can share a scheduler/environment while retaining independent truth, FSW, actuators, and messages. Relative state is a derived quantity with a chief/frame/time definition.

**Build exercise:** propagate a chief and deputy independently. Compute Hill-frame relative position/velocity using Basilisk utilities and an independently constructed DCM. Confirm axis signs with a hand-designed radial/along-track/cross-track offset.

**Exit criteria**

- Every spacecraft has unambiguous model/message tags and recorder ownership.
- You can explain shared versus per-spacecraft models and tasks.
- Chief/deputy choice, frame axes, epoch, and timestamp alignment are explicit.
- Direct cross-spacecraft subscriptions are labelled ideal information links.

**Premature jump:** adopting MultiSatBskSim for two spacecraft before a simple list/standalone structure becomes genuinely repetitive.

## Stage 9 — Formation control and rendezvous/proximity operations

**Study**

- [`scenarioFormationMeanOEFeedback.py`](../examples/scenarioFormationMeanOEFeedback.py).
- [`scenarioFormationReconfig.py`](../examples/scenarioFormationReconfig.py).
- [`scenarioDragRendezvous.py`](../examples/scenarioDragRendezvous.py).
- [`scenarioRendezVous.py`](../examples/scenarioRendezVous.py).
- Current BASILISK-X [`cooperative_geo_rendezvous.py`](../scenarios/cooperative_geo_rendezvous/cooperative_geo_rendezvous.py), with its stated low-fidelity assumptions.

**Why it matters:** these examples progress from ideal continuous formation force to finite-pulse reconfiguration and event-driven approach/pointing modes, while exposing how much a complete RPO system still requires.

**Build exercise:** define one bounded chief/deputy reconfiguration. Compare linear CW targeting with nonlinear Basilisk propagation, realize the command as finite pulses, disperse maneuver execution, and monitor terminal covariance/error, propellant, keep-out margin, and passive-safety behavior.

**Exit criteria**

- Relative orbit geometry, maneuver targeting, navigation, guidance, actuation, and safety are separate layers.
- No safety claim relies on truth-only relative state or ideal instantaneous impulses.
- Approach corridor, keep-out zone, abort condition, success criteria, and frame convention are executable metrics.

**Premature jump:** learning an RPO policy before deterministic guidance, navigation error, actuator realization, and safety supervision are validated.

## Stage 10 — Mission modes and reusable BskSim architecture

**Study**

- [`BskSim/scenarios/scenario_BasicOrbit.py`](../examples/BskSim/scenarios/scenario_BasicOrbit.py).
- [`BskSim/scenarios/scenario_AttGuidance.py`](../examples/BskSim/scenarios/scenario_AttGuidance.py).
- [`BskSim/scenarios/scenario_AttModes.py`](../examples/BskSim/scenarios/scenario_AttModes.py).
- [`BskSim/scenarios/scenario_AddRWFault.py`](../examples/BskSim/scenarios/scenario_AddRWFault.py).
- [`BskSim/BSK_masters.py`](../examples/BskSim/BSK_masters.py), [`BskSim/models/BSK_Dynamics.py`](../examples/BskSim/models/BSK_Dynamics.py), and [`BskSim/models/BSK_Fsw.py`](../examples/BskSim/models/BSK_Fsw.py).

**Why it matters:** BskSim separates a reusable vehicle/dynamics/FSW library from scenario configuration and uses gateway messages plus events to switch modes without rewiring consumers.

**Build exercise:** do not copy the whole framework. Trace one mode from `modeRequest` through event/task enablement and gateway authorship to the actuator. Then add one deliberately small mode or failure response in a disposable branch and test all transitions, including returning to standby.

**Exit criteria**

- You can state which repeated platform/mode problem justifies the abstraction.
- Disabled-task held outputs and gateway zeroing are understood.
- Every transition has entry, exit, invalid-request, and repeated-command tests.
- You have reviewed the documented local BskSim standby-event defect before reusing it.

**Premature jump:** wrapping three unrelated learning scenarios in a class hierarchy before their stable common interface exists.

## Stage 11 — Fleet architecture with MultiSatBskSim

**Study**

- [`MultiSatBskSim/scenariosMultiSat/scenario_BasicOrbitMultiSat.py`](../examples/MultiSatBskSim/scenariosMultiSat/scenario_BasicOrbitMultiSat.py).
- [`MultiSatBskSim/scenariosMultiSat/scenario_AttGuidMultiSat.py`](../examples/MultiSatBskSim/scenariosMultiSat/scenario_AttGuidMultiSat.py).
- [`MultiSatBskSim/scenariosMultiSat/scenario_StationKeepingMultiSat.py`](../examples/MultiSatBskSim/scenariosMultiSat/scenario_StationKeepingMultiSat.py).
- [`MultiSatBskSim/BSK_MultiSatMasters.py`](../examples/MultiSatBskSim/BSK_MultiSatMasters.py).

**Why it matters:** the architecture provides a shared environment process, indexed per-satellite dynamics/FSW processes, and optional formation-level models/rates.

**Build exercise:** configure three satellites with deliberately different FSW rates or platform parameters. Verify independent states, shared environment timing, formation-reference production, message ownership, and per-satellite resource logs.

**Exit criteria**

- Shared world and per-satellite responsibilities are explicit.
- Adding/removing a spacecraft does not rely on hidden indices 0–2.
- Relative/barycenter messages are not mislabeled as noisy relative navigation.
- Communications and centralized information assumptions are documented.

**Premature jump:** treating indexed lists and ideal cross-links as a distributed flight architecture.

## Stage 12 — Monte Carlo, uncertainty, and validation

**Study**

- [`MonteCarloExamples/scenarioBskSimAttFeedbackMC.py`](../examples/MonteCarloExamples/scenarioBskSimAttFeedbackMC.py).
- [`MonteCarloExamples/scenarioRerunMonteCarlo.py`](../examples/MonteCarloExamples/scenarioRerunMonteCarlo.py).
- [`scenarioMonteCarloAttRW.py`](../examples/scenarioMonteCarloAttRW.py).
- [`scenarioMonteCarloSpice.py`](../examples/scenarioMonteCarloSpice.py).

**Why it matters:** an uncertainty campaign is a deterministic simulation factory plus defensible distributions, reproducible seeds, bounded retention, failure capture, and mission-level statistics.

**Build exercise:** disperse initial orbit, attitude, sensor bias, and one actuator parameter in a validated deterministic scenario. Archive inputs/seeds, rerun one case exactly, and estimate success probability with confidence bounds and failure-mode categories.

**Exit criteria**

- Deterministic rerun is bitwise/numerically explainable within the expected environment.
- Marginals and correlations have engineering provenance.
- Retained data are sufficient for metrics and failure diagnosis without logging everything.
- Success/failure criteria are computed per run before aggregate plotting.

**Premature jump:** running thousands of cases before one nominal and several boundary cases pass physics/regression tests.

## Stage 13 — Optical navigation

**Study in this order**

1. [`OpNavScenarios/scenariosOpNav/scenario_OpNavPoint.py`](../examples/OpNavScenarios/scenariosOpNav/scenario_OpNavPoint.py).
2. [`scenario_OpNavOD.py`](../examples/OpNavScenarios/scenariosOpNav/scenario_OpNavOD.py).
3. [`scenario_OpNavAttOD.py`](../examples/OpNavScenarios/scenariosOpNav/scenario_OpNavAttOD.py).
4. Limb variants and [`scenario_OpNavHeading.py`](../examples/OpNavScenarios/scenariosOpNav/scenario_OpNavHeading.py).
5. [`scenario_faultDetOpNav.py`](../examples/OpNavScenarios/scenariosOpNav/scenario_faultDetOpNav.py).
6. CNN example only after its external model/build dependencies are resolved.

**Why it matters:** OpNav forces an end-to-end separation of truth, renderer, camera corruption, image processing, optical measurement, filter, navigation solution, guidance, and actuation.

**Build exercise:** initially make no new algorithm. Trace one pixel/limb measurement from spacecraft/planet truth through image output and conversion into the estimator. Record every frame, camera parameter, cadence, timestamp, covariance, and failure flag; test one injected invalid image.

**Exit criteria**

- Synthetic rendering, sensor acquisition/corruption, perception, measurement model, estimator, and FSW are distinct in your diagram and metrics.
- Vizard's role inside the sensor loop is understood.
- Current local version/import, hard-coded Vizard path, optional OpenCL/ONNX, and missing CNN asset issues are resolved or explicitly isolated.
- Estimator accuracy is evaluated separately from pointing and perception success.

**Premature jump:** training a CNN or autonomy policy before the camera/measurement/filter interface is reproducible.

## Stage 14 — MuJoCo branch for mechanisms and contact

This stage is optional. Take it only when the spacecraft problem needs topology/contact that standard Basilisk effectors do not represent economically.

**Study**

- [`mujoco/scenarioReactionWheel.py`](../examples/mujoco/scenarioReactionWheel.py).
- [`mujoco/scenarioAttitudeFeedbackRWMuJoCo.py`](../examples/mujoco/scenarioAttitudeFeedbackRWMuJoCo.py).
- [`mujoco/scenarioDeployPanels.py`](../examples/mujoco/scenarioDeployPanels.py).
- [`mujoco/scenarioArmWithThrusters.py`](../examples/mujoco/scenarioArmWithThrusters.py).
- [`mujoco/scenarioSimpleDocking.py`](../examples/mujoco/scenarioSimpleDocking.py) and [`mujoco/scenarioAsteroidLanding.py`](../examples/mujoco/scenarioAsteroidLanding.py).

**Why it matters:** MuJoCo supplies generic MJCF bodies, joints, sites, actuators, equality constraints, contact, and internal integration-stage dynamics while Basilisk environment/FSW messages can remain around it.

**Build exercise:** reproduce the same simple wheel-controlled bus in standard `Spacecraft` and MuJoCo. Compare attitude response, momentum/energy behavior, message interfaces, integrator settings, runtime, and modelling effort.

**Exit criteria**

- You can justify `MJScene` from contact/branching/mechanism requirements.
- XML geometry, inertial properties, joints, sites, actuators, constraints, and sensor adapters are validated independently.
- Simplified docking welds or landing forces are not called validated impact/contact models.

**Premature jump:** interpreting a more general multibody engine as automatically more accurate for orbital spacecraft dynamics.

## Stage 15 — Decision autonomy and external BSK-RL

**Study**

- [`14_bsk_rl_and_decision_autonomy.md`](14_bsk_rl_and_decision_autonomy.md).
- The official [BSK-RL API](https://avslab.github.io/bsk_rl/api_reference/index.html), starting with satellite, simulation, observation, action, data/reward, communication, and scenario concepts.
- Your validated deterministic mission from Stages 7–12.

**Why it matters:** RL adds a policy/search problem above Basilisk physics and FSW. It does not replace engineering requirements, a deterministic baseline, safety logic, or validation.

**Build exercise:** before installing or training anything, write one complete POMDP/SMDP specification for a single-satellite imaging mission or bounded inspection mission: hidden state, onboard observation, action primitives, transition/fidelity, reward units, hard constraints, uncertainty distributions, termination/truncation, and deterministic baseline.

**Exit criteria**

- No observation leaks unavailable truth or fleet knowledge.
- Every action maps to a tested FSW mode or explicitly idealized maneuver primitive.
- Safety constraints are enforced outside reward-only learning.
- Reset removes all Basilisk episode state and randomization is reproducible.
- Training and held-out engineering evaluation protocols are separate.
- The policy beats meaningful deterministic/optimization baselines on mission and safety metrics, not only return.

**Premature jump:** choosing PPO/SAC/DQN, tuning a reward, or scaling to multi-agent training before the aerospace decision problem and baseline are defined.

## Suggested capstone progression

Use one mission thread rather than a collection of unrelated demonstrations:

```text
1. Deterministic Earth orbit with validated perturbations
2. Nadir/target pointing with ideal torque
3. Reaction wheels, power, and momentum limits
4. Sensor/navigation error and target-access geometry
5. Image generation, storage, and downlink resource accounting
6. Rule-based mode scheduler with explicit success/safety metrics
7. Monte Carlo mission-success campaign
8. Add a deputy/inspector and relative-navigation uncertainty
9. Add bounded maneuver primitives and RPO safety supervision
10. Expose the validated decision layer to BSK-RL
```

At each step retain the previous model as a lower-fidelity regression oracle. If adding fidelity changes the result unexpectedly, explain the mechanism before proceeding.

## Evidence portfolio

The durable outcome of the roadmap should be a small engineering portfolio, not a large scenario count:

- execution/message diagram for each architecture used;
- requirements and fidelity rationale;
- frame/unit/interface table;
- conservation, convergence, and deterministic regression checks;
- estimator residual/covariance evidence;
- actuator and resource margins;
- mission success and safety metrics;
- Monte Carlo distributions, seeds, and rerun procedure;
- known limitations and out-of-family conditions;
- decision-autonomy baseline and held-out evaluation plan.

You are ready to move deeper when you can predict and explain results—not merely run the example successfully.
