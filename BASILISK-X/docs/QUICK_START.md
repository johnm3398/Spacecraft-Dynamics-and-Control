> **Draft status — AI-generated:** This page was drafted by OpenAI Codex on 2026-08-19. It has not yet been technically vetted by the BASILISK-X repository owner. The repository owner will review, correct, and maintain it. Verify APIs, units, frames, assumptions, and version compatibility before engineering use.

# Basilisk Quick Start for BASILISK-X

This guide is a compact engineering introduction to the Basilisk APIs used throughout this repository. It targets the locally pinned Basilisk 2.11.1 environment. It is not a substitute for the [official AVS Lab documentation](https://avslab.github.io/basilisk/) or a model's API page.

The best companion source files are:

- [`scenarioBasicOrbit.py`](../examples/scenarioBasicOrbit.py) for the smallest useful orbit-propagation pattern;
- [`scenarioAttitudeFeedback2T.py`](../examples/scenarioAttitudeFeedback2T.py) for a multi-rate attitude-control loop;
- [`scenarioAttitudeFeedbackRW.py`](../examples/scenarioAttitudeFeedbackRW.py) for a physical reaction-wheel command chain;
- [`scenarioAttitudePointingPy.py`](../examples/scenarioAttitudePointingPy.py) for a custom message-driven Python module.

## 1. The mental model

A Basilisk simulation is a deterministic scheduler containing executable models. Models exchange typed messages rather than calling one another directly.

```text
SimBaseClass
│
├── Process: groups related work and establishes process priority
│   ├── Task at rate A
│   │   ├── model/module, priority 30
│   │   ├── model/module, priority 20
│   │   └── recorder, priority 0
│   └── Task at rate B
│       └── model/module
│
├── Event: periodically evaluates mission-mode conditions
└── Clock: integer nanoseconds at scheduler boundaries
```

This separation exists so that truth dynamics, sensors, navigation, flight software, actuators, recording, and visualization can operate at different rates while retaining explicit interfaces.

## 2. Essential classes and functions

| API | What it does | Typical use |
|---|---|---|
| `SimulationBaseClass.SimBaseClass()` | Creates the simulation container and scheduler interface | First object created in a scenario |
| `sim.CreateNewProcess(name, priority=...)` | Creates a process used to group tasks | Separate shared environment, dynamics, and FSW when their ordering or execution policies differ |
| `sim.CreateNewTask(name, period_ns)` | Creates a fixed-period task | One task for each required execution rate |
| `process.addTask(task)` | Attaches a task to a process | A task is not scheduled until assigned to a process |
| `sim.AddModelToTask(task_name, model, ModelPriority=...)` | Schedules an executable model or recorder | Every model that must update needs task membership; some effectors must also be attached to `Spacecraft` |
| `spacecraft.Spacecraft()` | Integrates a spacecraft hub's translation and attitude and couples its effectors | Owns truth state and normally publishes `scStateOutMsg` |
| `spacecraft.hub` | Stores hub mass properties and initial conditions | Configure mass, inertia, position, velocity, attitude, and angular rate before initialization |
| `simIncludeGravBody.gravBodyFactory()` | Constructs and connects standard gravity bodies and optional SPICE support | `createEarth()`, set `isCentralBody`, then `addBodiesTo(spacecraft)` |
| `orbitalMotion.elem2rv(mu, oe)` | Converts classical orbital elements into inertial Cartesian state | Generate consistent initial position and velocity |
| `macros.sec2nano(x)` and related functions | Convert human-scale time to scheduler nanoseconds | Task periods and stop times |
| `input_reader.subscribeTo(output_message)` | Connects a consuming module to one producer | Wire every required input before initialization |
| `output_message.recorder(period_ns)` | Creates a scheduled model that samples a message | Add the recorder to a task after its producer |
| `sim.InitializeSimulation()` | Allocates messages and calls model initialization/reset hooks | Call only after models, configuration, subscriptions, and recorders exist |
| `sim.ConfigureStopTime(absolute_ns)` | Sets the absolute time at which execution stops | For a second phase, supply the new cumulative stop time—not merely a duration |
| `sim.ExecuteSimulation()` | Advances all scheduled models and events to the configured stop time | May be called repeatedly for phased missions |
| `vizSupport.enableUnityVisualization(...)` | Adds the Basilisk-to-Vizard interface | Optional playback or live visualization; it does not normally alter truth dynamics |

The official introductions to [processes/tasks](https://avslab.github.io/basilisk/Learn/bskPrinciples/bskPrinciples-1.html) and [typed messaging](https://avslab.github.io/basilisk/Learn/bskPrinciples/bskPrinciples-3.html) are worth reading alongside this page.

## 3. The standard construction lifecycle

Use this order until there is a specific reason to depart from it:

1. State the engineering question, output metrics, and minimum physical fidelity.
2. Create `SimBaseClass`, processes, and fixed-rate tasks.
3. Construct truth models: spacecraft, environment, and effectors.
4. Construct sensor, navigation, guidance, control, and actuator modules as required.
5. Configure model parameters and initial conditions.
6. Connect every required input message to its intended output message.
7. Create recorders and schedule them at deliberate rates and priorities.
8. Optionally add events, module loggers, and Vizard.
9. Call `InitializeSimulation()`.
10. Set an absolute stop time and call `ExecuteSimulation()`.
11. Read recorded arrays, transform them into engineering metrics, and validate the result.

Initialization is an architectural boundary. Do not treat it as a harmless formatting call: model self-initialization, reset logic, state registration, and message preparation occur there.

## 4. Minimal orbit-propagation skeleton

This is a memory aid, not a finished scenario. It deliberately uses point-mass Earth gravity and no sensor or FSW model.

```python
import numpy as np

from Basilisk.simulation import spacecraft
from Basilisk.utilities import (
    SimulationBaseClass,
    macros,
    orbitalMotion,
    simHelpers,
    simIncludeGravBody,
)

# Scheduler: one dynamics process containing one 10-second task.
sim = SimulationBaseClass.SimBaseClass()
process = sim.CreateNewProcess("dynamicsProcess")
task_rate = macros.sec2nano(10.0)
process.addTask(sim.CreateNewTask("dynamicsTask", task_rate))

# Truth spacecraft.
sc = spacecraft.Spacecraft()
sc.ModelTag = "spacecraft"
sc.hub.mHub = 100.0                                      # kg
sc.hub.IHubPntBc_B = simHelpers.np2EigenMatrix3d(
    [10.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 6.0]
)                                                           # kg m^2
sim.AddModelToTask("dynamicsTask", sc)

# Point-mass Earth. More bodies or harmonics are separate fidelity choices.
gravity = simIncludeGravBody.gravBodyFactory()
earth = gravity.createEarth()
earth.isCentralBody = True
gravity.addBodiesTo(sc)

# Inertial initial state from osculating classical elements.
oe = orbitalMotion.ClassicElements()
oe.a = 7_000_000.0                                      # m
oe.e = 0.001
oe.i = 51.6 * macros.D2R                                # rad
oe.Omega = 20.0 * macros.D2R
oe.omega = 30.0 * macros.D2R
oe.f = 40.0 * macros.D2R
r_N, v_N = orbitalMotion.elem2rv(earth.mu, oe)
sc.hub.r_CN_NInit = r_N                                 # m
sc.hub.v_CN_NInit = v_N                                 # m/s

# Record truth no faster than once per minute.
state_rec = sc.scStateOutMsg.recorder(macros.sec2nano(60.0))
sim.AddModelToTask("dynamicsTask", state_rec)

# Initialize once, then execute to an absolute simulation time.
sim.InitializeSimulation()
sim.ConfigureStopTime(macros.min2nano(30.0))
sim.ExecuteSimulation()

time_s = state_rec.times() * macros.NANO2SEC
position_N = np.asarray(state_rec.r_BN_N)                # m
velocity_N = np.asarray(state_rec.v_BN_N)                # m/s
```

Important interpretation:

- `Spacecraft` is still a 6-DOF truth object even when only orbit variables are analyzed.
- The initial position uses the spacecraft center-of-mass notation `C`; the output message commonly exposes the body reference-point notation `B`. These coincide for the simple rigid hub above but need not coincide when the center of mass moves.
- `N` denotes the inertial frame. Basilisk does not infer or convert frames from variable names.
- Setting Earth as the central body defines the central-body-relative interpretation used by the gravity system. Adding third bodies, harmonics, atmosphere, or SPICE changes the model; it is not a prerequisite for a valid two-body question.
- Validate energy, angular momentum, orbital period, or another physics metric before trusting plots.

## 5. Wiring a closed-loop attitude chain

The common control architecture is:

```text
Spacecraft truth
    │ SCStatesMsg
    v
SimpleNav ── NavAttMsg ────────────────┐
                                       v
inertial3D ── AttRefMsg ──> attTrackingError
                                       │ AttGuidMsg
                                       v
                                  mrpFeedback
                                       │ CmdTorqueBodyMsg
                                       v
                                ExtForceTorque
                                       │ physical torque
                                       └────────> Spacecraft truth
```

The essential configuration and subscriptions look like this:

```python
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import attTrackingError, inertial3D, mrpFeedback
from Basilisk.simulation import extForceTorque, simpleNav

nav = simpleNav.SimpleNav()
reference = inertial3D.inertial3D()
tracking = attTrackingError.attTrackingError()
controller = mrpFeedback.mrpFeedback()
torque_effector = extForceTorque.ExtForceTorque()

reference.sigma_R0N = [0.0, 0.0, 0.0]
controller.K = 3.5
controller.P = 30.0
controller.Ki = -1.0                                    # integral disabled

# A standalone configuration message is a constant data source.
inertia = [10.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 6.0]
vehicle_payload = messaging.VehicleConfigMsgPayload(ISCPntB_B=inertia)
vehicle_config = messaging.VehicleConfigMsg().write(vehicle_payload)

# Consumer input subscribes to producer output.
nav.scStateInMsg.subscribeTo(sc.scStateOutMsg)
tracking.attNavInMsg.subscribeTo(nav.attOutMsg)
tracking.attRefInMsg.subscribeTo(reference.attRefOutMsg)
controller.guidInMsg.subscribeTo(tracking.attGuidOutMsg)
controller.vehConfigInMsg.subscribeTo(vehicle_config)
torque_effector.cmdTorqueInMsg.subscribeTo(controller.cmdTorqueOutMsg)

# The effector must influence the spacecraft and must be scheduled to update.
sc.addDynamicEffector(torque_effector)
sim.AddModelToTask("dynamicsTask", torque_effector)
```

The example omits task creation for the added modules. A practical implementation normally schedules `nav`, `reference`, `tracking`, and `controller` explicitly—often with navigation on a dynamics/sensor task and FSW on a slower task. See [`scenarioAttitudeFeedback2T.py`](../examples/scenarioAttitudeFeedback2T.py) for that complete pattern.

The boundaries matter:

- `Spacecraft` publishes physical truth.
- `SimpleNav` creates navigation messages. Unless its errors are configured, it is essentially truth-like navigation; it is not evidence that an estimator has met navigation requirements.
- Guidance produces a desired reference motion.
- `attTrackingError` compares reference and navigation states using Basilisk's attitude conventions.
- `mrpFeedback` produces a requested body torque.
- `ExtForceTorque` realizes that request as ideal torque. It does not represent wheel momentum, wheel limits, thruster quantization, propellant, or power.
- For actuator fidelity, insert allocation and a reaction-wheel or thruster effector instead of connecting ideal torque directly.

## 6. Processes, tasks, priorities, and rates

Rates encode causal and numerical assumptions. A typical arrangement is:

```text
dynamics task       0.1 s: environment, effectors, spacecraft, sensor truth
navigation task     0.5 s: measurement processing and estimator
guidance task       1.0 s: target/reference generation
control task        0.1 s: tracking, feedback, allocation
mission task       10.0 s: mode logic and resource decisions
```

Do not copy those numbers blindly. Choose them from the fastest relevant physical time constant, controller bandwidth, sensor cadence, estimator requirements, command cadence, and integration-error study.

At a coincident scheduler time, larger priorities execute first within the applicable scheduling level. `AddModelToTask(..., ModelPriority=n)` is the most visible control. Equal-priority ordering follows insertion/creation order and is easy to disturb during maintenance.

Three rules prevent many timing defects:

1. Make required same-tick write-before-read relationships explicit with task/model priorities.
2. When rates differ, assume a consumer sees the producer's most recent message—a zero-order-held value—not a freshly interpolated state.
3. Inspect message timestamps and the simulation execution order when debugging a first-sample zero, one-cycle lag, or stale command.

Separate processes can improve organization and support advanced execution configurations, but they are not automatically concurrent operating-system processes. A single process with several tasks is often clearer for a small scenario.

## 7. Messages and configuration data

A message connection has three parts:

```text
producer output Msg  ── stores typed payload ──> consumer MsgReader
```

Typical forms are:

```python
# Dynamic producer-to-consumer connection.
consumer.inputMsg.subscribeTo(producer.outputMsg)

# Constant configuration source.
payload = messaging.SomeMsgPayload(...)
config_msg = messaging.SomeMsg().write(payload)
consumer.configInMsg.subscribeTo(config_msg)
```

Before initialization, verify required readers with `isLinked()` where the API exposes it. A linked message can still contain an unwritten/default payload on the first update if execution order is wrong.

Messages provide typed storage and timestamping; they do not provide:

- dimensional analysis;
- frame checking or frame conversion;
- automatic interpolation between task rates;
- realistic communication delay, bandwidth, packet loss, or network routing.

## 8. State effectors and dynamic effectors

Choose by physical role, not by the class name you remember:

| Concept | Physical meaning | Examples |
|---|---|---|
| `StateEffector` | Adds internal generalized state and exchanges momentum/energy/mass with the hub | Reaction wheels, hinged bodies, slosh particles, tanks, flexible components |
| `DynamicEffector` | Supplies external or commanded force/torque to the spacecraft | Drag, SRP, external force/torque, many thruster models, electrostatic forces |

Attaching an effector to a spacecraft and scheduling its update are separate responsibilities in many patterns:

```python
sc.addDynamicEffector(effector)       # include its force/torque in spacecraft dynamics
sim.AddModelToTask(task_name, effector)  # execute its message/state update
```

The corresponding method for a state effector is commonly `addStateEffector`. Confirm the exact module API because some coupled objects and MuJoCo models use different integration contracts.

## 9. Recording, module logging, and plotting

Use a message recorder for interface data:

```python
rec = module.outputMsg.recorder(macros.sec2nano(1.0))
sim.AddModelToTask(task_name, rec)

# After execution:
t_s = rec.times() * macros.NANO2SEC
values = rec.somePayloadField
```

Use a module variable logger for internal variables that are not published as message payloads. Use `bskLogger` for diagnostic text and severity-controlled runtime reporting. Plotting is ordinary Python post-processing and should compute an engineering metric before decorating a figure.

Recorder placement is part of model causality. If the recorder executes before its producer on the same tick, it can capture the previous value. A down-sampling period limits retained samples but does not change the producer's task rate.

Avoid recording every high-rate message for a long campaign. Estimate sample count and storage first, and retain only states needed for validation and metrics.

## 10. Events and phased execution

Events evaluate conditions on a configured period and execute actions such as:

- enabling or disabling tasks;
- switching a mission-mode request;
- changing a gateway message author;
- applying a deliberately ideal impulse;
- marking a failure or terminating the run.

The repository's BskSim examples show the recurring mode pattern:

```text
event fires
  → disable mutually exclusive FSW tasks
  → zero or redirect gateway messages
  → enable the selected guidance/control chain
```

Event conditions and actions execute as Python orchestration; they are not physical dynamics. Prefer callback-based current APIs where available. Some copied examples still use legacy string expressions, so verify the local 2.11.1 signature before adopting them.

For explicit mission phases, repeated execution is often sufficient:

```python
sim.ConfigureStopTime(macros.min2nano(10.0))
sim.ExecuteSimulation()

# Reconfigure an authorized command or mission-mode input here.
sim.ConfigureStopTime(macros.min2nano(20.0))  # absolute, not +10 minutes
sim.ExecuteSimulation()
```

Do not directly change integrated truth state unless the engineering abstraction is explicitly an ideal state discontinuity, such as an impulsive maneuver study. A propulsion-performance question requires actuator force and mass depletion instead.

## 11. Vizard

For a conventional scenario, Vizard consumes simulation outputs:

```python
from Basilisk.utilities import vizSupport

if vizSupport.vizFound:
    viz = vizSupport.enableUnityVisualization(
        sim,
        "dynamicsTask",
        sc,
        # saveFile=__file__,
        # liveStream=True,
    )
```

Keep these distinctions clear:

- playback writes visualization data for later viewing;
- live streaming adds synchronization and external-application concerns;
- Vizard normally observes truth and does not define it;
- in the specialized OpNav examples, Vizard renders synthetic camera imagery and therefore becomes part of the sensor loop.

Headless numerical execution should remain possible when visualization is optional.

## 12. Common first failures

| Symptom | Likely cause | First check |
|---|---|---|
| Input remains zero | Missing subscription, producer has not written, or consumer executes first | `isLinked()`, producer recorder, timestamps, execution order |
| One-cycle command lag | Recorder/consumer ordering or deliberately slower producer | Task periods and model priorities |
| Physically impossible orbit | Metres/kilometres error, degrees/radians error, wrong central body, or frame mismatch | Units and suffixes before tuning the integrator |
| Attitude points the wrong way | MRP/DCM direction or `B`, `R`, and `N` frame interpretation is reversed | Write the mapping represented by each attitude before connecting modules |
| Controller output looks correct but motion is unchanged | Command message is not connected to an attached/scheduled effector | Trace the chain all the way back to `Spacecraft` |
| Navigation performance looks perfect | FSW is consuming default `SimpleNav` or truth directly | Add the intended sensor errors and estimator, then compare truth and estimate separately |
| Energy or momentum drifts unexpectedly | Step too large, model/integrator mismatch, incorrect effector setup, or unintended force/torque | Run a step-size convergence and conservation test |
| Second execution stops immediately | `ConfigureStopTime()` received a duration smaller than current time | Pass cumulative absolute time |
| Plot arrays appear misaligned | Different task/recorder rates were joined by array index | Align using timestamps |
| Example fails to import locally | Copied example and installed Basilisk version differ | Check the repository's version/provenance notes before changing code |

## 13. What to learn next

After this quick start, work through these transitions in order:

1. Point-mass propagation → perturbation and integration convergence.
2. Ideal torque → reaction-wheel allocation, limits, and momentum.
3. Truth-like `SimpleNav` → sensor model and estimator.
4. One spacecraft → independent chief/deputy truth and explicit relative-frame conversion.
5. Ideal impulse → finite burn, pointing, allocation, and fuel depletion.
6. Deterministic success metric → repeatable Monte Carlo dispersion.
7. Deterministic modes → OpNav or decision-autonomy layers only when the underlying physics is validated.

Use the [module and message glossary](reference/module_and_message_glossary.md) while reading examples. Preserve the distinction between what an example demonstrates and what your engineering question requires.
