> **Draft status — AI-generated:** This page was drafted by OpenAI Codex on 2026-08-19. It has not yet been technically vetted by the BASILISK-X repository owner. The repository owner will review, correct, and maintain it. Verify APIs, units, frames, assumptions, and version compatibility before engineering use.

# Architecture, execution, and lifecycle

This chapter explains the software structure that turns a set of physical models and flight-software algorithms into a deterministic Basilisk simulation. It is deliberately about the mental model first and API calls second.

## Evidence and recommendations

Statements labelled **Observed in this repository** describe behavior found in the checked-in examples or the locally installed Basilisk 2.11.1 API. Statements labelled **Engineering recommendation** are BASILISK-X guidance inferred from those examples; they are not claims that Basilisk requires that architecture.

## Why Basilisk has a scheduler and messages

A spacecraft simulation is not one differential equation. It is a set of components with different physical and computational responsibilities:

- continuous spacecraft and environmental dynamics;
- sampled sensors and navigation algorithms;
- guidance and control laws;
- actuator command and physical actuator models;
- mission logic, telemetry, analysis, and visualization.

Those components often run at different rates. They also need to be replaceable: an ideal torque should be replaceable by reaction wheels without rewriting guidance, and perfect navigation should be replaceable by a filter without rewriting the spacecraft. Basilisk therefore separates **when something executes** from **how data moves between components**.

```text
Python scenario: construct, configure, connect, run, analyse
                              |
                              v
                   SimulationBaseClass.SimBaseClass
                              |
               +--------------+--------------+
               |                             |
          Process A                      Process B
          priority 200                   priority 100
               |                             |
       +-------+-------+               +-----+-----+
       |               |               |           |
   dynamics task   sensor task     guidance task  control task
      0.1 s           1.0 s            1.0 s        0.2 s
       |                                   |
   ordered models                       ordered models

Typed messages connect models without requiring them to share Python objects.
```

## The execution hierarchy

### `SimulationBaseClass.SimBaseClass`

`SimulationBaseClass.SimBaseClass` owns the simulation kernel, process/task registration, event handling, initialization state, stop time, progress reporting, and the common Basilisk logger.

```python
from Basilisk.utilities import SimulationBaseClass, macros

simulation = SimulationBaseClass.SimBaseClass()
process = simulation.CreateNewProcess("dynamicsProcess", 100)
task = simulation.CreateNewTask("dynamicsTask", macros.sec2nano(0.1))
process.addTask(task, 100)
```

This is the minimal scheduling shell. It does not yet contain a spacecraft or any other model.

### Processes

A process groups tasks and provides a priority boundary. A Basilisk process is a scheduler concept; creating one does **not** by itself create an operating-system process or guarantee concurrent execution.

```python
dynamics_process = simulation.CreateNewProcess("DynamicsProcess", 200)
fsw_process = simulation.CreateNewProcess("FSWProcess", 100)
```

**Observed in this repository:** most standalone examples use one process and one task. [`BskSim/BSK_masters.py`](../examples/BskSim/BSK_masters.py) creates separate dynamics and FSW processes. [`MultiSatBskSim/BSK_MultiSatMasters.py`](../examples/MultiSatBskSim/BSK_MultiSatMasters.py) creates a high-priority shared-environment process, per-spacecraft dynamics processes, an optional formation-barycenter process, and lower-priority per-spacecraft FSW processes.

**Engineering recommendation:** add processes because they express a useful scheduling or ownership boundary, not simply to mirror a block diagram. A one-off orbit or attitude study is usually clearer with one process.

### Tasks

A task is a fixed-rate list of models. The task period is an integer number of nanoseconds.

```python
dynamics_process.addTask(
    simulation.CreateNewTask(
        "dynamicsTask",
        macros.sec2nano(0.1),
        FirstStart=0,
    ),
    100,
)
```

`FirstStart` delays the task's first scheduled execution. The older `InputDelay` argument exists in the local API but is deprecated and non-functional.

Use separate tasks when models genuinely require different update rates or need to be enabled and disabled as a group. Mission modes in BskSim are implemented largely as selectable tasks.

### Models/modules

A model is a scheduled unit of behavior: a spacecraft, environment model, sensor, FSW algorithm, recorder, Vizard interface, or custom `SysModel`.

```python
simulation.AddModelToTask("dynamicsTask", spacecraft_object, 100)
simulation.AddModelToTask("dynamicsTask", navigation, 90)
```

The model priority is the last argument in this compact call form. The local method also supports its legacy `ModelData` argument for C modules, so check the installed signature before using less common positional forms.

[`scenarioAttitudePointingPy.py`](../examples/scenarioAttitudePointingPy.py) is the local template for a custom Python module. It subclasses `Basilisk.architecture.sysModel.SysModel`, calls `super().__init__()`, declares typed message endpoints, and implements `Reset(CurrentSimNanos)` and `UpdateState(CurrentSimNanos)`.

## Priority and execution order

When processes, tasks, or models are eligible at the same simulation time, **higher numeric priority executes first**. Priority exists to make read-after-write relationships deterministic.

```text
same scheduler time t

process priority:  dynamics 200  -->  FSW 100
task priority:     guidance 20   -->  control 10
model priority:    guidance 10   -->  tracking 9 --> controller 8
```

Models with equal/default priority should not be treated as an implicit engineering contract. The examples commonly rely on insertion order, but an explicit priority makes an important dependency visible and robust against later insertions.

**Observed in this repository:** [`BskSim/models/BSK_Fsw.py`](../examples/BskSim/models/BSK_Fsw.py) explicitly orders guidance before tracking and steering before rate servo and wheel allocation. [`BskSim/models/BSK_Dynamics.py`](../examples/BskSim/models/BSK_Dynamics.py) assigns explicit priorities to effectors, ephemerides, sensors, and navigation. BASILISK-X [`nadir_pointing.py`](../scenarios/nadir_pointing/nadir_pointing.py) wires the correct chain but mostly relies on addition order.

Use the scheduler itself to inspect the resolved arrangement:

```python
simulation.ShowExecutionOrder()
figure = simulation.ShowExecutionFigure(show_plots=False)
```

### Multi-rate semantics

Messages hold the most recently written payload. A faster consumer therefore reuses the latest slower producer output until a new value is published.

```text
time [s]               0.0   0.1   0.2   0.3   0.4   0.5 ... 1.0
dynamics, 0.1 s         X     X     X     X     X     X       X
control, 0.2 s           X           X           X             X
navigation, 1.0 s        X                                   X

Between navigation updates, control reads a held navigation sample.
At a coincident tick, priority decides whether it reads the new or previous sample.
```

This is deterministic sampled-data behavior, not an error. It becomes an error if the rates and latency are accidental or untested.

**Engineering recommendation:** for every important message path, record the producer period, consumer period, coincident-tick priority, and tolerable data age. Do not choose the dynamics integration step solely from the desired plot spacing.

## The construction and run lifecycle

The normal lifecycle is:

```text
1. Construct scheduler, processes, and tasks
2. Construct and configure models
3. Attach state/dynamic effectors to their spacecraft
4. Connect typed messages
5. Add recorders, loggers, and optional Vizard interface
6. InitializeSimulation()
7. ConfigureStopTime(absolute_time_ns)
8. ExecuteSimulation()
9. Read retained data and compute engineering metrics
```

A compact pattern is:

```python
simulation.InitializeSimulation()
simulation.ConfigureStopTime(macros.min2nano(10.0))
simulation.ExecuteSimulation()
```

`InitializeSimulation()` runs the kernel initialization sequence, including module self-initialization and reset. Configure and wire the simulation before calling it unless a specific module documents runtime reconfiguration.

`ConfigureStopTime()` takes an **absolute** simulation time, not a duration relative to the current time. This matters in segmented runs:

```python
simulation.ConfigureStopTime(macros.min2nano(5.0))
simulation.ExecuteSimulation()

# Continue from 5 to 8 minutes; do not pass only 3 minutes here.
simulation.ConfigureStopTime(macros.min2nano(8.0))
simulation.ExecuteSimulation()
```

The default stop condition runs only scheduled work that does not exceed the requested time. The local 2.11.1 API also accepts `StopCondition=">="` for the first scheduled time at or beyond the target; verify this API when changing Basilisk versions.

## Spacecraft truth and effectors

`spacecraft.Spacecraft` is a dynamic object containing the hub translational and rotational state. It integrates its state together with attached effectors and publishes an `SCStatesMsg` truth output.

```text
Spacecraft hub
  position, velocity, attitude, angular velocity
          |
          +-- StateEffectors
          |     coupled internal generalized states and mass properties
          |     examples: reaction wheels, hinges, flexible bodies, fuel/slosh
          |
          +-- DynamicEffectors
                externally applied forces and torques
                examples: drag, SRP, electrostatics, ideal external torque
```

A `StateEffector` participates in the spacecraft's coupled state, mass properties, momentum, and energy. A `DynamicEffector` primarily computes force/torque contributions from states or messages. The distinction is a dynamics contract, not merely a naming convention.

Typical attachment calls are:

```python
spacecraft_object.addStateEffector(reaction_wheel_state_effector)
spacecraft_object.addDynamicEffector(external_force_torque)
```

Many effectors must also be added to a task so their scheduled message handling or outputs run. Follow the module-specific example rather than assuming attachment alone is sufficient.

**Observed in this repository:** [`scenarioHingedRigidBody.py`](../examples/scenarioHingedRigidBody.py) attaches hinged rigid-body state effectors and an external dynamic effector. [`scenarioAttitudeFeedbackRW.py`](../examples/scenarioAttitudeFeedbackRW.py) uses a reaction-wheel state effector. [`scenarioDragDeorbit.py`](../examples/scenarioDragDeorbit.py) uses a drag dynamic effector.

## Events, modes, and gateway messages

Events provide Python-level mission logic evaluated during execution. An event has a condition, an action, an evaluation schedule, and optional terminal behavior.

```python
def condition(sim):
    return sim.TotalSim.CurrentNanos >= burn_start_ns

def action(sim):
    sim.enableTask("finiteBurnTask")

simulation.createNewEvent(
    "startBurn",
    macros.sec2nano(0.1),
    True,
    conditionFunction=condition,
    actionFunction=action,
)
```

When an event fires it deactivates unless its action reactivates it. The current API supports exact-rate, elapsed-interval, and condition-time checking. A condition cannot be detected more precisely than the scheduler times at which it is evaluated.

**Observed in this repository:** [`BskSim/models/BSK_Dynamics.py`](../examples/BskSim/models/BSK_Dynamics.py) uses function-based events for reaction-wheel faults. BskSim FSW is designed to use events to disable tasks, clear command gateways, and enable a selected mode. The generic `initiateStandby` callback in this checkout is a known exception: it returns strings instead of executing those actions; see [Modes, events, faults, and deterministic mission autonomy](10_modes_events_faults_and_mission_autonomy.md#known-local-defect-generic-bsksim-standby). Some older examples still use string-based conditions/actions; the local API marks those forms deprecated.

A gateway is a stable message endpoint with several possible authors:

```text
inertial guidance -----\
Hill guidance ----------> AttRef gateway --> tracking --> controller
velocity guidance -----/

Only the active task should author the gateway.
Mode transition actions should clear stale gateway payloads.
```

[`BskSim/models/BSK_Fsw.py`](../examples/BskSim/models/BSK_Fsw.py) creates `AttRefMsg_C` and `CmdTorqueBodyMsg_C` gateways, adds several algorithms as authors, and connects fixed downstream readers once. Its working mode callbacks zero gateways during transitions; the defective generic standby callback noted above does not.

**Engineering recommendation:** use a gateway when alternatives share one interface and are mutually exclusive. Do not use one to conceal ambiguous multiple-writer behavior. Make mode exclusivity, transition timing, and safe default commands explicit.

## Recording, logging, plotting, and visualization

These are separate mechanisms:

| Mechanism | Purpose | Scheduled? |
|---|---|---|
| Message recorder | Retain selected payload fields and timestamps | Yes; it is added as a model |
| Variable logger | Retain selected internal module variables | Yes; it is added as a model |
| `bskLogger` | Diagnostic/status/error text from modules | Emitted by module code |
| Plotting/analysis | Compute metrics and render figures after execution | Normally no |
| Vizard | Observe, stream, or replay visualization messages | Interface is scheduled |

```python
state_recorder = spacecraft_object.scStateOutMsg.recorder(
    macros.sec2nano(1.0)
)
simulation.AddModelToTask("dynamicsTask", state_recorder, 0)
```

Recorder priority matters. A recorder that executes before its producer at the same tick captures the previously held payload. A recorder's sampling interval limits storage; it does not change the producer's update rate.

Normal Vizard use is a one-way observer:

```text
spacecraft/environment messages --> vizInterface --> live Vizard or playback file
```

The OpNav architecture is different because rendering is in the sensor loop:

```text
spacecraft truth --> Vizard scene/render --> image message
       ^                                      |
       |                                      v
actuation <-- control <-- navigation <-- image processing/camera
```

[`OpNavScenarios/modelsOpNav/BSK_OpNavDynamics.py`](../examples/OpNavScenarios/modelsOpNav/BSK_OpNavDynamics.py) subscribes camera modules to `vizInterface.opnavImageOutMsgs`. Vizard availability, rendering cadence, port configuration, and image latency are therefore simulation dependencies, not presentation details.

## A practical architecture review checklist

Before trusting a run, answer these questions:

1. Which modules own truth states, sensor outputs, estimates, references, commands, and actuator states?
2. Which task executes each module, at what period and first-start time?
3. At coincident ticks, does each consumer run after the intended producer?
4. Which messages are intentionally held between updates, and what is their maximum age?
5. Have all effectors been both attached and scheduled as required?
6. Does every event have a known evaluation resolution and post-fire state?
7. Can two active tasks write the same gateway?
8. Do recorders sample after their producers and at a rate appropriate to the metric?
9. Is Vizard merely observing, or is it participating in sensor generation?
10. Are stop times absolute during segmented execution?

## Repository examples to read next

- Minimal scheduler and orbit: [`scenarioBasicOrbit.py`](../examples/scenarioBasicOrbit.py)
- Complete attitude command chain: [`scenarioAttitudeFeedbackRW.py`](../examples/scenarioAttitudeFeedbackRW.py)
- Custom Python scheduled module: [`scenarioAttitudePointingPy.py`](../examples/scenarioAttitudePointingPy.py)
- Reusable dynamics/FSW split: [`BskSim/BSK_masters.py`](../examples/BskSim/BSK_masters.py)
- Task modes and gateways: [`BskSim/models/BSK_Fsw.py`](../examples/BskSim/models/BSK_Fsw.py)
- Per-spacecraft scheduling: [`MultiSatBskSim/BSK_MultiSatMasters.py`](../examples/MultiSatBskSim/BSK_MultiSatMasters.py)
- Renderer inside a sensor loop: [`OpNavScenarios/modelsOpNav/BSK_OpNavDynamics.py`](../examples/OpNavScenarios/modelsOpNav/BSK_OpNavDynamics.py)
