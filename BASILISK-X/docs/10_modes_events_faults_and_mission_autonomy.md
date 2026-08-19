> **Draft status — AI-generated:** This page was drafted by OpenAI Codex on 2026-08-19. It has not yet been technically vetted by the BASILISK-X repository owner. The repository owner will review, correct, and maintain it. Verify APIs, units, frames, assumptions, and version compatibility before engineering use.

# Modes, events, faults, and deterministic mission autonomy

This chapter is about Basilisk's **control plane**: the logic that decides which already-configured algorithms run, when a maneuver starts, when a simulation stops, and how the system reacts to a fault or resource constraint. It is not a substitute for the dynamics, navigation, or control algorithms themselves.

The core engineering distinction is:

```text
data plane                                  control plane
----------                                  -------------
truth -> sensors -> nav -> guidance ->      mode request / state machine
tracking -> control -> actuators             events, guards, task activation
        |                                           |
        +------------ messages <-------------------+
```

An event can switch tasks, but it does not make a collection of event conditions into a well-designed flight state machine. Mission autonomy requires explicit states, guarded transitions, priorities, safe command handover, observability, and verification.

## 1. What was observed in this workspace

Unless marked **Recommendation**, statements in this section describe the checked-in examples and the installed Basilisk implementation used by this workspace.

### 1.1 Event lifecycle

`SimBaseClass.createNewEvent()` installs an `EventHandlerClass` with a name, activation flag, checking rule, optional terminal flag, a condition, and an action. The current API prefers Python callables:

```python
def enter_safe(sim):
    sim.fswProc.disableAllTasks()
    sim.FSWModels.zeroGateWayMsgs()
    sim.enableTask("sunSafePointTask")
    sim.enableTask("mrpSteeringRWsTask")
    sim.setAllButCurrentEventActivity("enterSafe", True)

sim.createNewEvent(
    "enterSafe",
    eventRate,
    True,
    conditionFunction=lambda sim: sim.modeRequest == "safe",
    actionFunction=enter_safe,
)
```

The older `conditionList` and `actionList` string interfaces remain visible in historical prose, but the inspected implementation marks them deprecated. Prefer callables: they are type-checkable, refactorable, and can use normal Python objects.

An active event follows this lifecycle:

```text
active
  |
  | check is due, condition false
  +---------------------------------------> active
  |
  | check is due, condition true
  v
deactivate event -> execute action -> increment occurrence count
  |                                      |
  | action reactivates it                | no reactivation
  v                                      v
active again                         dormant one-shot
```

This one-shot default is central to the BskSim mode idiom: the event for the entered mode deactivates itself, while its action reactivates the competing mode events with `setAllButCurrentEventActivity(...)`. A repeated event must explicitly call `setEventActivity(eventName, True)`, as demonstrated by the probabilistic wheel-friction injection in [`scenario_AddRWFault.py`](../examples/BskSim/scenarios/scenario_AddRWFault.py).

### 1.2 Three event timing modes

| Checking mode | Configuration | Observed behavior | Engineering use |
|---|---|---|---|
| Exact interval | default `exactRateMatch=True` | Checks when current simulation time is an exact multiple of `eventRate`; a check can be missed if the scheduler never lands there | Mode polling aligned to an existing task lattice |
| Elapsed interval | `exactRateMatch=False` | Checks after at least `eventRate` has elapsed since the previous check | Monitoring when task rates do not share a convenient exact multiple |
| Scheduled time | `conditionTime=t_event` | Triggers at the first scheduler opportunity at or after `t_event` | Deterministic fault injection or a known mission timeline |

Events are evaluated at the current simulation time before Basilisk advances scheduled tasks. Therefore, a condition that reads a message normally sees the payload produced on the preceding module execution, not a value that a task is about to publish at the same timestamp. This is often correct for sampled flight logic, but it must be included in latency and threshold tests.

`terminal=True` requests simulation termination after a condition triggers. [`scenarioDragDeorbit.py`](../examples/scenarioDragDeorbit.py) uses this to stop when radius falls below a deorbit-altitude threshold. A terminal event is a numerical stopping condition, not automatically a flight-mode transition.

**Recommendation:** Choose the event rate from the required detection latency, not from the dynamics integrator step by habit. Record the time at which the monitored quantity crossed its threshold and the time at which the event acted; verify the quantization delay explicitly.

### 1.3 Task enable/disable semantics

Processes own scheduled tasks; tasks own ordered models. BskSim constructs all candidate FSW tasks up front and disables them before execution. Mode events then enable a coherent subset.

Important consequences:

- Disabling a task stops its modules from executing; it does **not** retract or clear their last output messages.
- Enabling a task does not imply that its stateful modules are reset. `ResetTask(taskName)` exists, but reset-on-entry is a design choice, not an automatic mode-switch property.
- A disabled actuator-command producer can therefore leave a nonzero last command visible to its subscriber unless a gateway or actuator command is deliberately cleared.
- An enabled task resumes on the scheduler's task lattice. Do not assume a newly enabled producer executes instantaneously inside the event action.
- Module priority still controls same-task ordering after a task is enabled.

These semantics explain why the example mode handlers use gateway messages and explicit zeroing.

## 2. Gateway messages and safe command handover

[`BskSim/models/BSK_Fsw.py`](../examples/BskSim/models/BSK_Fsw.py) creates stable C-wrapped gateway messages for attitude references, attitude guidance, body torque, wheel torque, and burn commands. Candidate algorithms author these shared interfaces; downstream consumers subscribe once to the gateway rather than being rewired during a mode change.

```text
inertial guidance ----\
Hill guidance ---------+--> AttRef gateway --> tracking --> AttGuid gateway
sun-safe guidance -----/                              |
                                                      v
direct torque control ----\                    controllers
RW control ---------------+--> command gateways -----> dynamics effectors
burn logic ---------------/
```

This indirection solves two problems:

1. Message subscriptions remain static after configuration.
2. A transition can overwrite the shared output with a known neutral payload before another producer becomes active.

The generic BskSim `zeroGateWayMsgs()` writes empty payloads to the command/reference gateways. The MultiSat implementation also explicitly zeros wheel and thruster command outputs in [`BSK_MultiSatFsw.py`](../examples/MultiSatBskSim/modelsMultiSat/BSK_MultiSatFsw.py).

### Recommended transition transaction

Treat a mode change as a transaction rather than as a collection of unrelated task toggles:

```text
1  validate requested transition and safety guards
2  disable all old command authors
3  write neutral/hold-safe actuator commands to stable gateways
4  reset only the stateful algorithms whose entry semantics require it
5  load/validate the new reference or maneuver command
6  enable the new producer-to-actuator chain in dependency order
7  verify output validity and record mode-entry telemetry
8  re-arm only legal outgoing transitions
```

Do not blindly reset every controller. For example, preserving an estimator covariance or integral state can be correct; preserving stale burn-on-time commands can be dangerous. Define entry, exit, and retained-state semantics per mode.

## 3. The BskSim mode pattern

The generic BskSim FSW model uses a string `modeRequest` as the transition request. Each event tests one value, disables all FSW tasks, zeros gateways, enables the tasks for that mode, and re-arms the other events.

| Local mode example | Enabled behavior | Best source to inspect | Caveat |
|---|---|---|---|
| `inertial3D` | inertial reference + tracking/RW control | [`BSK_Fsw.py`](../examples/BskSim/models/BSK_Fsw.py), [`scenario_AttFeedback.py`](../examples/BskSim/scenarios/scenario_AttFeedback.py) | Mode request is an external Python attribute, not a flight message |
| `hillPoint` | Hill reference + tracking/RW control | [`BSK_Fsw.py`](../examples/BskSim/models/BSK_Fsw.py), [`scenario_AddRWFault.py`](../examples/BskSim/scenarios/scenario_AddRWFault.py) | Assumes the configured nav/reference chain is valid |
| `sunSafePoint` | CSS-based safe pointing + steering/RW control | [`scenario_AttEclipse.py`](../examples/BskSim/scenarios/scenario_AttEclipse.py) | The scenario selects the mode; it is not a complete low-power autonomous safing design |
| `lambertFirstDV`, `lambertSecondDV` | burn guidance plus attitude-control chain | [`scenario_LambertGuidance.py`](../examples/BskSim/scenarios/scenario_LambertGuidance.py) | Demonstrates sequencing, not a full propulsion authorization system |
| formation `spacecraftPointing` | relative pointing and direct torque | [`BSK_FormationFsw.py`](../examples/BskSim/models/BSK_FormationFsw.py), [`scenario_RelativePointingFormation.py`](../examples/BskSim/scenarios/scenario_RelativePointingFormation.py) | Fixed two-spacecraft architecture |
| per-satellite pointing | independent indexed task/event groups | [`BSK_MultiSatFsw.py`](../examples/MultiSatBskSim/modelsMultiSat/BSK_MultiSatFsw.py), [`scenario_AttGuidMultiSat.py`](../examples/MultiSatBskSim/scenariosMultiSat/scenario_AttGuidMultiSat.py) | External scenario still assigns each spacecraft's request |
| OpNav preparation/estimation | camera pointing, image processing, filters and RW control | [`BSK_OpNavFsw.py`](../examples/OpNavScenarios/modelsOpNav/BSK_OpNavFsw.py) | Specialized, large mode matrix coupled to the legacy OpNav stack |

The MultiSat model appends a spacecraft index to event and task names and calls `setAllButCurrentEventActivity(..., useIndex=True)`. This prevents a transition on spacecraft 0 from re-arming or disabling events for every other spacecraft. It also controls station keeping with a separate `stationKeeping` request, enabling or disabling only the reconfiguration task. Its comments warn that standby cannot coexist with an orbital correction burn because standby removes attitude control.

### Known local defect: generic BskSim standby

**Observed behavior in this checkout:** the `initiateStandby` event in [`BskSim/models/BSK_Fsw.py`](../examples/BskSim/models/BSK_Fsw.py) defines its `actionFunction` as a lambda returning a tuple of **string literals**:

```python
actionFunction=lambda self: (
    "self.fswProc.disableAllTasks()",
    "self.FSWModels.zeroGateWayMsgs()",
    "self.setAllButCurrentEventActivity('initiateStandby', True)",
)
```

Those strings are values, not executed calls. The event condition can trigger and deactivate the event, but the intended task disabling, gateway clearing, and event re-arming do not occur. This is not a general Basilisk event limitation; adjacent generic modes use real calls, and the formation, MultiSat, and OpNav standby handlers also use real calls.

**Engineering consequence:** do not use generic BskSim `standby` as evidence of safe command removal in this checkout. This affects examples that request standby, including the final phase of [`scenario_LambertGuidance.py`](../examples/BskSim/scenarios/scenario_LambertGuidance.py). Fix and regression-test this source before relying on that behavior; this chapter intentionally does not modify it.

## 4. External phasing versus onboard-style autonomy

Two valid architectures answer different questions.

### 4.1 External Python campaign conductor

[`scenario_AttModes.py`](../examples/BskSim/scenarios/scenario_AttModes.py) alternates mode strings in a Python loop and advances absolute stop time in ten-minute segments. [`scenarioHohmann.py`](../examples/scenarioHohmann.py) similarly uses mode events to select burn-pointing tasks while the surrounding scenario conducts the mission. BASILISK-X's [`cooperative_geo_rendezvous.py`](../scenarios/cooperative_geo_rendezvous/cooperative_geo_rendezvous.py) goes further: `execute_rendezvous_phases()` advances named phases and `apply_relative_velocity_impulse()` directly modifies the propagated state for idealized impulses.

Use this style for:

- deterministic trade studies and test-vector generation;
- ideal maneuver design before actuator details matter;
- test harnesses that deliberately place the simulation in a condition;
- campaign-level sequencing that is not claimed to represent onboard software.

Its main risk is semantic inflation: a Python `while` loop, direct truth access, or direct state mutation is a **scenario conductor**, not autonomous flight logic.

### 4.2 Onboard-style mission logic

Onboard-style logic consumes only available FSW messages, maintains its own state, publishes requested mode/command messages, and executes at a declared task rate. It does not read arbitrary truth attributes or call `ExecuteSimulation()` itself.

```text
estimated state + resource status + command validity + elapsed time
                              |
                              v
                   mission-manager module
                 [state, guards, timers, faults]
                              |
                 mode request / command message
                              |
                              v
                   event or mode dispatcher
                              |
                    enabled FSW task graph
```

The MuJoCo FSW example gives the clearest local event-driven state sequence: [`BSK_MujocoFSW.py`](../examples/mujoco/mujocoModels/BSK_MujocoFSW.py) transitions through joint motion, thruster firing, and coast using command-write timestamps, joint/rate tolerances, measured thruster forces, and deliberate event re-arming. It remains an example-specific controller, but its conditions and actions are much closer to explicit state-machine guards than a time-phased scenario loop.

### Decision rule

| Question | Appropriate owner |
|---|---|
| “What happens if the burn occurs at this prescribed time?” | External scenario conductor |
| “Can the controller recover from this injected state?” | External test harness can establish the state; FSW performs recovery |
| “Should the spacecraft burn now given estimated geometry and resources?” | Onboard-style mission-manager module |
| “Which of 10,000 mission cases should execute next?” | External campaign/Monte Carlo layer |
| “Can the same autonomy code make the decision without truth access?” | Onboard-style module with message-only inputs |

## 5. Designing a mission state machine

**Recommendation:** define the state machine independently of Basilisk APIs before wiring tasks.

For every state, specify:

| Contract element | Example questions |
|---|---|
| Entry guard | Is navigation valid? Is attitude within the burn cone? Is battery state above reserve? |
| Entry action | Which references are loaded? Which estimator/controller is reset? Which command is zeroed first? |
| Active tasks and rates | Which sensor, estimator, guidance, control, and actuator modules run? |
| Invariants | Must keep-out angle remain positive? Must wheel momentum remain below a limit? |
| Exit guards | Target achieved, timeout, command revoked, resource low, or fault detected? |
| Exit action | Disarm thrusters, latch result, clear gateway, preserve estimator state? |
| Failure transition | Which safe or degraded mode owns control, and with what priority? |
| Telemetry | Mode, transition reason, guard values, entry time, command sequence number |

A practical transition model is:

```text
             nav valid & resources good
   STANDBY ------------------------------> ACQUIRE
      ^                                      |
      | timeout/fault                        | geometry & attitude settled
      |                                      v
     SAFE <--------- constraint ---------- OPERATE
      ^                                      |
      +------- low power / stale nav --------+
```

Use hysteresis, dwell time, or consecutive-valid-sample counters around noisy thresholds. Without them, sampled conditions can chatter between modes. Give safety transitions higher priority than mission-progress transitions, and make simultaneous-guard resolution deterministic.

### Resource and safety gating

The repository contains the physical building blocks—eclipse-aware attitude behavior in [`scenario_AttEclipse.py`](../examples/BskSim/scenarios/scenario_AttEclipse.py), power nodes/storage in [`scenarioPowerDemo.py`](../examples/scenarioPowerDemo.py), ground access in [`scenarioGroundLocationImaging.py`](../examples/scenarioGroundLocationImaging.py), and attitude constraints in [`scenarioAttitudeConstraintViolation.py`](../examples/scenarioAttitudeConstraintViolation.py). It does not combine them into a generic production mission manager.

**Recommendation:** gate a maneuver on FSW-visible messages, not truth-only checks:

```text
authorize_operation =
    command_fresh
    and nav_valid
    and attitude_error < limit
    and keep_out_margin > margin
    and battery_energy > reserve
    and thermal_state_ok
    and actuator_available
```

Record every individual predicate. A single Boolean “authorized” flag is insufficient for explaining aborts or estimating mission success probability.

## 6. Fault injection is not FDIR

| Layer | Purpose | Local evidence |
|---|---|---|
| Fault injection | Change truth, sensor, actuator, or FSW parameters at a controlled time | [`scenario_AddRWFault.py`](../examples/BskSim/scenarios/scenario_AddRWFault.py), [`scenario_FaultList.py`](../examples/BskSim/scenarios/scenario_FaultList.py), [`BSK_Faults.py`](../examples/BskSim/models/BSK_Faults.py) |
| Detection | Infer from observable residuals/status that behavior is abnormal | OpNav-specific comparison chain in [`scenario_faultDetOpNav.py`](../examples/OpNavScenarios/scenariosOpNav/scenario_faultDetOpNav.py) |
| Isolation | Decide which component/failure mode is responsible | Only special-purpose demonstrations locally; no generic flight FDIR framework established |
| Recovery | Reconfigure tasks, algorithms, effectors, or mission plan | Can be built with mode/gateway patterns; not supplied as a general reusable policy |

`BSK_Faults.py` packages timed mutations as `FaultObject` subclasses. The local list includes reaction-wheel power/friction changes, debris impact on/off, CSS signal faults, encoder/bit-flip-like faults, and magnetometer faults/noise. `addFaultToSimulation()` creates a scheduled event with `conditionTime`, invokes the mutation once, prints status, and disables the event.

This architecture is useful because the fault campaign is separate from nominal mode logic. However, many injected faults directly mutate model or algorithm parameters. That is appropriate for a truth-side test harness, but the recovery logic must not be given the same privileged access.

### Fault-study discipline

1. Define the physical failure and its onset, duration, intermittency, and affected channel.
2. Decide whether it belongs in truth dynamics, a sensor model, a data link, an actuator, or FSW memory.
3. Inject it outside the algorithm under test.
4. Limit detector inputs to observable flight data.
5. Define detection latency, false-alarm rate, isolation accuracy, recovery time, and residual performance.
6. Include undetected, misisolated, and recovery-failed outcomes in Monte Carlo success logic.

## 7. Verification checklist for deterministic autonomy

- [ ] Every legal mode has an explicit task set; every illegal mode request goes to a defined safe response.
- [ ] At most one producer owns each actuator gateway in a mode.
- [ ] Mode exit clears persistent actuator commands before disabling their producers.
- [ ] Stateful modules have documented preserve/reset behavior.
- [ ] Event check rate and worst-case transition latency are verified.
- [ ] Guards use navigation/resource messages available to flight logic, not truth shortcuts.
- [ ] Thresholds have units, frames, hysteresis, dwell time, and boundary tests.
- [ ] Simultaneous guards have a deterministic priority order.
- [ ] Mode, transition reason, event occurrence, command validity, and gateway outputs are recorded.
- [ ] Timeouts and stale-message detection exist for operations that can wait indefinitely.
- [ ] Fault injection is independent of detection and recovery logic.
- [ ] Safe mode has been tested from every operational state, including during nonzero actuator commands.
- [ ] Multi-spacecraft event names are grouped per spacecraft; one vehicle cannot accidentally re-arm all event groups.
- [ ] The generic BskSim standby defect described above is fixed and regression-tested before use.

## 8. Recommended progression for BASILISK-X

1. Keep the current GEO rendezvous phase conductor as an explicitly ideal, deterministic mission-analysis layer.
2. Add structured phase/mode telemetry and guard-value logging before adding more modes.
3. Replace direct maneuver state changes with commanded finite-burn dynamics only when actuator performance is part of the engineering question.
4. Introduce a small message-driven mission-manager module for one bounded behavior—such as `WAIT -> ACQUIRE -> HOLD -> SAFE`—and test it with synthetic nav/resource messages.
5. Add resource and keep-out gates, timeouts, stale-data handling, and safe command handover.
6. Add fault injection and detector/recovery tests as separate components.
7. Only then place the deterministic autonomy inside the Monte Carlo campaign described in [Monte Carlo, uncertainty, and statistics](11_monte_carlo_uncertainty_and_statistics.md).

The important design principle is that events are a scheduler mechanism, gateways are command-interface infrastructure, and a mission state machine is an engineering specification. They complement one another, but they are not interchangeable.
