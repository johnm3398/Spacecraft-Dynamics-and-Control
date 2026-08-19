> **Draft status — AI-generated:** This page was drafted by OpenAI Codex on 2026-08-19. It has not yet been technically vetted by the BASILISK-X repository owner. The repository owner will review, correct, and maintain it. Verify APIs, units, frames, assumptions, and version compatibility before engineering use.

# Messages, time, ordering, frames, and units

Most subtle Basilisk mistakes are not failures of orbital mechanics. They are interface mistakes: a correctly computed quantity was produced at the wrong time, consumed in the wrong frame, interpreted in the wrong units, or confused with a truth value. This chapter establishes a disciplined way to reason about those interfaces.

## Evidence and recommendations

**Observed in this repository** refers to code or behavior present in the checked-in examples and the locally installed Basilisk 2.11.1 API. **Engineering recommendation** identifies a BASILISK-X practice inferred from that evidence. Always treat the defining module documentation and payload declaration for the installed Basilisk version as authoritative.

## A message is a typed interface contract

A Basilisk message normally has three related objects:

```text
payload type                   output message                 input reader
AttRefMsgPayload       <-->    AttRefMsg             <-->    AttRefMsgReader
fields and semantics          producer-owned                 consumer-owned
```

The output message owns a latest-value payload. A reader subscribes to a compatible output. The payload type enforces structure, but it cannot prove that the producer and consumer agree about physical meaning, frame, epoch, or units.

The subscription direction is always easiest to read as a sentence:

```python
consumer.input_message.subscribeTo(producer.output_message)
```

For example, the local nadir scenario connects truth to navigation, navigation to guidance, and control to an ideal torque effector:

```python
navigation.scStateInMsg.subscribeTo(spacecraft_object.scStateOutMsg)
guidance.transNavInMsg.subscribeTo(navigation.transOutMsg)
tracking.attRefInMsg.subscribeTo(guidance.attRefOutMsg)
tracking.attNavInMsg.subscribeTo(navigation.attOutMsg)
controller.guidInMsg.subscribeTo(tracking.attGuidOutMsg)
torque_effector.cmdTorqueInMsg.subscribeTo(controller.cmdTorqueOutMsg)
```

See [`scenarios/nadir_pointing/nadir_pointing.py`](../scenarios/nadir_pointing/nadir_pointing.py).

### Constant/configuration messages

Configuration data can be written once from Python and then subscribed like any other output:

```python
vehicle_payload = messaging.VehicleConfigMsgPayload(
    ISCPntB_B=inertia_flat
)
vehicle_message = messaging.VehicleConfigMsg().write(vehicle_payload)
controller.vehConfigInMsg.subscribeTo(vehicle_message)
```

Keep a Python reference to the stand-alone message for as long as the simulation needs it. A configuration message is still an interface contract: the consumer determines expected ordering, frame, and units for every field.

### Writing from a custom Python module

[`scenarioAttitudePointingPy.py`](../examples/scenarioAttitudePointingPy.py) demonstrates the normal pattern:

```python
self.guidInMsg = messaging.AttGuidMsgReader()
self.cmdTorqueOutMsg = messaging.CmdTorqueBodyMsg()

def UpdateState(self, CurrentSimNanos):
    guidance = self.guidInMsg()
    output = messaging.CmdTorqueBodyMsgPayload()
    output.torqueRequestBody = compute_torque(guidance)
    self.cmdTorqueOutMsg.write(output, CurrentSimNanos, self.moduleID)
```

Writing the current scheduler time and module ID preserves source and timestamp information. A custom algorithm should validate required inputs in `Reset()` where the module API supports `isLinked()` checks.

## Messages retain the latest sample

A message is not a continuously evaluated function and not automatically a queue. A reader gets the latest payload written by its subscribed source. This creates zero-order-hold behavior across task rates.

```text
producer period = 1.0 s
consumer period = 0.2 s

t [s]          0.0   0.2   0.4   0.6   0.8   1.0
producer       P0                            P1
consumer sees  P0    P0    P0    P0    P0    P1*

* only if producer executes before consumer at the coincident 1.0 s tick
```

This distinction matters for:

- camera images used by faster image-processing or control tasks;
- navigation estimates consumed by a faster controller;
- thruster commands held until the next allocator update;
- environment data generated more slowly than spacecraft dynamics;
- interspacecraft data treated as instantaneous in simplified examples.

**Engineering recommendation:** document an interface using at least this tuple:

```text
(payload type, physical meaning, source frame, units,
 producer rate, consumer rate, timestamp semantics, allowed age)
```

## Ordering and timestamp discipline

At a scheduler time shared by several processes/tasks/models, higher numeric priority executes first. A read-before-write ordering produces a deterministic one-sample delay.

```text
Desired same-tick flow

sensor (priority 30) --> estimator (20) --> controller (10) --> recorder (0)

If recorder priority is 40, it captures the previous held value.
If controller priority is 25, it may read the previous estimator output.
```

Do not assume equal array indices imply simultaneous physical data. Compare recorder timestamps and, where necessary, the source message's write time.

Useful reader/message diagnostics in the local API include `isLinked()`, `isWritten()`, and `timeWritten()`. Their exact availability differs between output objects and reader wrappers, so verify on the concrete message type.

### Recorder semantics

A recorder is a scheduled subscriber:

```python
recorder = producer.outputMsg.recorder(macros.sec2nano(1.0))
simulation.AddModelToTask("dynamicsTask", recorder, 0)
```

The optional recorder interval is a minimum retention spacing, not a producer update period and not a numerical integrator step. Recorder times are exposed in nanoseconds:

```python
time_seconds = recorder.times() * macros.NANO2SEC
```

[`scenarioBasicOrbit.py`](../examples/scenarioBasicOrbit.py) warns that a separately scheduled recorder can retain an older payload if its timing is not coordinated with the producer. [`scenarioAttitudeFeedback.py`](../examples/scenarioAttitudeFeedback.py) additionally warns that an input-reader recorder must be created only after that reader has subscribed to a source. Prefer recording the producer output unless the engineering question specifically concerns what a consumer received.

**Engineering recommendation:** put a recorder after its producer in the same task when exact same-tick sampling matters. When the recorder runs in another task, treat its samples as independently scheduled telemetry and align by timestamp.

### Loggers are different

A message recorder captures a public message contract. A variable logger captures selected internal module variables. `bskLogger` emits diagnostic text. Use the least invasive tool that answers the question:

- output performance or interface verification: message recorder;
- internal algorithm investigation: variable logger;
- setup errors and runtime diagnostics: `bskLogger`;
- engineering conclusions: post-run derived metrics with units.

## Gateway messages and multiple authors

BskSim uses gateway messages so several mission modes can share a downstream interface:

```text
mode A AttRefOutMsg --\
mode B AttRefOutMsg ----> AttRef gateway --> tracking error
mode C AttRefOutMsg --/
```

[`BskSim/models/BSK_Fsw.py`](../examples/BskSim/models/BSK_Fsw.py) creates C-wrapped `AttRefMsg_C` and `CmdTorqueBodyMsg_C` gateways, registers algorithm outputs as authors, and connects consumers to the gateway once. Its intended mode-transition pattern disables competing tasks and writes zero payloads. The generic `initiateStandby` callback in this checkout fails to execute that pattern because it returns strings rather than calling the actions; see [the dedicated defect note](10_modes_events_faults_and_mission_autonomy.md#known-local-defect-generic-bsksim-standby).

A gateway does not arbitrate contradictory active authors on engineering intent. Task enablement and scheduling must ensure one intended writer. A stale valid command can be more dangerous than a missing command, so safe mode transitions should explicitly establish the desired payload.

## Truth is not navigation

Keep this distinction visible even when a tutorial uses perfect information:

```text
physical truth                 sensing/navigation                 FSW

Spacecraft.SCStatesMsg ---> sensor or SimpleNav ---> NavTransMsg/NavAttMsg
        |                            |                       |
    integrated state          emulation/estimate       guidance and control
```

`spacecraft.Spacecraft.scStateOutMsg` is a truth-state interface. `simpleNav.SimpleNav` consumes that truth and publishes navigation-format messages. Without configured errors, `SimpleNav` can behave almost like a truth adapter; it is still an architectural boundary, not proof that a navigation estimator has been modelled.

This affects conclusions:

- A controller closed around default `SimpleNav` demonstrates control under near-perfect state knowledge.
- A relative-position calculation using two spacecraft truth outputs does not demonstrate relative navigation.
- Direct truth access in mission logic bypasses sensing, estimation, communication, and latency.

**Engineering recommendation:** name analysis variables with their epistemic role: `truth_position`, `measured_bearing`, `estimated_state`, and `commanded_delta_v`. Do not use the unqualified word `state` at an interface boundary.

## Reading Basilisk frame notation

Basilisk names often encode both geometric meaning and expression frame. For the common notation

```text
r_BN_N
```

read from left to right:

- `B`: point or frame whose position is being described;
- `N`: point/frame relative to which it is described;
- final `_N`: coordinates are expressed in frame `N`.

Common local names include:

| Symbol/name | Usual interpretation in these examples |
|---|---|
| `r_BN_N` | position of spacecraft/body point B relative to inertial origin N, expressed in N |
| `v_BN_N` | corresponding inertial-frame velocity components |
| `sigma_BN` | MRP attitude of body frame B relative to inertial frame N; associated DCM maps N-frame components into B-frame components |
| `omega_BN_B` | angular velocity of B relative to N, expressed in B |
| `sigma_RN` | reference frame R relative to N |
| `sigma_BR` | tracking attitude error of body B relative to reference R |
| `torqueRequestBody` | requested torque components in body coordinates |

This table is a reading aid, not a substitute for each payload/module contract. Some interfaces use camera, sensor, planet-fixed, or component frames and have their own labels.

### Hill/LVLH relative coordinates

The usual chief-centred Hill triad in the formation examples is constructed from chief inertial position and velocity:

```text
H1 = r_chief / |r_chief|                radial outward
H3 = (r_chief x v_chief) / |r x v|      orbit-normal
H2 = H3 x H1                            along-track for a prograde orbit
```

The deputy relative inertial vector must be rotated into this chief-defined frame. `orbitalMotion.rv2hill()` and `hill2rv()` encapsulate the local conversion used by formation/RPO examples. Check argument order: swapping chief and deputy changes both the origin and the frame.

[`scenarioFormationBasic.py`](../examples/scenarioFormationBasic.py), [`scenarioFormationReconfig.py`](../examples/scenarioFormationReconfig.py), and [`scenarioRendezVous.py`](../examples/scenarioRendezVous.py) are useful comparisons. They do not all represent the same navigation or actuation fidelity.

**Engineering recommendation:** every stored relative vector should encode both origin and expression frame, for example `r_deputy_chief_H_m`. Avoid a generic variable such as `relative_position`.

## Attitude representation hazards

The examples commonly use modified Rodrigues parameters (MRPs), DCMs, and occasionally quaternions in supporting utilities. These are not interchangeable arrays.

Before connecting or converting an attitude quantity, establish:

1. Which frame is oriented relative to which other frame?
2. Does the matrix map source-frame components to destination-frame components, or the inverse?
3. What quaternion element order and sign convention does the function use?
4. Is the angular velocity expressed in body, reference, or inertial components?
5. Is an MRP shadow-set switch possible or already managed by the module?

Never infer a DCM direction from the numeric shape. Test a nontrivial known rotation and its inverse. For an interface adapter, add round-trip and vector-rotation unit tests rather than testing only the identity attitude.

## Units

The repository generally follows SI internally:

| Quantity | Common Basilisk unit |
|---|---|
| position, semimajor axis, radius | m |
| velocity | m/s |
| mass | kg |
| inertia | kg m² |
| force | N |
| torque | N m |
| gravitational parameter | m³/s² |
| angles | rad |
| angular rate | rad/s |
| task/event/simulation time | integer ns |

Plot labels may convert to km, degrees, minutes, or hours. Keep those conversions at analysis/presentation boundaries.

```python
task_period_ns = macros.sec2nano(0.1)
duration_ns = macros.min2nano(10.0)
time_s = recorder.times() * macros.NANO2SEC
position_km = recorder.r_BN_N * 1.0e-3
```

Important exceptions and boundaries include date strings passed to SPICE, TLE formats, Vizard display scaling, image pixel coordinates, and model-specific empirical coefficients. Inspect the module documentation rather than assuming a unit from the variable's magnitude.

**Engineering recommendation:** BASILISK-X scenario configuration names should carry units (`altitude_m`, `duration_s`, `gain_n_m`, `omega_rad_s`) and conversions should happen once near the external boundary.

## Time and epoch are related but different

The scheduler uses elapsed simulation time in integer nanoseconds. A message write carries this elapsed timestamp. Calendar epoch information used for ephemerides, magnetic models, or Earth orientation is configured separately.

```text
simulation clock: 0 ns, 100000000 ns, ...
        +
configured epoch: UTC/calendar/ephemeris input
        =
physical time interpretation for epoch-dependent models
```

Do not assume that `t = 0` means J2000 or the start date displayed by Vizard. Verify which module owns the epoch message and which modules subscribe to it.

During segmented execution, `ConfigureStopTime()` remains absolute. During Monte Carlo, archive both the initial epoch and random seed; the same elapsed timeline at another epoch can encounter different Sun, Moon, magnetic-field, atmosphere, or access geometry.

## A message-interface worksheet

Complete one row for every high-consequence connection:

| Producer.output | Consumer.input | Payload | Meaning | Frame | Units | Producer period | Consumer period | Max age |
|---|---|---|---|---|---|---:|---:|---:|
| `Spacecraft.scStateOutMsg` | `SimpleNav.scStateInMsg` | `SCStatesMsg` | truth state | fields use N/B conventions | SI | dynamics | navigation | design value |
| `mrpFeedback.cmdTorqueOutMsg` | `ExtForceTorque.cmdTorqueInMsg` | `CmdTorqueBodyMsg` | ideal commanded body torque | B | N m | control | dynamics/effector | design value |

The first two rows are examples; replace rates and allowable ages with values from the actual scenario.

## Pre-run interface checks

1. Is every required input linked to the intended output type?
2. Has each one-time configuration message been retained and written before initialization?
3. Are truth, measurement, estimate, reference, command, and actuator state clearly separated?
4. Is each vector's origin and expression frame documented?
5. Are attitude direction and angular-rate expression frame explicit?
6. Are all physical fields in the units required by the receiving module?
7. At a coincident tick, does producer priority precede consumer priority?
8. Can the consumer tolerate the held sample between producer updates?
9. Do recorder timestamps correspond to the payload being compared?
10. Are epoch-dependent models subscribed to a consistent epoch source?
11. Are gateway writers mutually exclusive and safely initialized during transitions?
12. Have frame and unit adapters been tested with nontrivial cases?

## Repository examples to read next

- Basic message recording and time conversion: [`scenarioBasicOrbit.py`](../examples/scenarioBasicOrbit.py)
- Truth-to-navigation-to-control chain: [`scenarioAttitudeFeedbackRW.py`](../examples/scenarioAttitudeFeedbackRW.py)
- Local BASILISK-X chain: [`scenarios/nadir_pointing/nadir_pointing.py`](../scenarios/nadir_pointing/nadir_pointing.py)
- Custom typed Python messages: [`scenarioAttitudePointingPy.py`](../examples/scenarioAttitudePointingPy.py)
- Gateway messages and selectable modes: [`BskSim/models/BSK_Fsw.py`](../examples/BskSim/models/BSK_Fsw.py)
- Hill-frame relative motion: [`scenarioFormationReconfig.py`](../examples/scenarioFormationReconfig.py)
- Multi-rate image messages: [`OpNavScenarios/modelsOpNav/BSK_OpNavDynamics.py`](../examples/OpNavScenarios/modelsOpNav/BSK_OpNavDynamics.py)
