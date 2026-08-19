> **Draft status — AI-generated:** This page was drafted by OpenAI Codex on 2026-08-19. It has not yet been technically vetted by the BASILISK-X repository owner. The repository owner will review, correct, and maintain it. Verify APIs, units, frames, assumptions, and version compatibility before engineering use.

# Pitfalls, Debugging, and Validation

Basilisk usually does exactly what was scheduled and wired. That is why many mistakes produce smooth, plausible results instead of exceptions. This chapter organizes the most important failure modes found in the local examples and turns them into a repeatable debugging method.

## Debug in causal order

When a result is wrong, inspect the system in this order:

```text
configuration and initial state
        -> execution order and timestamps
        -> message links and payload semantics
        -> frame and unit interpretation
        -> truth/environment/dynamics
        -> sensor and navigation boundary
        -> guidance/control/allocation/actuator chain
        -> recorder and analysis code
        -> physical fidelity and uncertainty
```

Do not tune gains or add a higher-order model before the earlier layers have passed.

## 1. Hidden task and model ordering

### Failure

A consumer executes before its producer at the same scheduler instant and reads the previous payload or an unwritten default.

Typical symptoms:

- the first sample is zero;
- command histories lag errors by one tick;
- identical modules behave differently after refactoring;
- a recorder shows a command one sample later than expected.

### Cause

Higher numeric priority executes first. Equal/default priorities rely on insertion order. Different-rate tasks meet only at specific scheduler times, and messages retain the latest written value between updates.

### Response

1. Run `ShowExecutionOrder()` and inspect relevant task periods.
2. Record producer output, consumer-visible input if necessary, and timestamps.
3. State whether the intended contract is current-sample or held previous-sample data.
4. Assign explicit priorities only where causal order matters.
5. Repeat with a deliberately simplified single-rate arrangement to isolate the timing effect.

[`scenarioAttitudeFeedback2T.py`](../examples/scenarioAttitudeFeedback2T.py) is a useful clean multi-rate reference. BskSim and MultiSat use explicit priorities but can record FSW messages from earlier-running dynamics tasks, so arrays still require timestamp interpretation.

## 2. Wrong message direction or wrong producer

The correct grammatical form is:

```python
consumer.someInMsg.subscribeTo(producer.someOutMsg)
```

A link can have the correct payload type and still be semantically wrong: the wrong spacecraft, wrong mode gateway, wrong ephemeris index, or wrong navigation source can publish that type.

Check:

- `isLinked()` before initialization;
- `isWritten()` and `timeWritten()` when debugging first-sample behavior;
- producer `ModelTag` and spacecraft index;
- frame, units, and epistemic role of the payload;
- whether multiple enabled tasks can author the same gateway.

Do not solve a missing message by wiring the nearest compatible truth output if the architecture claims a sensor, estimator, or communication boundary.

## 3. Configuration-message lifetime and initialization

A Python-written configuration message must remain alive while a reader is subscribed to it:

```python
payload = messaging.VehicleConfigMsgPayload()
payload.ISCPntB_B = inertia
vehicle_config_msg = messaging.VehicleConfigMsg().write(payload)
controller.vehConfigInMsg.subscribeTo(vehicle_config_msg)
```

Keep `vehicle_config_msg` as a scenario/model attribute or a local variable that remains in scope through execution.

Finish physical attachment, configuration, subscriptions, model scheduling, recorders, and Vizard setup before `InitializeSimulation()`. Accessing registered dynamic state objects for an ideal mid-run reset is normally a post-initialization operation; assigning ordinary initial conditions is not.

## 4. Recorder placement and misleading histories

A recorder is a scheduled model. It records the value available when it executes, not a retroactively synchronized truth history.

Common problems:

- recorder priority precedes producer priority;
- separate recorder task samples just before a producer task;
- two histories at different rates are compared by array index;
- every high-rate payload is retained for a long mission;
- an input-reader recorder is created before the reader subscribes.

Use timestamps as the join key. Record the producer output unless the question specifically concerns the consumer-visible held sample. [`scenarioAttitudeFeedback.py`](../examples/scenarioAttitudeFeedback.py) warns that an input-message recorder should be created only after subscription.

## 5. Frame and sign mistakes

Smooth wrong-way motion is the characteristic frame bug.

Before changing a controller, write down:

- the point and origin of every position;
- the expression frame of every vector;
- which attitude maps which frame into which;
- the expression frame of angular velocity, torque, and force;
- the body axis that should align with the target;
- whether line of sight is target-minus-observer or its negative.

Test a nonidentity rotation and a known vector. Identity attitude tests cannot reveal a transpose error.

For Hill/LVLH work:

- define radial, along-track, and orbit-normal axes explicitly;
- define chief and deputy rather than relying on index order;
- use `rv2hill`/`hill2rv` and round-trip both position and velocity;
- remember that rotating-frame relative velocity is not just a rotated inertial velocity difference.

The current [`nadir_pointing.py`](../scenarios/nadir_pointing/nadir_pointing.py) contains a documentation inconsistency: its opening text names `+b1`, while its configured reference rotation, analysis, and reporting use `+b3`. The implementation and documentation must be reconciled before using it as a body-axis authority.

## 6. Unit mistakes

Basilisk generally expects SI quantities and integer nanoseconds for scheduler time. Common scale errors include:

- kilometres supplied as metres;
- degrees supplied where radians are expected;
- RPM used as rad/s;
- seconds passed directly to an API expecting nanoseconds;
- a gravity parameter in km³/s² combined with a state in metres;
- covariance entries interpreted without squared units;
- on-time, impulse, thrust, and delta-v confused with one another.

Use unit-bearing BASILISK-X names such as `altitude_m`, `omega_rad_s`, and `duration_s`. Convert for display at the plotting boundary.

Order-of-magnitude checks should precede simulation:

```text
orbital speed       ~ sqrt(mu/r)
orbital period      ~ 2*pi*sqrt(a^3/mu)
angular acceleration ~ torque/inertia
delta-v             ~ integral(thrust/mass dt)
energy              ~ integral(power dt)
```

## 7. Truth, measurement, and estimate confusion

`Spacecraft.scStateOutMsg` is truth. `SimpleNav` turns truth into navigation-format messages and can add configured error processes; default use is often nearly perfect.

Consequences:

- a controller closed around default `SimpleNav` demonstrates control with near-perfect navigation;
- two directly shared `NavTransMsg` objects do not demonstrate intersatellite sensing;
- `relativeNavigation` in MultiSat currently refers to a formation-barycenter service, not a range/bearing sensor and estimator;
- preparatory truth-assisted pointing in OpNav isolates a later OD algorithm but is not autonomous acquisition.

Use names such as `truth_r_N`, `measured_bearing`, `estimated_r_N`, and `commanded_force_B`. Record truth and estimate separately and compute residuals, error, covariance consistency, and data age.

## 8. Direct state manipulation disguised as actuation

Several examples deliberately call a dynamic state object's `setState()` to model an ideal impulse. This is valid for transfer geometry but bypasses:

- finite burn duration;
- attitude pointing and settling;
- thruster geometry and allocation;
- minimum impulse bit and valve dynamics;
- mass depletion;
- execution and navigation error;
- power, plume, thermal, and constraint effects.

Examples include `scenarioOrbitManeuver.py`, `scenarioHohmann.py`, `scenarioRendezVous.py`, BskSim Lambert guidance, and the BASILISK-X cooperative GEO rendezvous.

Label the abstraction “ideal impulse.” Preserve it as a regression baseline when adding a separate finite-burn realization.

## 9. Prescribed attitude or ideal force interpreted as GNC performance

`Spacecraft.attRefInMsg` can prescribe spacecraft attitude truth. `ExtForceTorque` can apply an ideal commanded force or torque directly.

These seams are powerful isolation tools:

- prescribed attitude isolates orbit/drag/payload geometry;
- ideal torque isolates guidance/tracking/control;
- ideal inertial force isolates a translational formation law.

They do not demonstrate actuator feasibility or closed-loop attitude performance. `scenarioDragRendezvous.py` prescribes attitude, while `scenarioFormationMeanOEFeedback.py` applies ideal force without thrust allocation, saturation, duty cycle, or propellant.

## 10. StateEffector and DynamicEffector setup errors

Typical mistakes include:

- attaching an effector but failing to schedule its update;
- scheduling an effector but attaching it to the wrong spacecraft;
- treating internal momentum exchange as an external torque;
- forgetting to connect a thruster set to a fuel tank while claiming depletion;
- using independent integration for physically coupled spacecraft;
- choosing a step that does not resolve a flexible, slosh, actuator, or contact mode.

Check the module-specific example because attachment and scheduling contracts differ. Use conservation tests for closed internal-effectors cases.

Cross-spacecraft constraints require special care. [`scenarioConstrainedDynamics.py`](../examples/scenarioConstrainedDynamics.py) uses synchronized dynamics integration and one constraint effector coupled to both spacecraft.

## 11. Integrator changes used to hide physical errors

A smaller step can reduce numerical error. It cannot fix:

- the wrong frame;
- a missing force;
- an incorrect density model;
- bad mass properties;
- stale navigation;
- an actuator shortcut;
- an invalid linearization.

Treat physical fidelity, task rate, integrator algorithm, internal substeps, and tolerances as separate decisions. Run convergence on engineering metrics and invariants, not only on a visually smooth trajectory.

Use [`scenarioIntegrators.py`](../examples/scenarioIntegrators.py), [`scenarioIntegratorsComparison.py`](../examples/scenarioIntegratorsComparison.py), and [`scenarioVariableTimeStepIntegrators.py`](../examples/scenarioVariableTimeStepIntegrators.py) as numerical-method references.

## 12. Event and gateway transition defects

Disabling a task does not erase its last message. A safe mode transition normally needs to:

1. disable incompatible writers;
2. establish safe gateway payloads;
3. enable the intended task chain;
4. reset module state if the contract requires it;
5. re-arm relevant events;
6. verify the first post-transition command and timestamp.

The copied [`BskSim/models/BSK_Fsw.py`](../examples/BskSim/models/BSK_Fsw.py) contains a likely real migration defect: `initiateStandby` returns a tuple of strings instead of calling its actions. Current callback handling discards that return, so the transition appears not to disable active FSW tasks or zero gateways. Other mode callbacks call methods correctly.

Do not copy legacy string-event syntax into new code. Use current callbacks and test every transition from every reachable source mode.

## 13. Multi-spacecraft indexing and false distribution

Adding processes per spacecraft improves ownership; it does not create independent onboard computers or a physical network.

Check:

- every module/configuration/recorder name includes the spacecraft identity;
- every environment output index feeds the intended spacecraft;
- each spacecraft receives its own eclipse and resource messages;
- chief changes update both origin and relative frame;
- direct message subscriptions are not described as communication;
- estimator and planner information sets match the claimed centralized/distributed architecture.

The copied MultiSat solar-panel setup appears to connect every vehicle to `eclipseOutMsgs[0]`, even though one output is created per spacecraft. Treat this as a probable wiring defect until a version-matched test proves otherwise.

Several MultiSat scenarios also hard-code indices 0, 1, and 2 even though the underlying model classes use lists. The architecture scales farther than those scenario configurations.

## 14. Vizard and external-dependency assumptions

Vizard is normally optional presentation. Keep numerical scenarios runnable headlessly and do not use a visually plausible animation as validation.

Live mode introduces:

- external-process discovery and lifecycle;
- ports and connection timeouts;
- synchronization and real-time factors;
- platform-specific paths;
- cleanup on exceptions.

OpNav is different: Vizard produces the image used by the sensor pipeline. Executable version, rendering cadence, scene configuration, camera calibration, port exclusivity, and image timestamp become engineering inputs.

The copied OpNav master hard-codes a macOS Vizard path and common port, making parallel image runs unsafe without explicit port management.

## 15. Version drift and missing assets

The repository pins `bsk[all,examples]==2.11.1`, but the copied examples do not record an upstream commit. Known mismatches include:

- `Basilisk.hasBuildFeature` imports absent from installed 2.11.1;
- a missing OpNav CNN `CAD.onnx` file;
- MuJoCo examples using models or properties absent from the installed package;
- legacy event prose and threading notes;
- material files referring to missing texture/material assets.

Do not “repair” an import by deleting the guard and assuming the feature exists. Record the intended upstream revision, optional package, external asset, and compatibility decision.

See [scope, versions, and provenance](00_scope_versions_and_source_provenance.md).

## 16. Monte Carlo before deterministic validation

An ensemble of wrong simulations is not uncertainty quantification.

Before dispersion:

- verify a deterministic baseline;
- define a numerical success metric;
- classify epistemic parameters versus stochastic processes;
- identify correlations and physical bounds;
- control and archive seeds;
- retain failed-run inputs and diagnostics;
- ensure attribute paths remain stable when model ordering changes.

The Monte Carlo controller frequently uses string paths into scenario objects. Paths such as `TaskList[0].TaskModels[0]` are fragile; named scenario attributes are preferable.

## 17. Plot-driven conclusions

A plot can hide:

- unit scaling;
- frame changes;
- aliasing and missed extrema;
- stale samples;
- truth/estimate confusion;
- failed runs removed from an ensemble;
- constraint violations between plotted samples.

Every figure should be backed by a computed metric with units, time interval, source message/state, and tolerance. For mission decisions, retain the metric data and configuration rather than only an image.

## Symptom-to-check table

| Symptom | First checks |
|---|---|
| All-zero input | `isLinked`, `isWritten`, producer scheduling, first-write priority |
| One-sample lag | producer/consumer periods, priorities, recorder location |
| Wrong pointing direction | body alignment axis, LOS sign, MRP/DCM direction |
| Orbit scale nonsensical | metres vs kilometres, `mu` units, central body |
| Control command but no response | complete allocation/actuator chain, physical attachment, scheduling |
| Perfect navigation | default `SimpleNav`, direct truth link, disabled noise |
| Unexpected momentum drift | external torque, effector coupling, step convergence, recorder semantics |
| Phase executes for zero time | absolute `ConfigureStopTime` smaller than current time |
| Spacecraft use identical environment data | output indexing and cross-spacecraft subscriptions |
| Headless run fails | unconditional Vizard/import/asset dependency |
| Example import fails | package/example version drift and optional feature availability |
| Monte Carlo cannot reproduce a run | archived seed/configuration, worker-local initialization, SPICE loading |

## Minimum validation package for a new scenario

Before calling a scenario an engineering baseline, provide:

1. a written claim and minimum-fidelity rationale;
2. a message/execution diagram;
3. a configuration record with units, frames, epoch, rates, and version;
4. at least one analytic or independent comparison;
5. step/rate convergence for the principal metric;
6. limiting-case and sign tests;
7. toleranced headless regression metrics;
8. explicit known omissions;
9. reproducible random seeds if stochastic behavior exists;
10. a distinction between exploration plots and acceptance evidence.

If a smooth result fails any of those tests, keep it labelled exploratory.
