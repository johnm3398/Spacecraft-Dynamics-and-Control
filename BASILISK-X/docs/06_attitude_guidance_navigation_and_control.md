> **Draft status — AI-generated:** This page was drafted by OpenAI Codex on 2026-08-19. It has not yet been technically vetted by the BASILISK-X repository owner. The repository owner will review, correct, and maintain it. Verify APIs, units, frames, assumptions, and version compatibility before engineering use.

# Attitude, Guidance, Navigation, and Control

This chapter explains the attitude-control architecture repeatedly demonstrated by the local Basilisk 2.11.1 examples. It focuses on how to reason about interfaces and fidelity. See the [quick-start guide](QUICK_START.md) for simulation construction and the [module/message glossary](reference/module_and_message_glossary.md) for terminology.

## 1. Start with the physical question

“Point the spacecraft” is not yet a control requirement. First declare:

- which body-fixed axis or sensor boresight must point where;
- whether roll about that axis matters;
- whether the target/reference is inertial, orbit-relative, celestial, surface-fixed, or another spacecraft;
- allowable steady-state error, jitter, rate, settling time, and constraint violation;
- whether the metric is based on truth, navigation estimate, or a physical sensor output;
- available torque, momentum storage, power, and command cadence;
- disturbances and flexible modes that matter over the interval of interest.

Those decisions determine the reference generator, navigation inputs, tracking variables, controller, actuator model, execution rates, and validation outputs.

## 2. Attitude notation is part of the interface

Basilisk names normally encode both the relative orientation and the expression frame.

| Quantity | Interpretation used in the examples |
|---|---|
| `sigma_BN` | Modified Rodrigues parameters (MRPs) describing body frame `B` relative to inertial frame `N` |
| `[BN]` | Direction cosine matrix that maps components expressed in `N` into components expressed in `B` |
| `omega_BN_B` | Angular velocity of `B` relative to `N`, expressed in `B` |
| `sigma_RN` | Reference frame `R` relative to inertial `N` |
| `omega_RN_N` | Reference angular velocity relative to `N`, expressed in `N` |
| `sigma_BR` | Body tracking error: `B` relative to reference `R` |
| `omega_BR_B` | Body/reference rate error expressed in `B` |

For a position-like vector, read a name such as `r_PQ_F` as “position of point `P` relative to point `Q`, expressed in frame `F`.” Never discard the suffix merely because two arrays both contain three numbers.

### MRPs

For principal rotation angle \(\Phi\) and axis \(\hat e\), the MRP vector is

\[
\boldsymbol\sigma = \hat e\tan\left(\frac{\Phi}{4}\right).
\]

MRPs are minimal and convenient for feedback, but they have a singularity at a full \(2\pi\) rotation. Basilisk's attitude dynamics support the MRP shadow-set mechanism; recorded state includes switch information where applicable. A jump in the MRP components can therefore be a valid representation switch rather than a physical attitude jump. Evaluate a physical pointing error or principal rotation angle in addition to raw component histories.

### DCM and quaternion boundaries

`RigidBodyKinematics.MRP2C(sigma_BN)` returns `[BN]` in the local usage, so its rows are body axes expressed against inertial coordinates and it maps an inertial-component vector into body components. An accidental transpose reverses that mapping.

The audited scenarios predominantly use MRPs at Basilisk message interfaces. Quaternions/Euler parameters are valuable at external interfaces and for custom algorithms, but an adapter must declare:

1. scalar-first or scalar-last ordering;
2. Hamilton or alternative multiplication convention;
3. active rotation or passive coordinate transformation;
4. direction—such as `BN` versus `NB`;
5. normalization and sign-continuity policy;
6. expression frame for associated angular rates.

Do conversions at a named boundary, test known 90-degree rotations, and keep Basilisk's native frame/message contract visible. A “quaternion helper” that hides these choices is more dangerous than an explicit MRP/DCM conversion.

## 3. The recurring GNC chain

```text
physical truth                 flight-software side                   truth

Spacecraft ──SCStatesMsg──> sensor/navigation ──NavAtt/NavTrans──┐
                                                               │
mission target/environment ───────────────> guidance ──AttRef──┤
                                                               v
                                                     attTrackingError
                                                           │ AttGuid
                                                           v
                                                     attitude control
                                                           │ requested torque
                                                           v
                                                   control allocation
                                                           │ device commands
                                                           v
                                                    actuator dynamics
                                                           │ force/torque
                                                           └────────> Spacecraft
```

The layers are replaceable because they communicate through typed messages.

| Layer | Common local module/output | What it means |
|---|---|---|
| Truth | `spacecraft.Spacecraft.scStateOutMsg` | Simulated physical attitude, rate, position, and velocity |
| Navigation emulation | `simpleNav.SimpleNav.attOutMsg`, `transOutMsg` | FSW-facing navigation state; often nearly truth in tutorials |
| Guidance | `inertial3D`, `hillPoint`, `velocityPoint` → `AttRefMsg` | Desired attitude and reference motion |
| Tracking | `attTrackingError` → `AttGuidMsg` | Body-relative-to-reference attitude/rate error |
| Combined pointing/tracking | `locationPointing` → `AttGuidMsg` | Direct error generation toward a location/celestial state |
| Control | `mrpFeedback` → `CmdTorqueBodyMsg` | Requested body control torque |
| Steering | `mrpSteering` → rate command, then `rateServoFullNonlinear` | Cascaded outer attitude and inner rate control |
| Allocation | `rwMotorTorque` → `ArrayMotorTorqueMsg` | Maps requested body torque to wheel motor torques |
| Actuation | `ReactionWheelStateEffector` or ideal `ExtForceTorque` | Applies physical or idealized torque to truth dynamics |

Minimal wiring for the common split architecture is:

```python
nav.scStateInMsg.subscribeTo(sc.scStateOutMsg)
guidance.transNavInMsg.subscribeTo(nav.transOutMsg)
tracking.attRefInMsg.subscribeTo(guidance.attRefOutMsg)
tracking.attNavInMsg.subscribeTo(nav.attOutMsg)
controller.guidInMsg.subscribeTo(tracking.attGuidOutMsg)
controller.vehConfigInMsg.subscribeTo(vehicle_config_msg)
allocator.vehControlInMsg.subscribeTo(controller.cmdTorqueOutMsg)
allocator.rwParamsInMsg.subscribeTo(rw_config_msg)
rw_effector.rwMotorCmdInMsg.subscribeTo(allocator.rwMotorTorqueOutMsg)
```

Do not connect truth directly to guidance merely because it is convenient. If the engineering question includes navigation performance, FSW must consume an estimator output with its actual cadence, latency, validity, and frame contract.

## 4. Choosing guidance

| Engineering objective | Module/pattern | Important inputs and assumptions | Local source |
|---|---|---|---|
| Hold a fixed inertial orientation | `inertial3D.inertial3D()` | Set `sigma_R0N`; no target geometry is needed | [`scenarioAttitudeFeedback2T.py`](../examples/scenarioAttitudeFeedback2T.py) |
| Align with the orbit/Hill frame | `hillPoint.hillPoint()` | Translational navigation plus optional central-body ephemeris; a zero ephemeris message places the body at the origin | [`scenarioAttitudeGuidance.py`](../examples/scenarioAttitudeGuidance.py) |
| Point relative to velocity/orbit geometry | `velocityPoint.velocityPoint()` | Translational navigation and gravitational parameter; verify the module's reference-axis definitions | [`scenarioOrbitManeuverTH.py`](../examples/scenarioOrbitManeuverTH.py) |
| Point a body boresight at a location, planet, Sun, or spacecraft | `locationPointing.locationPointing()` | Navigation plus `locationInMsg` or celestial ephemeris; configure `pHat_B` | [`scenarioGroundLocationImaging.py`](../examples/scenarioGroundLocationImaging.py), [`scenarioRendezVous.py`](../examples/scenarioRendezVous.py) |
| Sweep/scan about a reference | Reference guidance plus `eulerRotation` | Builds a time-varying reference; reference rates matter to tracking/feedforward | [`scenarioSweepingSpacecraft.py`](../examples/scenarioSweepingSpacecraft.py), [`scenarioInertialSpiral.py`](../examples/scenarioInertialSpiral.py) |
| Respect keep-in/keep-out cones during a slew | `ConstrainedAttitudeManeuver` plus boresight monitoring | Requires vehicle inertia, celestial directions, constrained body axes, and feasibility assessment | [`scenarioAttitudeConstrainedManeuver.py`](../examples/scenarioAttitudeConstrainedManeuver.py) |
| Rate-limited imaging scan | `locationPointing` → `attTrackingError` → `mrpSteering` → rate servo | Explicit outer/inner loop with wheel allocation | [`scenarioStripImaging.py`](../examples/scenarioStripImaging.py) |

`locationPointing` is architecturally unusual because it consumes attitude navigation and emits `AttGuidMsg` directly. It collapses reference construction and error calculation. That is useful for simple one-axis pointing, but a separate `AttRefMsg` is often clearer when references must be switched, logged, checked independently, or shared.

A pointing direction rarely determines roll uniquely. Inspect each module's secondary-axis convention and singular cases. “Nadir pointing” can still yield an unexpected yaw if the transverse-axis definition was never stated.

## 5. Tracking and feedback control

`attTrackingError` composes the navigation attitude with the guidance reference and publishes the error quantities controllers expect. The separation is valuable because the same controller can track an inertial, Hill, velocity, or mission-generated reference without learning how that reference was produced.

The elementary ideal-torque chain in [`scenarioAttitudeFeedback2T.py`](../examples/scenarioAttitudeFeedback2T.py) is:

```text
inertial3D → attTrackingError → mrpFeedback → ExtForceTorque
```

For the basic no-integral case, the local custom examples illustrate the conceptual law

\[
\mathbf L_r = -K\boldsymbol\sigma_{BR} - P\boldsymbol\omega_{BR,B}
\]

with additional feedforward, inertia, wheel-momentum, and integral terms handled by the production `mrpFeedback` module as configured. A negative `Ki` is used by these examples to disable integral feedback. When integral action is enabled, set and validate `integralLimit`; do not copy the example expression after changing gain signs without checking its resulting value.

`VehicleConfigMsg` supplies the controller's assumed inertia. Truth and FSW inertia can intentionally differ for robustness tests, but accidental mismatch is not a tuning method.

### Steering versus direct feedback

Direct MRP feedback produces torque from attitude/rate error in one controller. A steering architecture separates:

```text
attitude error → mrpSteering → commanded body rate
              → rateServoFullNonlinear → commanded torque
```

This is useful when rate limits and maneuver shaping are explicit requirements. [`scenarioAttitudeSteering.py`](../examples/scenarioAttitudeSteering.py) demonstrates the outer-loop/inner-loop separation and deliberately includes cases where an aggressive outer loop violates the sub-servo separation assumption. [`scenarioStripImaging.py`](../examples/scenarioStripImaging.py) connects the pattern to a time-varying observation task.

## 6. Reaction-wheel allocation and physical closure

The physical wheel chain is:

```text
CmdTorqueBodyMsg
       ↓
rwMotorTorque + RWArrayConfigMsg
       ↓ ArrayMotorTorqueMsg
ReactionWheelStateEffector
       ↓ equal-and-opposite bus torque; wheel momentum/speed state
Spacecraft
```

[`scenarioAttitudeFeedbackRW.py`](../examples/scenarioAttitudeFeedbackRW.py) is the principal template. `simIncludeRW.rwFactory()` creates device geometry and simulation parameters; `rwFactory.getConfigMessage()` supplies the matching FSW configuration. The wheel ordering, spin axes, inertias, limits, and command-array ordering must remain consistent.

The controller can also subscribe to wheel configuration and speed messages so its nonlinear feedforward accounts for stored wheel momentum. This does not remove the need to validate saturation and momentum accumulation. Optional voltage I/O in the example inserts `rwMotorVoltage` and a voltage interface between allocation and physical wheel torque.

## 7. Rates, priorities, and tuning

A control result depends on the complete sampled-data loop, not only gains.

```text
truth integration → sensor/nav sample → guidance sample
                  → tracking/control sample → actuator command hold
```

Engineering recommendations:

1. Estimate rigid-body and actuator bandwidths before choosing task rates.
2. Make control execution materially faster than the desired closed-loop bandwidth.
3. Keep guidance slower when its target geometry changes slowly, but preserve reference rates/accelerations.
4. Model sensor and command latency when it affects phase margin.
5. Perform a task-rate convergence study independently of integrator convergence.
6. Use explicit priorities for required same-tick write-before-read paths.
7. Tune with the actual actuator limits and inertia uncertainty, not only `ExtForceTorque`.

[`scenarioAttitudeFeedback2T.py`](../examples/scenarioAttitudeFeedback2T.py) deliberately uses dynamics and FSW at different rates. It also demonstrates the subtlety that process ordering can make the first FSW sample see an unwritten/default navigation payload. Treat initial transients and timestamps deliberately.

## 8. Validation ladder

Record and evaluate at least:

- principal attitude error and body-axis pointing error;
- rate error in the correct frame;
- settling time, overshoot, steady-state error, and jitter spectrum/RMS where relevant;
- commanded versus realized body torque;
- individual wheel torque, speed, momentum, and saturation margin;
- task/message timestamps and effective latency;
- constraint cone margin, not merely a Boolean violation flag;
- angular momentum and energy in an unforced truth-only test;
- robustness to inertia, disturbance, sensor, and initial-condition uncertainty.

A sensible fidelity ladder is:

| Level | Add only when required |
|---|---|
| 1 | Prescribed/reference attitude geometry without closed-loop dynamics |
| 2 | Rigid-body truth + ideal torque + truth-like navigation |
| 3 | Physical reaction wheels or thrusters with limits |
| 4 | Sensor errors, estimator, sampled-data latency, and command quantization |
| 5 | Disturbance torques, momentum management, wheel friction/jitter, power |
| 6 | Flexible appendages, fuel motion, structural modes, or constrained maneuvers |

Do not add every level to answer a reference-frame or geometry question. Conversely, an ideal-torque result cannot establish actuator feasibility.

## 9. Example-derived caveats

**Observed in the local sources:**

- Many attitude examples use default `SimpleNav` and therefore do not demonstrate estimator performance.
- `ExtForceTorque` is frequently the actuator; it supplies unlimited ideal torque unless the scenario constrains it externally.
- Several examples place dynamics, navigation, FSW, actuation, and recorders in one task. This is pedagogically compact but hides flight-like cadence and latency.
- `scenarioAttitudeFeedbackRW.py` is a much stronger actuator template than the ideal-torque examples, but its nominal sensor/nav chain remains simplified.
- The constrained-attitude examples use boresight calculators to verify geometry independently of controller error. That independent geometric check is a good production pattern.
- The local examples do not establish a standard quaternion-native FSW architecture. Quaternion interfaces in BASILISK-X should therefore be adapters with explicit conventions, not a parallel undocumented attitude system.

**Recommended interpretation:** begin with [`scenarioAttitudeGuidance.py`](../examples/scenarioAttitudeGuidance.py), then [`scenarioAttitudeFeedback2T.py`](../examples/scenarioAttitudeFeedback2T.py), then [`scenarioAttitudeFeedbackRW.py`](../examples/scenarioAttitudeFeedbackRW.py), and only then use steering, constraints, or flexible-spacecraft cases. At every transition, retain the simpler model as a regression oracle.
