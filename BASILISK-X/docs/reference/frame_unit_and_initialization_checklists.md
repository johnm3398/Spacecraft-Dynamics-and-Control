> **Draft status — AI-generated:** This page was drafted by OpenAI Codex on 2026-08-19. It has not yet been technically vetted by the BASILISK-X repository owner. The repository owner will review, correct, and maintain it. Verify APIs, units, frames, assumptions, and version compatibility before engineering use.

# Frame, Unit, Initialization, and Validation Checklists

Use these checklists during design reviews and before trusting a plot. They are intentionally redundant: frame, unit, timing, and truth/estimate mistakes often produce smooth, plausible trajectories.

## Read Basilisk symbols literally

Common Basilisk rigid-body notation follows this pattern:

| Symbol | Meaning to verify |
|---|---|
| `r_BN_N` | Position of point/body origin B relative to inertial origin N, expressed in N |
| `v_BN_N` | Velocity of B relative to N, expressed in N |
| `sigma_BN` | MRPs describing B relative to N; verify the corresponding DCM direction before transforming vectors |
| `omega_BN_B` | Angular velocity of B relative to N, expressed in B |
| `[BN]` | DCM that maps N-frame components to B-frame components under the convention used by `RigidBodyKinematics` |

For the common `[BN]` convention:

\[
{}^{B}\mathbf v = [BN],{}^{N}\mathbf v,
\qquad
{}^{N}\mathbf v = [BN]^T,{}^{B}\mathbf v.
\]

Do not rely on variable names copied from another library. Confirm the convention with the Basilisk module documentation and one known-vector unit test.

## Frame checklist

### Before implementation

- [ ] List every frame and origin used by the scenario: inertial, planet-fixed, body, sensor, actuator, Hill, reference, target, and any mechanism frames.
- [ ] Define the direction-cosine matrix notation and vector transformation direction.
- [ ] Define every relative-position direction: target from observer, observer from target, deputy from chief, or chief from deputy.
- [ ] Define whether a velocity is an inertial difference, a rotating-frame derivative, or a transport-corrected relative velocity.
- [ ] Define the time/epoch at which inertial and planet-fixed frames align.
- [ ] Record the source of body orientation and ephemeris: fixed assumption, analytic rotation, or SPICE.

### Attitude and body axes

- [ ] State which body axis is controlled or sensed: `+b1`, `-b3`, camera boresight, panel normal, thruster direction, and so on.
- [ ] Verify that the guidance module’s alignment vector points in the intended body-frame direction.
- [ ] Verify target-minus-spacecraft versus spacecraft-minus-target line-of-sight sign.
- [ ] Transform a known body basis vector into inertial coordinates and check it by hand.
- [ ] Check whether `MRP2C` output is used as `[BN]` or `[NB]`; transpose where required.
- [ ] Exercise an MRP shadow-set crossing or demonstrate why the test interval cannot encounter one.
- [ ] Do not mix scalar-first and scalar-last quaternion conventions at a custom interface.
- [ ] If a quaternion adapter is introduced, specify passive/active rotation, multiplication order, normalization, and sign-equivalence policy.

### Hill/LVLH and relative motion

- [ ] Name the chief and deputy explicitly; never infer them from array order alone.
- [ ] Define the Hill axes. The usual local convention is radial, along-track, orbit-normal, but “LVLH” is not universal.
- [ ] Use `orbitalMotion.rv2hill`/`hill2rv` for round-trip tests instead of rotating only the position vector.
- [ ] Verify that the relative velocity includes the rotating-frame transport term expected by the utility.
- [ ] Check `hill2rv(rv2hill(...))` against the original inertial deputy state within numerical tolerance.
- [ ] For CW/Hill equations, document circular-chief, small-separation, constant-mean-motion, and two-body assumptions.
- [ ] Compare the linear solution with nonlinear propagation over the actual separation and duration.
- [ ] Check whether a commanded “hold” is a dynamical equilibrium or merely zero instantaneous relative velocity.

### Planet-fixed and ground geometry

- [ ] Distinguish geocentric latitude from geodetic latitude.
- [ ] Distinguish equatorial reference radius from local ellipsoid or terrain altitude.
- [ ] Attach a real epoch/body orientation before labeling longitude as absolute.
- [ ] When using tesseral gravity terms, provide the rotating-body orientation required to evaluate them.
- [ ] Confirm SPICE `zeroBase`, target, observer, frame, kernel coverage, and time system.
- [ ] State whether atmosphere co-rotates and in which frame winds are represented.

## Unit checklist

Basilisk interfaces generally use SI units and nanoseconds for scheduler time. Do not infer units from a plot label; trace them to the message payload or module documentation.

- [ ] Position: meters, not kilometers.
- [ ] Velocity: meters per second.
- [ ] Acceleration: meters per second squared.
- [ ] Mass: kilograms.
- [ ] Inertia: kilograms meters squared, in the declared body frame and about the declared point.
- [ ] Force: newtons; torque: newton meters.
- [ ] Angular rate: radians per second, not degrees per second or RPM.
- [ ] Angles passed to Basilisk: radians unless the API explicitly states otherwise.
- [ ] Convert human-facing degrees with `macros.D2R` and label returned degrees explicitly.
- [ ] Simulation/task/event time: integer nanoseconds using `macros.sec2nano`, `min2nano`, or equivalent.
- [ ] Recorder times: convert with `macros.NANO2SEC` before physical analysis.
- [ ] Gravity parameter: meters cubed per second squared when position is in meters.
- [ ] Power: watts; energy/storage: joules or watt-seconds unless the module states otherwise.
- [ ] Reaction-wheel speed: radians per second internally; convert to RPM only for presentation.
- [ ] Momentum: N m s; minimum impulse bit: N s; thruster on-time: seconds.
- [ ] Covariances: square units and state ordering documented beside the matrix.
- [ ] Noise spectral density, per-sample standard deviation, and random-walk parameters are not interchanged.
- [ ] Every user-configurable symbol carries a unit suffix where practical.

### Dimensional sanity tests

- [ ] Compute orbital period independently from \(2\pi\sqrt{a^3/\mu}\).
- [ ] Check circular speed against \(\sqrt{\mu/r}\).
- [ ] Estimate control angular acceleration from \(I^{-1}L\).
- [ ] Estimate wheel momentum buildup from integrated torque.
- [ ] Estimate burn delta-v from integrated thrust divided by mass.
- [ ] Estimate energy usage from integrated power.
- [ ] Compare each result with an order-of-magnitude hand calculation before plotting.

## Simulation construction and initialization checklist

### Simulation container and scheduling

- [ ] Create the intended `SimBaseClass` instance once.
- [ ] Give every process and task a unique, meaningful name.
- [ ] Choose task periods from physical bandwidth and data rates, not convenience.
- [ ] Assign explicit process/task/model priorities when same-tick order matters.
- [ ] Inspect execution order for complex simulations with `ShowExecutionOrder()` or equivalent local tooling.
- [ ] Document current-sample versus one-cycle-old data at each control-loop edge.
- [ ] Confirm multi-rate tasks meet at expected scheduler times.
- [ ] Confirm event rate and task rate can represent the required transition timing.

### Spacecraft hub and integrated states

- [ ] Assign a unique `ModelTag` to every spacecraft and module.
- [ ] Set hub mass, center-of-mass offset, and inertia before initialization.
- [ ] Confirm inertia is about the intended point and expressed in the body frame.
- [ ] Confirm the inertia matrix is symmetric and positive definite.
- [ ] Set position, velocity, attitude, and angular-rate initial values with correct shapes and units.
- [ ] Check orbit validity: periapsis, eccentricity domain, central-body radius, and epoch.
- [ ] Avoid singular classical-element interpretations for circular/equatorial cases; validate the resulting Cartesian state.
- [ ] Use fresh factory instances where examples warn that state persists between repeated runs.

### Gravity and environment

- [ ] Mark exactly the intended central body with `isCentralBody`.
- [ ] Attach the required gravity bodies to every spacecraft separately.
- [ ] Add and schedule SPICE/environment modules before consumers require their messages.
- [ ] Check ephemeris indexing rather than assuming body order.
- [ ] Connect eclipse, atmosphere, magnetic field, SRP, ground, and power inputs to the correct spacecraft message.
- [ ] Document each omitted perturbation and why it is below the decision margin.

### Effectors and actuators

- [ ] Decide whether the physical component is a `StateEffector` or `DynamicEffector`.
- [ ] Attach each effector to the correct spacecraft in addition to scheduling any required module update.
- [ ] Give every actuator set a unique tag and message path.
- [ ] Check actuator position and direction in body coordinates.
- [ ] Check saturation, minimum command/on-time, deadband, and initial state.
- [ ] Connect tanks to thruster sets when propellant depletion is claimed.
- [ ] Confirm internal effectors conserve total momentum in the applicable limiting case.
- [ ] Label direct `dynManager.setState` changes as ideal state resets, never as physical actuation.

### Messages

- [ ] Draw the message graph before coding.
- [ ] Subscribe the reader input to the writer output, not the reverse.
- [ ] Verify payload type compatibility at every edge.
- [ ] Identify required versus optional input messages from module documentation.
- [ ] Keep standalone configuration-message objects alive for the duration of their subscriptions.
- [ ] Initialize and zero gateway messages intentionally during mode transitions.
- [ ] Ensure each satellite consumes its own configuration and state messages unless sharing is deliberate.
- [ ] Do not feed FSW a truth message when the architecture claims a sensor, estimator, or link boundary.
- [ ] Record the producer’s frame, units, time tag, and validity semantics with the interface.

### Navigation and FSW

- [ ] Label each navigation source as truth-derived, sensor-derived, or estimated.
- [ ] Record truth and navigation separately when navigation performance is evaluated.
- [ ] Connect guidance to navigation, not directly to truth, unless the test deliberately isolates downstream FSW.
- [ ] Verify the chain `guidance -> tracking error -> control -> allocation -> actuator` is complete.
- [ ] Check controller inertia/configuration messages match the truth vehicle only when perfect knowledge is intended.
- [ ] Confirm controller gains have the units and sign semantics expected by the module.
- [ ] Check disabled integral terms, limits, and state reset behavior during mode changes.
- [ ] Define safe command behavior when an input message is missing or invalid.

### Recorders and visualization

- [ ] Add recorders before `InitializeSimulation()`.
- [ ] Record the messages needed for metrics and debugging, not every available field.
- [ ] Choose a sampling interval that captures peaks without exhausting memory.
- [ ] Record commands and achieved actuator response separately.
- [ ] Record truth and estimate/reference separately.
- [ ] Confirm histories compared sample-by-sample have compatible timestamps.
- [ ] Treat Vizard as a consumer of simulation data, not validation evidence by itself.
- [ ] Keep visualization optional for headless tests and batch runs.
- [ ] Store configuration/version metadata with retained plots or playback files.

### Initialization and execution

- [ ] Finish model configuration, attachments, subscriptions, recorders, and Vizard setup before initialization.
- [ ] Call `InitializeSimulation()` before accessing registered dynamic state objects for runtime maneuvers.
- [ ] Configure a stop time in integer nanoseconds.
- [ ] For phased execution, maintain absolute simulation time rather than restarting phase time at zero.
- [ ] Check boundary semantics: whether the phase-end sample occurs before or after a command/state change.
- [ ] Protect external processes and retained files with cleanup/error handling.
- [ ] Verify repeated `run()` calls do not reuse mutable factory, message, or default-array state unexpectedly.

## Validation checklist

### Deterministic baseline

- [ ] Define success/failure metrics before running.
- [ ] Test invalid configurations and malformed vector shapes.
- [ ] Test a zero-input or zero-disturbance limiting case.
- [ ] Test a hand-computable geometry or dynamics case.
- [ ] Check conservation laws appropriate to the modeled closed system.
- [ ] Repeat with a smaller integration/task step and compare metrics.
- [ ] Check that all plotted labels match the underlying frame and units.
- [ ] Check initial and final samples manually.
- [ ] Freeze toleranced numerical regression metrics rather than image binaries alone.

### Fidelity increments

- [ ] Add one physical effect at a time.
- [ ] Preserve an option to disable it for an A/B comparison.
- [ ] Predict the expected sign and approximate magnitude before running.
- [ ] Explain discrepancies between the previous and new fidelity levels.
- [ ] Confirm new parameters have sources and uncertainty ranges.
- [ ] Remove fidelity that does not affect the decision but adds poorly known parameters.

### Navigation and uncertainty

- [ ] Verify random seeds are controllable and repetitions are reproducible.
- [ ] Separate truth, measurement, and estimate error distributions.
- [ ] Include correlations where physics or calibration creates them.
- [ ] Check estimator innovations/residuals and covariance consistency, not only final position error.
- [ ] Define failure handling and retain failed-run diagnostics.
- [ ] Compute confidence intervals or success probability with enough valid samples.
- [ ] Do not start Monte Carlo until the deterministic baseline passes.

## Multi-spacecraft checklist

- [ ] Give each spacecraft, process, task, module, state, and recorder a unique indexed name.
- [ ] Confirm every environment model receives all intended spacecraft state messages once.
- [ ] Confirm every spacecraft has its intended gravity bodies and effectors.
- [ ] Decide whether integrations are independent, synchronized, or physically coupled, and document why.
- [ ] Define chief/deputy or formation-reference selection, including handover behavior.
- [ ] Verify `rv2hill`/`hill2rv` round trips for each pair.
- [ ] Avoid mixing relative states referenced to different chiefs or epochs.
- [ ] Check cross-spacecraft message ownership and vehicle index at every subscription.
- [ ] Distinguish direct in-memory truth sharing from a modeled intersatellite link.
- [ ] For distributed control, write the information set available to each vehicle.
- [ ] Model link latency, loss, scheduling, and clocks when they affect the claim.
- [ ] Record per-vehicle truth, estimate, command, resource state, and constraint margins.
- [ ] Check collision/keep-out constraints continuously at adequate resolution, not only at phase endpoints.
- [ ] Exercise loss-of-navigation, loss-of-link, actuator failure, retreat, and abort cases where safety is claimed.
- [ ] Benchmark runtime and recorder memory as spacecraft count grows.

## Final pre-plot challenge

Before accepting a result, answer these questions without looking at the plot:

1. Which exact message or state produced the ordinate?
2. Is it truth, measurement, estimate, reference, command, or response?
3. What are its origin, axes, expression frame, units, and timestamp?
4. Which model ran immediately before its producer at that scheduler instant?
5. Which omitted effect is most likely to change the conclusion?
6. What independent calculation would reveal a sign, scale, or frame error?

If those answers are unavailable, the plot is exploratory—not engineering evidence.
