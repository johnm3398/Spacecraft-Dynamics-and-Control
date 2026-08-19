> **Draft status — AI-generated:** This page was drafted by OpenAI Codex on 2026-08-19. It has not yet been technically vetted by the BASILISK-X repository owner. The repository owner will review, correct, and maintain it. Verify APIs, units, frames, assumptions, and version compatibility before engineering use.

# Multi-Spacecraft Simulation, Relative Motion, and RPO

This chapter maps the multi-spacecraft architectures present in the Basilisk 2.11.1 example snapshot. It focuses on how to choose an architecture, how chief/deputy states are represented, and where a geometric rendezvous demonstration stops being an RPO system simulation.

## One physical world, many software architectures

“Multiple spacecraft” does not imply one required Basilisk architecture. The correct structure depends on what must scale independently:

- number of propagated rigid bodies;
- environment models;
- onboard FSW instances and modes;
- sensor and communication links;
- update rates and process ownership;
- campaign configuration and reuse.

The repository demonstrates three main choices.

| Style | Physical/software structure | Best use | Cost |
|---|---|---|---|
| Standalone scenario | Several `Spacecraft` objects assembled directly in one script, often in one task | One-off formation, relative-motion, or RPO studies | Lowest abstraction; repetition grows quickly |
| `BskSim` formation | Fixed chief/deputy dynamics and FSW model classes behind a scenario/master interface | Repeated two-spacecraft modes and shared internal configuration | More indirection and gateway/task logic |
| `MultiSatBskSim` | Indexed environment, dynamics, and FSW model instances; generally one dynamics and FSW process per satellite | Constellations or scalable N-spacecraft experiments | Highest setup complexity and strongest naming/index discipline |

Do not select `MultiSatBskSim` merely because there are two vehicles. Select it when the *software pattern* must scale to N vehicles.

## Architecture A: standalone multi-spacecraft scenarios

[scenarioFormationBasic.py](../examples/scenarioFormationBasic.py) constructs three spacecraft directly. [scenarioRendezVous.py](../examples/scenarioRendezVous.py) constructs a servicer and debris object, individual navigation modules, several pointing modes, reaction wheels, and ideal relative-orbit state changes. [scenarioSatelliteConstellation.py](../examples/scenarioSatelliteConstellation.py) uses loops to configure a larger set without introducing a framework.

A typical topology is:

```text
SimBaseClass
  `-- dynamics process
       `-- one task at dt
            |-- Spacecraft: chief
            |-- Spacecraft: deputy
            |-- chief navigation
            |-- deputy navigation
            |-- relative/pointing guidance
            |-- controllers and effectors
            `-- recorders
```

Each `Spacecraft` owns distinct hub and effector states. Sharing a task does not merge their states. If both receive the same gravity-body objects, each spacecraft still evaluates gravity for its own position.

This style is usually best for:

- a new chief/deputy concept;
- a fixed two- or three-vehicle comparison;
- relative-frame and targeting studies;
- experiments where seeing every subscription is more valuable than reuse.

It becomes awkward when each satellite needs a repeated suite of effectors, power, sensors, FSW modes, recorders, and unique names.

### Synchronized versus independent integration

[scenarioFormationBasic.py](../examples/scenarioFormationBasic.py) demonstrates `syncDynamicsIntegration` for two spacecraft and explicitly notes that it is not required for its uncoupled case. Independent spacecraft integrated at the same task rate are adequate when their dynamics do not require a shared integration stage. Synchronization becomes important when the numerical formulation or a coupled interaction requires states to advance together. Do not infer that same-task scheduling alone creates physical coupling.

## Architecture B: the `BskSim` formation variant

The formation-specific `BskSim` files are:

- [BSK_FormationDynamics.py](../examples/BskSim/models/BSK_FormationDynamics.py)
- [BSK_FormationFsw.py](../examples/BskSim/models/BSK_FormationFsw.py)
- [scenario_BasicOrbitFormation.py](../examples/BskSim/scenarios/scenario_BasicOrbitFormation.py)
- [scenario_RelativePointingFormation.py](../examples/BskSim/scenarios/scenario_RelativePointingFormation.py)

`BSK_FormationDynamics.BSKDynamicModels` creates two dynamics tasks, one per spacecraft, and assigns modules explicit priorities. `BSK_FormationFsw.BSKFswModels` creates separate task chains for each vehicle, gateway messages, and events that enable attitude or spacecraft-pointing modes.

```text
scenario class
   |
   +--> BSK master / modeRequest
   |       +--> events enable/disable FSW tasks
   |
   +--> formation dynamics model
   |       +--> chief dynamics task
   |       `--> deputy dynamics task
   |
   `--> formation FSW model
           +--> chief guidance/control tasks
           +--> deputy guidance/control tasks
           `--> shared/gateway command messages
```

This architecture solves configuration reuse and mode switching. It is useful when many scenarios share the same vehicle models but change initial conditions, task activation, disturbances, or mission phases.

It is unnecessary overhead for a single transparent trade study. The formation model is also a specific example architecture, not a universal production framework: it contains fixed assumptions, duplicated chief/deputy objects, and example-specific configuration.

## Architecture C: `MultiSatBskSim`

The scalable example framework begins at [BSK_MultiSatMasters.py](../examples/MultiSatBskSim/BSK_MultiSatMasters.py). Its `BSKSim` class accepts `numberSpacecraft` and creates:

- one environment process with higher example priority;
- one indexed dynamics process per spacecraft;
- one indexed FSW process per spacecraft;
- an optional relative-navigation process containing `FormationBarycenter`.

The reusable model families are:

- [BSK_EnvironmentEarth.py](../examples/MultiSatBskSim/modelsMultiSat/BSK_EnvironmentEarth.py)
- [BSK_EnvironmentMercury.py](../examples/MultiSatBskSim/modelsMultiSat/BSK_EnvironmentMercury.py)
- [BSK_MultiSatDynamics.py](../examples/MultiSatBskSim/modelsMultiSat/BSK_MultiSatDynamics.py)
- [BSK_MultiSatFsw.py](../examples/MultiSatBskSim/modelsMultiSat/BSK_MultiSatFsw.py)

`BSK_MultiSatDynamics.BSKDynamicModels` creates indexed spacecraft, navigation, reaction-wheel, thruster, tank, power, and mass-property objects. This breadth is why the abstraction is valuable for its station-keeping example and excessive for a first relative-motion script.

Study these scenarios in order:

1. [scenario_BasicOrbitMultiSat.py](../examples/MultiSatBskSim/scenariosMultiSat/scenario_BasicOrbitMultiSat.py) — basic indexed construction.
2. [scenario_AttGuidMultiSat.py](../examples/MultiSatBskSim/scenariosMultiSat/scenario_AttGuidMultiSat.py) — per-satellite FSW.
3. [scenario_StationKeepingMultiSat.py](../examples/MultiSatBskSim/scenariosMultiSat/scenario_StationKeepingMultiSat.py) — formation control, effectors, power, and fuel.
4. [scenario_constellationFromTle.py](../examples/MultiSatBskSim/scenariosMultiSat/scenario_constellationFromTle.py) — external orbit data and constellation setup.
5. [scenario_BasicOrbitMultiSat_MT.py](../examples/MultiSatBskSim/scenariosMultiSat/scenario_BasicOrbitMultiSat_MT.py) — special multithreading-oriented variant; do not adopt it before the ordinary case is understood and benchmarked.

### What the optional relative-navigation process is—and is not

`BSK_MultiSatMasters.BSKSim.add_relativeNavigation` creates a process around `formationBarycenter.FormationBarycenter`. This supplies a formation reference/barycenter architecture. Its class name does not make it a complete sensor-level relative navigation estimator. A real relative-navigation study must still define measurements, observability, noise, covariance, data age, and which spacecraft receives which estimate.

## Scheduling and data freshness

Multi-spacecraft simulations amplify ordering mistakes because a value can be stale in time *and* belong to the wrong vehicle.

For every cross-spacecraft edge, document:

| Item | Required statement |
|---|---|
| Producer | Vehicle, module, and output message |
| Consumer | Vehicle, module, and input reader |
| Frame and epoch | Coordinate frame, time tag, and origin |
| Rate | Producer and consumer task periods |
| Age | Current-tick, previous-tick, held, or delayed sample |
| Link model | Direct in-memory subscription or modeled communication |

The `MultiSatBskSim` master uses process priorities to order environment, dynamics, optional formation reference, and FSW processes. Within tasks, the models use priorities and indexed names. Those choices are part of the example’s timing contract; copying modules without their ordering can change behavior.

For a distributed architecture, do not connect an estimator or controller directly to another vehicle’s truth message merely for convenience. Place an explicit measurement or communication boundary between them.

## Chief/deputy and Hill-frame mechanics

For chief inertial position and velocity \(\mathbf r_c, \mathbf v_c\), define the common Hill triad

\[
\hat{\mathbf h}_r = \frac{\mathbf r_c}{\|\mathbf r_c\|}, \qquad
\hat{\mathbf h}_h = \frac{\mathbf r_c \times \mathbf v_c}{\|\mathbf r_c \times \mathbf v_c\|}, \qquad
\hat{\mathbf h}_\theta = \hat{\mathbf h}_h \times \hat{\mathbf h}_r.
\]

The usual component order is radial, along-track, orbit-normal:

```text
x_H : radial, away from the central body
y_H : along track, in the chief direction of motion
z_H : orbit normal, completing the right-handed frame
```

The local repository uses `orbitalMotion.rv2hill` and `orbitalMotion.hill2rv` rather than manually assembling only a position rotation. This matters because relative velocity in a rotating Hill frame includes the frame-rotation term. Treat `rhoPrime_H` as the derivative defined by the Basilisk utility, not simply the inertial velocity difference rotated into Hill components.

Hill and LVLH names are not universally interchangeable. Some organizations define LVLH with nadir as a principal axis or permute/sign-flip axes. Record the actual triad, not only the label “LVLH.”

### CW/Hill linear-model boundary

Clohessy-Wiltshire targeting assumes a circular chief orbit, small separation, and two-body relative dynamics with constant mean motion. It is valuable for:

- intuition about bounded relative ellipses;
- initial targeting estimates;
- analytic regression checks;
- exposing radial/along-track coupling.

It is not a substitute for nonlinear propagation when separation, eccentricity, perturbations, long duration, or safety margins invalidate its assumptions. A strong workflow is:

```text
CW design -> convert with hill2rv -> nonlinear Basilisk propagation
          -> recover with rv2hill -> quantify model mismatch
```

The current [cooperative GEO rendezvous](../scenarios/cooperative_geo_rendezvous/cooperative_geo_rendezvous.py) follows this pattern through `cw_targeting_velocity`, `hill2rv`, and `rv2hill`.

## Information architecture: centralized versus distributed

### Centralized truth-assisted study

```text
chief truth ----+
                +--> central Python/FSW logic --> deputy command
deputy truth ---+
```

This is appropriate for geometry development and algorithm verification. It is not evidence that the required information can be measured or communicated.

### Centralized estimated system

```text
vehicle measurements --> link/network --> central estimator/planner
                                            |
                                            v
                                  time-tagged vehicle commands
```

This requires link delays, dropouts, common time, estimator state, and command routing.

### Distributed system

```text
satellite i: local truth -> local sensors -> local nav -> local FSW -> local actuators
                                      ^          |
                                      |          v
                                 received data <-link-> peer messages
```

Each satellite should have an explicit information set. A shared Python object or direct truth subscription silently converts a distributed problem into a centralized one.

## Interpreting the current BASILISK-X RPO scenario

[cooperative_geo_rendezvous.py](../scenarios/cooperative_geo_rendezvous/cooperative_geo_rendezvous.py) correctly labels itself a low-fidelity sandbox. Its actual architecture is:

```text
point-mass Earth
  |-- target Spacecraft -> target SimpleNav ----+
  |                                             |
  `-- servicer Spacecraft -> servicer SimpleNav +--> locationPointing
                                                   -> attTrackingError
                                                   -> mrpFeedback
                                                   -> ideal body torque

external Python phase driver
  -> read registered chief/deputy states
  -> compute Hill state and CW target velocity
  -> replace servicer velocity state instantaneously
  -> advance to next phase
```

What it demonstrates well:

- two independently propagated spacecraft;
- chief/deputy Hill conversion;
- analytic targeting followed by nonlinear propagation;
- phase annotation and relative metrics;
- target-relative attitude guidance;
- clearly bounded keep-out and impulse checks.

What it does not model:

- a cooperative target controller or communications link;
- relative sensors or estimation;
- translational feedback guidance;
- thruster geometry, minimum impulse bit, finite burn, or mass depletion;
- execution errors or navigation covariance;
- approach corridors, passive-safety certification, plume constraints, or aborts;
- capture, docking, or contact.

Direct `dynManager` state replacement is an official-example shortcut used in [scenarioRendezVous.py](../examples/scenarioRendezVous.py), [scenarioOrbitManeuver.py](../examples/scenarioOrbitManeuver.py), and related maneuver examples. It is appropriate when the variable of interest is ideal transfer geometry. It must not be reused as an actuator abstraction.

## Engineering metrics for relative-motion studies

At minimum, compute and retain:

- Hill position and velocity components;
- range and signed range rate;
- maneuver times, vectors, and achieved rather than only commanded delta-v;
- minimum separation and time of closest approach;
- terminal position/velocity error;
- pointing error during sensor or burn windows;
- constraint margins, not only Boolean violations.

For higher fidelity add:

- navigation error and covariance consistency;
- line-of-sight angle/rate and sensor field-of-view margin;
- thrust duration, duty cycle, minimum impulse bit, and propellant;
- wheel momentum, attitude settling, and burn-pointing error;
- collision probability or uncertainty-set clearance;
- communication age, availability, and lost-message statistics;
- abort success from every required hold/approach state.

## Staged RPO development roadmap

Keep each stage executable as a regression baseline.

1. **Relative-frame unit tests** — hand-check `rv2hill`/`hill2rv`, axis signs, velocity definition, and round trips.
2. **Passive nonlinear truth** — two point-mass spacecraft; compare with CW over one orbit and map the validity region.
3. **Ideal targeting** — instantaneous impulses, delta-v accounting, hold-point error, and keep-out monitoring.
4. **Physical actuation** — attitude guidance, thruster allocation, finite burn, tank coupling, and execution timing.
5. **Relative sensing/navigation** — generate measurements, estimate relative state/covariance, and remove truth from guidance inputs.
6. **Closed-loop translational GNC** — guidance updates from estimated state, command constraints, and maneuver replanning.
7. **Mission modes and safety** — coast, acquire, hold, approach, retreat, abort, safing, and explicit event/task transitions.
8. **Uncertainty campaign** — injection, sensor, thrust, timing, mass-property, and environment dispersions with reproducible seeds.
9. **Distributed cooperation** — explicit communication products, delays, dropouts, clocks, and local information sets.
10. **Contact/servicing fidelity** — only if the mission question includes docking, manipulation, or contact; then study [scenarioSimpleDocking.py](../examples/mujoco/scenarioSimpleDocking.py) and [scenarioArmWithThrusters.py](../examples/mujoco/scenarioArmWithThrusters.py).

## Architecture selection rules

Use a standalone scenario when the system is fixed, the message graph is still being learned, or the main product is one engineering study.

Move to `BskSim`-style model/scenario separation when several scenarios reuse the same dynamics and FSW, or when task-based mode switching is itself part of the experiment.

Move to `MultiSatBskSim`-style indexed models when vehicle count varies, per-satellite FSW must scale, environment services are shared intentionally, and unique message/model ownership is testable.

For production internal tools, borrow the principles—explicit ownership, rates, interfaces, and configuration—from these examples. Do not assume the example base classes themselves are stable production APIs.
