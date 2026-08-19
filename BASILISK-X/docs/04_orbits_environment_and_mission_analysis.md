> **Draft status — AI-generated:** This page was drafted by OpenAI Codex on 2026-08-19. It has not yet been technically vetted by the BASILISK-X repository owner. The repository owner will review, correct, and maintain it. Verify APIs, units, frames, assumptions, and version compatibility before engineering use.

# Orbits, environment, and mission analysis

Orbit propagation is useful only when it is matched to an engineering question. A more elaborate force model is not automatically a better simulation: it can add uncertain inputs, longer run time, and new failure modes without changing the decision being made.

This chapter develops a question-driven workflow and maps it to the examples in this repository.

## Evidence and recommendations

Statements labelled **Observed in this repository** describe local source behavior. Statements labelled **Engineering recommendation** are proposed BASILISK-X practices inferred from those sources. The local examples include material from different Basilisk development periods, so verify each named API against the installed version.

## Start from the decision, not the propagator

An experienced workflow is:

```text
engineering decision
      |
      v
metric and required accuracy
      |
      v
states that determine the metric
      |
      v
minimum environment and force model
      |
      v
navigation, FSW, and actuator fidelity if relevant
      |
      v
numerical accuracy and sampling strategy
      |
      v
validation case and uncertainty study
```

For example, a first estimate of Hohmann transfer time does not require Earth orientation, atmospheric density, reaction wheels, or a camera. A month-long differential-drag rendezvous study may require all of Earth rotation, atmospheric and attitude geometry, relative states, actuator/pointing limitations, and density uncertainty.

### Mission-analysis worksheet

Before selecting modules, write down:

| Question | What must be specified |
|---|---|
| Decision | What choice will this simulation support? |
| Metric | Terminal position, eclipse fraction, fuel, revisit time, probability of success, etc. |
| Accuracy | How much error changes the decision? |
| Time horizon | Seconds, orbits, days, or mission life |
| Truth states | Translation only, 6-DOF attitude, mass, articulation, resources, multiple vehicles |
| Environment | Central gravity, harmonics, third bodies, atmosphere, SRP, eclipse, magnetic field |
| Execution | Ideal impulse, finite thrust, closed-loop actuator, or no maneuver |
| Outputs | Which messages and derived quantities establish success? |
| Validation | Analytic solution, independent propagator, conservation law, or convergence study |

## Minimum orbit state and baseline propagation

For point-mass orbit analysis, the essential truth state is inertial position and velocity:

```text
x_orbit = [r_CN_N, v_CN_N]
```

The normal local setup is:

```python
gravity_factory = simIncludeGravBody.gravBodyFactory()
earth = gravity_factory.createEarth()
earth.isCentralBody = True
gravity_factory.addBodiesTo(spacecraft_object)

spacecraft_object.hub.r_CN_NInit = r_initial_m
spacecraft_object.hub.v_CN_NInit = v_initial_m_s
```

[`scenarioBasicOrbit.py`](../examples/scenarioBasicOrbit.py) is the canonical starting point. It constructs `spacecraft.Spacecraft`, attaches gravity, converts classical elements with `orbitalMotion.elem2rv()`, records `scStateOutMsg`, and derives orbital energy and angular momentum.

Although several scenario descriptions call these cases “3-DOF,” they still instantiate the six-degree-of-freedom `Spacecraft` object. The intended simplification is that attitude dynamics do not affect the orbital result.

Useful baseline metrics are:

```text
specific mechanical energy       epsilon = |v|^2 / 2 - mu / |r|
specific angular momentum        h = r x v
osculating elements              (a, e, i, Omega, omega, f)
position/velocity error           delta-r(t), delta-v(t)
period or event-time error        delta-t
```

**Engineering recommendation:** make a two-body case the first executable validation for every new orbit-analysis architecture. Preserve it as a regression case after perturbations are added.

## Central-body choice and coordinate interpretation

`isCentralBody` affects the origin relative to which spacecraft translation is integrated and reported. It is not merely a label saying which body has the largest gravity.

[`scenarioCentralBody.py`](../examples/scenarioCentralBody.py) compares:

- Earth selected as the central body, giving Earth-relative spacecraft states; and
- Earth not selected as central, requiring the planet ephemeris to interpret inertial spacecraft states.

```text
central body selected
    spacecraft state is convenient for planet-centred orbit analysis

no central body selected
    spacecraft and planet states share the broader inertial origin
    planet-relative state must be formed explicitly
```

This matters for heliocentric transfers, moon/planet systems, SPICE comparisons, ground geometry, and patched-conic segments.

**Engineering recommendation:** state the inertial origin, central body, orientation frame, and epoch at the top of every mission-analysis configuration. A plot labelled only “ECI position” is insufficient when several bodies or epochs are involved.

## Gravity fidelity

### Point-mass gravity

Use point-mass gravity when the question is dominated by gross orbit geometry, transfer timing, initial phasing, or software architecture. It is cheap, interpretable, and easy to validate analytically.

Best local template: [`scenarioBasicOrbit.py`](../examples/scenarioBasicOrbit.py).

### Multiple gravitating bodies

[`scenarioOrbitMultiBody.py`](../examples/scenarioOrbitMultiBody.py) uses a SPICE interface to supply celestial-body locations, attaches several gravity bodies, and compares the propagated spacecraft trajectory with a SPICE spacecraft trajectory for selected cases.

```text
SPICE kernels and epoch
          |
          v
planet state messages --> gravity bodies --> Spacecraft acceleration
          |
          +-----------------------------> reference/analysis ephemeris
```

SPICE and gravity perform different jobs:

- SPICE supplies body or prescribed spacecraft ephemerides and orientations from kernels.
- The Basilisk gravity model computes acceleration from the configured bodies/field.
- Loading an ephemeris does not automatically add every physical perturbation.

Use multi-body gravity for long high-altitude arcs, lunar/planetary missions, flybys, and cases where third-body acceleration is comparable to the required error budget.

Related examples are [`scenarioJupiterArrival.py`](../examples/scenarioJupiterArrival.py), [`scenarioFlybySpice.py`](../examples/scenarioFlybySpice.py), [`scenarioHelioTransSpice.py`](../examples/scenarioHelioTransSpice.py), and [`scenarioPatchedConics.py`](../examples/scenarioPatchedConics.py).

### Spherical harmonics and planet orientation

The gravity-body interface can load a spherical-harmonic field:

```python
earth.useSphericalHarmonicsGravityModel(gravity_file, maximum_degree)
```

The degree/order required depends on altitude, duration, and metric. `J2` is often the first useful addition for secular node/perigee behavior in Earth orbit. Higher degree/order can matter for low-altitude, long-duration, or precision work.

[`scenarioOrbitConsistencyVerification.py`](../examples/scenarioOrbitConsistencyVerification.py) demonstrates a critical detail: tesseral and sectoral terms are defined in the rotating planet-fixed frame. Without an orientation message, evaluating them in a fixed inertial orientation produces spurious long-period/secular behavior. Zonal terms such as `J2` do not expose the same longitude error.

```text
SPICE planet orientation ----\
                              > spherical-harmonic gravity --> acceleration
planet-fixed coefficients ---/
```

Best comparisons:

- [`scenarioBasicOrbit.py`](../examples/scenarioBasicOrbit.py): point mass versus selected harmonic field;
- [`scenarioOrbitConsistencyVerification.py`](../examples/scenarioOrbitConsistencyVerification.py): rotating versus incorrectly non-rotating high-order field;
- [`scenarioGroundDownlink.py`](../examples/scenarioGroundDownlink.py): `J2`, epoch, and rotating-ground geometry in a mission context.

### Custom and small-body gravity

[`scenarioCustomGravBody.py`](../examples/scenarioCustomGravBody.py) shows `createCustomGravObject()` with user-supplied gravitational parameter and radius. [`scenarioAsteroidArrival.py`](../examples/scenarioAsteroidArrival.py) and [`scenarioSmallBodyNavUKF.py`](../examples/scenarioSmallBodyNavUKF.py) extend the pattern to small-body mission and navigation contexts using higher-order gravity.

A custom point mass or spherical harmonic model is not automatically a faithful irregular-body field. Near-surface asteroid work may need a validated polyhedral, mascon, or other shape-dependent model not demonstrated by `scenarioCustomGravBody.py` alone.

## Ephemerides, epochs, and SPICE

Epoch-dependent models need a consistent physical time reference:

```python
spice = gravity_factory.createSpiceInterface(
    time="2029 June 12 05:30:30.0",
    epochInMsg=True,
)
spice.zeroBase = "Earth"
simulation.AddModelToTask(task_name, spice)
```

The scheduler still begins at elapsed simulation time zero. The configured epoch maps elapsed time to ephemeris time for SPICE, Earth orientation, magnetic models, and other consumers.

Record the following in reproducibility metadata:

- epoch string and time scale;
- loaded kernel names and versions;
- SPICE `zeroBase`;
- central-body selection;
- gravity file, degree, and order;
- any epoch message shared with environment models or Vizard.

**Observed in this repository:** `scenarioGroundTracks.py` and `scenarioSatelliteConstellation.py` use SPICE/epoch information to interpret rotating-planet ground tracks. `scenarioHelioTransSpice.py` and `scenarioFlybySpice.py` load the local `spacecraft_21T01.bsp` as a prescribed reference trajectory.

**Engineering recommendation:** do not call a SPICE trajectory “truth” without recording the kernel provenance and clarifying whether it is a reconstructed/prescribed trajectory or a force-model prediction.

## Non-gravitational environment

### Atmosphere, wind, and drag

The drag architecture is modular:

```text
spacecraft state + planet state + epoch/space weather
                       |
                       v
                 atmosphere model -----> density message
                       |                         |
                 optional wind                  v
                                           drag effector
                                      geometry/attitude/Cd/area
                                                 |
                                                 v
                                         spacecraft force/torque
```

[`scenarioDragDeorbit.py`](../examples/scenarioDragDeorbit.py) compares `ExponentialAtmosphere` and `MsisAtmosphere`, connects density to `DragDynamicEffector`, and optionally uses a wind model. [`scenarioDragRendezvous.py`](../examples/scenarioDragRendezvous.py) uses faceted drag and attitude to produce differential drag. [`scenarioStochasticDragSpacecraft.py`](../examples/scenarioStochasticDragSpacecraft.py) demonstrates stochastic atmospheric forcing, but its `igbmNoiseStateEffector` import is absent from the pinned Basilisk 2.11.1 installation and the file is not runnable unchanged here. [`scenarioAerocapture.py`](../examples/scenarioAerocapture.py) uses tabular atmosphere and aerodynamic deceleration in a short high-dynamic arc.

Increasing atmospheric model sophistication is not useful if ballistic coefficient, projected area, attitude history, or space-weather inputs dominate uncertainty.

### Solar radiation pressure

The examples show two useful levels:

- a cannonball-style `radiationPressure.RadiationPressure` in [`scenarioSmallBodyNav.py`](../examples/scenarioSmallBodyNav.py);
- attitude- and geometry-dependent `FacetSRPDynamicEffector` in [`scenarioSepMomentumManagement.py`](../examples/scenarioSepMomentumManagement.py).

Use a faceted model when torque, articulated geometry, or attitude-dependent acceleration is part of the question. Use a simpler area/reflectivity model for first-order translational studies.

### Eclipse and albedo

`eclipse.Eclipse` computes an illumination factor from Sun, occulting-body, and spacecraft geometry. Downstream consumers must explicitly subscribe to its output.

[`scenarioPowerDemo.py`](../examples/scenarioPowerDemo.py) connects eclipse to solar-panel generation and battery state. [`scenarioCSS.py`](../examples/scenarioCSS.py) demonstrates eclipse as an optional coarse-sun-sensor input. [`scenarioAlbedo.py`](../examples/scenarioAlbedo.py) compares average/data-driven planetary albedo and optional eclipse treatment.

An eclipse module alone does not create a power constraint, thermal response, or mission-mode change. Those consequences require the corresponding resource and mission-logic models.

### Magnetic environment

[`scenarioMagneticFieldCenteredDipole.py`](../examples/scenarioMagneticFieldCenteredDipole.py) supplies a simple centred-dipole field. [`scenarioMagneticFieldWMM.py`](../examples/scenarioMagneticFieldWMM.py) uses WMM coefficients and requires an epoch. [`scenarioTAMcomparison.py`](../examples/scenarioTAMcomparison.py) compares their downstream magnetometer implications.

A field output does not produce spacecraft torque until it is connected to a magnetic dipole/actuator or disturbance model. [`scenarioMtbMomentumManagementSimple.py`](../examples/scenarioMtbMomentumManagementSimple.py) provides that larger chain.

## Physical fidelity versus numerical fidelity

These are independent questions:

```text
physical-model error                 numerical error
--------------------                 ---------------
missing J2 or drag                   step too large
wrong atmosphere                     inappropriate integrator
wrong attitude/area                  loose adaptive tolerance
incorrect planet orientation         unsynchronised coupled dynamics
```

[`scenarioIntegrators.py`](../examples/scenarioIntegrators.py) compares Euler, RK2, default RK4, RKF45, RKF78, and custom Runge–Kutta forms. [`scenarioVariableTimeStepIntegrators.py`](../examples/scenarioVariableTimeStepIntegrators.py) exposes relative/absolute tolerance choices. [`scenarioIntegratorsComparison.py`](../examples/scenarioIntegratorsComparison.py) compares accuracy and cost.

For fixed-step dynamics, the task period is also the nominal integration interval. Adaptive integrators may take internal stages/steps, but messages and recorders remain governed by the simulation scheduler. Fast internal dynamics such as slosh or flexible modes can force a much smaller step than the orbit alone requires.

**Engineering recommendation:** perform two separate convergence exercises:

1. **Numerical convergence:** hold physics fixed and reduce step/tolerances until the mission metric stabilizes.
2. **Model convergence:** hold numerical error acceptably small and add one physical effect at a time.

Never use agreement between two runs with the same missing physics as validation.

## A practical fidelity ladder

| Level | Include | Appropriate questions |
|---:|---|---|
| 0 | Analytic two-body or relative-motion equations outside/alongside Basilisk | Hand checks, rough transfer and phasing design |
| 1 | `Spacecraft` plus point-mass central gravity | Short propagation, scheduler learning, gross geometry |
| 2 | `J2` or selected harmonics with correct planet orientation | Secular LEO behavior, ground-track drift, formation sensitivity |
| 3 | SPICE third bodies, SRP, eclipse | High-altitude/long arcs, lunar/interplanetary geometry, power opportunities |
| 4 | Atmosphere/wind, attitude-dependent facets, stochastic inputs | Drag lifetime, differential drag, precision low orbit |
| 5 | Navigation, closed-loop FSW, physical actuators, mass/resources | Executability, station keeping, autonomous operations |
| 6 | Dispersions and Monte Carlo | Robustness and success probability |

Move upward only when the current level cannot answer the metric within the required error.

## Translating common mission questions

| Mission question | Required states/environment | Outputs and metrics | Best local starting points | Important caveat |
|---|---|---|---|---|
| Orbit propagation | `r,v`, central gravity; add attitude only if forces depend on it | element histories, energy/angular momentum, terminal error | `scenarioBasicOrbit.py`, `scenarioCentralBody.py` | “3-DOF” examples still instantiate `Spacecraft` |
| Perturbation budget | same ICs across controlled force-model increments | delta-elements, secular rates, along-track error | `scenarioOrbitConsistencyVerification.py`, `scenarioDragDeorbit.py` | isolate one perturbation before combining |
| Phasing | chief/deputy `r,v`, relative frame, maneuver epochs | phase angle, Hill state, time/fuel to target | [`scenarioFormationBasic.py`](../examples/scenarioFormationBasic.py), [`scenarioFormationReconfig.py`](../examples/scenarioFormationReconfig.py); BASILISK-X [`cooperative_geo_rendezvous.py`](../scenarios/cooperative_geo_rendezvous/cooperative_geo_rendezvous.py) | CW/HCW targeting and ideal impulses have limited validity |
| Ideal transfer | pre/post-burn `r,v`, point mass | analytic versus propagated transfer time and terminal orbit | `scenarioOrbitManeuver.py`, `scenarioHohmann.py` | direct state change bypasses thrust, attitude, mass, and errors |
| Finite burn | 6-DOF state, pointing, thruster force and command timing | delivered delta-v, burn loss, pointing error, terminal miss | `scenarioOrbitManeuverTH.py`, `scenarioFormationReconfig.py` | Hohmann finite-burn example assumes constant mass |
| Low thrust | long-duration `r,v`, mass, thrust direction, power/eclipses | delivered impulse, propellant, power feasibility, terminal state | `scenarioSepMomentumManagement.py` for articulated SEP mechanics | it is momentum-management research, not a general low-thrust optimizer |
| Deployment/separation | multiple 6-DOF states, interface geometry, separation impulse/force | minimum clearance, relative orbit, tip-off rates | `scenarioFormationBasic.py` for multiple truth objects; `scenarioOrbitManeuver.py` for an ideal state jump | no dedicated end-to-end deployment/separation-device example was found |
| Station keeping | relative state/OE, perturbations, estimator, thrusters, fuel/power | dead-band violations, burn count, fuel, duty cycle | `MultiSatBskSim/scenariosMultiSat/scenario_StationKeepingMultiSat.py` | local relative navigation is idealized and key setup is three-spacecraft-specific |
| Ground access | orbit, rotating planet/epoch, site location, elevation/range limits | access windows, duration, range/elevation, revisit | `scenarioGroundMapping.py`, `scenarioGroundDownlink.py` | access geometry is not automatically an RF link budget |
| Intersatellite access | two 6-DOF states, antenna boresight/FOV/range | opportunity intervals and pointing margin | `scenarioSpacecraftLocation.py` | uses truth states and simplified instantaneous geometry |
| Eclipse/power | Sun/planet ephemerides, occultation, attitude, panel and battery | eclipse fraction, minimum charge, mode feasibility | `scenarioPowerDemo.py` | power affects operations only if mission logic consumes it |
| Constellation layout | list of spacecraft `r,v`, rotating Earth if ground metrics matter | coverage/revisit, plane spacing, access statistics | `scenarioSatelliteConstellation.py`, `MultiSatBskSim/scenariosMultiSat/scenario_constellationFromTle.py` | orbit visualization alone is not a coverage analysis |

## What to record

Record only data needed for verification and the engineering decision:

- `SCStatesMsg` position/velocity and, when force geometry matters, attitude/rate;
- relevant planet state/orientation and epoch provenance;
- force/acceleration contributions when building a perturbation budget;
- orbital elements and Hill-relative states as derived data with timestamps;
- maneuver commands, actual thrust, accumulated delta-v, and fuel mass;
- access/eclipses as event intervals rather than only dense Boolean traces;
- resource minima/maxima and constraint violations;
- terminal success metrics and the configuration that produced them.

Avoid logging every message at the fastest dynamics rate for a multi-day trade study. Preserve enough high-rate data around discontinuities such as burns, eclipse transitions, and close approach.

## Validation and stopping rules

A mission-analysis result should have at least one validation anchor:

| Model | Useful validation |
|---|---|
| Point mass | analytic conic, energy/angular momentum conservation |
| `J2` | analytic secular rate or trusted independent propagator |
| High-order gravity | independent tool with matching field, orientation, epoch, and truncation |
| Drag | ballistic-coefficient hand check, density history, model sensitivity |
| Finite burn | integrated thrust/mass delta-v and ideal-impulse limiting case |
| Relative motion | nonlinear truth compared with linear model inside its validity region |
| Access | independent geometry at selected epochs and boundary cases |
| Constellation | symmetry checks and independent site-access calculation |

Stop increasing fidelity when:

- the decision is unchanged across plausible model increments;
- numerical error is well below the engineering tolerance;
- a newly added model is less certain than its influence on the metric;
- required inputs cannot be justified or validated;
- computational cost prevents the uncertainty analysis that matters more.

## Important local caveats

- `scenarioOrbitManeuver.py` and `scenarioHohmann.py` directly modify translational state for ideal impulses; burn-pointing animation does not make the impulse attitude-dependent.
- `scenarioOrbitManeuverTH.py` models finite thrust and attitude but states a constant-mass burn-duration assumption.
- High-order rotating gravity material in `scenarioOrbitConsistencyVerification.py` is newer than much of the catalogue; verify compatibility with the pinned Basilisk version.
- Ground/access examples generally demonstrate geometry and simple data flow, not atmospheric attenuation, antenna gain, link margin, routing, or network latency.
- `scenarioSatelliteConstellation.py` demonstrates Walker initialization and propagation, not complete global-coverage optimization.
- `scenarioDragRendezvous.py` combines simplified atmosphere, a precomputed controller, truth-like navigation, and attitude-driven differential drag; it is a research template, not a validated operational design.
- `scenarioSepMomentumManagement.py` is a specialized articulated SEP momentum-management architecture. Do not treat it as the default low-thrust mission-analysis pattern.
- No dedicated deployment-dynamics or separation-device scenario was identified in the local example set.

## Recommended reading path

1. [`scenarioBasicOrbit.py`](../examples/scenarioBasicOrbit.py)
2. [`scenarioCentralBody.py`](../examples/scenarioCentralBody.py)
3. [`scenarioIntegrators.py`](../examples/scenarioIntegrators.py)
4. [`scenarioOrbitConsistencyVerification.py`](../examples/scenarioOrbitConsistencyVerification.py)
5. [`scenarioOrbitMultiBody.py`](../examples/scenarioOrbitMultiBody.py)
6. [`scenarioDragDeorbit.py`](../examples/scenarioDragDeorbit.py)
7. [`scenarioOrbitManeuver.py`](../examples/scenarioOrbitManeuver.py), then [`scenarioOrbitManeuverTH.py`](../examples/scenarioOrbitManeuverTH.py)
8. [`scenarioGroundDownlink.py`](../examples/scenarioGroundDownlink.py)
9. [`scenarioFormationReconfig.py`](../examples/scenarioFormationReconfig.py)
10. [`MultiSatBskSim/scenariosMultiSat/scenario_StationKeepingMultiSat.py`](../examples/MultiSatBskSim/scenariosMultiSat/scenario_StationKeepingMultiSat.py)
