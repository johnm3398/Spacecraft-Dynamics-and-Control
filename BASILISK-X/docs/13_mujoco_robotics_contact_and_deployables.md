> **Draft status — AI-generated:** This page was drafted by OpenAI Codex on 2026-08-19. It has not yet been technically vetted by the BASILISK-X repository owner. The repository owner will review, correct, and maintain it. Verify APIs, units, frames, assumptions, and version compatibility before engineering use.

# MuJoCo, Robotics, Contact, and Deployables

## Why Basilisk has two multibody mental models

Basilisk's ordinary `Spacecraft` model is a hub-centred spacecraft dynamics architecture. Its StateEffectors and DynamicEffectors are designed for familiar spacecraft subsystems: wheels, hinged bodies, tanks/slosh, flexible elements, thrusters, and environmental forces. That architecture is usually the clearest and best-validated choice for a conventional vehicle.

`mujoco.MJScene` is an alternative Basilisk `DynamicObject`. It delegates a general tree of rigid bodies, joints, collision geometries, sites, actuators, and constraints to MuJoCo while retaining Basilisk scheduling, messaging, environment models, recorders, FSW algorithms, and Vizard interfaces around it. It exists for topology and interaction problems that are awkward to express as a hub plus specialised effectors.

The upstream [Basilisk MuJoCo dynamics guide](https://avslab.github.io/basilisk/Learn/makingModules/advancedTopics/mujocoDynObject.html) labels this support **beta/work in progress**, says extensive validation has not been performed, and warns that features and APIs may change. This is not boilerplate: the copied local examples contain development-era APIs that do not all match the pinned 2.11.1 package.

Labels in this chapter:

- **Observed locally** identifies checked-in code or a smoke check against `basiliskx_env` on 2026-08-19.
- **Engineering recommendation** is BASILISK-X guidance inferred from that evidence.
- **Unverified** means an example illustrates an architecture but was not established as an engineering-valid model.

## Standard `Spacecraft` versus `MJScene`

```text
STANDARD BASILISK                         MUJOCO

Spacecraft hub                            MJScene DynamicObject
  + StateEffectors                          + body tree / free bodies
  + DynamicEffectors                        + hinge/slide/free joints
  + gravity effectors                       + geoms and collision
  + registered states                       + sites / local frames
                                            + actuators
                                            + equalities/constraints

specialised spacecraft coupling           general multibody solver/topology
well matched to normal 6-DOF missions      well matched to mechanisms/contact
```

| Engineering need | Prefer standard `Spacecraft` | Consider `MJScene` |
|---|---:|---:|
| Rigid spacecraft and orbit propagation | yes | unnecessary |
| Reaction wheels/thrusters/tanks with known Basilisk models | yes | only for a topology study |
| One or several simple hinges/flexible appendages | usually | when joint network/contact exceeds effector model |
| Arbitrary branched robotic mechanism | difficult | yes |
| Many general revolute/prismatic joints | difficult | yes |
| Collision/contact/surface interaction | not its normal abstraction | yes |
| Docking capture constraint | custom coupling required | yes, but validate contact/capture model |
| Multiple non-contacting spacecraft | separate `Spacecraft` objects are simpler | usually unnecessary |
| Coupled bodies that must share one contact solver | no | one common `MJScene` |

**Engineering recommendation:** move to MuJoCo because a stated topology, collision, or constraint requirement demands it. "More bodies" or "more fidelity" alone is not a reason. MuJoCo replaces one model contract with another; it does not automatically add accurate gravity, actuators, sensors, material properties, or spacecraft FSW.

## 1. The `MJScene` object model

The first local example to read is [`scenarioReactionWheel.py`](../examples/mujoco/scenarioReactionWheel.py), backed by [`sat_w_wheel.xml`](../examples/mujoco/sat_w_wheel.xml). Its MJCF hierarchy is conceptually:

```text
world
`-- hub body
    |-- free joint             -> six free-flight coordinates
    |-- box geom               -> mass/inertia + visual/collision shape
    `-- wheel body
        |-- cylinder geom
        `-- hinge joint        -> wheel spin coordinate

actuator: scalar motor -> wheel hinge
```

The essential MJCF concepts are:

| XML/Python object | Physical/software meaning | Local use |
|---|---|---|
| `<body>` / `MJBody` | rigid body whose origin is defined relative to its parent | hubs, panels, wheels, tanks, arms |
| `<freejoint>` | six-DOF body relative to the world | one spacecraft or several independent vehicles |
| hinge/slide joint / `MJScalarJoint` | relative degree of freedom with state and rate messages | wheels, deployable panels, robot-arm joints |
| `<geom>` | shape contributing visual, inertial, and/or collision properties according to attributes | bus boxes, cylinders, panels, Itokawa mesh |
| `<site>` / `MJSite` | named body-fixed point/frame for state output, force application, docking, or sensors | panel centroids, thruster points, contact frames |
| `<actuator>` / `MJ*Actuator` | maps a command into joint or site force/torque coordinates | wheel motors, thrusters, joint motors, drag/SRP adapters |
| `<equality>` | solver-enforced kinematic/dynamic constraint | manually activated docking weld |
| contact geometry | collision detection plus contact constraint solution | asteroid landing; explicitly disabled in simple docking |

A site's `stateOutMsg` is `SCStatesMsg`-like and carries inertial position, velocity, attitude, and rate of that frame. Every body provides origin and centre-of-mass sites, and arbitrary named sites can be retrieved with `getSite`. This is the primary bridge from MuJoCo kinematics into ordinary Basilisk sensor, navigation, force, recorder, and visualization modules.

Actuation can be declared in XML and retrieved with `getSingleActuator`, or added from Python after retrieving a joint/site. The installed API exposes scalar `MJSingleActuator`, vector `MJForceActuator`, `MJTorqueActuator`, and `MJForceTorqueActuator` classes. A scalar actuator subscribes to `SingleActuatorMsg`; vector adapters use the corresponding force/torque messages. Direction and sign depend on the joint axis, site frame, and MJCF `gear`, so a plausible animation is not a sign-convention test.

### Initialization contract

The examples create and configure the scene, add it to a Basilisk task, call `InitializeSimulation()`, and only then set free-body and joint initial states with methods such as `setPosition`, `setVelocity`, `setAttitude`, `setAttitudeRate`, and joint `setPosition`/`setVelocity`. [`scenarioDeployPanels.py`](../examples/mujoco/scenarioDeployPanels.py) calls this order out explicitly.

Do not assume MJCF defaults share the same attitude, joint-angle, or frame convention as another Basilisk model. Record whether an XML angle is interpreted in degrees or radians by its compiler settings, while runtime Basilisk joint states and rates should be verified against the installed API and example outputs.

## 2. Two scheduling levels and the integrator-stage trap

An `MJScene` is scheduled in a normal Basilisk task, but it owns an internal dynamics task evaluated during integration.

```text
TOP-LEVEL BASILISK TASK at t_k

    MJScene.UpdateState(t_k)
       |
       +-- integrator stage 1 at provisional (t, x)
       |      forward kinematics
       |      internal dynamics-task models
       |      assemble forces / state derivatives
       |
       +-- integrator stage 2 ...
       +-- adaptive/retried stages ...
       `-- commit x(t_k)

    ordinary task models and recorders
```

`scene.AddModelToDynamicsTask(model, priority)` does **not** mean "run once per top-level task tick." It calls the model at each integrator sub-step. With an adaptive integrator, calls can occur at provisional times, repeat, or be rejected. Models used here normally compute state-dependent force/torque or environment quantities and must not treat `UpdateState` as a monotonic discrete-time callback.

Examples include:

- state-dependent SRP in [`scenarioSRPInPanels.py`](../examples/mujoco/scenarioSRPInPanels.py);
- joint reference interpolators and an analog controller in [`scenarioDeployPanels.py`](../examples/mujoco/scenarioDeployPanels.py);
- atmosphere, drag, orbital-element conversion, and formation force control in [`scenarioFormationFlyingWithDrag.py`](../examples/mujoco/scenarioFormationFlyingWithDrag.py);
- ephemeris and gravity in [`scenarioMJEarthMoonGravity.py`](../examples/mujoco/scenarioMJEarthMoonGravity.py).

For ordinary `SysModel` callbacks, outputs should be a memoryless function of current inputs, stage state, and stage time. The special `StatefulSysModel` pattern in `scenarioDeployPanels.py` registers the PID integral error as a scene-integrated state; its derivative participates in the same ODE rather than being accumulated by assuming fixed callback intervals.

### Priorities inside the dynamics task

Higher numeric priority executes first. The Earth-Moon source documents this sequence:

```text
forward kinematics (implicit priority 10000)
        -> ephemeris (75)
        -> NBodyGravity (factory default -1)
```

This ordering ensures the current provisional body state and current planetary states exist before gravity is evaluated. `scenarioDeployPanels.py` similarly schedules position interpolation at 50, velocity interpolation at 49, and control at 25. Insertion order in the Python file is not a substitute for explicit priority when one output feeds another.

### `extraEoMCall`

After the integrator commits its final state, a dynamics-task message may still contain values from the last provisional stage. Setting:

```text
scene.extraEoMCall = True
```

asks `MJScene` to perform another equations-of-motion evaluation at the committed top-level time. It refreshes forward kinematics and dynamics-task outputs without advancing the integrated state. This is why the panel and SRP examples put recorders in the **top-level** task and enable `extraEoMCall`, rather than recording every provisional stage.

```text
integrator stages -> commit state -> extra EOM evaluation -> top-level recorders/Vizard
```

This flag is an output-coherency mechanism, not a second physical integration step. It also means dynamics-task models must tolerate an evaluation whose purpose is message refresh.

### Stochastic dynamics

`AddModelToDiffusionDynamicsTask` is the stochastic counterpart for the diffusion term in an SDE. [`scenarioStochasticDrag.py`](../examples/mujoco/scenarioStochasticDrag.py) places an atmospheric-density state model in both drift and diffusion tasks and selects a stochastic integrator. This is a specialised continuous stochastic-state contract, not a substitute for ordinary Monte Carlo parameter dispersion.

## 3. Connecting normal Basilisk environment and FSW

MuJoCo supplies plant dynamics and kinematics. The surrounding spacecraft stack still communicates through Basilisk messages.

### Standard attitude FSW adapter

[`scenarioAttitudeFeedbackRWMuJoCo.py`](../examples/mujoco/scenarioAttitudeFeedbackRWMuJoCo.py) is the clearest bridge:

```text
MJ hub centre-of-mass stateOutMsg
              |
              v
          SimpleNav
              |
inertial3D -> attTrackingError -> mrpFeedback -> rwMotorTorque
                                                   |
                                                   v
                                array torque to scalar adapters
                                                   |
                                      saturation per wheel
                                                   |
                                                   v
                                      MuJoCo joint motors
```

`scalarJointStatesToRWSpeed` converts MuJoCo wheel rates into the standard `RWSpeedMsg`; `arrayMotorTorqueToSingleActuators` converts the FSW wheel array command to one scalar message per MuJoCo motor; `saturationSingleActuator` provides explicit command limits. The example demonstrates interface reuse, not identity between a MuJoCo wheel and `ReactionWheelStateEffector`. Inertia, friction, speed limits, torque saturation, sign, and momentum conservation still need comparison.

### Gravity and ephemerides

`simIncludeGravBody.gravBodyFactory().addBodiesTo(scene)` creates the MuJoCo-compatible `NBodyGravity` model and registers scene bodies as gravity targets. [`scenarioAttitudeFeedbackRWMuJoCo.py`](../examples/mujoco/scenarioAttitudeFeedbackRWMuJoCo.py) uses central Earth gravity. [`scenarioMJEarthMoonGravity.py`](../examples/mujoco/scenarioMJEarthMoonGravity.py) moves SPICE or analytic ephemeris execution into the internal dynamics task so gravity sees planet states at each integrator stage.

Gravity is not implied by `<option gravity="0 0 0"/>`; most local XML files explicitly disable MuJoCo's uniform world gravity because orbital gravity should come from Basilisk. Conversely, [`scenarioAsteroidLanding.py`](../examples/mujoco/scenarioAsteroidLanding.py) intentionally applies a constant `-200 N` inertial force through a site actuator. That is a contact demonstration near a surface, not an Itokawa gravity model.

### State-dependent environmental forces

The reusable adapter pattern is:

```text
MJ body/site stateOutMsg -> Basilisk environment/force model
                                      |
                                      v
                              force/torque message
                                      |
                                      v
                     MJ site force/torque actuator
```

[`scenarioSRPInPanels.py`](../examples/mujoco/scenarioSRPInPanels.py) uses panel-centroid site attitudes to compute a simplified SRP scalar force. [`scenarioFormationFlyingWithDrag.py`](../examples/mujoco/scenarioFormationFlyingWithDrag.py) connects atmospheric density and body state to drag, then converts the command to a site force. These models run inside the dynamics task because force depends on provisional state.

### Vizard and mixed plant types

`vizSupport.enableUnityVisualization` accepts an `MJScene` and represents its bodies. [`scenarioMJSceneVizard.py`](../examples/mujoco/scenarioMJSceneVizard.py) places two independent MuJoCo scenes and a normal `Spacecraft` in one Basilisk task and one Vizard view. This proves coexistence and visualization plumbing; it does not dynamically couple the three plants.

If two objects must exchange contact or equality forces, put them in one `MJScene`. Two separately integrated `MJScene` objects cannot share MuJoCo's contact solver merely because they are in one Basilisk task. For standard multiple `Spacecraft` objects, `syncDynamicsIntegration` is a separate Basilisk pattern; the local MuJoCo examples do not demonstrate cross-scene synchronization or contact.

## 4. What each local example actually teaches

Read the examples in increasing order of modelling commitment.

| Order | Source | Reusable lesson | Important limit |
|---:|---|---|---|
| 1 | [`scenarioReactionWheel.py`](../examples/mujoco/scenarioReactionWheel.py), [`sat_w_wheel.xml`](../examples/mujoco/sat_w_wheel.xml) | scene/body/joint/site/actuator/message basics; angular-momentum check | ideal motor and simple body model |
| 2 | [`scenarioAttitudeFeedbackRWMuJoCo.py`](../examples/mujoco/scenarioAttitudeFeedbackRWMuJoCo.py) | adapters to standard navigation and attitude FSW | not a validated replacement for the standard RW effector |
| 3 | [`scenarioArmWithThrusters.py`](../examples/mujoco/scenarioArmWithThrusters.py), [`sat_w_deployable_thruster.xml`](../examples/mujoco/sat_w_deployable_thruster.xml) | branched arms, constrained joint profiles, forces at end-effector sites | joints are prescribed rather than sensor-driven robotic control |
| 4 | [`scenarioDeployPanels.py`](../examples/mujoco/scenarioDeployPanels.py), [`sat_w_deployable_panels.xml`](../examples/mujoco/sat_w_deployable_panels.xml) | continuous integrated PID, joint limits, internal-task priorities, committed-state logging | long idealized deployment; controller is an example model |
| 5 | [`scenarioBranchingPanels.py`](../examples/mujoco/scenarioBranchingPanels.py), [`sat_w_branching_panels.xml`](../examples/mujoco/sat_w_branching_panels.xml) | staged branching topology, constraints, controller composition | prescribed sequencing is not a deployment reliability model |
| 6 | [`scenarioUnbalancedThrusters.py`](../examples/mujoco/scenarioUnbalancedThrusters.py), [`sat_w_thrusters.xml`](../examples/mujoco/sat_w_thrusters.xml) | off-centre site forces and tank mass-property derivatives | simplified depletion/actuator physics |
| 7 | [`scenarioSRPInPanels.py`](../examples/mujoco/scenarioSRPInPanels.py) | state-dependent external force at each integrator stage | fixed Sun direction and simple panel SRP |
| 8 | [`scenarioSimpleDocking.py`](../examples/mujoco/scenarioSimpleDocking.py), [`sats_dock.xml`](../examples/mujoco/sats_dock.xml) | two free bodies and runtime activation of a weld equality | **contact is disabled; docking is manually declared** |
| 9 | [`scenarioAsteroidLanding.py`](../examples/mujoco/scenarioAsteroidLanding.py), [`sat_ast_landing.xml`](../examples/mujoco/sat_ast_landing.xml) | mesh collision, surface contact, vector force actuator, Vizard mesh adapter | constant inertial "gravity," no realistic asteroid field/soil/landing control |
| 10 | [`scenarioFormationFlyingWithDrag.py`](../examples/mujoco/scenarioFormationFlyingWithDrag.py) | two free bodies in one scene, gravity, atmosphere, drag, continuous formation force | MuJoCo is unnecessary for most non-contact formation studies |
| 11 | [`scenarioStochasticDrag.py`](../examples/mujoco/scenarioStochasticDrag.py) | drift/diffusion tasks and stochastic integrators | currently fails to import under the pinned package |
| 12 | [`scenarioMJEarthMoonGravity.py`](../examples/mujoco/scenarioMJEarthMoonGravity.py) | ephemeris priority, third-body gravity, `extraEoMCall` | analytic branch uses a newer property absent in 2.11.1 |
| 13 | [`scenarioMJSceneVizard.py`](../examples/mujoco/scenarioMJSceneVizard.py) | two MJScenes and one standard spacecraft sharing environment/visualization | visual coexistence is not force coupling |
| 14 | [`scenarioThrArmControl.py`](../examples/mujoco/scenarioThrArmControl.py), [`BSK_mujocoMasters.py`](../examples/mujoco/BSK_mujocoMasters.py), [`mujocoModels`](../examples/mujoco/mujocoModels/) | BSK-style dynamics/FSW split, outer-loop/joint/firing/coast events, arm allocation | high abstraction and local-import assumptions; study last |

## 5. Contact, capture, and docking are different claims

The distinction is critical for RPO engineering.

```text
approach navigation/control
        -> first geometric contact
        -> collision impulse / compliance / friction
        -> capture mechanism engages
        -> constrained docked stack
```

[`scenarioSimpleDocking.py`](../examples/mujoco/scenarioSimpleDocking.py) demonstrates only a scripted approach followed by a constraint transition. Its XML explicitly sets `<flag contact="disable"/>`. The Python script thrusts two bodies together, turns thrust off, then manually calls `scene.getEquality("dock").setActive(True)` on a weld between two sites. Later it manually disables the weld. There is no contact detection triggering capture, no impact impulse, compliance, rebound, friction, latch logic, sensor uncertainty, misalignment envelope, or abort guidance.

Calling this scenario "contact dynamics" would be incorrect. It is a useful **post-capture equality-constraint** example.

[`scenarioAsteroidLanding.py`](../examples/mujoco/scenarioAsteroidLanding.py) does exercise collision between spacecraft geometry and an Itokawa mesh loaded from [`dataForExamples/Itokawa`](../examples/dataForExamples/Itokawa/). It is the better starting point for contact mechanics, but its force field and commanded descent are intentionally artificial. A docking model that needs real impact and capture should combine collision-enabled geometries and calibrated contact properties with a separately defined latch/equality transition and explicit detection logic.

## 6. Application boundaries

### Robotic spacecraft and arms

MuJoCo is useful when arbitrary joint topology, end-effector sites, joint constraints, and forces transmitted through a mechanism are the question. Add encoder/sensor models, motor dynamics and limits, structural flexibility if relevant, and a robotics controller around the joint states. The local arm examples mostly prescribe joint trajectories or demonstrate allocation; they do not establish manipulator estimation, collision avoidance, flexible modes, backlash, gearbox friction, or contact-rich manipulation.

### Deployables

Use standard hinged rigid-body/flexible effectors when one or a few appendages and their coupling answer the question. Use MuJoCo when branching, hard stops, geometric interference, latch constraints, or many general joints drive the result. Deployment success requires more than reaching an angle: assess hinge torque margin, rate/impact at stops, latch loads, collision clearance, body attitude disturbance, flexible response, and failure cases.

### Docking and servicing

MuJoCo becomes relevant when geometry, impact, constraints, or manipulation affect loads and capture. Before contact, standard Basilisk multi-spacecraft truth/navigation/GNC is usually simpler. A practical hybrid study can use standard models for far- and mid-field RPO and introduce a validated common MuJoCo scene for terminal contact, but state/frame transfer at that boundary must conserve position, velocity, attitude, rate, mass properties, and time.

### Landing and surface interaction

MuJoCo provides mesh collision and contact solution, not the celestial-body model. Shape scale/orientation, gravity, body rotation, regolith/contact parameters, foot compliance, thrusters, navigation, and terrain uncertainty remain mission inputs. Validate contact parameters against experiments or a trusted model before interpreting loads or stability.

### Formation flying

Ordinary separate `Spacecraft` objects are preferable for phasing, relative orbit design, differential drag, station keeping, communications, and constellation problems without mechanical interaction. The MuJoCo formation example is informative because it shows multiple free bodies and force adapters in one scene, but using a contact-capable multibody engine does not improve orbital fidelity by itself.

## 7. Fidelity and validation plan

### Mechanical model checks

Before adding control or environment:

1. Extract total mass, centre of mass, inertia, joint axes, limits, and zero configurations; compare them with the intended engineering model.
2. Apply no external force/torque and check linear/angular momentum conservation.
3. Exercise each actuator independently; verify sign, frame, lever arm, equal-and-opposite internal reactions, saturation, and work/energy.
4. Check joint units and orientation at representative poses, not only the initial pose.
5. Compare a one-joint or one-wheel case with an analytic solution and, where possible, the equivalent standard Basilisk effector.

MJCF `geom` density/size can silently define mass and inertia. A visually correct model can therefore have the wrong plant. Conversely, a visual mesh and a collision/inertial representation may need different geometries.

### Numerical checks

- Refine top-level step and integrator tolerances until mission metrics converge.
- Inspect whether contact duration and the fastest joint/control dynamics are resolved.
- Remember that adaptive integrator stage callbacks are not uniformly spaced or guaranteed monotonic.
- Keep top-level recorders after `MJScene`; use `extraEoMCall` when recording state-dependent internal-task outputs.
- Re-test after changing solver, contact parameters, joint limits, or topology; numerical contact parameters are part of the physical model.

### Contact checks

Contact requires documented geometry, friction, normal compliance/solver parameters, restitution/damping behaviour, timestep sensitivity, penetration, impulse/load outputs, and transition logic. Validate simple cases first: single-body drop, normal impact, oblique friction case, then multi-point capture. A manually activated weld must be reported as a commanded constraint, not discovered capture.

### Closed-loop checks

Validate the adapter seams independently:

| Seam | Required check |
|---|---|
| MJ site state -> navigation | origin, frame direction, MRP convention, angular-rate expression, timestamp |
| FSW torque -> MJ actuator | sign, body/joint axes, scalar ordering, saturation |
| environment force -> site | inertial/body/site frame transformation and lever arm |
| joint state -> sensor/FSW | unit, zero, direction, latency/noise |
| multiple plants -> Vizard | visualization identity versus actual dynamical coupling |

## 8. Local version and execution caveats

### What was confirmed

The repository pins Basilisk 2.11.1. `Basilisk.simulation.mujoco` imports in `basiliskx_env`, and [`scenarioReactionWheel.py`](../examples/mujoco/scenarioReactionWheel.py) completed its headless `run(False)` smoke test during this audit. This proves a minimal local execution path, not validation of the MuJoCo subsystem.

### Confirmed mismatches

- [`scenarioStochasticDrag.py`](../examples/mujoco/scenarioStochasticDrag.py) imports both `MJStochasticAtmDensity` and `MJIgbmAtmDensity`. `MJIgbmAtmDensity` is absent from the installed 2.11.1 package, so the file fails at import even when its default `useIgbm=False` branch would not use that class.
- [`scenarioMJEarthMoonGravity.py`](../examples/mujoco/scenarioMJEarthMoonGravity.py) is dated 2026 and corresponds to development documentation. Its analytic `run(False, useSpice=False)` branch fails locally when it assigns `PlanetEphemeris.zeroBase`; that property is absent from the installed 2.11.1 class. The SPICE branch was not established here as a substitute validation.
- [`scenarioThrArmControl.py`](../examples/mujoco/scenarioThrArmControl.py) uses bare imports of `BSK_mujocoMasters`, `BSK_MujocoDynamics`, and `BSK_MujocoFSW`. It imports when executed with the MuJoCo example directory on `sys.path`, but a generic import from elsewhere fails unless that path is arranged. This is packaging/path fragility rather than a MuJoCo physics failure.

The copied tree therefore does not correspond cleanly to one tagged Basilisk release. Select a version-matched example set before repairing any source. Record the Basilisk/MuJoCo versions and the exact XML/assets with every retained result.

### Example-specific simplifications

- Most XML files set uniform MuJoCo gravity to zero; environment forces must be wired explicitly.
- `scenarioSimpleDocking` disables contact and toggles a weld manually.
- `scenarioAsteroidLanding` uses constant inertial force instead of asteroid gravity.
- `scenarioSRPInPanels` assumes a fixed Sun direction and simple panel law.
- Several panel/arm examples prescribe or constrain joint motion, bypassing actuator/sensor dynamics.
- `scenarioFormationFlyingWithDrag` uses MuJoCo for an otherwise non-contact orbital control problem; it should not be the default formation architecture.
- Solver/contact/material parameter validation is not supplied by these tutorial scripts.
- MuJoCo support remains upstream-labelled beta, with API change and incomplete validation risk.

## 9. Engineering recommendation for BASILISK-X

Keep MuJoCo optional and behind explicit plant adapters. A reusable BASILISK-X boundary could eventually expose:

```text
named body/site state messages
named joint state messages
standard navigation adapters
standard force/torque command adapters
versioned MJCF/asset manifest
mechanical and conservation validation tests
```

Do not place mission-specific MJCF geometry, joint names, contact properties, or controller gains in a generic utility merely because multiple objects exist. Those are plant and experiment definitions until stable common contracts emerge.

For any new robotics/contact study, start with this decision sequence:

```text
What mechanical interaction changes the answer?
        |
        +-- none -> standard Spacecraft/multi-spacecraft model
        |
        `-- topology/contact/constraint matters
                  -> smallest MJCF plant
                  -> analytic/conservation validation
                  -> Basilisk environment adapters
                  -> sensor/FSW adapters
                  -> contact/capture validation
                  -> mission scenario
```

The strongest near-term learning exercise is to build the same three-wheel attitude-control case with standard `Spacecraft` and `MJScene`, then compare mass/inertia, momentum, commanded torque, attitude response, numerical convergence, runtime, and modelling effort. Only after that baseline should BASILISK-X use MuJoCo for a deployable, arm, docking, or landing research model.

## Source index

- MuJoCo example directory: [`examples/mujoco`](../examples/mujoco/)
- Introductory scene: [`scenarioReactionWheel.py`](../examples/mujoco/scenarioReactionWheel.py)
- Standard FSW bridge: [`scenarioAttitudeFeedbackRWMuJoCo.py`](../examples/mujoco/scenarioAttitudeFeedbackRWMuJoCo.py)
- Continuous joint dynamics: [`scenarioDeployPanels.py`](../examples/mujoco/scenarioDeployPanels.py)
- Environment at integration stages: [`scenarioSRPInPanels.py`](../examples/mujoco/scenarioSRPInPanels.py), [`scenarioFormationFlyingWithDrag.py`](../examples/mujoco/scenarioFormationFlyingWithDrag.py), and [`scenarioMJEarthMoonGravity.py`](../examples/mujoco/scenarioMJEarthMoonGravity.py)
- Constraint-only docking: [`scenarioSimpleDocking.py`](../examples/mujoco/scenarioSimpleDocking.py) and [`sats_dock.xml`](../examples/mujoco/sats_dock.xml)
- Collision/landing: [`scenarioAsteroidLanding.py`](../examples/mujoco/scenarioAsteroidLanding.py) and [`sat_ast_landing.xml`](../examples/mujoco/sat_ast_landing.xml)
- BSK-style wrapper: [`BSK_mujocoMasters.py`](../examples/mujoco/BSK_mujocoMasters.py), [`BSK_MujocoDynamics.py`](../examples/mujoco/mujocoModels/BSK_MujocoDynamics.py), and [`BSK_MujocoFSW.py`](../examples/mujoco/mujocoModels/BSK_MujocoFSW.py)
- Official references: [Basilisk multi-body dynamics with MuJoCo](https://avslab.github.io/basilisk/Learn/makingModules/advancedTopics/mujocoDynObject.html), [MuJoCo XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html), and [development `MJScene` API](https://avslab.github.io/basilisk/develop/Documentation/simulation/mujocoDynamics/_GeneralModuleFiles/MJScene.html)
