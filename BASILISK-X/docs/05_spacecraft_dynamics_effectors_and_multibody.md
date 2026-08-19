> **Draft status — AI-generated:** This page was drafted by OpenAI Codex on 2026-08-19. It has not yet been technically vetted by the BASILISK-X repository owner. The repository owner will review, correct, and maintain it. Verify APIs, units, frames, assumptions, and version compatibility before engineering use.

# Spacecraft dynamics, effectors, and multibody systems

Basilisk's standard spacecraft model is deliberately **hub-centric**. A central rigid hub carries the vehicle's translational and attitude coordinates, while effectors add forces, torques, internal states, moving mass properties, and coupled momentum. This is an efficient and physically meaningful architecture for most spacecraft, but it is not the only multibody architecture in the repository.

## Evidence and recommendations

Statements labelled **Observed in this repository** describe local source behavior. Statements labelled **Engineering recommendation** are proposed BASILISK-X practices based on that evidence. Some of the newest standard-dynamics and MuJoCo examples may target a later development state than the pinned Basilisk runtime; verify availability before adopting their APIs.

## First-principles model of a spacecraft

A rigid spacecraft needs six configuration degrees of freedom:

```text
translation:  inertial position r_CN_N and velocity v_CN_N
rotation:     attitude sigma_BN and angular velocity omega_BN_B
```

The standard `spacecraft.Spacecraft` model adds all attached effectors to this hub description:

```text
                         Spacecraft
                             |
                    rigid hub properties
                  m, centre of mass, inertia
                             |
          +------------------+------------------+
          |                                     |
     StateEffectors                        DynamicEffectors
 internal coordinates,                 forces and torques from
 mass/momentum coupling                commands/environment/states
          |                                     |
          +------------------+------------------+
                             |
                 coupled equations of motion
                             |
                         integrator
                             |
                             v
                       SCStatesMsg
```

The important idea is that an attached component is part of the equations of motion, not merely a scheduled Python object.

## Hub configuration and truth outputs

A typical hub setup is:

```python
spacecraft_object = spacecraft.Spacecraft()
spacecraft_object.hub.mHub = mass_kg
spacecraft_object.hub.r_BcB_B = r_body_origin_to_hub_com_b_m
spacecraft_object.hub.IHubPntBc_B = inertia_about_hub_com_b_kg_m2

spacecraft_object.hub.r_CN_NInit = initial_com_position_n_m
spacecraft_object.hub.v_CN_NInit = initial_com_velocity_n_m_s
spacecraft_object.hub.sigma_BNInit = initial_mrp_bn
spacecraft_object.hub.omega_BN_BInit = initial_rate_b_rad_s
```

| Field | Physical role |
|---|---|
| `mHub` | rigid-hub mass, not automatically the total mass of attached state effectors |
| `r_BcB_B` | hub centre-of-mass offset from body-frame origin, expressed in B |
| `IHubPntBc_B` | hub inertia about the hub centre of mass, expressed in B |
| `r_CN_NInit`, `v_CN_NInit` | initial total-system centre-of-mass translation in N |
| `sigma_BNInit` | initial MRP attitude of B relative to N |
| `omega_BN_BInit` | initial angular velocity of B relative to N, expressed in B |

The output `scStateOutMsg` includes both body-origin quantities (`r_BN_N`, `v_BN_N`) and centre-of-mass quantities (`r_CN_N`, `v_CN_N`) along with attitude, angular rate, angular acceleration, accumulated delta-v, and non-conservative acceleration fields. When the body origin and centre of mass coincide, many examples make the distinction invisible; moving masses make it important.

**Engineering recommendation:** draw the body origin B, hub centre of mass Bc, and current total-system centre of mass C before assigning a nonzero offset. Validate a nonzero-offset case instead of relying on comments copied from an example.

## What the integrator actually couples

At an integration stage, the dynamic object conceptually performs work like:

```text
current hub and effector states
          |
          v
update effector mass properties and geometry
          |
          v
evaluate gravity, state-effector coupling, external forces/torques
          |
          v
solve coupled translational, rotational, and internal-state derivatives
          |
          v
integrator advances the complete registered state vector
```

An RK4 step evaluates derivatives at several internal stages even though the spacecraft's scheduled `UpdateState` is called at the task tick. A variable-step integrator can evaluate more stages. Force models that must participate at every stage need the proper dynamics/effector contract; putting a continuous force calculation only in a normal Python `UpdateState()` can leave it held across the integration interval.

Commands and sensor/FSW messages are normally sampled at their task rates and held while the dynamics integrator evaluates its stages. This is how a discrete controller drives continuous dynamics.

[`scenarioIntegrators.py`](../examples/scenarioIntegrators.py) shows `setIntegrator()`. [`scenarioHingedRigidBody.py`](../examples/scenarioHingedRigidBody.py) gives the practical warning that a step acceptable for orbit motion can be much too large for fast panel modes.

## State effectors versus dynamic effectors

### State effectors

A `StateEffector` adds or prescribes internal configuration and participates in coupled mass properties, momentum, energy, and state derivatives.

Typical examples include:

| Capability | Local module/pattern | Best examples |
|---|---|---|
| Reaction wheels | `ReactionWheelStateEffector` with wheel devices | [`scenarioAttitudeFeedbackRW.py`](../examples/scenarioAttitudeFeedbackRW.py) |
| Single hinged panel | `HingedRigidBodyStateEffector` | [`scenarioHingedRigidBody.py`](../examples/scenarioHingedRigidBody.py) |
| General articulated chain | `spinningBodyOneDOFStateEffector`, `spinningBodyTwoDOFStateEffector`, `spinningBodyNDOFStateEffector` | [`scenarioRoboticArm.py`](../examples/scenarioRoboticArm.py), [`scenarioFlexiblePanel.py`](../examples/scenarioFlexiblePanel.py) |
| Deploying/articulated arrays | linked spinning bodies | [`scenarioDeployingSolarArrays.py`](../examples/scenarioDeployingSolarArrays.py), [`scenarioSepMomentumManagement.py`](../examples/scenarioSepMomentumManagement.py) |
| Translating appendage | translating-body state effector | [`scenarioExtendingBoom.py`](../examples/scenarioExtendingBoom.py) |
| Tank and slosh | `FuelTank` plus `LinearSpringMassDamper` particles | [`scenarioFuelSlosh.py`](../examples/scenarioFuelSlosh.py) |
| Propellant depletion | tank state effector connected to thruster set | [`MultiSatBskSim/modelsMultiSat/BSK_MultiSatDynamics.py`](../examples/MultiSatBskSim/modelsMultiSat/BSK_MultiSatDynamics.py) |
| Finite-thrust transient | `ThrusterStateEffector` | [`scenarioOrbitManeuverTH.py`](../examples/scenarioOrbitManeuverTH.py) |
| Prescribed moving body | `PrescribedMotionStateEffector` with profile generators | [`scenarioPrescribedScrewMotion.py`](../examples/scenarioPrescribedScrewMotion.py) |

### Dynamic effectors

A `DynamicEffector` contributes force and torque based on state, geometry, environment, or command messages without representing the same kind of internal generalized coordinate set.

| Capability | Local module/pattern | Best examples |
|---|---|---|
| Ideal commanded force/torque | `ExtForceTorque` | [`scenarioAttitudeFeedback.py`](../examples/scenarioAttitudeFeedback.py) |
| Atmospheric drag | `DragDynamicEffector`, `FacetDragDynamicEffector` | [`scenarioDragDeorbit.py`](../examples/scenarioDragDeorbit.py), [`scenarioDragRendezvous.py`](../examples/scenarioDragRendezvous.py) |
| Solar-radiation pressure | cannonball or faceted SRP dynamic effector | [`scenarioSmallBodyNav.py`](../examples/scenarioSmallBodyNav.py), [`scenarioSepMomentumManagement.py`](../examples/scenarioSepMomentumManagement.py) |
| Gravity-gradient torque | gravity-gradient dynamic effector | [`scenarioAttitudeGG.py`](../examples/scenarioAttitudeGG.py) |
| Finite thrusters | `ThrusterDynamicEffector` | [`scenarioFormationReconfig.py`](../examples/scenarioFormationReconfig.py) |
| Inter-spacecraft constraint | shared `ConstraintDynamicEffector` | [`scenarioConstrainedDynamics.py`](../examples/scenarioConstrainedDynamics.py) |
| Electrostatic interaction | `MsmForceTorque` feeding per-spacecraft `ExtForceTorque` | [`scenarioTwoChargedSC.py`](../examples/scenarioTwoChargedSC.py) |

The category does not by itself indicate fidelity. `ExtForceTorque` can represent a deliberate external disturbance or an unrealistically ideal actuator. A dynamic thruster can model force geometry while still omitting valve transients, plume interaction, pressure decay, or propulsion-system thermodynamics.

## Attaching and scheduling are different operations

An effector commonly has two registrations:

```python
spacecraft_object.addDynamicEffector(drag_effector)
simulation.AddModelToTask("dynamicsTask", drag_effector, model_priority)
```

Attachment tells `Spacecraft` to include the effector in the equations of motion. Task registration allows scheduled message reads, command processing, resets, and outputs. State effectors follow the same broad pattern, although factories such as reaction-wheel and thruster factories may perform attachment internally.

**Engineering recommendation:** for each effector, explicitly verify:

1. how it is attached to its dynamic object;
2. whether it must also be scheduled;
3. which task and priority it uses;
4. which messages must be linked before reset;
5. which states and outputs prove it is active.

A simulation that runs without throwing an exception does not prove that an unattached or unwired effector influenced dynamics.

## Reaction wheels

The physical chain in [`scenarioAttitudeFeedbackRW.py`](../examples/scenarioAttitudeFeedbackRW.py) is:

```text
attitude error --> body torque controller --> wheel torque allocation
        --> RW motor command --> ReactionWheelStateEffector
        --> equal-and-opposite hub response + wheel-speed state
```

This is materially different from applying the controller output directly through `ExtForceTorque`:

- wheel geometry determines achievable body torque;
- wheel inertia creates speed dynamics and stored momentum;
- torque/speed limits, friction, voltage interfaces, and power can be added;
- saturation turns attitude regulation into a momentum-management problem.

[`scenarioMomentumDumping.py`](../examples/scenarioMomentumDumping.py) and [`scenarioMtbMomentumManagementSimple.py`](../examples/scenarioMtbMomentumManagementSimple.py) add unloading paths. [`scenarioAttitudeFeedbackRWPower.py`](../examples/scenarioAttitudeFeedbackRWPower.py) exposes power consequences.

Validation should include body-plus-wheel angular momentum, applied motor torque sign, wheel-axis mapping, saturation, and the zero-external-torque limit.

## Hinges, flexible bodies, and articulated mechanisms

### Hinged rigid bodies

[`scenarioHingedRigidBody.py`](../examples/scenarioHingedRigidBody.py) adds two single-axis panel coordinates to a six-DOF hub, creating eight coupled degrees of freedom. It demonstrates hub-panel reaction and the need to resolve the panel natural frequency numerically.

### Lumped flexible/articulated models

[`scenarioFlexiblePanel.py`](../examples/scenarioFlexiblePanel.py) represents bending and torsion with an N-DOF chain of spinning rigid sub-bodies. More segments can reproduce more modes, but also create higher-frequency dynamics and greater computational cost.

```text
continuum panel
      |
      v  discretise
rigid subpanel -- spring/damper joint -- rigid subpanel -- ...
      |
      v
finite set of coupled modal-like coordinates
```

Mesh refinement is not automatically convergence. Compare frequencies, mode shapes, hub response, and conserved/dissipated energy as segment count and integration step change.

### Branching and prescribed motion

[`scenarioPrescribedMotionWithRotationBranching.py`](../examples/scenarioPrescribedMotionWithRotationBranching.py) attaches dynamically responding solar panels to prescribed rotating trusses. [`scenarioPrescribedMotionWithTranslationBranching.py`](../examples/scenarioPrescribedMotionWithTranslationBranching.py) does the analogous translation hierarchy.

Prescribed motion answers, “What reaction does this commanded kinematic profile produce?” It does not predict whether an actuator has enough torque, force, power, or bandwidth to realize that profile unless those limitations are modelled separately.

Use prescribed motion for known deployment profiles, scan laws, or mechanism disturbances. Use a dynamically actuated joint when actuator feasibility and closed-loop motion are part of the question.

## Tanks, slosh, and propellant mass

[`scenarioFuelSlosh.py`](../examples/scenarioFuelSlosh.py) couples a `FuelTank` to spring-mass-damper slosh particles. With zero damping, it checks orbital and rotational energy/angular-momentum conservation; with damping, rotational energy should dissipate while angular momentum remains governed by external torque.

This example is primarily a slosh/conservation demonstration. It does not by itself establish realistic propellant depletion during a burn.

The MultiSat dynamics model uses a `FuelTankModelUniformBurn`, attaches the tank as a state effector, and connects the thruster set through `addThrusterSet()`. [`scenario_StationKeepingMultiSat.py`](../examples/MultiSatBskSim/scenariosMultiSat/scenario_StationKeepingMultiSat.py) records fuel mass alongside thrust and power.

For propulsion analysis distinguish:

```text
tank mass model       how vehicle mass and centre of mass change
slosh model           internal propellant motion and damping
thruster force model  force/torque, geometry, transient, command
depletion law         mass flow and specific impulse assumptions
feed system           pressure/flow coupling, usually not demonstrated here
```

## Coupling multiple spacecraft

Independent `Spacecraft` objects normally advance their own equations when their scheduled model executes. This is adequate when they communicate only through sampled messages and do not share an integration-stage force law.

For direct dynamic coupling, Basilisk provides synchronized integration:

```python
spacecraft_1.syncDynamicsIntegration(spacecraft_2)
integrator = svIntegrators.svIntegratorRKF45(spacecraft_1)
spacecraft_1.setIntegrator(integrator)
```

The primary dynamic object then advances the synchronized state sets at common integrator stages. Do not assign an independent integrator to the secondary object after synchronization.

[`scenarioFormationBasic.py`](../examples/scenarioFormationBasic.py) demonstrates synchronization even though its vehicles are dynamically independent and explains that it is optional there. [`scenarioConstrainedDynamics.py`](../examples/scenarioConstrainedDynamics.py) uses it because one constraint effector acts on both spacecraft.

```text
unsynchronised sampled coupling       synchronised stage coupling

SC 1 --state msg--> model             SC 1 <--- shared effector ---> SC 2
SC 2 --state msg-->/                           one integrated system
  force held to next task tick          forces evaluated at common stages
```

**Engineering recommendation:** use synchronized integration when a shared effector's force on one body depends directly on the simultaneous state/acceleration of another. Do not use it merely because two spacecraft appear in the same scenario.

### Constraint dynamics

`ConstraintDynamicEffector` applies forces/torques to maintain specified relative position and attitude relationships. In [`scenarioConstrainedDynamics.py`](../examples/scenarioConstrainedDynamics.py), stabilization gains `alpha` and `beta` trade constraint residual against numerical stiffness and run time. The order in which the shared effector is attached to the spacecraft is called out as significant in the source.

Constraint quality must be measured, not assumed. Record translational and rotational residuals, forces/torques, energy behavior, and step-size sensitivity. The companion component, frequency, and maneuver analyses explore these effects:

- [`scenarioConstrainedDynamicsComponentAnalysis.py`](../examples/scenarioConstrainedDynamicsComponentAnalysis.py)
- [`scenarioConstrainedDynamicsFrequencyAnalysis.py`](../examples/scenarioConstrainedDynamicsFrequencyAnalysis.py)
- [`scenarioConstrainedDynamicsManeuverAnalysis.py`](../examples/scenarioConstrainedDynamicsManeuverAnalysis.py)

### Electrostatic interaction

[`scenarioTwoChargedSC.py`](../examples/scenarioTwoChargedSC.py) loads multi-sphere geometry from `dataForExamples/GOESR_bus_80_sphs.csv`, computes electrostatic force/torque with `MsmForceTorque`, and sends the results through per-spacecraft `ExtForceTorque` modules.

That is message-level coupling at the scheduled task rate, not the same architecture as a shared integration-stage constraint. Its force update rate, geometry resolution, charge/potential assumption, and action-reaction balance are part of the fidelity statement.

## Standard Basilisk dynamics or MuJoCo?

The local MuJoCo examples introduce `mujoco.MJScene`, an alternative Basilisk `DynamicObject`. It loads a MuJoCo XML/MJCF model containing bodies, joints, sites, geometries, actuators, contacts, and equality constraints.

| Choose standard `Spacecraft` when | Choose `MJScene` when |
|---|---|
| one hub is a natural vehicle root | no single body is naturally privileged |
| existing state/dynamic effectors match the hardware | generic joint/body topology is the central problem |
| orbit, attitude, resources, and conventional GNC dominate | contact, landing, docking, or surface interaction dominates |
| conservation and spacecraft-specific module heritage matter | arbitrary application points and constraint/contact solvers matter |
| large campaign/Monte Carlo cost should remain modest | XML model flexibility justifies additional complexity |

The architecture changes fundamentally:

```text
standard Spacecraft                    MuJoCo MJScene

hub + StateEffectors                   peer rigid bodies in MJCF tree
    + DynamicEffectors                 joints, sites, geoms, actuators
Spacecraft integrator                  MJScene integrator/contact solver
scheduled effector interfaces          internal dynamics task at integrator stages
```

`MJScene.AddModelToDynamicsTask()` adds models evaluated within the scene's integrator-stage dynamics task, not an ordinary fixed-period Basilisk task.

Recommended local progression:

1. [`mujoco/scenarioReactionWheel.py`](../examples/mujoco/scenarioReactionWheel.py): bodies, a hinge joint, scalar actuator, and momentum check.
2. [`mujoco/scenarioAttitudeFeedbackRWMuJoCo.py`](../examples/mujoco/scenarioAttitudeFeedbackRWMuJoCo.py): adapter from a MuJoCo body/site state to standard Basilisk navigation and FSW.
3. [`mujoco/scenarioDeployPanels.py`](../examples/mujoco/scenarioDeployPanels.py): joint interpolation/control inside the dynamics task.
4. [`mujoco/scenarioSimpleDocking.py`](../examples/mujoco/scenarioSimpleDocking.py): multiple free bodies and a weld equality.
5. [`mujoco/scenarioAsteroidLanding.py`](../examples/mujoco/scenarioAsteroidLanding.py): mesh collision and force-vector actuators.

Important caveats:

- `scenarioSimpleDocking.py` disables contact and activates a weld equality; it demonstrates constraint capture, not a validated impact/contact docking model.
- `scenarioAsteroidLanding.py` deliberately uses constant inertial “gravity” near the asteroid; it is not a high-fidelity small-body gravity example.
- MuJoCo gravity, atmosphere, SRP, navigation, and FSW are not automatically inherited from standard spacecraft examples. They require explicit scene dynamics-task models and adapters.
- Contact parameters, integrator tolerances, joint stiffness/damping, geometry, and solver settings can dominate results.

**Engineering recommendation:** prototype a conventional spacecraft with standard effectors first. Move to MuJoCo when topology/contact requirements are demonstrated, not because “multibody” sounds more realistic.

## Dynamics validation strategy

### 1. Mass-property checks

- Confirm total mass, centre of mass, and inertia at the initial configuration.
- Move each appendage through a simple configuration and independently recompute the expected centre-of-mass shift.
- Confirm units and reference points for every inertia and position vector.

### 2. Conservation and dissipation checks

With no external force/torque and conservative internal elements:

- total linear momentum should be constant;
- total angular momentum should be constant;
- total mechanical energy should remain within integration error.

With dampers, friction, drag, or inelastic contact, predict which quantity should dissipate and which should still be conserved. [`scenarioFuelSlosh.py`](../examples/scenarioFuelSlosh.py) is the strongest local conservation-pattern template.

### 3. Limiting cases

- Zero appendage mass should recover the rigid hub.
- Infinite/large joint stiffness should approach a rigidly attached body within numerical limits.
- Zero wheel torque should preserve wheel and hub rates absent other torques.
- Zero potential should remove electrostatic forces.
- Zero atmosphere density or area should remove drag.
- Very short finite burn should approach an ideal impulse with the same integrated delta-v.

### 4. Numerical convergence

Reduce the step size and tighten adaptive tolerances until:

- conserved quantities stabilize;
- flexible/slosh modal frequency and amplitude stabilize;
- constraint residuals stabilize;
- contact impulse and peak loads stabilize appropriately;
- mission-level metrics no longer change materially.

### 5. Interface checks

Record commanded and realized force/torque separately. Confirm direction and moment arm in a nontrivial attitude. Verify equal-and-opposite internal reactions and action-reaction forces for coupled vehicles.

## Common failure modes

- Setting hub inertia about the wrong point or in the wrong frame.
- Confusing body origin B with current total-system centre of mass C.
- Creating an effector but failing to attach it to `Spacecraft`.
- Attaching an effector but failing to schedule required message/update behavior.
- Using an ideal `ExtForceTorque` command and interpreting it as actuator performance.
- Choosing a time step from orbital period while ignoring the fastest hinge, flexible, slosh, actuator, or contact mode.
- Applying independent integrators to spacecraft connected by a shared dynamic effector.
- Prescribing motion while claiming actuator feasibility.
- Treating a fuel tank as proof that thrust depletion is connected.
- Comparing hub angular momentum alone when wheel/appendage momentum is present.
- Using a message-coupled interaction without accounting for its sample/hold delay.
- Moving to MuJoCo without a validation case for contact/joint parameters and state-frame adapters.

## Model-selection checklist

1. Is one rigid hub a natural reference body?
2. Which additional coordinates are physically necessary for the metric?
3. Are those coordinates dynamic, prescribed, or adequately represented by a force/torque only?
4. Does an existing state or dynamic effector implement the required physics?
5. What is the highest physical frequency, and what step/tolerance resolves it?
6. Are multiple dynamic objects independent, message-coupled, or directly stage-coupled?
7. Which mass, momentum, energy, constraint, and actuator metrics validate the model?
8. Does contact or arbitrary multibody topology justify MuJoCo?
9. Which simplifications are intentional, and where are they recorded?

## Recommended reading path

1. [`scenarioAttitudeFeedback.py`](../examples/scenarioAttitudeFeedback.py): rigid hub plus ideal dynamic torque.
2. [`scenarioAttitudeFeedbackRW.py`](../examples/scenarioAttitudeFeedbackRW.py): physical internal actuator state.
3. [`scenarioHingedRigidBody.py`](../examples/scenarioHingedRigidBody.py): simple coupled appendages.
4. [`scenarioFuelSlosh.py`](../examples/scenarioFuelSlosh.py): moving mass and conservation.
5. [`scenarioFlexiblePanel.py`](../examples/scenarioFlexiblePanel.py): articulated N-DOF approximation.
6. [`scenarioPrescribedMotionWithRotationBranching.py`](../examples/scenarioPrescribedMotionWithRotationBranching.py): prescribed/dynamic branching.
7. [`scenarioConstrainedDynamics.py`](../examples/scenarioConstrainedDynamics.py): shared effector and synchronized integration.
8. [`scenarioTwoChargedSC.py`](../examples/scenarioTwoChargedSC.py): sampled message-level interaction.
9. [`mujoco/scenarioReactionWheel.py`](../examples/mujoco/scenarioReactionWheel.py): alternate multibody architecture.
10. [`mujoco/scenarioSimpleDocking.py`](../examples/mujoco/scenarioSimpleDocking.py): equality-constrained multiple bodies.

