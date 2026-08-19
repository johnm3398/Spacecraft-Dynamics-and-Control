> **Draft status — AI-generated:** This page was drafted by OpenAI Codex on 2026-08-19. It has not yet been technically vetted by the BASILISK-X repository owner. The repository owner will review, correct, and maintain it. Verify APIs, units, frames, assumptions, and version compatibility before engineering use.

# Actuators, Propulsion, and Resources

This chapter connects requested spacecraft motion to physical devices, propellant, power, data, and thermal consequences. Its central rule is simple:

> A valid guidance or control command is not evidence that a spacecraft can execute it.

## 1. Close the physical command chain

```text
mission/guidance
      ↓
requested body torque or force
      ↓
allocation and command shaping
      ↓
device command
      ↓
actuator dynamics and limits
      ↓
force, torque, momentum, mass flow, power, heat
      ↓
spacecraft truth and resource states
      ↓
telemetry and mission gating
```

Each missing layer is an assumption. That can be appropriate, but it must match the engineering question.

## 2. Ideal versus physical actuation

| Representation | What it answers | What it omits | Local template |
|---|---|---|---|
| Direct truth-state change | Ideal impulsive targeting or prescribed deployment | Pointing, force history, actuator limits, mass loss, disturbances | [`scenarioOrbitManeuver.py`](../examples/scenarioOrbitManeuver.py), [`scenarioHohmann.py`](../examples/scenarioHohmann.py) |
| `ExtForceTorque` | Required ideal body/inertial force or torque and closed-loop rigid-body response | Allocation, saturation, momentum, valves, propellant, power | [`scenarioAttitudeFeedback2T.py`](../examples/scenarioAttitudeFeedback2T.py) |
| Reaction-wheel state effector | Wheel/body momentum exchange and wheel state | Momentum removal unless unloading devices are added | [`scenarioAttitudeFeedbackRW.py`](../examples/scenarioAttitudeFeedbackRW.py) |
| Thruster effector with on-time commands | Geometry, finite force/torque, pulse scheduling, transients as configured | Tank depletion unless explicitly connected; many propulsion-system details | [`scenarioAttitudeFeedback2T_TH.py`](../examples/scenarioAttitudeFeedback2T_TH.py) |
| Thruster plus tank/depletion | Finite thrust with time-varying propellant mass | Feed pressure, mixture ratio, thermal/valve faults unless added | MultiSat [`BSK_MultiSatDynamics.py`](../examples/MultiSatBskSim/modelsMultiSat/BSK_MultiSatDynamics.py) |

Use the lowest row that contains the output metric of interest. Ideal state changes are often the correct first model for transfer geometry; they are the wrong final model for burn execution error or propellant sizing.

## 3. Effectors and attachment

A state effector contributes internal state and mass/momentum/energy coupling to the spacecraft hub. A dynamic effector contributes force and torque without that same hub-coupled generalized-state role.

```python
# Dynamic effector pattern
sc.addDynamicEffector(thruster_or_ext_force)
sim.AddModelToTask(task_name, thruster_or_ext_force)

# State effector pattern
sc.addStateEffector(tank_or_internal_device)
sim.AddModelToTask(task_name, tank_or_internal_device)
```

Physical attachment and scheduler registration are distinct. A model can update messages yet fail to affect the spacecraft if it was not attached; an attached model can retain stale commands if it was not scheduled as required.

## 4. Reaction wheels

The complete wheel pattern is:

```text
mrpFeedback.cmdTorqueOutMsg
              ↓
rwMotorTorque + RWArrayConfigMsg
              ↓ ArrayMotorTorqueMsg
ReactionWheelStateEffector
              ↓ rwSpeedOutMsg and per-wheel RWConfigLogMsg
Spacecraft truth
```

Construction normally uses `simIncludeRW.rwFactory()`:

1. Create each wheel with model, spin axis, initial speed, momentum/torque parameters, and location where relevant.
2. Create `ReactionWheelStateEffector` and use the factory to attach its wheel set to `Spacecraft`.
3. Schedule the state effector.
4. Generate the FSW `RWArrayConfigMsg` from the same factory or intentionally construct a dispersed FSW configuration.
5. Connect controller torque → `rwMotorTorque` → state effector motor command.
6. Record requested and realized motor torque, speed, momentum, and saturation flags/margins.

The array order is part of the interface. Reordering factory creation while retaining old command/configuration arrays can command the wrong physical wheel.

[`scenarioAttitudeFeedbackRW.py`](../examples/scenarioAttitudeFeedbackRW.py) also demonstrates optional voltage mapping. [`scenarioAttitudeFeedbackRWPower.py`](../examples/scenarioAttitudeFeedbackRWPower.py) adds one `ReactionWheelPower` node per wheel and connects those nodes to a battery.

### Wheel fidelity ladder

```text
ideal body torque
  → ideal wheel torque allocation
  → wheel momentum and speed state
  → torque/speed/saturation limits
  → friction, voltage drive, jitter/imbalance
  → electrical power and thermal consequences
```

Stop when the quantities needed for the requirement are represented. A body-pointing trade does not always need imbalance jitter; a long-duration fine-pointing or momentum-budget study may.

## 5. Thruster command chains

Attitude-control thruster examples use a staged chain:

```text
CmdTorqueBodyMsg
       ↓
thrForceMapping
       ↓ THRArrayCmdForceMsg
thrFiringSchmitt or momentum-dump scheduler
       ↓ THRArrayOnTimeCmdMsg
thruster effector
       ↓ force/torque + THROutputMsg per device
Spacecraft
```

[`scenarioAttitudeFeedback2T_TH.py`](../examples/scenarioAttitudeFeedback2T_TH.py) demonstrates `thrForceMapping` followed by `thrFiringSchmitt` and a `ThrusterDynamicEffector`. It includes on-pulsing/off-pulsing cases. The Schmitt trigger converts continuously requested force into on-time commands and introduces quantization/hysteresis behavior absent from ideal torque.

Thruster configuration must keep these consistent:

- body-frame location and thrust direction;
- maximum thrust and specific impulse;
- array ordering in FSW and dynamics;
- minimum on-time/pulse policy;
- on/off transient model;
- mounting on the hub versus an auxiliary body;
- which tank, if any, receives mass-flow information.

### `ThrusterDynamicEffector` versus `ThrusterStateEffector`

The names are easy to misinterpret. In Basilisk 2.11.1:

| Module | Dynamic behavior | Numerical implication |
|---|---|---|
| `ThrusterDynamicEffector` | Computes on/off behavior using configured ramps; default can represent instantaneous on/off | Official documentation warns it is incompatible with variable-step integrators |
| `ThrusterStateEffector` | Integrates a thrust-factor state \(\kappa\) with a first-order ODE governed by `cutoffFrequency` | Compatible with variable-step integration; cannot create mathematically instantaneous transitions |

The state-effector dynamics are approximately

\[
\dot\kappa = \omega_c(1-\kappa) \quad \text{on},
\qquad
\dot\kappa = -\omega_c\kappa \quad \text{off}.
\]

Verify current details in the official [thruster state-effector](https://avslab.github.io/basilisk/Documentation/simulation/dynamics/Thrusters/thrusterStateEffector/thrusterStateEffector.html) and [thruster dynamic-effector](https://avslab.github.io/basilisk/Documentation/simulation/dynamics/Thrusters/thrusterDynamicEffector/thrusterDynamicEffector.html) pages.

Observed examples:

- [`scenarioOrbitManeuverTH.py`](../examples/scenarioOrbitManeuverTH.py) and [`scenarioAttitudeFeedback2T_stateEffTH.py`](../examples/scenarioAttitudeFeedback2T_stateEffTH.py) use `ThrusterStateEffector` and variable-step integration.
- [`scenarioAttitudeFeedback2T_TH.py`](../examples/scenarioAttitudeFeedback2T_TH.py), BskSim, MultiSat, and momentum dumping use `ThrusterDynamicEffector`.

Do not choose based only on which example was copied first; choose the transient and integrator contract required by the study.

## 6. Finite burns

A finite-burn simulation needs more than a nonzero thruster command:

```text
targeting Δv and burn epoch
        ↓
attitude reference aligned with desired thrust direction
        ↓
tracking and attitude actuation
        ↓
burn-enable condition and on-time/force command
        ↓
thruster force and mass flow during propagation
        ↓
terminal state and propellant error metrics
```

[`scenarioOrbitManeuverTH.py`](../examples/scenarioOrbitManeuverTH.py) is the clearest finite-thrust starting point. It uses `velocityPoint`, an attitude loop, a `ThrusterStateEffector`, and phased execution. Between phases it reads the registered position/velocity states to calculate the next burn; it does not replace those states with an ideal impulse. It does not connect a fuel tank, and its burn-duration calculation uses constant hub mass. Treat it as a finite-force and transient demonstration—not an end-to-end propulsion-performance model.

For a serious burn, record:

- commanded and realized thrust vector in a declared frame;
- attitude/boresight error during the burn;
- start/stop timing and achieved impulse;
- acceleration and integrated \(\Delta v\);
- propellant mass, center of mass, and inertia history;
- terminal Cartesian/orbital-element targeting error;
- duty cycle, minimum pulse effects, and failed/misaligned thrusters where required.

## 7. Fuel tanks, depletion, and slosh

MultiSat's [`BSK_MultiSatDynamics.py`](../examples/MultiSatBskSim/modelsMultiSat/BSK_MultiSatDynamics.py) shows the essential depletion topology:

```text
ThrusterDynamicEffector ── mass-flow contribution ──┐
                                                   v
FuelTank(FuelTankModelUniformBurn) ── fuelTankOutMsg
                 │ state effector
                 └────────> Spacecraft mass properties
```

The source configures `FuelTankModelUniformBurn`, sets initial/max fuel mass and tank geometry, attaches the tank with `addStateEffector`, and connects the thruster set with `addThrusterSet`. [`scenario_StationKeepingMultiSat.py`](../examples/MultiSatBskSim/scenariosMultiSat/scenario_StationKeepingMultiSat.py) records `fuelMass` alongside thrust and formation states.

[`scenarioFuelSlosh.py`](../examples/scenarioFuelSlosh.py) addresses tank/slosh coupling and energy/momentum behavior. It is not by itself a depletion-under-thrust template. Separate these questions:

- propellant bookkeeping and changing mass;
- center-of-mass/inertia migration;
- liquid slosh modes and damping;
- propulsion feed/system performance.

Adding one does not imply the others.

## 8. Momentum management

Reaction wheels store disturbance and maneuver momentum; they do not remove total spacecraft angular momentum. Long-duration control therefore needs an unloading path.

### Thruster unloading

[`scenarioMomentumDumping.py`](../examples/scenarioMomentumDumping.py) implements:

```text
wheel speeds + RW configuration
        ↓
thrMomentumManagement → desired ΔH
        ↓
thrForceMapping → required thruster impulses
        ↓
thrMomentumDumping → discrete on-times with wait/min-fire logic
        ↓
ThrusterDynamicEffector
```

The normal RW attitude loop remains active. Validate attitude disturbance during unloading, achieved wheel-momentum change, propellant use, plume/translation effects, and time between dump opportunities.

### Magnetic-torque-bar unloading

[`scenarioMtbMomentumManagement.py`](../examples/scenarioMtbMomentumManagement.py) connects magnetic-field truth, a magnetometer, `tamComm`, `mtbMomentumManagement`, and `MtbEffector`. [`scenarioMtbMomentumManagementSimple.py`](../examples/scenarioMtbMomentumManagementSimple.py) makes the mapping and feedforward layers explicit with `dipoleMapping` and `mtbFeedforward`.

Magnetic torque is constrained by \(\mathbf L=\mathbf m\times\mathbf B\); torque parallel to the local field is unavailable instantaneously. Orbit geometry and field-model accuracy therefore matter to unloading authority.

[`scenarioSepMomentumManagement.py`](../examples/scenarioSepMomentumManagement.py) is a specialized, high-coupling example involving a solar-electric-propulsion pointing architecture, reaction wheels, articulating arrays/platform, and thruster use. Study it after the simpler unloading examples; it is not a generic first propulsion template.

## 9. Power resources

The simple resource architecture is additive:

```text
Sun + eclipse + spacecraft attitude
        ↓
SimpleSolarPanel ── PowerNodeUsageMsg (+)

device loads ───── PowerNodeUsageMsg (-)
        │
        v
SimpleBattery ── PowerStorageStatusMsg
```

[`scenarioPowerDemo.py`](../examples/scenarioPowerDemo.py) configures a solar panel, constant sink, eclipse, and battery. `addPowerNodeToModel()` registers each source/sink with storage. Capacities and charge are in joules; example comments convert watt-hours using \(1\,\mathrm{Wh}=3600\,\mathrm{J}\).

[`scenarioAttitudeFeedbackRWPower.py`](../examples/scenarioAttitudeFeedbackRWPower.py) derives wheel electrical power from each wheel's state and an efficiency assumption. This is the better template when a GNC trade includes energy.

The battery aggregates power; it does not automatically make mission decisions. Unless a controller subscribes to resource status and disables a device/mode, a depleted battery need not stop an instrument or actuator in the simulation.

## 10. Data resources and access-triggered payloads

[`scenarioDataDemo.py`](../examples/scenarioDataDemo.py) illustrates data nodes, partitions, storage, and a transmitter:

```text
instrument DataNodeUsageMsg (+bits/s) ─┐
                                       v
                             storage unit status
                                       ^
transmitter DataNodeUsageMsg (-bits/s) ┘
```

[`scenarioGroundLocationImaging.py`](../examples/scenarioGroundLocationImaging.py) adds useful operational gating:

```text
attitude error + target access
             ↓
simpleInstrumentController → DeviceCmdMsg
             ↓
instrument → partitioned storage

ground-station access + storage status
             ↓
spaceToGroundTransmitter → negative data rate
```

This is a reusable mission pattern: physical/geometry conditions command a payload, while data production and depletion remain resource models. The example does not also close the power/thermal loop; add those conditions only when they affect mission feasibility.

## 11. Thermal resources

[`scenarioSensorThermal.py`](../examples/scenarioSensorThermal.py) couples spacecraft/Sun geometry, absorptivity/emissivity, area, mass, specific heat, and internal sensor power to a temperature output. It reverses a pointing vector between two execution phases to demonstrate orientation-dependent heating.

This is a lumped thermal model, not a general thermal network. Escalate fidelity from constant allowable temperature → lumped node → coupled nodes/conduction/radiation only when temperature affects performance or survival metrics.

## 12. Resource gating pattern

Resource simulation becomes mission engineering only when status affects commands:

```text
navigation/geometry ─┐
power status ────────┤
data storage ────────┤
thermal status ──────┤→ mission-mode logic → device/actuator commands
propellant estimate ─┤
fault status ────────┘
```

Recommended behavior includes hysteresis and recovery thresholds, command acknowledgement, validity/timestamp checks, and explicit safe modes. Avoid instantaneous chatter at a single threshold.

## 13. Fidelity-selection checklist

| Question | Minimum useful model |
|---|---|
| Does the ideal trajectory close? | Ideal impulses/state discontinuities |
| Can a finite force achieve the terminal state? | Thruster geometry, attitude, finite force, timing |
| Will control saturate? | Physical actuator allocation and limits |
| How much propellant is used? | Thruster mass flow plus connected fuel tank |
| Will momentum remain manageable? | Disturbances, RW momentum, unloading actuator and logic |
| Is the mission energy-positive? | Attitude/eclipses, sources, all material loads, storage |
| Can data be collected and returned? | Access, pointing, payload generation, storage, downlink depletion |
| Is a device thermally viable? | Duty/orientation-dependent thermal model at adequate order |

**Observed caveat:** the examples often plot power, data, thermal, or fuel states without feeding them back into mission logic. Treat those as resource accounting demonstrations. **Engineering recommendation:** define an operational constraint, connect status to commands, and measure both resource margin and achieved mission value.
