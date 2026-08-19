> **Draft status — AI-generated:** This page was drafted by OpenAI Codex on 2026-08-19. It has not yet been technically vetted by the BASILISK-X repository owner. The repository owner will review, correct, and maintain it. Verify APIs, units, frames, assumptions, and version compatibility before engineering use.

# Reusable Patterns and Minimal Skeletons

This chapter condenses recurring Basilisk arrangements into patterns that are small enough to remember. The snippets are intentionally incomplete: each shows an architectural seam, not a new all-purpose scenario. Confirm exact members against Basilisk 2.11.1 and the cited local example before use.

## Pattern 1: build a deterministic simulation shell

Use one process and one task until the engineering question requires different rates or ownership boundaries.

```python
from Basilisk.utilities import SimulationBaseClass, macros

sim = SimulationBaseClass.SimBaseClass()
process = sim.CreateNewProcess("dynamicsProcess")
task = sim.CreateNewTask("dynamicsTask", macros.sec2nano(0.1))
process.addTask(task)
```

What each object does:

- `SimBaseClass` owns orchestration, the scheduler, events, initialization, and execution.
- A process groups tasks and participates in priority ordering.
- A task executes registered models at one fixed scheduler period.

Why use it: this is the smallest explicit scheduler configuration and keeps causal order visible.

Source: [`scenarioBasicOrbit.py`](../examples/scenarioBasicOrbit.py).

## Pattern 2: add a rigid spacecraft and central gravity

```python
from Basilisk.simulation import spacecraft
from Basilisk.utilities import simIncludeGravBody

sc = spacecraft.Spacecraft()
sc.ModelTag = "spacecraft"
sc.hub.mHub = 100.0
sc.hub.IHubPntBc_B = [[10.0, 0.0, 0.0],
                      [0.0, 8.0, 0.0],
                      [0.0, 0.0, 6.0]]
sim.AddModelToTask("dynamicsTask", sc)

gravity = simIncludeGravBody.gravBodyFactory()
earth = gravity.createEarth()
earth.isCentralBody = True
gravity.addBodiesTo(sc)
```

Configure the initial Cartesian state before initialization:

```python
sc.hub.r_CN_NInit = r_N       # m
sc.hub.v_CN_NInit = v_N       # m/s
sc.hub.sigma_BNInit = sigma_BN
sc.hub.omega_BN_BInit = omega_BN_B  # rad/s
```

Why use it: `Spacecraft` supplies the normal hub-centric 6-DOF truth model. Point-mass gravity is the correct baseline for deciding whether more environmental fidelity is required.

Watch for:

- `C`, `B`, and `Bc` do not remain coincident when internal mass moves;
- gravity attachment does not automatically add SPICE, harmonics, drag, eclipse, or SRP;
- the inertia point, expression frame, symmetry, and units must be explicit.

## Pattern 3: derive an inertial initial state from orbital elements

```python
from Basilisk.utilities import macros, orbitalMotion

oe = orbitalMotion.ClassicElements()
oe.a = 7_000_000.0
oe.e = 0.001
oe.i = 51.6 * macros.D2R
oe.Omega = 20.0 * macros.D2R
oe.omega = 30.0 * macros.D2R
oe.f = 40.0 * macros.D2R

r_N, v_N = orbitalMotion.elem2rv(earth.mu, oe)
sc.hub.r_CN_NInit = r_N
sc.hub.v_CN_NInit = v_N
```

Why use it: element inputs are readable mission configuration; Cartesian states are what the integrator needs.

Validate the resulting periapsis, energy, period, and Cartesian state. Circular/equatorial cases contain classically undefined angles even when the Cartesian state is valid.

## Pattern 4: connect a producer to a consumer

```python
consumer.someInMsg.subscribeTo(producer.someOutMsg)
```

Read this as: “the consumer’s input subscribes to the producer’s output.” A direct subscription supplies the latest written typed payload. It does not add frame conversion, unit conversion, interpolation, delay, or packet transport.

Before initialization, check required readers where supported:

```python
if not consumer.someInMsg.isLinked():
    raise RuntimeError("required input is not connected")
```

Why use it: typed message boundaries let truth, sensors, navigation, FSW, and actuators be replaced independently.

## Pattern 5: publish constant configuration through a message

```python
from Basilisk.architecture import messaging

payload = messaging.VehicleConfigMsgPayload()
payload.ISCPntB_B = [10.0, 0.0, 0.0,
                     0.0, 8.0, 0.0,
                     0.0, 0.0, 6.0]
vehicle_config_msg = messaging.VehicleConfigMsg().write(payload)
controller.vehConfigInMsg.subscribeTo(vehicle_config_msg)
```

Keep `vehicle_config_msg` alive for the simulation lifetime.

Why use it: a static Python configuration then obeys the same typed interface as a dynamic producer. This is common for vehicle inertia, reaction-wheel layouts, and thruster layouts.

Source: [`scenarioAttitudeFeedback.py`](../examples/scenarioAttitudeFeedback.py).

## Pattern 6: schedule and down-sample a recorder

```python
record_period = macros.sec2nano(1.0)
state_recorder = sc.scStateOutMsg.recorder(record_period)
sim.AddModelToTask("dynamicsTask", state_recorder, 0)
```

After execution:

```python
time_s = state_recorder.times() * macros.NANO2SEC
position_N_m = state_recorder.r_BN_N
velocity_N_m_s = state_recorder.v_BN_N
```

Why use it: a recorder retains a public interface with timestamps. The optional period limits retained samples; it does not change producer rate.

Watch for:

- place the recorder after its producer when same-tick data matters;
- align histories by timestamp rather than array index;
- create an input-reader recorder only after that reader has subscribed;
- retain only data needed for debugging, validation, or metrics.

## Pattern 7: build the basic attitude loop

```text
Spacecraft truth
    -> SimpleNav
    -> attitude reference generator
    -> attTrackingError
    -> mrpFeedback
    -> ideal torque or physical allocation
    -> actuator/effector
    -> Spacecraft truth
```

Representative wiring:

```python
nav.scStateInMsg.subscribeTo(sc.scStateOutMsg)
tracking.attNavInMsg.subscribeTo(nav.attOutMsg)
tracking.attRefInMsg.subscribeTo(guidance.attRefOutMsg)
controller.guidInMsg.subscribeTo(tracking.attGuidOutMsg)
controller.vehConfigInMsg.subscribeTo(vehicle_config_msg)
```

Use `inertial3D`, `hillPoint`, `velocityPoint`, `locationPointing`, or another reference generator according to the desired frame. Do not connect a controller straight to a reference: tracking converts the navigation/reference pair into the body-relative error contract expected by the controller.

Sources: [`scenarioAttitudeGuidance.py`](../examples/scenarioAttitudeGuidance.py) and [`scenarioAttitudeFeedback2T.py`](../examples/scenarioAttitudeFeedback2T.py).

## Pattern 8: start with ideal torque, then replace the seam

The minimal control realization is:

```python
torque_effector.cmdTorqueInMsg.subscribeTo(controller.cmdTorqueOutMsg)
sc.addDynamicEffector(torque_effector)
sim.AddModelToTask("dynamicsTask", torque_effector)
```

This is appropriate for checking reference signs, controller behavior, settling time, and demanded torque. It is not an actuator model.

The reaction-wheel realization replaces only the downstream seam:

```text
CmdTorqueBodyMsg
    -> rwMotorTorque allocation
    -> ArrayMotorTorqueMsg
    -> ReactionWheelStateEffector
    -> hub/wheel momentum exchange
```

It additionally requires a wheel factory/configuration message, wheel geometry and limits, initial speeds, physical attachment, and appropriate recorders.

Source: [`scenarioAttitudeFeedbackRW.py`](../examples/scenarioAttitudeFeedbackRW.py).

## Pattern 9: realize a torque request with thruster pulses

```text
CmdTorqueBodyMsg
    -> thrForceMapping
    -> thrFiringSchmitt
    -> THRArrayOnTimeCmdMsg
    -> thruster dynamic/state effector
    -> spacecraft force and torque
```

Why use it: allocation and pulse logic expose geometry, minimum on-time/deadband, and torque/translation coupling that ideal torque hides.

Source comparison:

- [`scenarioAttitudeFeedback2T_TH.py`](../examples/scenarioAttitudeFeedback2T_TH.py) uses the dynamic-effector route.
- [`scenarioAttitudeFeedback2T_stateEffTH.py`](../examples/scenarioAttitudeFeedback2T_stateEffTH.py) demonstrates the state-effector route and thrust-state behavior.

Do not infer propellant depletion merely because a thruster fires. Add and connect an appropriate tank/mass-flow model when mass is a metric.

## Pattern 10: distinguish an ideal impulse from a finite burn

An ideal impulse changes the registered velocity state at a defined instant. It is a boundary condition for mission mechanics:

```text
target delta-v -> instantaneous truth-state discontinuity -> coast
```

A finite-burn model is a closed execution chain:

```text
target delta-v
    -> burn timing and attitude reference
    -> attitude tracking/control
    -> thruster command
    -> finite force and torque
    -> optional mass depletion
    -> achieved delta-v reconstructed from telemetry
```

Use the ideal form for transfer geometry and analytic regression. Use the finite form for pointing, duration, execution error, disturbance, power, or fuel questions.

Sources: [`scenarioOrbitManeuver.py`](../examples/scenarioOrbitManeuver.py) and [`scenarioOrbitManeuverTH.py`](../examples/scenarioOrbitManeuverTH.py).

## Pattern 11: add a sensor without confusing it with navigation

```text
truth/environment -> sensor physics/noise -> measurement message
                                         -> estimator -> navigation message
```

Build and validate these boundaries separately:

1. Does the sensor produce the expected ideal geometry?
2. Are noise, bias, saturation, occultation, and validity modeled at the intended cadence?
3. Does the estimator consume only available measurements?
4. Are truth error, residuals/innovations, covariance, and navigation output recorded separately?

Use `SimpleNav` as a truth-derived navigation baseline when studying downstream FSW. Replace it with measurement-driven estimation when navigation performance or autonomous safety is the question.

Sources: [`scenarioCSS.py`](../examples/scenarioCSS.py), [`scenarioCSSFilters.py`](../examples/scenarioCSSFilters.py), and [`scenarioSmallBodyNavUKF.py`](../examples/scenarioSmallBodyNavUKF.py).

## Pattern 12: create several independent spacecraft explicitly

```python
spacecraft_objects = []
state_recorders = []

for index in range(number_spacecraft):
    vehicle = spacecraft.Spacecraft()
    vehicle.ModelTag = f"spacecraft_{index}"
    gravity.addBodiesTo(vehicle)
    sim.AddModelToTask("dynamicsTask", vehicle)

    recorder = vehicle.scStateOutMsg.recorder(record_period)
    sim.AddModelToTask("dynamicsTask", recorder)

    spacecraft_objects.append(vehicle)
    state_recorders.append(recorder)
```

Why use it: a list-based standalone pattern is transparent and scales far enough for many fixed constellation or formation studies.

Move to a MultiSat-style abstraction only when per-spacecraft dynamics/FSW stacks, heterogeneous types, separate rates, or campaign reuse justify it.

Source: [`scenarioFormationBasic.py`](../examples/scenarioFormationBasic.py).

## Pattern 13: compute relative state through an explicit chief frame

```python
rho_H, rho_prime_H = orbitalMotion.rv2hill(
    chief_r_N,
    chief_v_N,
    deputy_r_N,
    deputy_v_N,
)
```

Use `hill2rv` for the inverse construction and test round trips. Do not compute Hill relative velocity by rotating only the inertial velocity difference; the rotating-frame transport term matters.

Why use it: this keeps chief choice, origin, axes, and velocity definition visible.

Source: [`scenarioRendezVous.py`](../examples/scenarioRendezVous.py).

## Pattern 14: perform explicit phased execution

```python
sim.ConfigureStopTime(macros.min2nano(5.0))
sim.ExecuteSimulation()

# Apply a declared mission-level reconfiguration here.

sim.ConfigureStopTime(macros.min2nano(8.0))  # absolute time
sim.ExecuteSimulation()
```

Why use it: sequential mission-analysis phases often need no event framework. It also makes the exact boundary at which an ideal command or state reset occurs easy to inspect.

Move phase logic into events/tasks or message-driven mission modules when it is intended to behave like onboard logic, requires high-rate state-dependent transitions, or is reused across scenarios.

## Pattern 15: switch reusable FSW modes through gateways

```text
mode event
   -> disable mutually exclusive tasks
   -> clear gateway commands
   -> enable chosen guidance/control tasks
   -> re-arm the other mode events
```

Guidance tasks write through an attitude-reference gateway; tracking and control subscribe to that stable endpoint once.

Why use it: gateway messages decouple downstream consumers from alternative algorithm authors and reduce rewiring during mode changes.

Source: [`BskSim/models/BSK_Fsw.py`](../examples/BskSim/models/BSK_Fsw.py).

Important local caveat: the copied `initiateStandby` callback returns strings instead of invoking its actions under the current callback API. Treat the pattern as sound but that implementation as unverified until corrected and tested.

## Pattern 16: wrap a deterministic scenario in Monte Carlo

```python
from Basilisk.utilities.MonteCarlo.Controller import Controller
from Basilisk.utilities.MonteCarlo.RetentionPolicy import RetentionPolicy

controller = Controller()
controller.setSimulationFunction(create_simulation)
controller.setExecutionFunction(execute_simulation)
controller.setExecutionCount(number_runs)
controller.setShouldDisperseSeeds(True)
controller.setArchiveDir(archive_directory)

# controller.addDispersion(...)

retention = RetentionPolicy()
retention.addMessageLog("messageName", ["payloadField"])
controller.addRetentionPolicy(retention)

failures = controller.executeSimulations()
```

Why use it: the deterministic model remains intact while an external controller owns repetition, dispersion, seeds, retention, and failures.

Before using it:

- define deterministic pass/fail metrics;
- expose stable named parameter paths;
- separate uncertain parameters from stochastic seeds;
- model correlations and physical bounds;
- archive version/configuration provenance;
- ensure the scenario has created the recorders expected by the retention policy.

Source: [`MonteCarloExamples/scenarioBskSimAttFeedbackMC.py`](../examples/MonteCarloExamples/scenarioBskSimAttFeedbackMC.py).

## Pattern 17: add Vizard as an optional observer

```python
from Basilisk.utilities import vizSupport

if vizSupport.vizFound:
    viz = vizSupport.enableUnityVisualization(
        sim,
        "dynamicsTask",
        sc,
        # saveFile=__file__,
        # liveStream=True,
    )
```

Why use it: playback and live 3-D visualization help interpret geometry while preserving a headless numerical path.

Vizard is not validation evidence. In ordinary scenarios it observes truth. In OpNav it can generate imagery returned to the camera pipeline, making executable version, port, render cadence, and latency part of the sensor configuration.

## Pattern 18: write a custom message-driven Python module

```python
from Basilisk.architecture import messaging, sysModel

class MyAlgorithm(sysModel.SysModel):
    def __init__(self):
        super().__init__()
        self.input_msg = messaging.NavAttMsgReader()
        self.output_msg = messaging.CmdTorqueBodyMsg()

    def Reset(self, CurrentSimNanos):
        if not self.input_msg.isLinked():
            self.bskLogger.error("MyAlgorithm.input_msg is not linked")

    def UpdateState(self, CurrentSimNanos):
        input_payload = self.input_msg()
        output_payload = messaging.CmdTorqueBodyMsgPayload()
        output_payload.torqueRequestBody = self.compute(input_payload)
        self.output_msg.write(
            output_payload,
            CurrentSimNanos,
            self.moduleID,
        )
```

Why use it: an algorithm that participates in simulation timing or message flow belongs inside the scheduler rather than in a Python loop that reaches into truth state.

The skeleton leaves `compute` undefined. The full local pattern is [`scenarioAttitudePointingPy.py`](../examples/scenarioAttitudePointingPy.py).

## Pattern-selection guide

| Need | Smallest fitting pattern |
|---|---|
| One orbit or control trade | Standalone shell |
| Two or three fixed spacecraft | Standalone list pattern |
| Reused vehicle and mode library | BskSim-style model/scenario split |
| Variable or heterogeneous fleet | MultiSat-style indexed world/dynamics/FSW |
| Articulated topology without contact | Standard `Spacecraft` plus state effectors |
| General joints, constraints, or contact | MuJoCo `MJScene` |
| Repeated uncertainty campaign | Monte Carlo wrapper around a deterministic scenario |
| Image-based navigation loop | OpNav layered architecture |
| High-level tasking policy | External BSK-RL environment after deterministic validation |

## Patterns not to introduce prematurely

Avoid creating a generic framework merely because the same five setup lines appear twice. A reusable abstraction is justified when it has:

- at least two real consumers with the same semantic contract;
- explicit units, frames, configuration ownership, and failure behavior;
- deterministic tests at the abstraction boundary;
- a versioned public interface smaller than the code it hides;
- a reason to evolve independently from a mission scenario.

Until then, explicit scenario-local configuration is valuable engineering documentation.
