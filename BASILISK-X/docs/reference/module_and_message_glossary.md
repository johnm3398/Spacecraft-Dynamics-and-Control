> **Draft status — AI-generated:** This page was drafted by OpenAI Codex on 2026-08-19. It has not yet been technically vetted by the BASILISK-X repository owner. The repository owner will review, correct, and maintain it. Verify APIs, units, frames, assumptions, and version compatibility before engineering use.

# Module and Message Glossary

This is a working vocabulary for the Basilisk 2.11.1 APIs and architectural patterns encountered in BASILISK-X. Names ending in `Msg`, `MsgPayload`, `InMsg`, and `OutMsg` are representative conventions; consult the module's API because exact member names vary.

See the [quick-start guide](../QUICK_START.md) for executable construction patterns and the [official Basilisk documentation](https://avslab.github.io/basilisk/) for generated module APIs.

## Simulation kernel and scheduling

| Term | Meaning and engineering use |
|---|---|
| `SimulationBaseClass.SimBaseClass` | Python-facing simulation container. Creates processes/tasks, schedules models, initializes the simulation, evaluates events, sets stop times, and executes the underlying simulation kernel. It is orchestration, not spacecraft physics. |
| `TotalSim` | Lower-level simulation object held by `SimBaseClass`. Most ordinary scenarios should use the higher-level `SimBaseClass` methods rather than manipulating it directly. |
| Process | Ordered grouping of tasks. Processes help separate environment, dynamics, FSW, or per-spacecraft execution. They are not automatically OS processes or independent simulated computers. Higher process priority executes first at coincident times. |
| Task | Fixed-period list of executable models. Use different tasks for genuine execution-rate differences—not merely to make the file look organized. |
| Task period/rate | Scheduler interval expressed in integer nanoseconds. It represents algorithm cadence; a dynamics object's numerical integrator can evaluate derivatives internally between scheduler updates. |
| Model/module | An executable simulation, sensor, FSW, logging, or interface object scheduled in a task. In Python, modules generally expose initialization/reset/update lifecycle methods through Basilisk's `SysModel` architecture. |
| `ModelTag` | Human-readable model identifier used in diagnostics, graphs, and some configuration paths. Assign unique, stable tags; it does not create message wiring. |
| Model priority | Ordering within a task when several models are due. Higher numeric priority executes earlier. Explicit priorities are valuable for same-tick write-before-read requirements. |
| Equal priority | Models with the same priority normally retain insertion/creation ordering. Relying on this implicitly makes a scenario fragile when a new model is inserted. |
| `CreateNewProcess()` | Constructs a process wrapper. A process becomes useful when at least one task is added. |
| `CreateNewTask()` | Constructs a periodic task using a period in nanoseconds. The task must be attached to a process. |
| `AddModelToTask()` | Registers an executable object or recorder with a named task. For an effector, this is often necessary in addition to attaching the effector physically to `Spacecraft`. |
| `InitializeSimulation()` | Runs the initialization lifecycle after configuration and subscriptions are complete. State registration, message self-initialization, and resets occur across this boundary. |
| `ConfigureStopTime()` | Sets an absolute scheduler time in nanoseconds. Successive mission segments must use increasing cumulative values. |
| `ExecuteSimulation()` | Advances scheduled work to the configured absolute stop time. It can be called repeatedly without rebuilding the simulation. |
| `ShowExecutionOrder()` / execution graph | Diagnostic facilities for inspecting processes, tasks, priorities, and message connections. Use them when tracing stale inputs or hidden timing assumptions. |
| `SysModel` | Base architecture for executable Basilisk models. A custom Python module typically subclasses `sysModel.SysModel`, calls `super().__init__()`, defines typed readers/messages, and implements reset/update behavior. See [`scenarioAttitudePointingPy.py`](../../examples/scenarioAttitudePointingPy.py). |
| `SelfInit` / self-initialization | Lifecycle phase in which a model prepares its own outputs/internal resources. Usually managed by `InitializeSimulation()`. |
| `Reset` | Lifecycle hook that validates links/configuration and initializes time-dependent internal state at a supplied simulation time. Do not use it as an arbitrary mid-run reset unless the module contract supports that operation. |
| `UpdateState` | Periodic model callback that reads inputs, computes, and writes outputs at a scheduled time. Exact language bindings can expose capitalization variants. |

## Spacecraft truth, environment, and dynamics

| Term | Meaning and engineering use |
|---|---|
| Truth state | The simulated physical state used as the reference reality. `Spacecraft.scStateOutMsg` is a common truth output. “Truth” is a semantic role, not a special message type or guarantee of physical fidelity. |
| `spacecraft.Spacecraft` | Hub-centric translational and rotational dynamic object. Registers/integrates hub states, collects effector contributions, and publishes spacecraft state. Appropriate for most conventional spacecraft simulations. |
| Hub | The base rigid body in a `Spacecraft`: mass, inertia, body reference point, initial position/velocity, attitude, and angular rate. State effectors couple to it. |
| `r_CN_NInit`, `v_CN_NInit` | Initial spacecraft center-of-mass position and velocity expressed in inertial frame `N`, normally SI units. Names must be interpreted literally; do not assume all `r_*` quantities refer to the same point. |
| `sigma_BNInit` | Initial modified Rodrigues parameters describing the body/inertial attitude relationship under Basilisk's convention. Confirm mapping direction before converting to a DCM or quaternion. |
| `omega_BN_BInit` | Initial angular velocity of body `B` relative to inertial `N`, expressed in `B`, in rad/s. |
| `SCStatesMsg` | Common spacecraft truth payload. Typical fields include inertial position/velocity, MRPs, body angular rate, and accumulated attitude-switch count. Exact semantics follow payload field names. |
| Gravity-body factory | `simIncludeGravBody.gravBodyFactory()` helper for constructing standard bodies, attaching gravity to spacecraft, and optionally building a SPICE interface. It is configuration convenience, not the gravity solver itself. |
| Central body | Gravity body selected as the reference origin for central-body-relative dynamics and outputs. Set intentionally when using planetary orbits. |
| Point-mass gravity | Lowest useful central-force model for many orbit questions. It should be the baseline against which harmonics and third bodies are added. |
| Spherical-harmonic gravity | Gravity field using degree/order coefficients. Tesseral and sectoral terms require correct body orientation; adding a high degree without planet rotation can be physically inconsistent. |
| SPICE interface | Uses kernels to publish celestial ephemeris and orientation messages. SPICE supplies states/orientations; it does not automatically enable gravity, occultation, SRP, or every environmental effect. |
| Ephemeris converter | Converts ephemeris representations/messages into forms required by downstream modules. It should not be confused with a trajectory propagator or estimator. |
| Integrator | Numerical method used by a dynamic object to propagate continuous states. Integrator choice, tolerances, and task rate are separate from physical-model fidelity. |
| `StateEffector` | Adds internal generalized states and couples mass, energy, and momentum to the spacecraft hub. Examples include reaction wheels, hinged bodies, fuel/slosh states, and flexible components. Typically attached with `addStateEffector()`. |
| `DynamicEffector` | Computes force and/or torque applied to a dynamic object without the same hub-coupled generalized-state role. Examples include drag, SRP, electrostatics, external force/torque, and many thruster effectors. Typically attached with `addDynamicEffector()`. |
| `ExtForceTorque` | Dynamic effector accepting or configuring external body/inertial forces and torques. Often used as an ideal actuator or disturbance. It bypasses physical actuator limits and resources unless those are modeled elsewhere. |
| Reaction-wheel state effector | Models wheel angular momentum and its exchange with the bus. It must be paired with wheel configuration/allocation and command messages for a closed-loop actuator chain. |
| Thruster dynamic/state effector | Converts on-time or force commands into physical force/torque. Different thruster modules have different transient and state assumptions; confirm whether fuel depletion is actually connected. |
| Fuel tank / depletion | State effector and mass-flow configuration that changes spacecraft mass properties as propellant is consumed. A thruster firing example is not automatically a fuel-depletion example. |
| Atmosphere model | Publishes density/environment information at spacecraft locations. Drag requires a separate area/aerodynamic effector connected to that atmosphere output. |
| Eclipse model | Computes occultation/illumination geometry. Power or sensor consequences occur only when consumers subscribe to its output. |
| SRP model | Computes solar-radiation pressure force/torque using a chosen optical/geometry model. Vizard visual geometry is not automatically SRP geometry. |

## Typed messaging

| Term | Meaning and engineering use |
|---|---|
| Message payload | Typed data structure containing named fields and their documented semantics. Construct with `SomeMsgPayload()`. Field names often encode points and frames. |
| Output message / `SomeMsg` | Message object owned or written by a producer. It holds the most recently written typed payload and associated write metadata. |
| Input reader / `SomeMsgReader` | Consumer-side typed handle. It must subscribe to a compatible output before it can receive producer data. |
| `subscribeTo()` | Connects one input reader to one output message: `consumer.input.subscribeTo(producer.output)`. Direction errors are a common source of failures. |
| `isLinked()` | Reader diagnostic indicating that a subscription exists. Linked does not necessarily mean that the producer has already written a non-default payload. |
| `isWritten()` | Diagnostic indicating whether the subscribed message has been written. Useful for first-sample and initialization debugging. |
| `timeWritten()` | Write timestamp associated with the latest message. Use it to diagnose stale inputs and multi-rate latency. |
| Last-value hold | A slower or non-coincident producer's most recent payload remains available to consumers. Messages do not automatically interpolate continuous values between writes. |
| Configuration message | Usually a constant message written once from Python, such as `messaging.VehicleConfigMsg().write(payload)`. It uses the same typed interface as a dynamic producer. Keep the Python message object alive for the scenario lifetime. |
| Gateway message | Stable intermediate output used to switch among several possible FSW authors without rewiring every downstream consumer. BskSim uses gateways heavily for mission modes. |
| Message direction | Defined by information ownership: truth/sensor/reference/command producer → consuming algorithm/effector. A variable named `InMsg` should not be used as a producer. |
| Message frame/unit contract | Semantic contract documented by the payload and module APIs. The messaging layer does no dimensional analysis or coordinate conversion. |
| Communication model | A physical or logical delay/loss/bandwidth system. A direct Basilisk subscription is not a realistic spacecraft communications link. |

## Navigation, guidance, control, and actuators

| Term | Meaning and engineering use |
|---|---|
| Sensor model | Converts truth/environment into a measurement, normally with geometry, cadence, bias, noise, limits, and validity behavior. A sensor output is neither truth nor automatically a state estimate. |
| `SimpleNav` | Navigation emulation module that publishes attitude and translational navigation messages from spacecraft truth, with configurable error processes. Default or lightly configured use is often truth-like and should not be called a validated estimator. |
| `NavTransMsg` | Typical translational navigation solution containing position/velocity information and associated timing/quality fields defined by its payload. Determine whether the producer supplies truth-like data, measurements, or an estimate. |
| `NavAttMsg` | Typical attitude navigation solution containing attitude/rate information. It is usually consumed by tracking, pointing, or control algorithms rather than by truth dynamics. |
| Estimator/filter | Algorithm that combines measurements and dynamics to produce a navigation state and uncertainty. Examples include CSS filters and the UKFs in the small-body and OpNav scenarios. |
| Guidance | Generates a desired reference trajectory or attitude from mission objectives and navigation information. It should not directly alter truth state. |
| `AttRefMsg` | Attitude-reference payload, commonly containing reference attitude, angular rate, and angular acceleration. Produced by modules such as `inertial3D`, `hillPoint`, `locationPointing`, or `velocityPoint`. |
| Attitude tracking | Computes body-relative-to-reference attitude and rate error from an attitude reference plus navigation. `attTrackingError` commonly publishes `AttGuidMsg`. |
| `AttGuidMsg` | Tracking-error payload passed to attitude controllers. Typical fields include `sigma_BR` and `omega_BR_B`; verify frame direction and expression frame. |
| Controller | Converts estimated/reference errors into a requested generalized command, such as body torque. It does not by itself guarantee actuator feasibility. |
| `mrpFeedback` | Nonlinear MRP feedback controller producing a body torque request from `AttGuidMsg` and vehicle inertia/configuration. Gains, integral enable/limit, unmodeled torques, and actuator saturation all affect interpretation. |
| `CmdTorqueBodyMsg` | Requested control torque expressed in the body frame. It can feed an ideal torque effector or an allocation algorithm. The payload is a request, not measured actuator torque. |
| Control allocation | Maps a generalized torque/force request into individual actuator commands. Examples include `rwMotorTorque` for reaction wheels and thruster-force/on-time mapping algorithms. |
| `ArrayMotorTorqueMsg` | Per-wheel motor torque commands. Wheel ordering must match the reaction-wheel configuration message and the state effector. |
| `THRArrayOnTimeCmdMsg` | Per-thruster firing durations/on-times. Ordering and interpretation must match the thruster configuration. Minimum impulse bit and valve/transient behavior require explicit models. |
| Actuator dynamics | Physical module converting allocated commands into forces, torques, momentum, power, and possibly mass flow. It closes the command chain back into truth dynamics. |
| Mission logic | Mode selection, command sequencing, resource/safety gating, and phase transitions. Keep it outside truth dynamics and connect it through messages/events. |

## Recording, diagnostics, events, and visualization

| Term | Meaning and engineering use |
|---|---|
| Message recorder | Scheduled model returned by `outputMsg.recorder()`. Retains payload fields and timestamps for post-processing. Its task placement and priority determine which write it samples. |
| Recorder sampling interval | Optional minimum interval between retained samples. It reduces stored data but does not change the producer's computation rate. |
| Variable logger | Scheduled logger for selected internal module attributes not exposed on messages. Use it sparingly; message interfaces are usually more stable. |
| `bskLogger` | Runtime diagnostic logging facility with severity levels. It is distinct from numerical telemetry recording. |
| Event | Periodically evaluated condition plus actions. Common uses are mode switching, failure injection, command changes, and termination. It is Python orchestration, not continuous physical dynamics. |
| Task enable/disable | Scheduler-level mode mechanism. Disabling a task can leave its last output held, so gateways or explicit zero commands may also need updating. |
| Recorder timestamp | Simulation time at which the recorder sampled. Join multi-rate outputs by timestamp, not by assuming equal array indices. |
| Plotting | Python analysis performed after or during a simulation. A plot is not validation; compute units, residuals, conservation quantities, uncertainties, and success metrics explicitly. |
| Vizard | AVS Lab visualization application interfaced through `vizSupport`/`vizInterface`. Usually consumes truth for playback/live display. In OpNav, it can render imagery returned to camera-processing modules. |
| Playback | Simulation writes a Vizard data file for later interactive viewing. Usually the most reproducible visualization mode. |
| Live stream | Simulation and Vizard exchange data during execution. It adds ports, synchronization, external-process, and headless-test considerations. |

## Time, units, and frames

| Term | Meaning and engineering use |
|---|---|
| Simulation time | Scheduler time represented as integer nanoseconds. Use `macros` conversions instead of raw floating-point seconds in task/stop APIs. |
| Absolute stop time | `ConfigureStopTime()` target measured from simulation epoch. It is not a duration from the most recent `ExecuteSimulation()` call. |
| `macros.sec2nano`, `min2nano`, `hour2nano` | Convert common time units into scheduler nanoseconds. Reverse constants such as `NANO2SEC` support analysis. |
| SI convention | Basilisk generally uses metres, seconds, kilograms, radians, newtons, and newton-metres. Exceptions are module-specific. Unit suffixes in local variable names are valuable but are not checked by the framework. |
| Degrees/radians | Dynamics and algorithm angles are generally radians. Use `macros.D2R` and `macros.R2D` at explicit user/input/output boundaries. |
| `N` frame | Inertial/reference frame notation used throughout spacecraft states. The exact inertial origin/orientation depends on the configured celestial system. |
| `B` frame | Spacecraft body-fixed frame. `r_BN_N` denotes point `B` relative to `N`, expressed in `N`; `omega_BN_B` is body relative to inertial, expressed in `B`. |
| `R` frame | Guidance reference frame. Tracking commonly describes `B` relative to `R`. It is not universally Hill/LVLH unless the selected guidance module defines it that way. |
| Hill/LVLH frame | Chief/orbit-relative rotating frame. Basilisk utilities and relative-motion modules document axis definitions; do not assume another tool's RIC/RTN/LVLH signs and ordering are identical. |
| Point/frame notation | A compact convention such as `r_PQ_F`: position of point `P` relative to point `Q`, components expressed in frame `F`. Read all three parts before using a vector. |
| MRP | Modified Rodrigues parameter attitude representation used widely in Basilisk FSW. MRPs have a shadow set; switch count and short/long rotation behavior can matter during analysis. |
| Quaternion/EP | Euler-parameter representation. Component ordering, scalar position, active/passive interpretation, and mapping direction must be declared at every external adapter. |
| DCM | Direction cosine matrix. The subscript order defines which coordinate representation is mapped into which. Do not infer direction from a generic variable named `C`. |

## Repository architecture families

| Term | Meaning and appropriate use |
|---|---|
| Standalone scenario | One Python file explicitly builds a question-specific simulation. Best for learning, trade studies, and modest one-off analyses. See [`scenarioBasicOrbit.py`](../../examples/scenarioBasicOrbit.py). |
| BskSim | Reusable platform architecture dividing scenario, spacecraft dynamics, FSW tasks, gateways, and events. Useful when many missions reuse one vehicle/mode library; unnecessary for a small isolated question. See [`BskSim/BSK_masters.py`](../../examples/BskSim/BSK_masters.py). |
| MultiSatBskSim | Shared-world plus indexed per-satellite dynamics/FSW processes, with optional formation-level models. Useful for reusable fleet studies, but direct message links do not imply a physical network. See [`MultiSatBskSim/BSK_MultiSatMasters.py`](../../examples/MultiSatBskSim/BSK_MultiSatMasters.py). |
| OpNavScenarios | BskSim-derived research architecture linking truth, Vizard-generated images, camera degradation, Hough/limb/CNN processing, measurement conversion, navigation filters, and FSW. It has optional/external dependencies and local version drift. See [`OpNavScenarios/BSK_OpNav.py`](../../examples/OpNavScenarios/BSK_OpNav.py). |
| MuJoCo integration | Alternative `MJScene` dynamic object using MJCF bodies, joints, actuators, constraints, contact, and MuJoCo integration-stage callbacks. Prefer standard `Spacecraft` for conventional hub/effectors; use MuJoCo for general mechanisms/contact. See [`mujoco/scenarioReactionWheel.py`](../../examples/mujoco/scenarioReactionWheel.py). |
| Monte Carlo controller | Infrastructure that repeatedly constructs deterministic simulations, applies seeded dispersions, executes runs, retains selected outputs, archives parameters, and invokes callbacks/analysis. See [`MonteCarloExamples/scenarioBskSimAttFeedbackMC.py`](../../examples/MonteCarloExamples/scenarioBskSimAttFeedbackMC.py). |
| BSK-RL | Separate AVS Lab framework layered over Basilisk for Gymnasium/PettingZoo satellite tasking. It defines observations, actions, rewards, reset/episode behavior, data stores, and agent coordination. It is not present in this local example tree. See the [official BSK-RL API](https://avslab.github.io/bsk_rl/api_reference/index.html). |
| BASILISK-X | This repository's learning/experimentation layer around Basilisk. Reusable, tested cross-scenario behavior belongs in `src/basiliskx`; mission assumptions and one-off configurations should remain scenario-local until repetition proves an abstraction. |

## A compact interface-reading checklist

For every module added to a simulation, record these facts before wiring it:

1. What physical or algorithmic role does it represent: truth, sensor, navigation, guidance, control, actuator, resource, or analysis?
2. Which task runs it, at what period and priority?
3. What is every input payload's producer?
4. What point, frame, direction, units, timestamp, and validity semantics does each field have?
5. What outputs does it own, and which downstream consumers use them?
6. Does it need both scheduler registration and physical attachment to a dynamic object?
7. What parameters and message links must exist before `InitializeSimulation()`?
8. What simplifying assumptions separate this module from the intended engineering system?
9. What truth quantity, interface telemetry, residual, or conservation metric will validate it?

If those questions cannot be answered from source or official documentation, treat the interface as unresolved rather than filling the gap by assumption.
