> **Draft status — AI-generated:** This page was drafted by OpenAI Codex on 2026-08-19. It has not yet been technically vetted by the BASILISK-X repository owner. The repository owner will review, correct, and maintain it. Verify APIs, units, frames, assumptions, and version compatibility before engineering use.

# BSK-RL and Decision Autonomy

## 1. Scope and provenance

**BSK-RL is not present in this repository.** The audited `BASILISK-X/examples/`, `src/basiliskx/`, `scenarios/`, tests, dependency files, and local source tree contain no BSK-RL environment, satellite agent, Gymnasium/PettingZoo integration, rewarder, or training code. Basilisk examples with events or mode switches demonstrate deterministic autonomy, not reinforcement learning.

This chapter therefore describes the separately maintained AVS Lab framework from its official documentation and source. The version channels are not identical as of 2026-08-19: the hosted documentation identifies itself as **v1.3.4** and labels that line **Development**, [PyPI lists v1.3.2](https://pypi.org/project/bsk-rl/) as the latest published package, and the [GitHub releases page](https://github.com/AVSLab/bsk_rl/releases) marks v1.3.0 as the latest tagged release. The API descriptions below follow the hosted development documentation. Pin a chosen release and use its matching source/documentation before implementation.

No PPO, SAC, DQN, or other learning algorithm is selected or implemented here. The aerospace problem formulation must come first.

## 2. Where Basilisk ends and the RL problem begins

```text
                         BSK-RL / decision layer

scenario opportunities ─┐
local knowledge ─────────┤
resource/navigation obs ─┤→ observation → policy → high-level action
                         │                         │
reward/termination ──────┘                         v
                                               FSW mode/task
                                                   │
                         Basilisk layer             v

world/environment → spacecraft dynamics → sensors/navigation
        ↑                    │              │
        │                    └→ resources ←─┘
        │                           │
        └──── force/torque ← actuators ← low-level FSW
```

| Basilisk should own | The autonomy formulation should own |
|---|---|
| Truth state propagation and environmental physics | Information actually exposed to the decision maker |
| Sensor, navigation, actuator, power, storage, and thermal consequences at selected fidelity | Action semantics and decision cadence |
| Low-level guidance, control, allocation, and FSW modes | Mission value/reward definition |
| Continuous-time evolution between decisions | Constraints, safety supervision, and allowed operating envelope |
| Physical and numerical failures | Episode initialization, randomization, termination, and evaluation protocol |

The policy should ordinarily command a validated FSW mode or bounded maneuver request. It should not replace stable low-level attitude control merely because a neural network can emit torques. That is a different research question with a larger verification burden.

## 3. Official BSK-RL architecture

The official simulation package divides the underlying Basilisk model into shared **World**, per-satellite **Dynamics**, and per-satellite **FSW** models. Its `Simulator` subclasses Basilisk `SimBaseClass` and is reconstructed on each environment reset. [Official simulation architecture](https://avslab.github.io/bsk_rl/api_reference/sim/index.html)

```text
GeneralSatelliteTasking / SatelliteTasking / ConstellationTasking
│
├── Scenario
│     targets, inspection points, or other data opportunities
│
├── WorldModel
│     shared epoch, gravity, atmosphere, ground stations, etc.
│
├── Satellite agent A
│     ├── observation_spec
│     ├── action_spec
│     ├── DynamicsModel  ─ physical platform, actuators, payload/resources
│     ├── FSWModel       ─ low-level modes and algorithms
│     └── DataStore      ─ local mission knowledge/data
│
├── Satellite agent B ...
│
├── GlobalReward / composed rewarders
└── CommunicationMethod
      decides which local DataStores exchange information
```

### Primary objects

| Object | Official role | Engineering interpretation |
|---|---|---|
| `Satellite` | Base agent class; subclass defines `observation_spec`, `action_spec`, and selected dynamics/FSW model types | Configuration boundary between an agent and its simulated spacecraft |
| `AccessSatellite` | Computes ordered future ground-location access opportunities | Look-ahead opportunity service, not an onboard orbit-determination or RF model |
| `ImagingSatellite` | Extends access behavior with target imaging and event-driven completion/miss behavior | Taskable EO-style spacecraft abstraction |
| `WorldModelABC` | Shared environment required by the fleet | Centralized simulation world; it need not correspond to information known onboard |
| `DynamicsModelABC` | Per-satellite physical platform, actuators, instrument, power, and storage | Basilisk truth/resource plant selected for the problem |
| `FSWModelABC` | Per-satellite low-level actuator and instrument control | Command realization and modes beneath the policy |
| `Scenario` | Defines targets/data-generating environment | Mission opportunity generator, distinct from spacecraft physics |
| `Simulator` | Basilisk `SimBaseClass` wrapper constructed for an episode | Advances continuous physics between decision epochs |

The official satellite API permits `sat_args` values to be constants or functions evaluated on reset, and a fleet-level `sat_arg_randomizer` can create correlated overrides. [Official satellite API](https://avslab.github.io/bsk_rl/api_reference/sats/index.html)

## 4. Environment interfaces

The official API provides three base environments:

| Environment | API | Agent count | Intended interface |
|---|---|---:|---|
| `SatelliteTasking` | Gymnasium | 1 | Direct single-agent observation/action |
| `GeneralSatelliteTasking` | Gymnasium | 1 or more | Joint tuple of observations/actions; useful for multi-satellite testing and centralized interfaces |
| `ConstellationTasking` | PettingZoo parallel API | 1 or more | Per-agent dictionaries for multi-agent training |

`ConstellationTasking` can also group satellites under meta-agents, but agent grouping does not itself solve credit assignment, communication, heterogeneity, or decentralized execution.

The environment is naturally semi-Markov when actions have variable duration or Basilisk events end a step early. Decision interval must therefore be part of discounting, observation, reward normalization, and baseline comparison—not hidden as if every action consumed one identical time step.

## 5. What reset does

The official implementation deletes the previous simulator and associated Basilisk objects, seeds/reset state, randomizes the time limit and model arguments, invokes reset hooks, creates the new `Simulator`, initializes its world/dynamics/FSW, initializes data stores, and returns the first observation.

```text
env.reset(seed)
   ↓
delete previous Simulator/Basilisk model graph
   ↓
reset Scenario, rewarder, communicator, satellites
   ↓
sample world_args and sat_args, including correlated overrides
   ↓
pre-initialization hooks
   ↓
construct World + per-satellite Dynamics + per-satellite FSW
   ↓
finish Basilisk initialization
   ↓
post-initialization hooks and initial DataStore state
   ↓
observation, info
```

Reconstruction is an important correctness feature: an episode should not inherit wheel speed, battery charge, integrator state, events, recorders, or messages from the previous episode. It also makes reset cost and object cleanup material to high-throughput training.

Randomization is not automatically statistically correct. Correlated orbit injection, shared environmental errors, fleet manufacturing batches, sensor calibration, and navigation covariance require deliberate joint distributions and seed management.

## 6. What step does

The official source implements the central loop in this order:

```text
satellite.set_action(action)
          ↓
Simulator.run() until max duration, time limit, or enabled event
          ↓
each DataStore.update_from_logs()
          ↓
rewarder.reward(new_data)
          ↓
communicator.communicate()
          ↓
new observation, reward, terminated, truncated, info
```

[Official environment source](https://avslab.github.io/bsk_rl/_modules/bsk_rl/gym.html)

This ordering has consequences. Reward is calculated from newly logged data before same-step communication updates local knowledge; subsequent observations can reflect communicated knowledge. Confirm ordering again for the installed version before building assumptions into a policy or metric.

The simulator can stop early on an event—for example, target image success, missed opportunity, or a mode completion—so `info["d_ts"]`/step duration matters when interpreting reward rate and discounting.

## 7. Observations: what the policy is allowed to know

Official observation components include:

- `SatProperties`: selected properties from dynamics or FSW;
- `RelativeProperties`: properties relative to another satellite;
- `Time`;
- `OpportunityProperties`: look-ahead data for upcoming targets/access points;
- `Eclipse`;
- `ResourceRewardWeight` for randomized resource tradeoffs.

They are assembled through a satellite subclass's `observation_spec`. [Official observation API](https://avslab.github.io/bsk_rl/api_reference/obs/index.html)

### State is not observation

For a partially observable mission:

```text
true physical/environment state x_t
             ↓ sensor + estimator + information history
policy observation o_t
```

Do not expose a truth property through `SatProperties` merely because it is available in the simulator. Examples of leakage include:

- true orbit when the mission is navigation-limited;
- exact target state when the spacecraft has angles-only measurements;
- other satellites' buffers without a communication path;
- future access/priority information beyond onboard prediction capability;
- hidden failure state that has not been detected;
- global “already serviced” knowledge when agents have only local knowledge.

Observation design should specify normalization, bounds, frame, timestamp/age, missing-data encoding, permutation handling for opportunity lists, and whether a history or recurrent policy is needed. A padded opportunity vector is an interface choice with potential ordering and aliasing effects.

## 8. Actions: decisions, not magical outcomes

The official action set includes discrete FSW actions such as `Charge`, `Drift`, `NadirPoint`, `Desat`, `Downlink`, `Image`, and `Scan`. Continuous actions include `ImpulsiveThrust`, chief-relative `ImpulsiveThrustHill`, and an MRP `AttitudeSetpoint`. [Official action API](https://avslab.github.io/bsk_rl/api_reference/act/index.html)

An action should define:

- command semantics and frame;
- bounds and quantization;
- action duration and interruption/preemption rules;
- conditions for success, failure, or early retasking;
- actuator/FSW model that realizes it;
- resource, time, and safety consequences;
- what happens when the action is infeasible.

`ImpulsiveThrust` changes velocity instantaneously in the documented action. It is appropriate for high-level maneuver policy studies where ideal impulses are the declared abstraction. It is not evidence of thrust pointing, collision avoidance during a burn, propellant feasibility, plume safety, or navigation performance. RPO research should eventually replace or supervise it with bounded maneuver primitives realized through validated guidance and propulsion.

## 9. Scenario and access opportunities

Official scenario types include `UniformTargets`, `CityTargets`, and `UniformNadirScanning` for Earth observation, plus `SphericalRSO` for inspection research. [Official scenario API](https://avslab.github.io/bsk_rl/api_reference/scene/index.html)

`AccessSatellite` calculates and orders upcoming opportunities based on per-location minimum elevation. It can extend its look-ahead horizon and filter opportunities using local knowledge. This is useful for event-driven tasking, but it remains a predictive geometry abstraction. For flight-representative autonomy, decide whether those windows are:

- computed onboard from an estimated orbit and target catalogue;
- uploaded from the ground;
- uncertain because of navigation, attitude-settling, weather, or target-state error;
- revised during the episode.

## 10. Data, local knowledge, and reward

The official reward architecture separates three concepts:

```text
Data       mission fact/value produced or learned
  ↓
DataStore  each satellite's local knowledge and accumulated data
  ↓
GlobalReward  omniscient evaluator/critic that scores new Data
```

At reset, a rewarder can provide initial knowledge. After each step, a data store compares current Basilisk logs with the prior log state to infer newly generated `Data`. Communication then merges data stores between permitted satellite pairs. [Official data and reward API](https://avslab.github.io/bsk_rl/api_reference/data/index.html)

Available reward systems documented there include unique target images, nadir scanning time, arbitrary resource change, and RSO inspection coverage. A `GlobalReward` may know facts hidden from individual agents. That is a valid evaluator design: reward need not be part of observation. It must not be mistaken for a realizable onboard signal when claiming decentralized autonomy.

### Reward is an engineering specification

A useful reward should reflect mission value after constraints, not merely whatever telemetry is easy to count:

\[
R = V_{\text{mission products}}
  - C_{\text{resources}}
  - C_{\text{time/opportunity}}
  - C_{\text{risk/violations}}.
\]

Keep units and scales interpretable. Compare learned behavior against simple deterministic policies before adding shaping terms.

Common reward-hacking failures include:

| Reward shortcut | Possible exploit | Better specification |
|---|---|---|
| Reward every image command | Repeatedly command impossible/duplicate images | Reward verified unique data products generated in valid geometry |
| Reward time in a productive mode | Remain in the mode without producing useful data | Derive `Data` from payload/storage changes and quality criteria |
| Small penalty for low battery | Spend battery to gain reward, then terminate | Treat survival/resource floors as hard constraints or large terminal consequences |
| Reward proximity to target | Approach unsafely or loiter at collision risk | Reward task progress subject to keep-out, passive-safety, and closing-rate constraints |
| Reward total fleet output only | One agent monopolizes tasks; others fail | Add survival/coverage/fairness metrics only if mission requirements demand them |
| Reward immediate downlink volume | Dump low-value data and neglect future access | Score delivered mission value with time/resource opportunity cost |

## 11. Communication is an information model

Official communication options include no communication, free all-to-all communication, direct line-of-sight communication, multi-degree paths, and line-of-sight multi-hop behavior. They determine which satellite `DataStore` objects share knowledge after a step. [Official communication API](https://avslab.github.io/bsk_rl/api_reference/comm/index.html)

This is not automatically a physical radio/network simulation. In particular, documented multi-hop behavior can propagate information instantaneously through a connected path within a step. Add a separate model if the research question depends on:

- antenna pointing or link budgets;
- message size, bandwidth, latency, queues, or packet loss;
- protocol/routing behavior;
- stale or out-of-order knowledge;
- adversarial, intermittent, or asymmetric links;
- energy cost of communication.

For decision research where only “knowledge exchange possible under LOS” matters, the built-in abstraction may be exactly the right fidelity.

## 12. Formulate the aerospace POMDP before selecting an algorithm

Use this worksheet.

### State \(x_t\)

The Markov state contains everything required to predict the future under an action, even if the policy cannot observe it:

- truth orbit, attitude, actuator, propellant, power, storage, and thermal states;
- environment/target/other-agent state;
- sensor biases and navigation-filter state;
- mode timers, command queues, faults, and communication state;
- mission-product and local/global knowledge state.

If the state omits a persistent variable such as sensor bias or wheel momentum, the simulated problem may not be Markov even if the observation vector looks large.

### Observation \(o_t\)

Specify only onboard/authorized information: navigation estimates and covariance, resource telemetry, visible opportunity metadata, local data store, received messages, mode status, and time. Define latency, noise, dropout, normalization, ordering, and history.

### Action \(a_t\)

Choose the decision authority being studied:

- FSW mode selection;
- target or ground-station choice;
- bounded maneuver primitive;
- communication/task-allocation decision;
- resource allocation or dwell time.

Avoid mixing high-level target choice with raw wheel torque unless cross-layer control is the explicit research contribution.

### Dynamics \(p(x_{t+1}|x_t,a_t)\)

State which portions are Basilisk physics, deterministic mission logic, stochastic sensor/environment processes, and deliberately simplified action effects. Include variable action duration if applicable.

### Reward \(r_t\)

Tie every term to a measurable mission objective or cost. Decide whether reward is per decision, per unit time, event-based, delivered-product-based, local, shared, or centralized.

### Constraints

Separate:

- hard invariants: collision, keep-out, minimum power, thermal survival, actuator/propellant bounds;
- chance constraints under state uncertainty;
- soft preferences that may appear in reward;
- supervisory action shields or fallback modes.

Safety should not depend only on the policy discovering that violations receive negative reward.

### Uncertainty

Define distributions and correlations for initial state, navigation, target knowledge, environment, device performance, faults, and communication. Separate training randomization from evaluation cases and preserve seeds/provenance.

### Termination and truncation

Use termination for true terminal outcomes such as satellite failure, collision, completed inspection, or an absorbing mission state. Use truncation for external episode limits such as time or compute horizon. Preserve the distinction during bootstrapping and evaluation.

## 13. Single-agent and multi-agent choices

| Architecture | Appropriate when | Risk to watch |
|---|---|---|
| Single satellite, single policy | Task scheduling or resource trade on one spacecraft | Hiding ground/fleet coordination in omniscient scenario inputs |
| Centralized joint policy | Small fleet with shared observation/action and centralized operations | Action/observation dimension growth and unrealistic global knowledge |
| Independent per-satellite policies | Homogeneous fleet and decentralized execution | Non-stationarity, duplicate work, weak coordination |
| Centralized training, decentralized execution | Training can use global critic while onboard policies use local information | Leakage of centralized information into actor observations or evaluation |
| Hierarchical/meta-agent | Fleet allocation above satellite-level mode selection | Ambiguous authority, timing, and credit across layers |

PettingZoo compatibility supplies an interface, not a coordination solution. Define which information and actions belong to each agent, how shared reward is assigned, and what failures remove an agent.

## 14. Mapping research problems into BSK-RL

| Research objective | Observation | Action | Mission value | Required extensions/cautions |
|---|---|---|---|---|
| Earth-observation scheduling | Estimated orbit/time, power/storage, next target windows/priorities, known imaged set | Image target, charge, downlink, drift | Unique delivered value, timeliness, survival | Weather/clouds, slew time, ground capacity if material |
| Continuous nadir scanning | Resource states, illumination, current coverage | Scan, charge, downlink | Valid scanned area/time | Avoid rewarding mode time without verified data |
| Space-domain awareness/inspection | Relative navigation and covariance, visible/illuminated surface points, fuel/power, keep-out margins | Viewpoint or bounded maneuver primitive | New validated surface coverage/information | `SphericalRSO` is an abstraction; add target dynamics, sensor/estimator, safety |
| RPO decision making | Relative estimate/covariance, approach corridor, passive-safety and actuator margins | Hold, retreat, waypoint/burn primitive | Task progress with propellant/time cost | Do not use truth-relative state or unconstrained impulses for safety claims |
| Constellation tasking | Local opportunities/resources, local/received task knowledge | Target/mode/communication choice | Fleet mission value | Communications are data-store sharing unless physical network added |
| Distributed coordination | Local state plus delayed peer messages | Bid/task/relay/execute | Coverage, latency, resilience | Model message age, bandwidth, failures, and asymmetric knowledge as required |
| Navigation-aware planning | Estimate, covariance/quality, future geometry | Observe landmark, maneuver, service task | Mission value minus uncertainty/risk | Covariance must influence consequences, not merely appear in observation |

## 15. Training versus engineering evaluation

Training needs throughput; engineering evaluation needs evidence.

### Training model

- Use the minimum physics that preserves the decision trade.
- Keep stable low-level FSW below the policy.
- Randomize only uncertainties with a defensible distribution.
- Profile reset and event behavior; do not sacrifice causality for speed silently.
- Track seeds, environment version, configuration, normalization, and reward definition.

### Evaluation model

- Freeze the policy and run held-out seeds/scenarios.
- Compare against deterministic rules, optimization baselines, and simple heuristics.
- Test out-of-distribution but plausible conditions and individual failure modes.
- Escalate physics/sensor/actuator/communications fidelity and measure policy degradation.
- Report mission success probability, constraint violations, tail risk, resources, and confidence intervals—not only mean return.
- Replay failures with full telemetry and inspect whether the observation contained sufficient warning.
- Test reproducibility across process counts and environment resets.

A policy that scores well only under the exact training reward and truth-like observation is a simulator exploit, not demonstrated spacecraft autonomy.

## 16. Safe adoption path for BASILISK-X

Before adding the external package:

1. Complete deterministic, testable Basilisk models for the intended mission.
2. Define a non-learning baseline and mission-success function.
3. Identify the FSW modes or maneuver primitives the policy may command.
4. Define a policy observation that contains no unavailable truth.
5. Add safety supervision outside the learned policy.
6. Create reset/randomization and evaluation provenance.
7. Keep BSK-RL an optional, version-pinned research dependency rather than a required core dependency.
8. Place reusable physical/FSW extensions in `src/basiliskx`; keep experiment-specific observations, actions, and rewards with the autonomy experiment until reuse is demonstrated.

Only after this should algorithm choice and hyperparameter work begin.

## 17. Official sources for this external framework

- [Release notes and development-version status](https://avslab.github.io/bsk_rl/release_notes.html)
- [Published Python package versions](https://pypi.org/project/bsk-rl/)
- [Tagged GitHub releases](https://github.com/AVSLab/bsk_rl/releases)
- [API and environment classes](https://avslab.github.io/bsk_rl/api_reference/index.html)
- [Satellite agents and access opportunities](https://avslab.github.io/bsk_rl/api_reference/sats/index.html)
- [World, dynamics, FSW, and simulator architecture](https://avslab.github.io/bsk_rl/api_reference/sim/index.html)
- [Observations](https://avslab.github.io/bsk_rl/api_reference/obs/index.html)
- [Actions](https://avslab.github.io/bsk_rl/api_reference/act/index.html)
- [Data stores and rewards](https://avslab.github.io/bsk_rl/api_reference/data/index.html)
- [Communication](https://avslab.github.io/bsk_rl/api_reference/comm/index.html)
- [Scenarios](https://avslab.github.io/bsk_rl/api_reference/scene/index.html)
- [Environment source used to verify reset/step ordering](https://avslab.github.io/bsk_rl/_modules/bsk_rl/gym.html)
