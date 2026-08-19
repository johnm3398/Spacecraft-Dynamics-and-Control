> **Draft status — AI-generated:** This page was drafted by OpenAI Codex on 2026-08-19. It has not yet been technically vetted by the BASILISK-X repository owner. The repository owner will review, correct, and maintain it. Verify APIs, units, frames, assumptions, and version compatibility before engineering use.

# BASILISK-X Architecture and Extension Policy

This chapter defines how BASILISK-X should grow around Basilisk without becoming a competing simulation framework. It evaluates the present repository and provides criteria for deciding what remains scenario-specific, what belongs in `src/basiliskx`, and when a larger architecture such as BskSim or MultiSat becomes justified.

## Project boundary

BASILISK-X is an engineering learning and experimentation layer:

```text
AVS Lab Basilisk
  scheduler, messages, dynamics, FSW modules, utilities, Vizard interface
                         |
                         v
BASILISK-X
  scenario integration, experiments, engineering metrics,
  narrowly reusable helpers, custom algorithms, tests, documentation
```

BASILISK-X should not duplicate `SimulationBaseClass`, message types, standard gravity factories, spacecraft dynamics, or existing FSW modules merely to present a different namespace.

The upstream project remains the authority for core Basilisk behavior. BASILISK-X owns the correctness of its configuration, custom algorithms, adapters, analysis, and claims.

## Current repository architecture

```text
BASILISK-X/
├── examples/        copied upstream teaching/research material
├── scenarios/       BASILISK-X question-specific simulations
├── src/basiliskx/   reusable BASILISK-X behavior
├── tests/           current reusable-utility tests
├── docs/            canonical field-manual Markdown
├── pyproject.toml
└── requirements.txt
```

### Current scenarios

[`basic_earth_orbit.py`](../scenarios/basic_earth_orbit/basic_earth_orbit.py) is a standalone point-mass orbit baseline. It separates construction, execution, visualization, derived quantities, plotting, and reporting.

[`nadir_pointing.py`](../scenarios/nadir_pointing/nadir_pointing.py) adds the canonical message chain:

```text
Spacecraft -> SimpleNav -> hillPoint -> attTrackingError
           -> mrpFeedback -> ideal ExtForceTorque -> Spacecraft
```

[`cooperative_geo_rendezvous.py`](../scenarios/cooperative_geo_rendezvous/cooperative_geo_rendezvous.py) constructs two independent spacecraft, computes Hill/CW targeting in Python, advances explicit phases, and applies ideal velocity discontinuities.

These are appropriately standalone. They answer different questions and do not yet share a stable vehicle or FSW product line that would justify BskSim-style master classes.

### Current reusable code

[`vizard_launcher.py`](../src/basiliskx/visualization/vizard_launcher.py) is presently the clearest reusable subsystem. It centralizes executable discovery, launch modes, process state, wait, termination, and cleanup without changing spacecraft physics.

That is a good extraction because:

- all three scenarios need the behavior;
- the operating-system lifecycle is independent of mission physics;
- it has a narrow interface;
- it has dedicated tests;
- failures can be stated clearly.

## What belongs in a scenario

Keep behavior scenario-local when it expresses the engineering experiment:

- spacecraft/environment topology;
- mission-specific initial states and epochs;
- controller gains selected for one vehicle;
- actuator layout and resource sizing for one concept;
- chief/deputy selection and maneuver sequence;
- mission success, safety, and constraint metrics;
- fidelity toggles whose meaning is specific to that study;
- plots and reports tied to those metrics;
- ideal state resets used as an explicit mission-analysis abstraction;
- Vizard camera/layout choices used only for presentation.

Scenario-local code is not inferior code. It preserves the traceability between a question, its assumptions, and its result.

## What can belong in `src/basiliskx`

A candidate reusable component should satisfy most of these tests:

1. It has two or more real consumers with the same semantics.
2. Its API can state frames, units, time, ownership, and failure behavior explicitly.
3. It can be tested without running an unrelated full mission.
4. It hides incidental mechanism rather than hiding engineering assumptions.
5. It does not duplicate a stable Basilisk utility or module.
6. It can evolve without forcing several scenarios to change their mission meaning.
7. Its name describes behavior rather than a current scenario.

Likely future candidates are below.

### Visualization session support

Build on the current launcher with a small session abstraction only if it removes repeated, tested live/playback lifecycle code:

```text
scenario supplies Vizard configuration
        -> reusable session owns launch/wait/cleanup
        -> scenario remains responsible for visualization semantics
```

Do not move spacecraft/Vizard model configuration into a universal helper unless several scenarios genuinely share it.

### Structured run results

Scenarios currently return `None`. A small immutable result/configuration boundary could support:

- headless regression tests;
- programmatic parameter studies;
- reproducible metadata;
- a consistent distinction between raw recorder objects and derived engineering metrics.

A result type should not prescribe one universal state vector. Prefer scenario-specific result dataclasses that can share small protocols or metadata structures.

### Validated frame and mission-analysis utilities

Potential examples include:

- Hill-state history conversion with explicit chief, frame, units, and timestamps;
- CW targeting with domain checks and analytic tests;
- ground-track conversion with explicit epoch/body rotation/geodetic assumptions;
- delta-v reconstruction from finite thrust and mass histories;
- common numerical comparison or convergence metrics.

Extract these only after their assumptions are documented and tested. A generic `relative_position()` helper with an implicit chief or frame would reduce safety.

### Sensor/navigation adapters

An adapter is justified when several scenarios must convert the same external or custom representation into a standard Basilisk message contract. It must declare:

- source and destination frames;
- units and timestamp semantics;
- covariance/state ordering;
- invalid/stale-data behavior;
- whether it is a physical sensor, measurement conversion, or navigation estimate.

Do not create an adapter solely to rename a standard Basilisk payload.

### Custom FSW modules

Write a custom module when Basilisk lacks the required algorithm or the research question is the algorithm itself. It should:

- subclass the appropriate `SysModel` architecture;
- expose typed input readers and output messages;
- validate required links/configuration in `Reset`;
- execute through `UpdateState` at a declared task rate;
- avoid direct mutation of truth state;
- publish enough telemetry for independent validation;
- have unit tests outside a full mission scenario.

[`scenarioAttitudePointingPy.py`](../examples/scenarioAttitudePointingPy.py) is the basic structural reference.

## Abstractions that are premature now

### A replacement simulation base class

BASILISK-X does not need its own wrapper around every process, task, and message call. Such a layer would make upstream documentation harder to apply and could obscure task order.

### Generic spacecraft/environment factories

The three current scenarios do not share a single vehicle/environment contract. A large factory would either expose nearly every Basilisk option or silently choose fidelity.

Use Basilisk's existing factories directly until a stable BASILISK-X product configuration appears repeatedly.

### BskSim-style master classes

BskSim is valuable when several campaigns reuse one broad dynamics and FSW stack with mode switching. It is unnecessary for three deliberately different learning scenarios.

### A universal multi-spacecraft framework

A fixed chief/deputy experiment does not need indexed per-satellite processes. Adopt a MultiSat-style abstraction when fleet size, heterogeneous spacecraft, reusable per-vehicle stacks, and independently selected rates are actual requirements.

### A parallel quaternion attitude framework

Basilisk FSW interfaces frequently use MRPs. Quaternion-based research can be valuable, but a BASILISK-X quaternion layer should solve a concrete interface or algorithm need, define conventions exactly, and translate through tested adapters. It should not duplicate every MRP algorithm merely for representational preference.

### Generic mission or plotting frameworks

Mission modes and figures encode question-specific meaning. First extract small state-machine, metric, or plotting primitives that have repeated semantics; do not start with an all-purpose framework.

## Recommended package layering

If reusable code grows, keep dependencies directed inward:

```text
basiliskx.visualization
    external-process and Vizard lifecycle; no mission physics

basiliskx.analysis
    pure numerical transforms and metrics with frames/units explicit

basiliskx.adapters
    typed boundary conversion into/out of Basilisk messages

basiliskx.fsw
    custom scheduled algorithms with Basilisk message interfaces

basiliskx.testing
    reusable assertions/fixtures for numerical and interface validation

scenarios
    compose Basilisk and basiliskx pieces into mission questions
```

These are proposed boundaries, not a request to create empty packages. Add a package only with its first proven consumer and tests.

## Public API policy

The current `basiliskx.__init__` files are empty, so no deliberate public API exists. When one is introduced:

- export only stable, documented entry points;
- keep Basilisk objects visible rather than wrapping them opaquely;
- use semantic names and unit suffixes;
- include `__all__` only when exports are intentional;
- document supported Basilisk/Python versions;
- deprecate before removing a used interface;
- avoid importing optional GUI or heavy dependencies at package import time.

## Testing strategy

The existing Vizard launcher has strong focused tests. Scenario physics needs a complementary test pyramid.

### Pure unit tests

- frame transformations and round trips;
- CW/targeting equations and domain checks;
- unit conversions;
- metric calculations;
- configuration validation;
- quaternion/MRP adapters if introduced.

### Module/interface tests

- required-message checks;
- known payload in -> known payload out;
- timestamp and reset behavior;
- stale/invalid input behavior;
- actuator/configuration ordering.

### Deterministic scenario smoke tests

- headless import/build/initialize/short execution;
- finite recorded states;
- expected message writes;
- stable toleranced end metrics.

### Physics regression tests

- two-body energy/angular momentum and period;
- attitude settling, rate, and demanded torque;
- Hill conversion and CW/nonlinear comparison;
- finite-burn achieved delta-v and mass change;
- conservation for internal effectors;
- time-step/rate convergence.

### Optional integration tests

- Vizard launch/stream/playback;
- SPICE/custom asset availability;
- OpNav rendering;
- MuJoCo/contact;
- multiprocessing Monte Carlo.

Optional tests need explicit markers, timeouts, and dependency diagnostics. The current GUI-stream helper should have an overall timeout so a live but non-progressing pair of processes cannot wait indefinitely.

## Current issues to resolve during vetting

The field manual should track, but not silently fix, these findings:

- `nadir_pointing.py` describes `+b1` in one place while implementing/analyzing `+b3`;
- some configuration summaries report module globals rather than run-time overrides;
- the cooperative RPO data extraction can normalize an input array in place;
- RPO input validation is incomplete;
- execution priorities are mostly implicit in BASILISK-X scenarios;
- run functions return no structured results;
- numerical scenario regressions are absent;
- Vizard setup/cleanup remains partly duplicated;
- default runs are GUI-oriented rather than headless-first;
- generated PNG/playback outputs are tracked without complete run provenance.

These are normal findings in an experimental repository. Resolve them through focused issues/tests rather than an architectural rewrite.

## Packaging and reproducibility policy

Current packaging declares the BASILISK-X package but not its runtime dependency. A future packaging pass should decide whether:

- Basilisk is a required dependency in `pyproject.toml`;
- GUI, OpNav, MuJoCo, Monte Carlo visualization, and development tools are optional groups;
- supported Python bounds match the selected Basilisk release;
- tests, linting, typing, and coverage belong in a development extra;
- a lock or reproducible environment record is maintained;
- the copied examples are retained, replaced by an upstream reference, or given an exact source manifest.

Do not update dependency versions merely to make a copied example import. First choose and document the intended version baseline.

## Recommended near-term evolution

```text
1. return structured results and add headless numerical regression metrics
2. make execution order and command latency explicit
3. add a physical reaction-wheel version of the nadir baseline
4. preserve ideal RPO targeting, then add a separate finite-burn realization
5. introduce relative measurement/navigation and remove truth from guidance
6. add mission constraints, aborts, and resource gating
7. add deterministic uncertainty inputs and then Monte Carlo
8. extract utilities only after repeated, tested semantics appear
```

This sequence grows engineering credibility before framework surface area.

## Extraction decision record

Before moving code from a scenario into `src/basiliskx`, answer:

```text
Candidate behavior:
Current scenario consumers:
Expected future consumers:
Semantic contract:
Frames and units:
Time/timestamp behavior:
Failure/invalid-input behavior:
Existing Basilisk equivalent checked:
Why scenario-local code is no longer adequate:
Unit/module tests:
Supported Basilisk versions:
Public API owner:
```

If the record is mostly blank, leave the behavior in the scenario and keep learning from it.
