> **Draft status — AI-generated:** This page was drafted by OpenAI Codex on 2026-08-19. It has not yet been technically vetted by the BASILISK-X repository owner. The repository owner will review, correct, and maintain it. Verify APIs, units, frames, assumptions, and version compatibility before engineering use.

# Monte Carlo, uncertainty, and statistics

A deterministic Basilisk scenario answers, “What does this model do for these inputs?” A Monte Carlo campaign answers a different question: “Across a defensible population of uncertain inputs and stochastic processes, how is an engineering metric distributed, and how often are requirements met?”

Basilisk's Monte Carlo utilities automate fresh simulation construction, parameter modification, process-level parallel execution, data retention, and reruns. They do **not** choose credible uncertainty distributions, create correlations automatically, define mission success, prove statistical convergence, or distinguish model-form error from random variability. Those remain engineering responsibilities.

## 1. The ensemble architecture

```text
campaign definition
  |-- creation function
  |-- optional configure function
  |-- execution function
  |-- run count / process count
  |-- dispersions and seed policy
  |-- retention policies
  v
Controller
  |
  +--> run 0: fresh sim -> modify -> initialize/execute -> retain
  +--> run 1: fresh sim -> modify -> initialize/execute -> retain
  +--> ...                                      |
  +--> run N-1                                  v
                                         archived parameters,
                                         run data, aggregate data,
                                         failure indices
                                                  |
                                                  v
                                      engineering metrics/statistics
```

The best general local example is [`scenarioMonteCarloAttRW.py`](../examples/scenarioMonteCarloAttRW.py). It constructs a complete standalone simulation, exposes objects as attributes of the simulation instance, creates message recorders, disperses spacecraft/RW/interface properties, archives runs, loads retained data, and demonstrates rerun APIs. The shorter BskSim wrapper is [`scenarioBskSimAttFeedbackMC.py`](../examples/MonteCarloExamples/scenarioBskSimAttFeedbackMC.py).

## 2. Creation, configuration, and execution are separate contracts

The `Controller` receives three callables:

| Callable | Required? | Contract |
|---|---:|---|
| `setSimulationFunction(create)` | yes | No-argument callable returning a **new**, fully structured simulation instance for every run |
| `setConfigureFunction(configure)` | no | Per-run hook called after dispersion values are generated but before they are applied |
| `setExecutionFunction(execute)` | yes | Callable that initializes, sets stop time, and executes the supplied instance |

The inspected worker follows this order:

```text
seed Python random and NumPy from run index
        |
create a fresh simulation
        |
discover/generate model RNG seeds
        |
generate dispersion values and archive the modification dictionary
        |
call optional configure function
        |
apply archived/generated modifications by attribute path
        |
install RetentionPolicy variable logs
        |
call execution function
        |
extract recorder/variable/custom data and archive it
```

Two consequences are easy to miss:

1. Objects targeted by a dispersion, and models whose `RNGSeed` should be discovered, must already exist after the creation function. The optional configure hook is too late to create a path needed during dispersion generation.
2. Dispersed values are applied before the execution function normally calls `InitializeSimulation()`. This is the desired order for initial conditions and module configuration.

The standalone attitude/RW example explicitly stores `scObject`, wheel objects, factories, interfaces, and even a hub reference on `scSim`. This both gives the Controller a resolvable path and keeps Python references to wrapped C++ objects alive.

### Compact controller skeleton

```python
mc = Controller()
mc.setSimulationFunction(create_simulation)   # returns a fresh SimBaseClass
mc.setExecutionFunction(execute_simulation)   # initializes and runs it
mc.setExecutionCount(n_runs)
mc.setThreadCount(n_processes)
mc.setArchiveDir(unique_campaign_directory)
mc.setShouldDisperseSeeds(True)

mc.addDispersion(NormalDispersion("scObject.hub.mHub", mean, sigma, bounds))

policy = RetentionPolicy()
policy.addMessageLog("nav", ["r_BN_N", "v_BN_N"])
mc.addRetentionPolicy(policy)                 # create sim.msgRecList["nav"] first

execution_failures = mc.executeSimulations()
```

This is infrastructure only. A useful campaign also archives the uncertainty specification, requirements, derived-metric code version, and interpretation of each run.

## 3. Attribute paths and why scenario structure matters

Dispersions identify targets with paths rooted at the returned simulation instance. The local examples use:

- scheduler-internal paths such as `TaskList[0].TaskModels[0].hub.mHub`;
- named scenario attributes such as `RW1.Omega` and `rwVoltageIO.voltage2TorqueGain[0]`;
- zero-argument accessors such as `get_DynModel().scObject.hub.r_CN_NInit` in the OpNav campaign.

The installed path resolver supports attributes, integer indices, and zero-argument method calls. Modification values archived as strings are parsed with literal parsing before assignment.

**Recommendation:** prefer stable, named attributes on the scenario (`scObject`, `RW1`, `cameraModel`) over `TaskList[i].TaskModels[j]`. The latter couples an uncertainty definition to process/task construction and model ordering. Add a preflight test that resolves every path and checks its nominal type, shape, unit, and physical bounds before launching a large campaign.

## 4. Dispersion models

The installed `Basilisk.utilities.MonteCarlo.Dispersions` module contains the following useful families:

| Family | Local class examples | Use | Important caveat |
|---|---|---|---|
| Scalar | `UniformDispersion`, `NormalDispersion` | mass, gains, scalar noise levels | Normal bounds are implemented by clipping to a boundary, not rejection sampling; this creates point mass at limits |
| Cartesian vector | `UniformVectorCartDispersion`, `NormalVectorCartDispersion` | position/rate/bias components | Confirm whether components are intended to be independent and what frame they occupy |
| Direction/angle | `UniformVectorAngleDispersion`, `NormalVectorAngleDispersion`, `NormalThrusterUnitDirectionVectorDispersion` | boresight or thrust misalignment | Direction uncertainty belongs on a manifold; do not independently perturb components and then forget normalization |
| Attitude | `UniformEulerAngleMRPDispersion`, `MRPDispersionPerAxis` | initial orientation | Verify whether bounds are Euler-angle offsets or direct MRPs; avoid interpreting MRP components as physical angles |
| Inertia | `InertiaTensorDispersion` | principal values/orientation | Enforce symmetry and positive definiteness; validate mass-property correlations |
| Orbit | `OrbitalElementDispersion` | jointly generate Cartesian position and velocity from element draws | Element singularities, angle wrapping, and physically invalid combinations need explicit treatment |

[`scenarioMonteCarloAttRW.py`](../examples/scenarioMonteCarloAttRW.py) demonstrates units-aware dispersions for MRPs, body rates, mass, center-of-mass offset, inertia orientation, wheel axes, wheel speed, and voltage-to-torque gains. It correctly converts wheel-speed bounds from RPM at factory input to rad/s for the stored wheel state.

[`OpNavMC/MonteCarlo.py`](../examples/OpNavScenarios/scenariosOpNav/OpNavMC/MonteCarlo.py) uses `OrbitalElementDispersion` to create consistent `r_CN_NInit` and `v_CN_NInit` values and separately disperses camera field of view and filter noise scale.

### Correlation is part of the model

Separate scalar/vector dispersion objects are normally separate draws. Real uncertainties are often correlated:

- injection position and velocity errors share launch-state covariance;
- thrust magnitude, specific impulse, and burn duration can share calibration or temperature drivers;
- mass, center of mass, and inertia derive from the same as-built configuration;
- camera focal length, principal point, and distortion share calibration covariance;
- biases may be constant within a run but vary between runs.

The framework has structured multi-output dispersions, such as orbital elements producing both position and velocity, and its worker contains support for co-dependent multi-output dispersion objects. That support is implementation-sensitive and is not a general covariance specification.

**Recommendation:** for a correlated campaign, generate a versioned run manifest from a joint distribution, validate every row physically, and use archived per-run modification JSON as the reproducibility contract. Do not silently approximate a covariance matrix with independent one-dimensional draws.

## 5. Seeds and reproducibility

In the inspected implementation, each worker seeds Python's `random` module and NumPy with `run_index * 10`. With `setShouldDisperseSeeds(True)`, it also scans task models for an `RNGSeed` attribute, generates a value for each, applies those seeds before initialization, and stores them in the run's modification JSON.

```text
run index
   |
   +--> Python random seed
   +--> NumPy random seed
   +--> generated per-model RNGSeed values --> archived runN.json
   +--> parameter draws ---------------------> archived runN.json
```

This makes the built-in draws repeatable by run index in the checked implementation. Reproducibility still requires more:

- every stochastic module must expose and use a controlled seed;
- custom Python code must use a controlled generator rather than operating-system entropy;
- external renderers and hardware-accelerated image pipelines may not be bitwise deterministic;
- process scheduling must not determine file names or shared-state updates;
- Basilisk, Python, NumPy, SPICE kernels, platform, and scenario revision must be archived;
- numerical reproducibility may be tolerance-based rather than bitwise across platforms.

**Recommendation:** treat `runN.json` as necessary but not sufficient provenance. Store a campaign manifest with the source revision, Basilisk version, dependency lock, kernel checksums, uncertainty-model revision, and metric/requirement revision.

## 6. Multiprocessing is process-level execution

Although the API is named `setThreadCount`, the inspected Controller uses `multiprocessing.Pool`; the value is a worker **process** count. The default is the detected CPU count. Each run receives a fresh simulation in a worker, and a separate data-writer process aggregates retained data.

Benefits include isolation of most simulation state and good throughput for CPU-bound cases. Constraints include:

- creation/execution callables and their imported modules must work under multiprocessing start semantics;
- global factories, mutable module-level configuration, and singleton external libraries are hazards;
- every worker needs unique output paths and must not bind the same TCP port;
- memory, not CPU count, may determine the safe worker count;
- visualization and image generation can dominate GPU or filesystem resources;
- the Controller currently batches pools as a workaround for a noted memory-leak concern in its own source comments.

Start with one process and a small run count. Verify parameter variation, seed capture, recorder shape, and cleanup. Then increase workers while watching memory, file collisions, external processes, and deterministic reruns.

## 7. RetentionPolicy: what is and is not retained

A `RetentionPolicy` names message fields, logged variables, optional custom retention functions, and an optional post-run callback.

### Message retention requires recorders

`addMessageLog("nav", ["r_BN_N"])` does not create a recorder. The creation/configuration code must:

1. create the output-message recorder at the desired sampling interval;
2. add it to a task;
3. store it in `sim.msgRecList["nav"]`.

The Controller later reads `sim.msgRecList["nav"].times()` and the named recorder field. The retained message array has time in nanoseconds prepended as column zero.

`RetentionPolicy(rate=...)` controls the default rate used by `addVariableLog()`. It does not override the sampling interval of an already-created message recorder. In [`scenarioBskSimAttFeedbackMC.py`](../examples/MonteCarloExamples/scenarioBskSimAttFeedbackMC.py), a local `samplingTime` is assigned but not passed to the policy or recorder in that file; the underlying BskSim scenario's recorder setup remains authoritative.

### Retain engineering evidence, not everything

| Retain | Why |
|---|---|
| Run inputs and seeds | Provenance and rerun |
| Requirement-driving time histories | Diagnose boundary cases and failures |
| Event/mode/command history | Explain why a run succeeded or failed |
| Endpoint and worst-case metrics | Efficient statistics |
| Validity flags and constraint margins | Prevent invalid samples from looking successful |
| A small diagnostic subset for all runs, richer data for selected runs | Control storage without losing traceability |

Avoid logging high-rate truth, every sensor sample, and imagery for every run unless the question requires it. A two-stage campaign is often better: retain compact metrics across the ensemble, then rerun selected tails and failures with richer logging.

## 8. Archive layout, loading, and reruns

With `setArchiveDir(directory)`, the checked Controller writes:

| Artifact | Meaning |
|---|---|
| `MonteCarlo.data` | gzip-pickled Controller/campaign object |
| `runN.json` | parameter modifications and model RNG seeds for run N |
| `runN.data` | gzip-pickled retained dictionary for run N |
| `<message.field>.data` | aggregated pandas data indexed by time with `(runNum, varIdx)` columns |
| `runNmag.txt` | optional standardized/percentage dispersion magnitudes |
| `failures.txt` | worker execution-error indices, when present |

`Controller.load(directory)` restores the archived controller; `getRetainedData(N)` and `getParameters(N)` load per-run output and inputs. Aggregate files support the Bokeh viewer in [`scenarioVisualizeMonteCarlo.py`](../examples/MonteCarloExamples/scenarioVisualizeMonteCarlo.py).

**Destructive-path caveat:** in the inspected implementation, `executeSimulations()` removes an existing archive directory before creating it. Always use an explicitly validated, campaign-specific output path. Never point it at a directory containing source or irreplaceable results.

### Two rerun mechanisms are not equivalent

| Mechanism | Observed behavior | Appropriate use |
|---|---|---|
| `Controller.reRunCases([N])` | Loads `runN.json`, disables new seed dispersion, and removes retention policies before executing sequentially | Re-execute a case for side effects/debugging; it does not create a fresh retained comparison in the inspected implementation |
| `setICDir(...)` + `setICRunFlag(True)` + `runInitialConditions([...])` | Loads archived JSON initial conditions and can write retained results to a different archive directory | Recompute selected cases with revised logging or compare outputs |

[`scenarioRerunMonteCarlo.py`](../examples/MonteCarloExamples/scenarioRerunMonteCarlo.py) demonstrates the second pattern and deliberately chooses a new `rerun` directory to protect the original initial-condition files.

**Observed test caveat:** [`scenarioMonteCarloAttRW.py`](../examples/scenarioMonteCarloAttRW.py) calls `reRunCases()` and then compares `getRetainedData()` with the previous data. Because the current `reRunCases()` implementation removes retention policies, the archived run data is not replaced; that comparison does not independently demonstrate output reproducibility. Use the initial-condition rerun path into a new directory for a meaningful numerical comparison.

Pickled Controller objects and pandas files are convenient local artifacts, not stable long-term interchange formats. Preserve simple JSON/CSV/Parquet summaries and provenance separately if results must outlive the precise Python/Basilisk environment.

## 9. Failure handling: execution failure is not mission failure

`executeSimulations()` returns indices whose workers raised an exception or reported failure. It may also write `failures.txt`. The aggregate writer reindexes missing run columns, allowing missing cases to appear as `NaN` in ensemble files.

That mechanism detects software/execution failures. A run that completes with a collision, missed imaging opportunity, filter divergence, empty battery, or violated keep-out zone is still an execution success.

```text
all planned runs
  |-- infrastructure failures -> diagnose/rerun; never count as mission outcomes
  |
  +-- valid completed runs
        |-- mission success
        |-- mission failure
        +-- indeterminate/invalid measurement or metric
```

**Recommendation:** keep three separate statuses:

1. `execution_status`: did the software run and produce complete data?
2. `validity_status`: was the modeled case physically and numerically admissible?
3. `mission_status`: did explicit engineering requirements pass?

Fail the campaign if an unexpected execution failure occurs. Archive its inputs and traceback context, rerun it sequentially with verbose logging, and never silently remove it from a probability denominator.

## 10. From deterministic scenario to uncertainty study

Use this sequence:

```text
validated deterministic baseline
        |
define decision metric and pass/fail requirements
        |
build uncertainty register: source, type, unit, frame, distribution, correlation
        |
separate aleatory variability from epistemic/model uncertainty
        |
select run design and sample count
        |
preflight a few archived, single-process runs
        |
execute ensemble and audit failures/missing data
        |
derive per-run metrics and mission status
        |
estimate distributions/probabilities with confidence bounds
        |
inspect tails and rerun boundary/failure cases
        |
repeat after fidelity or requirement changes
```

### Example uncertainty register

| Engineering case | Uncertain inputs | Outputs/requirements |
|---|---|---|
| Orbit injection | position/velocity covariance, epoch, vehicle mass | achieved orbit, perigee margin, acquisition Δv |
| Deployment | separation impulse and direction, attitude/rate, timing | collision clearance, relative-orbit envelope |
| Finite burn | thrust scale/alignment, latency, mass properties, navigation | terminal position/velocity error, propellant, constraint margins |
| Attitude acquisition | initial attitude/rate, inertia, wheel axes, sensor noise | settle time, peak error, wheel momentum, power |
| Navigation | truth initial state, bias, noise, dropouts, measurement geometry | NEES/NIS, error percentiles, divergence rate |
| RPO | injection/nav/thrust error, comm latency/dropout, target motion | keep-out compliance, approach time, fuel, docking conditions |
| Autonomous mission | opportunity timing, resources, faults, nav validity | completed tasks, reward/utility, safety violations, success probability |

Do not disperse a parameter merely because the API makes it easy. Each distribution needs a source: requirement, calibration result, covariance, supplier tolerance, environmental model, or explicitly labeled assumption.

## 11. Statistical interpretation

The demonstration run counts in this repository—often 2, 4, 10, or 12—exercise infrastructure; they are not generally enough to estimate tail probabilities.

For binary success, report

\[
\hat p = \frac{k}{n}
\]

with a binomial confidence interval, not only the point estimate. A Wilson interval is robust for modest `n`. If no failures are observed, the approximate 95% upper bound on failure probability is still `3/n` (“rule of three”); zero observed failures is not proof of zero risk.

For continuous metrics:

- report median and engineering-relevant percentiles, not only mean and standard deviation;
- attach uncertainty to percentile estimates, for example with bootstrap intervals;
- plot empirical CDFs and requirement margins;
- check convergence by increasing sample count or using repeated batches;
- stratify by meaningful regimes instead of averaging incompatible populations;
- inspect influential inputs with scatter/rank-correlation or a designed sensitivity method.

Pearson correlation detects only linear association and can be misleading for thresholds or strongly nonlinear dynamics. Rank correlation is often a useful first screen, but neither proves causality. Variance-based sensitivity, importance sampling, Latin hypercube sampling, or rare-event methods are analysis designs layered around Basilisk; they are not demonstrated as turnkey features in the local Controller examples.

For navigation filters, compare actual error with predicted covariance using consistency metrics such as NIS/NEES where the required truth and innovation data exist. Plotting a 3σ envelope without coverage statistics is not a filter-consistency assessment.

## 12. Visualization and aggregation

Retention callbacks can plot one archived run at a time after execution. `executeCallbacks()` loads retained cases and invokes each policy's callback. This is suitable for small ensembles and diagnostic overlays.

[`scenarioVisualizeMonteCarlo.py`](../examples/MonteCarloExamples/scenarioVisualizeMonteCarlo.py) uses `MonteCarloPlotter` and Bokeh to load aggregate `.data` files and provide interactive component/run selection, zooming, and optional server operation. It is a data-exploration interface, not a statistical report. For an engineering campaign, build a separate derived table with one row per run:

```text
run, execution_ok, valid, mission_success,
min_keepout_margin_m, final_range_m, delta_v_mps,
max_attitude_error_rad, min_battery_j, transition_count, ...
```

That table is the natural input to confidence intervals, sensitivity plots, failure clustering, and requirement traceability. Keep links from each row back to `runN.json` and selected retained histories.

## 13. SPICE and OpNav campaign caveats

### Python SPICE

[`scenarioMonteCarloSpice.py`](../examples/scenarioMonteCarloSpice.py) deliberately shows that Python `pyswice` kernels should be loaded inside each worker's simulation construction/execution context. Its commented counterexample loads kernels in the Controller constructor and is documented to cause a kernel-loading error. SPICE kernel pools and multiprocessing/global library state require disciplined per-process setup and cleanup.

**Recommendation:** pin kernel files and checksums, load/unload within worker ownership, verify epoch/frame strings, and test serial versus multiprocess results before scaling.

### OpNav and Vizard-generated imagery

[`OpNavMC/MonteCarlo.py`](../examples/OpNavScenarios/scenariosOpNav/OpNavMC/MonteCarlo.py) uses one process and calls [`scenario_OpNavAttODMC.py`](../examples/OpNavScenarios/scenariosOpNav/OpNavMC/scenario_OpNavAttODMC.py), whose execution function starts an external Vizard process on a fixed TCP endpoint, runs preparation and OpNav modes, then kills Vizard. This is a specialized image-in-the-loop campaign:

- multiple workers would contend for the fixed port unless each receives an isolated endpoint;
- external processes require `try/finally` cleanup for failed runs;
- rendering/GPU behavior may not be bitwise reproducible;
- image paths and filenames must be unique per run;
- camera cadence, FSW cadence, message validity, and retained truth must be time-aligned;
- `PROCESSES = 1` in the local OpNav driver is therefore an important architectural constraint, not merely a conservative performance choice.

The CNN image-generation driver [`CNN_ImageGen/OpNavMonteCarlo.py`](../examples/OpNavScenarios/scenariosOpNav/CNN_ImageGen/OpNavMonteCarlo.py) is legacy/special-purpose material, not a drop-in current template. **Observed inconsistencies in this checkout** include post-processing that expects separate `.times` message keys even though the installed RetentionPolicy prepends time as column zero, a `self.scRecmsgRecList` attribute typo in [`scenario_CNNImages.py`](../examples/OpNavScenarios/scenariosOpNav/CNN_ImageGen/scenario_CNNImages.py), and a defined dark-current dispersion path that is not added to the Controller. Validate and repair this pipeline before using it to create a research dataset.

## 14. Recommended robust BASILISK-X campaign pattern

1. Make the deterministic scenario a factory that returns a new self-contained simulation with named objects and recorders.
2. Keep scenario creation free of cross-run mutable globals; construct fresh factories and wrapped models per run.
3. Define a versioned uncertainty register and generate a validated run manifest, including correlations.
4. Use a unique immutable archive directory; never reuse a source or previous-results path.
5. Run 3–5 cases sequentially and inspect resolved paths, archived JSON, seed differences, recorder times/shapes, units, and metrics.
6. Separate minimal all-run retention from rich diagnostic retention.
7. Scale worker count based on memory and external-resource limits; keep SPICE/Vizard ownership process-local.
8. Treat Controller failure indices as infrastructure failures, not mission failures.
9. Compute a one-row-per-run metric table with explicit validity and success definitions.
10. Attach confidence bounds, convergence evidence, and distribution/source provenance to reported results.
11. Rerun tails, failures, and threshold-near cases from archived input JSON into a new directory with richer logs.
12. Regression-test a small fixed-seed campaign whenever Basilisk, dependencies, scenario structure, or metric logic changes.

The Controller turns a scenario into an ensemble. The engineering study begins only after the uncertain population, metrics, validity rules, and statistical evidence have been defined.
