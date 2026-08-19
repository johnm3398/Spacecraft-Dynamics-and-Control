> **Draft status — AI-generated:** This page was drafted by OpenAI Codex on 2026-08-19. It has not yet been technically vetted by the BASILISK-X repository owner. The repository owner will review, correct, and maintain it. Verify APIs, units, frames, assumptions, and version compatibility before engineering use.

# Example Tree and Asset Map

## Purpose and reading rule

This page explains what the major branches of [`examples/`](../../examples/) are for and how their support data enters a simulation. It is a routing map, not a file-by-file catalogue. Use the [example capability index](example_capability_index.md) when starting from an engineering capability, and use this page when deciding which example architecture or asset family you are looking at.

At the 2026-08-19 audit, the root of `examples/` contained 106 standalone `scenario*.py` scripts plus one Python helper. The copied tree has no manifest identifying its exact upstream Basilisk tag or commit, so counts and compatibility observations describe this checkout only.

Interpret the map using three evidence levels:

- **Observed locally:** a path, subscription, file consumer, or runtime result was inspected in this checkout.
- **Demonstrated by an example:** the source illustrates a pattern but does not validate it for another mission.
- **Engineering recommendation:** guidance about reuse, provenance, or verification is an inference for BASILISK-X.

```text
examples/
|-- scenario*.py                 focused standalone studies
|-- BskSim/                      reusable one/fixed-few-spacecraft framework
|-- MultiSatBskSim/              indexed N-spacecraft framework
|-- MonteCarloExamples/          Monte Carlo around BskSim scenarios
|-- OpNavScenarios/              rendered-image optical-navigation stack
|-- mujoco/                      MJCF multibody/contact examples
|-- Support/                     retained Monte Carlo initial-condition data
|-- dataForExamples/             local trajectories, kernels, gains, meshes, textures
`-- index.rst                    upstream-style documentation index
```

## Executive inventory

| Branch | What it contains | Architectural reason to use it | Do not infer |
|---|---|---|---|
| Root standalone scenarios | A `run(...)` function normally constructs, configures, executes, records, and plots one study | Fastest way to isolate a module chain or answer one engineering question | Every script is a production template or shares one preferred architecture |
| [`BskSim/`](../../examples/BskSim/) | master classes, reusable dynamics/FSW model classes, scenario subclasses, shared plotting | Reuse one configured spacecraft stack across modes, faults, and related scenarios | Basilisk requires this wrapper; it is an optional example framework |
| [`MultiSatBskSim/`](../../examples/MultiSatBskSim/) | shared environment, per-spacecraft dynamics/FSW processes, optional relative-navigation process, TLE inputs | Make N spacecraft and heterogeneous model lists systematic | Same scheduler automatically implies communications, coupling, or distributed autonomy |
| [`MonteCarloExamples/`](../../examples/MonteCarloExamples/) | ensemble controller setup, retained-data callbacks, rerun and visualization workflows | Apply uncertainty infrastructure to an existing `BskSim` scenario | The deterministic model or its dispersions have already been validated |
| [`OpNavScenarios/`](../../examples/OpNavScenarios/) | Mars truth dynamics, Vizard/camera exchange, perception, measurement conversion, filters, GNC, modes, MC image generation | Keep a long rendered-image navigation chain reusable across experiments | It is locally turnkey; external Vizard, version matching, and missing CNN assets remain issues |
| [`mujoco/`](../../examples/mujoco/) | standalone MuJoCo scenarios, MJCF/XML plants, and a BSK-style dynamics/FSW wrapper | Represent general joints, branched mechanisms, collision, contact, and equality constraints | MuJoCo automatically provides spacecraft environment, qualified contact data, or higher orbital fidelity |
| [`Support/`](../../examples/Support/) | three JSON Monte Carlo initial-condition cases | Supply deterministic dispersed inputs to one MC replay path | It is a general Python utility package; there is no local helper code in this directory |
| [`dataForExamples/`](../../examples/dataForExamples/) | local static and binary inputs consumed by selected scenarios | Keep example-specific assets beside the copied examples | Every referenced support asset or its provenance is present |

## Standalone scenarios: the default learning unit

The root scenarios are deliberately broad in subject but usually narrow in architecture. A reader can see scheduler construction, message wiring, configuration, initial conditions, recorders, derived metrics, and plotting in one place. Representative routes are:

| Question family | Representative local sources |
|---|---|
| Minimal scheduler/orbit/recorder | [`scenarioBasicOrbit.py`](../../examples/scenarioBasicOrbit.py), [`scenarioBskLog.py`](../../examples/scenarioBskLog.py), [`scenarioDataDemo.py`](../../examples/scenarioDataDemo.py) |
| Numerical integration | [`scenarioIntegrators.py`](../../examples/scenarioIntegrators.py), [`scenarioIntegratorsComparison.py`](../../examples/scenarioIntegratorsComparison.py), [`scenarioVariableTimeStepIntegrators.py`](../../examples/scenarioVariableTimeStepIntegrators.py) |
| Gravity, SPICE, and mission mechanics | [`scenarioOrbitMultiBody.py`](../../examples/scenarioOrbitMultiBody.py), [`scenarioSpiceSpacecraft.py`](../../examples/scenarioSpiceSpacecraft.py), [`scenarioHohmann.py`](../../examples/scenarioHohmann.py), [`scenarioLambertSolver.py`](../../examples/scenarioLambertSolver.py) |
| Attitude GNC and actuators | [`scenarioAttitudeGuidance.py`](../../examples/scenarioAttitudeGuidance.py), [`scenarioAttitudeFeedbackRW.py`](../../examples/scenarioAttitudeFeedbackRW.py), [`scenarioAttitudeSteering.py`](../../examples/scenarioAttitudeSteering.py), [`scenarioMomentumDumping.py`](../../examples/scenarioMomentumDumping.py) |
| Standard articulated/internal dynamics | [`scenarioHingedRigidBody.py`](../../examples/scenarioHingedRigidBody.py), [`scenarioFuelSlosh.py`](../../examples/scenarioFuelSlosh.py), [`scenarioConstrainedDynamics.py`](../../examples/scenarioConstrainedDynamics.py), [`scenarioRoboticArm.py`](../../examples/scenarioRoboticArm.py) |
| Sensors and environment | [`scenarioCSS.py`](../../examples/scenarioCSS.py), [`scenarioMagneticFieldWMM.py`](../../examples/scenarioMagneticFieldWMM.py), [`scenarioAlbedo.py`](../../examples/scenarioAlbedo.py), [`scenarioSensorThermal.py`](../../examples/scenarioSensorThermal.py) |
| Formations/RPO/constellations | [`scenarioFormationBasic.py`](../../examples/scenarioFormationBasic.py), [`scenarioRendezVous.py`](../../examples/scenarioRendezVous.py), [`scenarioFormationReconfig.py`](../../examples/scenarioFormationReconfig.py), [`scenarioSatelliteConstellation.py`](../../examples/scenarioSatelliteConstellation.py) |
| Mission payload/resources | [`scenarioGroundLocationImaging.py`](../../examples/scenarioGroundLocationImaging.py), [`scenarioGroundDownlink.py`](../../examples/scenarioGroundDownlink.py), [`scenarioPowerDemo.py`](../../examples/scenarioPowerDemo.py), [`scenarioStripImaging.py`](../../examples/scenarioStripImaging.py) |
| Python extension/performance | [`scenarioAttitudePointingPy.py`](../../examples/scenarioAttitudePointingPy.py), [`scenarioAttitudePointingNumba.py`](../../examples/scenarioAttitudePointingNumba.py), [`scenarioBenchmarkNumba.py`](../../examples/scenarioBenchmarkNumba.py) |
| Direct Monte Carlo | [`scenarioMonteCarloAttRW.py`](../../examples/scenarioMonteCarloAttRW.py), [`scenarioMonteCarloSpice.py`](../../examples/scenarioMonteCarloSpice.py) |

Standalone does not mean simplistic. Several scripts contain long integrated studies or multiple option branches. It means that scenario orchestration and model construction remain locally visible instead of being delegated to framework classes. Prefer this style until repeated experiments genuinely need a shared plant, FSW stack, or mode system.

## Framework branches

### `BskSim`

[`BSK_masters.py`](../../examples/BskSim/BSK_masters.py) subclasses `SimulationBaseClass`, creates dynamics and FSW processes on demand, and defines a scenario interface for initial conditions, logging, and output extraction. [`BSK_Dynamics.py`](../../examples/BskSim/models/BSK_Dynamics.py) and [`BSK_Fsw.py`](../../examples/BskSim/models/BSK_Fsw.py) own reusable module construction and wiring. [`BSK_Faults.py`](../../examples/BskSim/models/BSK_Faults.py) adds fault injection. Separate fixed-formation models appear in [`BSK_FormationDynamics.py`](../../examples/BskSim/models/BSK_FormationDynamics.py) and [`BSK_FormationFsw.py`](../../examples/BskSim/models/BSK_FormationFsw.py).

The scenarios then specialize configuration and sequencing: [`scenario_AttModes.py`](../../examples/BskSim/scenarios/scenario_AttModes.py) for task/event modes, [`scenario_FeedbackRW.py`](../../examples/BskSim/scenarios/scenario_FeedbackRW.py) for the RW control chain, [`scenario_AddRWFault.py`](../../examples/BskSim/scenarios/scenario_AddRWFault.py) for a fault, and [`scenario_LambertGuidance.py`](../../examples/BskSim/scenarios/scenario_LambertGuidance.py) for a planning/guidance chain.

Use this split when several studies must share the same named model graph. It becomes overhead when the shared graph is still changing faster than the scenarios.

### `MultiSatBskSim`

[`BSK_MultiSatMasters.py`](../../examples/MultiSatBskSim/BSK_MultiSatMasters.py) introduces process priority bands: shared environment, one dynamics process per spacecraft, optional formation-barycentre relative navigation, and one FSW process per spacecraft. Model lists can be homogeneous or heterogeneous. Environment is separated into [`BSK_EnvironmentEarth.py`](../../examples/MultiSatBskSim/modelsMultiSat/BSK_EnvironmentEarth.py) and [`BSK_EnvironmentMercury.py`](../../examples/MultiSatBskSim/modelsMultiSat/BSK_EnvironmentMercury.py); vehicle factories live in [`BSK_MultiSatDynamics.py`](../../examples/MultiSatBskSim/modelsMultiSat/BSK_MultiSatDynamics.py) and [`BSK_MultiSatFsw.py`](../../examples/MultiSatBskSim/modelsMultiSat/BSK_MultiSatFsw.py).

Start with [`scenario_BasicOrbitMultiSat.py`](../../examples/MultiSatBskSim/scenariosMultiSat/scenario_BasicOrbitMultiSat.py), then [`scenario_AttGuidMultiSat.py`](../../examples/MultiSatBskSim/scenariosMultiSat/scenario_AttGuidMultiSat.py) and [`scenario_StationKeepingMultiSat.py`](../../examples/MultiSatBskSim/scenariosMultiSat/scenario_StationKeepingMultiSat.py). [`scenario_BasicOrbitMultiSat_MT.py`](../../examples/MultiSatBskSim/scenariosMultiSat/scenario_BasicOrbitMultiSat_MT.py) and [`scenario_constellationFromTle.py`](../../examples/MultiSatBskSim/scenariosMultiSat/scenario_constellationFromTle.py) add thread assignment and external orbit data.

The abstraction scales construction and ownership; it does not supply intersatellite links, relative sensors, command authority, or consensus. Those must be explicit messages/modules.

### `MonteCarloExamples`

[`scenarioBskSimAttFeedbackMC.py`](../../examples/MonteCarloExamples/scenarioBskSimAttFeedbackMC.py) wraps a `BskSim` attitude scenario with parameter dispersions, seed dispersion, retained messages, and a callback. [`scenarioRerunMonteCarlo.py`](../../examples/MonteCarloExamples/scenarioRerunMonteCarlo.py) demonstrates archived-case replay, while [`scenarioVisualizeMonteCarlo.py`](../../examples/MonteCarloExamples/scenarioVisualizeMonteCarlo.py) separates visualization from execution.

This directory adds campaign orchestration, not new physics. Attribute-string dispersions such as `TaskList[0].TaskModels[0]...` are coupled to model ordering and are less stable than named scenario attributes.

### `OpNavScenarios`

[`BSK_OpNav.py`](../../examples/OpNavScenarios/BSK_OpNav.py), [`BSK_OpNavDynamics.py`](../../examples/OpNavScenarios/modelsOpNav/BSK_OpNavDynamics.py), and [`BSK_OpNavFsw.py`](../../examples/OpNavScenarios/modelsOpNav/BSK_OpNavFsw.py) separate a Mars truth/camera plant from image processing, measurement conversion, filters, pointing, RW control, and event-selected modes. The scenario directory contains Hough-circle, limb, CNN, heading, fault, and MC variants. See the dedicated [optical-navigation chapter](../12_optical_navigation.md) before attempting to run this branch.

### `mujoco`

The root MuJoCo scripts are mostly standalone learning units; the XML files define physical bodies, joints, sites, inertial/collision geometry, actuators, and equalities. [`scenarioReactionWheel.py`](../../examples/mujoco/scenarioReactionWheel.py) with [`sat_w_wheel.xml`](../../examples/mujoco/sat_w_wheel.xml) is the entry point. [`BSK_mujocoMasters.py`](../../examples/mujoco/BSK_mujocoMasters.py) plus [`mujocoModels/`](../../examples/mujoco/mujocoModels/) show the heavier reusable framework used by [`scenarioThrArmControl.py`](../../examples/mujoco/scenarioThrArmControl.py). See the [MuJoCo chapter](../13_mujoco_robotics_contact_and_deployables.md) for the standard-`Spacecraft` boundary and contact caveats.

## Support utilities, plotting, and tests

The capitalized local [`Support/`](../../examples/Support/) directory is easily confused with Basilisk utility code. In this checkout it contains only [`run_MC_IC`](../../examples/Support/run_MC_IC/): three JSON dictionaries of dispersed attributes used by case 2 of [`scenarioMonteCarloAttRW.py`](../../examples/scenarioMonteCarloAttRW.py) through `Controller.setICDir`. They are executable configuration data and must match the scenario's attribute paths and units.

Shared local plotting code exists only where an example framework benefits from it:

- [`BskSim/plotting/BSK_Plotting.py`](../../examples/BskSim/plotting/BSK_Plotting.py);
- [`MultiSatBskSim/plottingMultiSat/BSK_MultiSatPlotting.py`](../../examples/MultiSatBskSim/plottingMultiSat/BSK_MultiSatPlotting.py);
- [`OpNavScenarios/plottingOpNav/OpNav_Plotting.py`](../../examples/OpNavScenarios/plottingOpNav/OpNav_Plotting.py);
- [`SunLineKF_test_utilities.py`](../../examples/SunLineKF_test_utilities.py), a specialised covariance/error plotting helper.

Most standalone scenarios keep compact plotting/metric functions in the same script. Plotting helpers are analysis consumers; they are not flight messages or verification by themselves.

Many scenario `run(...)` entry points contain assertions or branches intended for an upstream pytest/documentation harness. BASILISK-X does not carry that upstream example-test suite. The current project tests, [`test_auto_vizard_stream.py`](../../tests/test_auto_vizard_stream.py) and [`test_vizard_launcher.py`](../../tests/test_vizard_launcher.py), cover BASILISK-X visualization helpers rather than the copied example set. An example importing or producing a figure is not evidence that its engineering result is valid.

## Static and binary asset map

Static files enter at different layers. Treat a file as physical model input only when the consuming code proves it.

| Asset family | Local path and representative consumer | Role in the simulation |
|---|---|---|
| Monte Carlo initial conditions | [`Support/run_MC_IC`](../../examples/Support/run_MC_IC/), consumed by [`scenarioMonteCarloAttRW.py`](../../examples/scenarioMonteCarloAttRW.py) | writes dispersed values into named model attributes before a run |
| TLE sets | [`MultiSatBskSim/tleData`](../../examples/MultiSatBskSim/tleData/), consumed by [`scenario_constellationFromTle.py`](../../examples/MultiSatBskSim/scenariosMultiSat/scenario_constellationFromTle.py) | external epoch/orbit initialization; TLE frame/time interpretation remains part of the model |
| Spacecraft SPICE BSP | [`spacecraft_21T01.bsp`](../../examples/dataForExamples/Spice/spacecraft_21T01.bsp), consumed by [`scenarioHelioTransSpice.py`](../../examples/scenarioHelioTransSpice.py) | binary ephemeris states over its kernel coverage interval |
| Recorded trajectories | [`scHoldTraj_rotating_MRP.csv`](../../examples/dataForExamples/scHoldTraj_rotating_MRP.csv) and [`scHoldTraj_rotating_EP.csv`](../../examples/dataForExamples/scHoldTraj_rotating_EP.csv), consumed by [`scenarioDataToViz.py`](../../examples/scenarioDataToViz.py) | replays externally stored state/attitude data into visualization messages |
| Controller gain | [`static_lqr_controlGain.npz`](../../examples/dataForExamples/static_lqr_controlGain.npz), consumed by [`scenarioDragRendezvous.py`](../../examples/scenarioDragRendezvous.py) | precomputed LQR gain; its state ordering and design assumptions are as important as the numbers |
| Electrostatic geometry | [`GOESR_bus_80_sphs.csv`](../../examples/dataForExamples/GOESR_bus_80_sphs.csv), consumed by [`scenarioTwoChargedSC.py`](../../examples/scenarioTwoChargedSC.py) | sphere locations/radii used by the Multi-Sphere force/torque model, so it affects physics |
| Spacecraft meshes/materials | [`Aura_27.obj`](../../examples/dataForExamples/Aura_27.obj), [`Loral-1300Com-main.obj`](../../examples/dataForExamples/Loral-1300Com-main.obj), and [`texture/`](../../examples/dataForExamples/texture/) in [`scenarioDataToViz.py`](../../examples/scenarioDataToViz.py) | principally Vizard appearance; does not redefine the propagated spacecraft mass/inertia |
| Panel mesh | [`triangularPanel.obj`](../../examples/dataForExamples/triangularPanel.obj), used by [`scenarioDeployingSolarArrays.py`](../../examples/scenarioDeployingSolarArrays.py) | Vizard custom-model geometry, separate from the state-effector dynamics |
| Itokawa shape/texture | [`dataForExamples/Itokawa`](../../examples/dataForExamples/Itokawa/), used by [`scenarioCustomGravBody.py`](../../examples/scenarioCustomGravBody.py) and [`mujoco/scenarioAsteroidLanding.py`](../../examples/mujoco/scenarioAsteroidLanding.py) | visualization in the former; MuJoCo collision mesh plus visualization in the latter; gravity remains separately configured |
| MuJoCo MJCF/XML | [`mujoco/sat_w_deployable_panels.xml`](../../examples/mujoco/sat_w_deployable_panels.xml), [`mujoco/sats_dock.xml`](../../examples/mujoco/sats_dock.xml), and peers | physical topology, mass/inertia, joints, collision, sites, actuators, and constraints; these are model definitions, not decorative assets |
| Package support data | calls to `supportDataTools.dataFetcher`, for example in [`BSK_OpNavDynamics.py`](../../examples/OpNavScenarios/modelsOpNav/BSK_OpNavDynamics.py) | resolves gravity coefficients, SPICE kernels, magnetic/planet data from the installed Basilisk support-data system rather than this folder |

For an OBJ, image, BSP, NPZ, or large CSV, inspect the consumer and metadata needed to establish role; do not dump binary content into documentation. Retain units, coordinate frame, epoch, scale, state ordering, source revision, and checksum with an engineering result.

## Provenance and compatibility caveats

1. [`requirements.txt`](../../requirements.txt) pins `bsk[all,examples]==2.11.1`, but the copied tree contains newer APIs. `Basilisk.hasBuildFeature` imports in OpNav, SPICE reconstruction, and Vesta examples are absent from the installed package; newer stochastic and MuJoCo classes/properties also cause confirmed local failures. See [scope and provenance](../00_scope_versions_and_source_provenance.md).
2. Asset completeness is not guaranteed. [`scenarioFlybySpice.py`](../../examples/scenarioFlybySpice.py) references `max_21T01.bsp`, which is not present locally, and its unload path differs from its load path. [`triangularPanel.obj`](../../examples/dataForExamples/triangularPanel.obj) names a missing `triangularPanel.mtl`. The Loral material file references texture subdirectories/files not present in `dataForExamples`.
3. The OpNav CNN expects `CAD.onnx`, which is absent. Vizard is an external application, not a checked-in asset, and the OpNav launcher hard-codes a machine path and port.
4. Relative paths are inconsistent: some visualization calls use paths relative to the current working directory while comments show file-relative alternatives. Launching from a different directory can therefore change asset resolution without changing the scenario source.
5. `dataForExamples` has no co-located README/licence/provenance manifest. Do not assume that a filename establishes source, permission, scale, coordinate frame, or validity.
6. Visual meshes and textures normally affect presentation only. Exceptions must be explicit: the Itokawa mesh enters MuJoCo collision, the GOESR sphere CSV enters electrostatic physics, the gain NPZ enters control, trajectory CSVs drive replayed states, and BSP/TLE files define ephemerides or initial orbits.
7. Generated archives, plots, Vizard binaries, and saved camera frames are outputs, not source assets. Keep them out of the canonical example tree unless they are deliberately versioned test fixtures with provenance.

## Practical routing checklist

Before copying an example, answer:

```text
Is the question isolated?                 -> start with a standalone scenario
Must several scenarios share one plant?   -> inspect BskSim
Is spacecraft count/configuration varied? -> inspect MultiSatBskSim
Is this an uncertainty campaign?          -> add Monte Carlo after validation
Do rendered pixels drive navigation?      -> inspect OpNavScenarios and dependencies
Do topology/contact/constraints matter?   -> inspect mujoco
Does the source load a file?               -> trace its units/frame/epoch/provenance
```

Then verify the exact Basilisk version, import every optional module, resolve each input asset from the intended working directory, and run a headless deterministic smoke case before using the example as an engineering baseline.
