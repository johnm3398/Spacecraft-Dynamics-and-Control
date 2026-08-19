> **Draft status — AI-generated:** This page was drafted by OpenAI Codex on 2026-08-19. It has not yet been technically vetted by the BASILISK-X repository owner. The repository owner will review, correct, and maintain it. Verify APIs, units, frames, assumptions, and version compatibility before engineering use.

# Basilisk Example Capability Index

This is a routing index, not a file-by-file encyclopedia. It identifies the smallest useful set of local examples to open for an engineering capability and records the main reason not to copy each architecture blindly.

The local examples are associated with the project’s Basilisk 2.11.1 dependency pin. The repository does not contain a separate manifest proving the exact upstream commit of the copied example tree, so confirm behavior against the installed module documentation before engineering use.

## Simulation architecture and interfaces

| Capability | Best local starting points | What to learn | Main caveat |
|---|---|---|---|
| Minimal simulation construction | [scenarioBasicOrbit.py](../../examples/scenarioBasicOrbit.py) | `SimBaseClass`, process, task, spacecraft, gravity, recorder, initialize/execute | Contains several option branches; extract the branch relevant to the question |
| Live Vizard streaming | [scenarioBasicOrbitStream.py](../../examples/scenarioBasicOrbitStream.py) | `ClockSynch`, live/broadcast setup, two-way Vizard input | GUI/network-dependent and unsuitable as a default automated-test path |
| Logging and data movement | [scenarioBskLog.py](../../examples/scenarioBskLog.py), [scenarioDataDemo.py](../../examples/scenarioDataDemo.py), [scenarioDataToViz.py](../../examples/scenarioDataToViz.py) | Basilisk logging, recorder/data patterns, visualization from data | These solve different layers; do not treat plotting data as a flight message |
| Custom Python module | [scenarioAttitudePointingPy.py](../../examples/scenarioAttitudePointingPy.py) | Python-authored algorithm in the scheduler/message architecture | A custom module needs lifecycle, message-contract, and unit tests |
| Numba module/performance experiments | [scenarioAttitudePointingNumba.py](../../examples/scenarioAttitudePointingNumba.py), [scenarioBenchmarkNumba.py](../../examples/scenarioBenchmarkNumba.py) | Python acceleration pattern and benchmark structure | Optimize only after profiling; Numba constraints alter normal Python design |
| Reusable dynamics/FSW architecture | [BSK_masters.py](../../examples/BskSim/BSK_masters.py), [BSK_Dynamics.py](../../examples/BskSim/models/BSK_Dynamics.py), [BSK_Fsw.py](../../examples/BskSim/models/BSK_Fsw.py) | Model/scenario separation, explicit priorities, gateway messages, events/modes | Example framework with fixed assumptions, not a required wrapper for Basilisk |

## Orbit, gravity, and mission mechanics

| Capability | Best local starting points | Important modules/utilities | Main caveat |
|---|---|---|---|
| Point-mass orbit | [scenarioBasicOrbit.py](../../examples/scenarioBasicOrbit.py) | `spacecraft`, `gravBodyFactory`, `orbitalMotion` | Keplerian baseline only |
| Central-body and frame choices | [scenarioCentralBody.py](../../examples/scenarioCentralBody.py) | Central-body selection and relative ephemeris setup | Frame/origin choices must be carried into all downstream analysis |
| Custom gravity body | [scenarioCustomGravBody.py](../../examples/scenarioCustomGravBody.py) | Custom body creation and gravity parameters | Parameter provenance becomes the user’s responsibility |
| Spherical-harmonic gravity | [scenarioBasicOrbit.py](../../examples/scenarioBasicOrbit.py), [scenarioOrbitConsistencyVerification.py](../../examples/scenarioOrbitConsistencyVerification.py) | Harmonic model configuration and consistency checks | Tesseral terms require correct body orientation/epoch |
| Multi-body gravity | [scenarioOrbitMultiBody.py](../../examples/scenarioOrbitMultiBody.py) | Several gravitating bodies and SPICE-backed states | More bodies do not automatically improve an incorrectly framed model |
| SPICE ephemerides | [scenarioSpiceSpacecraft.py](../../examples/scenarioSpiceSpacecraft.py), [scenarioFlybySpice.py](../../examples/scenarioFlybySpice.py), [scenarioSpiceReconstruction.py](../../examples/scenarioSpiceReconstruction.py) | Epoch, kernel-driven ephemerides, zero-base choices | Kernel coverage, time system, target/observer, and frame must be checked; `scenarioSpiceReconstruction.py` currently imports a newer build-feature API absent from the pinned 2.11.1 package |
| Interplanetary/patched conics | [scenarioPatchedConics.py](../../examples/scenarioPatchedConics.py), [scenarioHelioTransSpice.py](../../examples/scenarioHelioTransSpice.py), [scenarioJupiterArrival.py](../../examples/scenarioJupiterArrival.py) | Mission-phase mechanics and celestial transitions | Several examples use deliberate state resets as idealized phase changes |
| Lagrange and halo trajectories | [scenarioLagrangePointOrbit.py](../../examples/scenarioLagrangePointOrbit.py), [scenarioHaloOrbit.py](../../examples/scenarioHaloOrbit.py) | Multi-body geometry and special orbit initialization | Not generic station-keeping designs |
| Integrator selection | [scenarioIntegrators.py](../../examples/scenarioIntegrators.py), [scenarioIntegratorsComparison.py](../../examples/scenarioIntegratorsComparison.py), [scenarioVariableTimeStepIntegrators.py](../../examples/scenarioVariableTimeStepIntegrators.py) | Fixed/variable-step choices and numerical comparison | Compare mission metrics and invariants, not runtime alone |
| Ideal impulsive maneuvers | [scenarioOrbitManeuver.py](../../examples/scenarioOrbitManeuver.py), [scenarioHohmann.py](../../examples/scenarioHohmann.py) | Analytic targeting followed by state change | No finite-burn pointing, actuator, or propellant behavior |
| Lambert targeting | [scenarioLambertSolver.py](../../examples/scenarioLambertSolver.py), [scenario_LambertGuidance.py](../../examples/BskSim/scenarios/scenario_LambertGuidance.py) | Planner/solver/validator chain and burn insertion | Feasible Lambert geometry is not a complete maneuver execution design |
| Finite thrust | [scenarioOrbitManeuverTH.py](../../examples/scenarioOrbitManeuverTH.py), [scenarioFormationReconfig.py](../../examples/scenarioFormationReconfig.py) | Thruster effector, burn duration/on-time commands, attitude coupling | Verify thrust direction, command timing, and mass-flow connection; neither example alone establishes a complete propulsion system |
| Aerocapture/impact | [scenarioAerocapture.py](../../examples/scenarioAerocapture.py), [scenarioImpact.py](../../examples/scenarioImpact.py) | Atmospheric/terminal trajectory mechanics | Environment and vehicle aerodynamics dominate validity |
| Drag and lifetime | [scenarioDragSensitivity.py](../../examples/scenarioDragSensitivity.py), [scenarioDragDeorbit.py](../../examples/scenarioDragDeorbit.py), [scenarioStochasticDragSpacecraft.py](../../examples/scenarioStochasticDragSpacecraft.py) | Drag models, density sensitivity, stochastic forcing | Density uncertainty can dominate apparent model detail; the stochastic example imports `igbmNoiseStateEffector`, which is absent from the pinned Basilisk 2.11.1 installation |
| SRP and complex geometry | [scenarioDeployingSolarArrays.py](../../examples/scenarioDeployingSolarArrays.py), [scenarioSRPInPanels.py](../../examples/mujoco/scenarioSRPInPanels.py) | Attitude/articulation-dependent illumination and force | MuJoCo and standard spacecraft dynamics have different roles and assumptions |

## Attitude, guidance, control, and actuators

| Capability | Best local starting points | Important chain | Main caveat |
|---|---|---|---|
| Attitude reference generation | [scenarioAttitudeGuidance.py](../../examples/scenarioAttitudeGuidance.py), [scenarioAttGuideHyperbolic.py](../../examples/scenarioAttGuideHyperbolic.py) | Navigation to `AttRefMsg` | A reference generator is not a controller |
| Inertial/target/location pointing | [scenarioAttitudePointing.py](../../examples/scenarioAttitudePointing.py), [scenarioAttLocPoint.py](../../examples/scenarioAttLocPoint.py), [scenarioSpacecraftLocation.py](../../examples/scenarioSpacecraftLocation.py) | Reference, tracking-error, navigation messages | Check body-axis definition and target/observer direction |
| Ideal attitude feedback | [scenarioAttitudeFeedback.py](../../examples/scenarioAttitudeFeedback.py), [scenarioAttitudeFeedbackNoEarth.py](../../examples/scenarioAttitudeFeedbackNoEarth.py) | `attTrackingError -> mrpFeedback -> ExtForceTorque` | Ideal torque cannot size or validate actuators |
| Reaction-wheel control | [scenarioAttitudeFeedbackRW.py](../../examples/scenarioAttitudeFeedbackRW.py), [scenario_FeedbackRW.py](../../examples/BskSim/scenarios/scenario_FeedbackRW.py) | Controller, wheel allocation, motor command, wheel state effector | Include wheel geometry, limits, initial speeds, and momentum |
| Steering/rate-servo chain | [scenarioAttitudeSteering.py](../../examples/scenarioAttitudeSteering.py) | MRP steering, rate servo, torque allocation | Rate/acceleration limits and task ordering affect behavior |
| Constrained attitude maneuvers | [scenarioAttitudeConstrainedManeuver.py](../../examples/scenarioAttitudeConstrainedManeuver.py), [scenarioAttitudeConstraintViolation.py](../../examples/scenarioAttitudeConstraintViolation.py) | Constraint geometry and maneuver/reference generation | Constraint satisfaction requires continuous-time margin checks |
| Gravity-gradient attitude | [scenarioAttitudeGG.py](../../examples/scenarioAttitudeGG.py) | Environmental torque and passive dynamics | Sensitive to inertia ordering and frame convention |
| Thruster attitude control | [scenarioAttitudeFeedback2T_TH.py](../../examples/scenarioAttitudeFeedback2T_TH.py), [scenarioAttitudeFeedback2T_stateEffTH.py](../../examples/scenarioAttitudeFeedback2T_stateEffTH.py) | Two-stage guidance/control mapped to thrusters | Compare dynamic- and state-effector assumptions carefully |
| Momentum dumping | [scenarioMomentumDumping.py](../../examples/scenarioMomentumDumping.py), [scenarioSepMomentumManagement.py](../../examples/scenarioSepMomentumManagement.py) | Wheel momentum, unloading command, external actuator | Unloading authority depends on environment and actuator geometry |
| Magnetorquer management | [scenarioMtbMomentumManagement.py](../../examples/scenarioMtbMomentumManagement.py), [scenarioMtbMomentumManagementSimple.py](../../examples/scenarioMtbMomentumManagementSimple.py) | Magnetic field, dipole command, wheel unloading | Field strength/direction and actuator saturation are orbit-dependent |
| Wheel power | [scenarioAttitudeFeedbackRWPower.py](../../examples/scenarioAttitudeFeedbackRWPower.py) | Mechanical state to electrical load | Example power parameters are illustrative |
| Prescribed attitude/motion | [scenarioAttitudePrescribed.py](../../examples/scenarioAttitudePrescribed.py), [scenarioPrescribedScrewMotion.py](../../examples/scenarioPrescribedScrewMotion.py) | Prescribed-state interfaces | Prescribed motion is not the response to modeled actuator forces |
| Flexible/articulated state effectors | [scenarioHingedRigidBody.py](../../examples/scenarioHingedRigidBody.py), [scenarioFlexiblePanel.py](../../examples/scenarioFlexiblePanel.py), [scenarioDeployingPanel.py](../../examples/scenarioDeployingPanel.py), [scenarioFuelSlosh.py](../../examples/scenarioFuelSlosh.py) | Coupled hub and internal generalized coordinates | Resolve the fastest mode and validate conservation/coupling |
| Control-moment gyroscopes | No dedicated local example identified | Inspect installed Basilisk module documentation before use | Do not infer a preferred CMG architecture from unrelated wheel examples |

## Sensors, navigation, and estimation

| Capability | Best local starting points | What to learn | Main caveat |
|---|---|---|---|
| Truth-derived/simple navigation | [scenarioAttitudeFeedback.py](../../examples/scenarioAttitudeFeedback.py), [scenarioFormationBasic.py](../../examples/scenarioFormationBasic.py) | `SCStatesMsg` to navigation outputs | Default/no-error navigation is not a flight sensor suite |
| Coarse Sun sensors | [scenarioCSS.py](../../examples/scenarioCSS.py), [scenarioCSSFilters.py](../../examples/scenarioCSSFilters.py) | Sensor geometry, illumination, filtering/estimation | Albedo, eclipse, noise, and calibration affect realism |
| Magnetometer and field | [scenarioTAM.py](../../examples/scenarioTAM.py), [scenarioTAMcomparison.py](../../examples/scenarioTAMcomparison.py), [scenarioMagneticFieldWMM.py](../../examples/scenarioMagneticFieldWMM.py) | Field truth, body measurement, model comparison | Epoch/location and body-frame transformations are essential |
| Albedo | [scenarioAlbedo.py](../../examples/scenarioAlbedo.py) | Reflected-light environment and sensor impact | Surface model and geometry assumptions can dominate |
| Thermal sensor | [scenarioSensorThermal.py](../../examples/scenarioSensorThermal.py) | Sensor/thermal state coupling | Demonstration parameters are not hardware qualification data |
| Stochastic processes | [scenarioGaussMarkovRandomWalk.py](../../examples/scenarioGaussMarkovRandomWalk.py) | Bounded Gauss-Markov errors and seeds | Match covariance, correlation time, bounds, and units to the real error source |
| Small-body navigation | [scenarioSmallBodyNav.py](../../examples/scenarioSmallBodyNav.py), [scenarioSmallBodyNavUKF.py](../../examples/scenarioSmallBodyNavUKF.py) | Measurement, filter, truth/estimate comparison | Study each measurement model and observability assumption, not only filter type |
| Measurement-driven attitude estimation | [scenarioTempMeasurementAttitude.py](../../examples/scenarioTempMeasurementAttitude.py) | Estimator driven by an unconventional measurement | Special-purpose example, not a generic attitude-navigation template |

## Relative motion, formations, and constellations

| Capability | Best local starting points | What to learn | Main caveat |
|---|---|---|---|
| Multiple independent spacecraft | [scenarioFormationBasic.py](../../examples/scenarioFormationBasic.py) | Multiple hubs, optional synchronized integration, per-vehicle modules | Same task does not create physical or information coupling |
| Hill-frame conversion | [scenarioFormationBasic.py](../../examples/scenarioFormationBasic.py), [scenarioRendezVous.py](../../examples/scenarioRendezVous.py) | `hill2rv`, `rv2hill`, chief/deputy convention | Hill/LVLH axis conventions and rotating-frame velocity must be explicit |
| Ideal rendezvous sequence | [scenarioRendezVous.py](../../examples/scenarioRendezVous.py), [BASILISK-X cooperative GEO rendezvous](../../scenarios/cooperative_geo_rendezvous/cooperative_geo_rendezvous.py) | Relative geometry, pointing modes, ideal state changes | Neither is flight-representative closed-loop RPO |
| Formation reconfiguration | [scenarioFormationReconfig.py](../../examples/scenarioFormationReconfig.py) | Attitude-coupled finite-pulse thruster reconfiguration logic | This example does not add a fuel tank; verify relative-navigation assumptions and add depletion separately if propellant is a metric |
| Mean orbital-element control | [scenarioFormationMeanOEFeedback.py](../../examples/scenarioFormationMeanOEFeedback.py) | Relative mean-element feedback | Mean/osculating elements and perturbation model must be consistent |
| Differential-drag rendezvous | [scenarioDragRendezvous.py](../../examples/scenarioDragRendezvous.py) | Attitude/area modulation for relative control | Atmosphere and ballistic-coefficient uncertainty are central |
| Fixed two-spacecraft reusable architecture | [BSK_FormationDynamics.py](../../examples/BskSim/models/BSK_FormationDynamics.py), [BSK_FormationFsw.py](../../examples/BskSim/models/BSK_FormationFsw.py) | Separate vehicle tasks, gateways, modes | Specific example design, not automatically better than a standalone script |
| Scalable multi-satellite framework | [BSK_MultiSatMasters.py](../../examples/MultiSatBskSim/BSK_MultiSatMasters.py), [scenario_BasicOrbitMultiSat.py](../../examples/MultiSatBskSim/scenariosMultiSat/scenario_BasicOrbitMultiSat.py) | Indexed dynamics/FSW processes and shared environment | Naming, ownership, ordering, and cost scale with N |
| Station keeping with resources | [scenario_StationKeepingMultiSat.py](../../examples/MultiSatBskSim/scenariosMultiSat/scenario_StationKeepingMultiSat.py) | Formation reference, thrusters, fuel, wheels, power | Broad integrated example; isolate subsystems before reuse |
| Constellation setup | [scenarioSatelliteConstellation.py](../../examples/scenarioSatelliteConstellation.py), [scenario_constellationFromTle.py](../../examples/MultiSatBskSim/scenariosMultiSat/scenario_constellationFromTle.py) | Patterned or externally supplied initial orbits | A constellation propagation is not yet coverage or network analysis |

## Ground geometry, payloads, power, and thermal

| Capability | Best local starting points | What to learn | Main caveat |
|---|---|---|---|
| Ground tracks | [scenarioGroundTracks.py](../../examples/scenarioGroundTracks.py) | Planet-fixed transformation and map output | Absolute longitude needs a real epoch/body orientation |
| Ground access/location | [scenarioGroundLocationImaging.py](../../examples/scenarioGroundLocationImaging.py), [scenarioSpacecraftLocation.py](../../examples/scenarioSpacecraftLocation.py) | Access geometry and target visibility | Access is not a complete sensor or communication performance model |
| Ground mapping | [scenarioGroundMapping.py](../../examples/scenarioGroundMapping.py) | Instrument footprint/ground-point geometry | Terrain, imaging quality, and scheduling may be absent |
| Strip imaging | [scenarioStripImaging.py](../../examples/scenarioStripImaging.py) | Attitude/reference behavior for a scanning payload | Validate line timing, field of view, and agility constraints |
| Downlink/data handling | [scenarioGroundDownlink.py](../../examples/scenarioGroundDownlink.py) | Access-gated data transfer/storage pattern | Not automatically an RF link-budget or network model |
| Power system | [scenarioPowerDemo.py](../../examples/scenarioPowerDemo.py), [scenarioAttitudeFeedbackRWPower.py](../../examples/scenarioAttitudeFeedbackRWPower.py) | Source, load, storage, eclipse/attitude coupling | Illustrative component values and simplified efficiency models |
| Deploying solar arrays | [scenarioDeployingSolarArrays.py](../../examples/scenarioDeployingSolarArrays.py) | Articulation, collection geometry, dynamics | Deployment and electrical models have separate validation needs |
| Thermal behavior | [scenarioSensorThermal.py](../../examples/scenarioSensorThermal.py) | Lumped thermal/sensor response | Not a detailed spacecraft thermal network |

## Monte Carlo, modes, faults, and autonomy

| Capability | Best local starting points | What to learn | Main caveat |
|---|---|---|---|
| Standalone Monte Carlo | [scenarioMonteCarloAttRW.py](../../examples/scenarioMonteCarloAttRW.py), [scenarioMonteCarloSpice.py](../../examples/scenarioMonteCarloSpice.py) | Dispersion, repeats, retention, plotting | Verify deterministic case and seed behavior first |
| `BskSim` Monte Carlo | [scenarioBskSimAttFeedbackMC.py](../../examples/MonteCarloExamples/scenarioBskSimAttFeedbackMC.py), [scenarioRerunMonteCarlo.py](../../examples/MonteCarloExamples/scenarioRerunMonteCarlo.py), [scenarioVisualizeMonteCarlo.py](../../examples/MonteCarloExamples/scenarioVisualizeMonteCarlo.py) | Reusable scenario execution, rerun, aggregation | File retention and multiprocessing assumptions require operational planning |
| FSW mode switching | [scenario_AttModes.py](../../examples/BskSim/scenarios/scenario_AttModes.py), [BSK_Fsw.py](../../examples/BskSim/models/BSK_Fsw.py) | Events, task enable/disable, gateway messages | Gateway zeroing and transition timing are safety-critical |
| Reaction-wheel faults | [scenario_AddRWFault.py](../../examples/BskSim/scenarios/scenario_AddRWFault.py), [scenario_FaultList.py](../../examples/BskSim/scenarios/scenario_FaultList.py) | Fault injection through example architecture | Injected faults are not a validated reliability model |
| Constraint monitoring | [scenarioAttitudeConstraintViolation.py](../../examples/scenarioAttitudeConstraintViolation.py) | Geometry and violation reporting | Monitoring does not provide recovery or safety assurance |
| Reinforcement learning / BSK-RL | No local implementation or dependency identified | Treat official `AVSLab/bsk_rl` as a separate external layer | Do not infer observations, actions, rewards, or training support from this repository |

## Optical navigation

| Capability | Best local starting points | What to learn | Main caveat |
|---|---|---|---|
| OpNav master architecture | [BSK_OpNav.py](../../examples/OpNavScenarios/BSK_OpNav.py), [BSK_OpNavDynamics.py](../../examples/OpNavScenarios/modelsOpNav/BSK_OpNavDynamics.py), [BSK_OpNavFsw.py](../../examples/OpNavScenarios/modelsOpNav/BSK_OpNavFsw.py) | Truth, camera/Vizard, image processing, filter, and modes | Specialized architecture with external Vizard/OpNav dependencies |
| Center/heading OpNav | [scenario_OpNavHeading.py](../../examples/OpNavScenarios/scenariosOpNav/scenario_OpNavHeading.py), [scenario_OpNavOD.py](../../examples/OpNavScenarios/scenariosOpNav/scenario_OpNavOD.py) | Optical measurement to heading/orbit determination | Establish which stages use rendered images versus analytic measurements |
| Limb-based OpNav | [scenario_OpNavODLimb.py](../../examples/OpNavScenarios/scenariosOpNav/scenario_OpNavODLimb.py), [scenario_OpNavAttODLimb.py](../../examples/OpNavScenarios/scenariosOpNav/scenario_OpNavAttODLimb.py) | Limb extraction feeding navigation | Lighting, apparent-body geometry, and detection failures matter |
| CNN image processing | [scenario_CNNAttOD.py](../../examples/OpNavScenarios/scenariosOpNav/scenario_CNNAttOD.py), [scenario_CNNImages.py](../../examples/OpNavScenarios/scenariosOpNav/CNN_ImageGen/scenario_CNNImages.py) | Learned image-processing boundary and image generation | Model weights/data validity and external dependencies must be established separately |
| OpNav fault detection | [scenario_faultDetOpNav.py](../../examples/OpNavScenarios/scenariosOpNav/scenario_faultDetOpNav.py) | Redundant measurement/filter fault logic | Example thresholds are not mission-qualified |
| OpNav Monte Carlo | [OpNavMC](../../examples/OpNavScenarios/scenariosOpNav/OpNavMC/), [OpNavMonteCarlo.py](../../examples/OpNavScenarios/scenariosOpNav/CNN_ImageGen/OpNavMonteCarlo.py) | Image/filter dispersion campaigns | Rendering and multiprocessing can dominate runtime and reproducibility |

## MuJoCo and contact-capable multibody simulation

| Capability | Best local starting points | What to learn | Main caveat |
|---|---|---|---|
| MuJoCo master/model split | [BSK_mujocoMasters.py](../../examples/mujoco/BSK_mujocoMasters.py), [BSK_MujocoDynamics.py](../../examples/mujoco/mujocoModels/BSK_MujocoDynamics.py), [BSK_MujocoFSW.py](../../examples/mujoco/mujocoModels/BSK_MujocoFSW.py) | XML-defined multibody plant integrated with Basilisk messages/FSW | Separate engine, configuration language, and validation burden |
| Reaction wheel in MuJoCo | [scenarioReactionWheel.py](../../examples/mujoco/scenarioReactionWheel.py), [scenarioAttitudeFeedbackRWMuJoCo.py](../../examples/mujoco/scenarioAttitudeFeedbackRWMuJoCo.py) | Actuator/FSW mapping into MuJoCo plant | Compare signs, inertias, and solver behavior with standard Basilisk baseline |
| Robotic arm | [scenarioArmWithThrusters.py](../../examples/mujoco/scenarioArmWithThrusters.py), [scenarioThrArmControl.py](../../examples/mujoco/scenarioThrArmControl.py) | Articulated robot and spacecraft actuation | Coupled control/contact can be numerically stiff |
| Deployable/branching panels | [scenarioDeployPanels.py](../../examples/mujoco/scenarioDeployPanels.py), [scenarioBranchingPanels.py](../../examples/mujoco/scenarioBranchingPanels.py) | XML joints, branching topology, deployment | Use normal Basilisk state effectors when they adequately represent the mechanism |
| Docking/capture constraint | [scenarioSimpleDocking.py](../../examples/mujoco/scenarioSimpleDocking.py) | Multiple bodies and manual activation of a weld equality | Demonstrates constrained capture topology, not sensor-driven approach, compliant impact/contact, or latch dynamics |
| Surface landing | [scenarioAsteroidLanding.py](../../examples/mujoco/scenarioAsteroidLanding.py) | Gravity plus contact/surface interaction | Shape, gravity, and contact properties are highly mission-specific |
| Formation with drag | [scenarioFormationFlyingWithDrag.py](../../examples/mujoco/scenarioFormationFlyingWithDrag.py) | Multiple MuJoCo bodies in an orbital environment | MuJoCo is unnecessary if standard uncoupled spacecraft propagation answers the question |

## Current BASILISK-X baselines

| Scenario | Capability represented | Recommended official comparison |
|---|---|---|
| [basic_earth_orbit.py](../../scenarios/basic_earth_orbit/basic_earth_orbit.py) | Point-mass orbit, derived elements/ground track, Vizard modes | [scenarioBasicOrbit.py](../../examples/scenarioBasicOrbit.py) |
| [nadir_pointing.py](../../scenarios/nadir_pointing/nadir_pointing.py) | Hill reference, tracking error, MRP feedback, ideal torque | [scenarioAttitudeGuidance.py](../../examples/scenarioAttitudeGuidance.py), [scenarioAttitudeFeedback.py](../../examples/scenarioAttitudeFeedback.py) |
| [cooperative_geo_rendezvous.py](../../scenarios/cooperative_geo_rendezvous/cooperative_geo_rendezvous.py) | Two-body chief/deputy propagation, CW targeting, truth target pointing | [scenarioRendezVous.py](../../examples/scenarioRendezVous.py), [scenarioFormationBasic.py](../../examples/scenarioFormationBasic.py) |
