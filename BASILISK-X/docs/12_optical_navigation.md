> **Draft status — AI-generated:** This page was drafted by OpenAI Codex on 2026-08-19. It has not yet been technically vetted by the BASILISK-X repository owner. The repository owner will review, correct, and maintain it. Verify APIs, units, frames, assumptions, and version compatibility before engineering use.

# Optical Navigation

## What this chapter establishes

Optical navigation is not one model. It is a chain of physical truth, image formation, perception, measurement geometry, estimation, and spacecraft response. The local [`OpNavScenarios`](../examples/OpNavScenarios/) tree is valuable because it exposes that complete architectural chain around a Mars orbiter. It is not currently a turnkey BASILISK-X subsystem: the copied sources and the pinned Basilisk 2.11.1 installation have known incompatibilities and external dependencies documented below.

Labels used here have deliberate meanings:

- **Observed locally** means the statement comes from checked-in source or a smoke check against `basiliskx_env` on 2026-08-19.
- **Engineering recommendation** means guidance inferred from those sources, not a guarantee made by Basilisk.
- **Unverified** means the architecture is visible, but this repository has not demonstrated an end-to-end execution of it.

The principal sources are the master [`BSK_OpNav.py`](../examples/OpNavScenarios/BSK_OpNav.py), dynamics model [`BSK_OpNavDynamics.py`](../examples/OpNavScenarios/modelsOpNav/BSK_OpNavDynamics.py), FSW model [`BSK_OpNavFsw.py`](../examples/OpNavScenarios/modelsOpNav/BSK_OpNavFsw.py), and the scenario scripts under [`scenariosOpNav`](../examples/OpNavScenarios/scenariosOpNav/). Current installation guidance describes optical-navigation components as the optional `bsk-opnav` package, included by the `all` extra used by this project; the copied master's older `conanfile.py --opNav True` instructions belong to a source-build workflow. See the [official Basilisk installation page](https://avslab.github.io/basilisk/Install.html).

## The end-to-end mental model

The most important distinction is between **what exists physically** and **what flight software believes**.

```text
PHYSICAL / TRUTH SIDE
SPICE epoch and body ephemerides
             |
             v
Mars-centered gravity ---> Spacecraft 6-DOF truth state
                                  |
                                  +-- attitude and camera mounting
                                  +-- Mars/Sun/scene geometry
                                  v
                         Vizard scene renderer
                                  |
                                  v
                         ideal synthetic image
                                  |
                                  v
                    Camera acquisition/degradation

FLIGHT-SOFTWARE SIDE
degraded image
     |
     +--> Hough circle ----+
     +--> limb points -----+--> optical measurement --> estimator
     +--> CNN circle ------+                           |
                                                        v
                                            navigation solution/validity
                                                        |
                                                        v
                                             pointing and RW control
```

This separation exists so each claim can be tested independently. A filter can be tested with generated measurements without claiming realistic imagery. An image processor can be tested against labelled images without claiming an accurate orbit model. A closed-loop autonomy claim requires both, plus timing, failure handling, and actuator dynamics.

| Layer | Question answered | Local implementation | What it is not |
|---|---|---|---|
| Truth dynamics | Where is the spacecraft, and how does it move? | `Spacecraft`, gravity factory, SPICE, reaction wheels | A navigation estimate |
| Scene rendering | What reaches an ideal virtual camera from the configured geometry? | external Vizard process through `vizInterface` | A calibrated radiometric detector |
| Camera sensor | When is an image acquired, and how is it degraded? | two `camera.Camera` objects; only the first feeds FSW | Image understanding |
| Perception | What image features were found? | `HoughCircles`, `LimbFinding`, `CenterRadiusCNN` | Position/velocity estimation |
| Measurement geometry | What LOS/range-like optical quantity follows from pixels and calibration? | `pixelLineConverter`, `horizonOpNav` | A propagated navigation state |
| Estimation | What state and covariance are consistent over time? | `relativeODuKF`, `pixelLineBiasUKF`, `headingSuKF` | Truth |
| Guidance/control | Where should the camera point, and what wheel torques achieve it? | `opNavPoint`, `mrpFeedback`, `rwMotorTorque` | Mission decision logic |
| Modes/fault logic | Which chain is active and which measurement is accepted? | events, gateway messages, `faultDetection` | A complete FDIR/autonomy system |

## 1. Truth dynamics and the rendered scene

### Mars-centred physical model

**Observed locally:** [`BSK_OpNavDynamics.py`](../examples/OpNavScenarios/modelsOpNav/BSK_OpNavDynamics.py) constructs one 750 kg spacecraft with body inertia `diag(900, 800, 600) kg m^2`, a four-wheel pyramid, an external force/torque effector, eight ACS thrusters, coarse Sun sensors, eclipse, and `SimpleNav`. It creates Sun, Earth, Mars barycentre, and Jupiter barycentre gravity bodies. Mars is central and uses the GGM2B spherical-harmonic model to degree 2. SPICE is initialized at `2019 DECEMBER 12 18:00:00.0`, with `J2000` as reference and Mars barycentre as `zeroBase`.

This means `r_BN_N` in the examples is a Mars-relative inertial truth state, not an Earth-centred or spacecraft-relative vector. The filters separately hard-code Mars' gravitational parameter as `42828.314e9 m^3/s^2` and use `planetIdInit = 2`. Changing the target body therefore requires coordinated changes to truth, ephemeris, measurement geometry, filter dynamics, radii used in analysis, and imagery.

`SimpleNav` subscribes to `scObject.scStateOutMsg`. Position and velocity error processes are configured, while attitude and attitude-rate walk bounds are approximately `1e-18` degrees. Thus the image-geometry and control chain receives effectively perfect attitude even though translation is perturbed. This is a strong observability-isolation assumption, not a realistic combined attitude/OpNav navigation result.

### Vizard is part of the sensor in these scenarios

For ordinary Basilisk visualization, Vizard is a presentation consumer. In this tree it closes a two-way sensor loop:

```text
Spacecraft truth + SPICE bodies + camera configuration
                         |
                         v
                 vizInterface / ZeroMQ
                         |
                         v
             external Vizard renderer (:5556)
                         |
                         v
          vizInterface.opnavImageOutMsgs[0]
                         |
                         v
             cameraMod.imageInMsg
                         |
             blur/noise/cosmic-ray model
                         |
                         v
             cameraMod.imageOutMsg
```

[`SetVizInterface`](../examples/OpNavScenarios/modelsOpNav/BSK_OpNavDynamics.py) calls `vizSupport.enableUnityVisualization`, registers both camera configuration messages with `addCamMsgToModule`, and sets headless display and a black skybox. [`SetCamera`](../examples/OpNavScenarios/modelsOpNav/BSK_OpNavDynamics.py) subscribes the first camera to `opnavImageOutMsgs[0]`; `SetCamera2` similarly connects a second view, but the FSW image processors use only `cameraMod.imageOutMsg`.

The first camera is configured as 512 x 512 pixels, 55 degree edge-to-edge field of view, a nominal 10 mm square sensor, position `[0, 0.2, 2.2] m` in the spacecraft body, and identity camera-to-body MRP. Its default blur parameter is 3. Gaussian noise, dark current, salt-and-pepper noise, cosmic rays, and image saving are exposed; individual scenarios enable some of them. The camera model therefore handles acquisition/configuration and post-render degradation. Vizard produces the synthetic scene image.

The official [Vizard live-communication documentation](https://avslab.github.io/basilisk/Vizard/vizardAdvanced/vizardLiveComm.html) distinguishes lockstep `liveStream` from headless `noDisplay`, in which Vizard renders when Basilisk requests an OpNav image. Renderer mode, assets, version, port, and timing are therefore part of the sensor configuration and must be recorded with any result.

## 2. From images to optical measurements

### Circle/Hough branch

[`SetImageProcessing`](../examples/OpNavScenarios/modelsOpNav/BSK_OpNavFsw.py) connects:

```text
CameraImageMsg
      |
      v
HoughCircles -- centre [pixel], radius [pixel], valid --> OpNavCirclesMsg
      |
      v
pixelLineConverter + CameraConfigMsg + attitude NavAttMsg
      |
      v
OpNavMsg: target LOS/position-style measurement, covariance, validity, time tag
```

The Hough configuration assumes one expected circle and sets Canny, vote, blur, minimum-distance, and minimum-radius thresholds. These are algorithm/tuning choices for this synthetic Mars image set; they are not universal camera parameters.

`pixelLineConverter` supplies the camera geometry that a feature detector deliberately does not know. It combines the detected centre/radius with focal geometry and attitude and writes through the shared `OpNavMsg_C` gateway. Apparent angular radius provides range information only because the target radius/model is assumed known. A centroid-only or unresolved target does not provide that same instantaneous range observable.

### Limb branch

[`SetLimbFinding`](../examples/OpNavScenarios/modelsOpNav/BSK_OpNavFsw.py) and [`SetHorizonNav`](../examples/OpNavScenarios/modelsOpNav/BSK_OpNavFsw.py) form a separate chain:

```text
CameraImageMsg -> LimbFinding -> OpNavLimbMsg (edge/limb points)
                                      |
                                      v
          horizonOpNav + camera config + attitude -> OpNavMsg
```

The separation is architecturally important. `LimbFinding` is perception; `horizonOpNav` is the measurement model that interprets detected limb geometry. Keeping them separate permits tests with synthetic limb points and makes calibration/frame errors distinguishable from edge-detection errors.

### CNN branch

[`SetCNNOpNav`](../examples/OpNavScenarios/modelsOpNav/BSK_OpNavFsw.py) feeds `cameraMod.imageOutMsg` to `CenterRadiusCNN`, which emits the same circle payload used by `pixelLineConverter`. The downstream geometry, estimator, and control interfaces therefore do not need to know whether a classical Hough transform or a learned model detected the circle.

That interchangeable feature-message boundary is the reusable pattern:

```text
classical detector ---+
learned detector -----+--> common feature payload --> common geometry/filter chain
recorded test data ---+
```

It does **not** make the detectors equivalent. Training distribution, confidence calibration, out-of-distribution behaviour, and image preprocessing must be validated separately.

## 3. Estimation and navigation outputs

The local FSW creates three unscented-filter families.

| Module | Inputs in this architecture | State/configuration observed | Local use |
|---|---|---|---|
| `relativeODuKF` | shared `OpNavMsg` | six-state Mars-relative position/velocity, covariance and process noise | orbit determination in Hough and limb scenarios |
| `pixelLineBiasUKF` | circle pixels, camera configuration, attitude | nine-state position/velocity plus three biases | bias-estimating alternative in OD scenarios |
| `headingSuKF` | shared `OpNavMsg`, camera configuration | five-state heading/rate-style filter | filtered line-of-sight pointing in [`scenario_OpNavHeading.py`](../examples/OpNavScenarios/scenariosOpNav/scenario_OpNavHeading.py) |

The scenario recorders use each filter's `filtDataOutMsg`; the heading case also records `opnavDataOutMsg` and explicitly rewires `opNavPoint.opnavDataInMsg` to it.

A subtle but essential observation is that the integrated `OpNavAttOD` and limb tasks do not generally feed the `relativeODuKF` state back into pointing. `opNavPoint` consumes the current shared optical measurement, while the OD filter executes later and its state is recorded for assessment. The heading scenario is the clear local example of a filtered measurement driving pointing. Do not draw a generic "estimator drives all GNC" arrow unless the actual subscription proves it.

The filters' covariance is an internal consistency statement conditioned on their models and tuning. It does not include every error in the Vizard renderer, camera calibration, target shape, attitude solution, processing bias, or timing unless those uncertainties are explicitly represented.

## 4. Guidance, control, and actuator chain

The closed attitude chain is:

```text
OpNavMsg / headingUKF.opnavDataOutMsg
                 |
                 v
             opNavPoint <--- SimpleNav attitude + CameraConfigMsg
                 |
                 v
          AttGuidMsg gateway
                 |
                 v
            mrpFeedback <--- vehicle inertia + RW speeds/config
                 |
                 v
           rwMotorTorque
                 |
                 v
ReactionWheelStateEffector.rwMotorCmdInMsg -> spacecraft truth
```

`opNavPoint` aligns camera axis `[0, 0, 1]`, uses a 1000 s image timeout, and defines a search angular rate. Loss of a valid/recent target can therefore transition the pointing behaviour into search, but this is not a complete acquisition/reacquisition state machine.

The `prepOpNav` and OD-only modes use a deliberate **cheat-pointing** chain: `hillPoint` consumes truth-derived `SimpleNav` translation and Mars ephemeris; `attTrackingError` applies a fixed camera rotation; reaction-wheel control keeps Mars in view. This isolates perception/OD performance from initial target-acquisition performance. It should not be reported as autonomous OpNav pointing.

## 5. Execution rates, priorities, modes, and gateways

### Rates and ordering

The standalone scenario entry points normally instantiate `BSKSim(fswRate=0.5, dynRate=0.5)`. The architecture then schedules:

| Scope | Rate/priority observed | Consequence |
|---|---|---|
| Dynamics process | process priority 100 | executes ahead of FSW process priority 10 at coincident times |
| `DynamicsTask` | 0.5 s, task priority 1000 | spacecraft/environment/nav/Vizard exchange |
| `CameraTask` | fixed 60 s, task priority 999 | both camera models |
| FSW tasks | 0.5 s, task priorities 20 to 5 | selected mode pipeline and RW controller |
| Camera `renderRate` | 60 s | nominal new-image cadence |

Within `DynamicsTask`, the source assigns RW 301, external force/torque 300, CSS 299, eclipse 204, spacecraft 201, SPICE 200, ephemeris converter 199, `SimpleNav` 109, and Vizard interface 100. Higher numeric priority executes first.

This creates a multi-rate system: dynamics and FSW execute 120 times for each nominal camera acquisition. The image and measurement modules are still scheduled at the FSW rate. Whether each algorithm rejects repeated/stale payloads depends on its time-tag logic; the scenario orchestration does not itself perform a "new frame" guard. Timestamp, validity, image sequence, processing latency, and held-data behaviour must be checked before changing rates.

### Gateway messages

[`setupGatewayMsgs`](../examples/OpNavScenarios/modelsOpNav/BSK_OpNavFsw.py) creates shared attitude-guidance, optical-measurement, primary/secondary optical-measurement, and circle-feature messages. Producers register as authors; downstream consumers subscribe once to the gateway. On every mode change, `zeroGateWayMsgs` writes zero payloads, all FSW tasks are disabled, and the event enables the selected chain.

```text
mode-specific producer A --+
mode-specific producer B --+--> gateway --> common controller/filter
mode-specific producer C --+
```

This solves mode-dependent rewiring. It is safe only if the enabled task set guarantees the intended authors and execution order. A zero payload is not automatically equivalent to a formally invalid measurement; consumers must honour validity and timestamps.

### Modes represented

The FSW defines standby, preparation, image generation, Hough/heading/limb pointing, Hough/limb/bias OD, integrated pointing plus OD, CNN integrated OD, and fault-detection events. Every task is disabled initially. Scenario scripts set `modeRequest`, initialize, and often run a short preparation phase before continuing to an absolute final stop time.

The reusable lesson is to make image acquisition, perception choice, estimator choice, and attitude control explicit mode composition. The local implementation uses 15 tasks with repeated module instances, so it is better studied as a mode-composition example than copied as a production state machine.

## 6. Fault branch

[`scenario_faultDetOpNav.py`](../examples/OpNavScenarios/scenariosOpNav/scenario_faultDetOpNav.py) registers the limb/horizon solution as the primary OpNav author and the Hough/pixel-line solution as secondary. `faultDetection` receives both, plus camera configuration and attitude, and writes the selected result to the common `OpNavMsg` gateway. The scenario injects camera cosmic rays and records `valid` and `faultDetected`.

```text
image -> limb -> horizon measurement --- primary --+
                                                    +--> faultDetection --> OpNav gateway
image -> Hough -> pixel-line measurement - secondary+
```

This demonstrates algorithmic consistency checking and fallback. It does not model camera hardware redundancy, correlated common-mode errors, missed detections under identical image corruption, formal isolation/recovery, or safe-mode mission consequences. Both branches share the same image, camera calibration, attitude input, renderer, and target model.

## 7. Monte Carlo and image generation

[`OpNavMC/MonteCarlo.py`](../examples/OpNavScenarios/scenariosOpNav/OpNavMC/MonteCarlo.py) configures two single-threaded runs, disperses orbital elements, field of view, and filter noise scaling, requests dispersed seeds, retains truth/measurement/filter messages, and registers a plotting callback. Conceptually it converts the deterministic chain into:

```text
sample orbit/calibration/filter parameters
        -> start Vizard-backed simulation
        -> retain truth, validity, estimate, covariance
        -> compute error and consistency metrics
```

[`CNN_ImageGen/OpNavMonteCarlo.py`](../examples/OpNavScenarios/scenariosOpNav/CNN_ImageGen/OpNavMonteCarlo.py) instead disperses orbit, camera/spacecraft alignment, Gaussian noise, salt-and-pepper noise, cosmic rays, and blur to create labelled images for CNN training. This is a data-generation architecture, not a trained-network validation architecture. A credible learned-perception campaign must split runs by geometry and nuisance condition, retain seeds/configuration, prevent truth leakage, and assess false positives and out-of-distribution cases.

## 8. Best local study sequence

| Order | Example | What to isolate |
|---:|---|---|
| 1 | [`scenario_OpNavPoint.py`](../examples/OpNavScenarios/scenariosOpNav/scenario_OpNavPoint.py) | Hough circle -> pixel geometry -> pointing -> reaction wheels |
| 2 | [`scenario_OpNavPointLimb.py`](../examples/OpNavScenarios/scenariosOpNav/scenario_OpNavPointLimb.py) | limb points versus circle abstraction |
| 3 | [`scenario_OpNavOD.py`](../examples/OpNavScenarios/scenariosOpNav/scenario_OpNavOD.py) | cheat pointing plus Hough OD; compare six-state and bias filters |
| 4 | [`scenario_OpNavODLimb.py`](../examples/OpNavScenarios/scenariosOpNav/scenario_OpNavODLimb.py) | limb measurement feeding the same relative-OD filter |
| 5 | [`scenario_OpNavHeading.py`](../examples/OpNavScenarios/scenariosOpNav/scenario_OpNavHeading.py) | filtered LOS/heading explicitly fed back to pointing |
| 6 | [`scenario_OpNavAttOD.py`](../examples/OpNavScenarios/scenariosOpNav/scenario_OpNavAttOD.py) | integrated Hough pointing and OD; inspect which outputs actually close loops |
| 7 | [`scenario_OpNavAttODLimb.py`](../examples/OpNavScenarios/scenariosOpNav/scenario_OpNavAttODLimb.py) | integrated limb equivalent |
| 8 | [`scenario_faultDetOpNav.py`](../examples/OpNavScenarios/scenariosOpNav/scenario_faultDetOpNav.py) | dual perception/measurement paths and common-mode faults |
| 9 | [`scenario_CNNAttOD.py`](../examples/OpNavScenarios/scenariosOpNav/scenario_CNNAttOD.py) | learned detector at the common feature-message boundary; currently blocked by missing weights |
| 10 | [`OpNavMC`](../examples/OpNavScenarios/scenariosOpNav/OpNavMC/) and [`CNN_ImageGen`](../examples/OpNavScenarios/scenariosOpNav/CNN_ImageGen/) | uncertainties, retention, image labels, and external-process scaling |

Study each layer with recorded intermediate messages before enabling the next. A final position-error plot alone cannot tell whether an error came from rendering, feature extraction, camera geometry, attitude, filter tuning, or a time-tag defect.

## 9. Adapting the architecture

| Application | Preserve | Replace or add | Dominant observability/fidelity issue |
|---|---|---|---|
| Earth limb navigation | renderer/camera/perception/geometry separation | Earth-centred truth, Earth radius/oblateness, cloud/atmosphere/texture cases, Earth filter dynamics | a spherical sharp limb is not the radiometric Earth horizon |
| Planetary OpNav | common feature and `OpNavMsg` seams | target ephemeris, radius/shape, illumination, gravity, filter constants | phase angle, irregular shape, albedo and target-model bias |
| Angles-only navigation | attitude-calibrated LOS measurement | remove apparent-radius range assumption; add time history and manoeuvre geometry | range is weak/unobservable instantaneously; manoeuvres and dynamics create observability |
| Relative spacecraft navigation | image pipeline and mode concept | second spacecraft truth, target mesh/attitude/light, keypoint or silhouette detector, relative pose/state estimator and new messages | scale/range, target attitude, latency, occlusion and target ephemeris exchange |
| Camera-based rendezvous | relative-spacecraft additions | approach guidance, keep-out constraints, plume/contact sensor models, safety monitor | navigation uncertainty must couple to closing-rate and collision constraints |
| Inspection | target rendering and relative pose | landmark/coverage map, view-quality and lighting metrics, target tumble | seeing the target is not measuring coverage or identifying defects |
| Landmark navigation | detector/measurement/estimator layers | landmark catalogue, data association, bearing/PnP measurement model | false association and map/frame error; also study [`scenarioSmallBodyLandmarks.py`](../examples/scenarioSmallBodyLandmarks.py) |
| Autonomous navigation | all validated lower layers | acquisition/reacquisition, validity gates, covariance-aware modes, resource/latency model | autonomy cannot compensate for an unmodelled sensor failure or inconsistent time base |

For relative spacecraft work, the local Mars examples provide architecture but no complete target-relative camera navigation implementation. Do not rename the Mars-centre vector as a deputy-relative state; introduce an explicit relative-state payload and define its origin, axes, time, covariance, and target identity.

## 10. Minimum useful fidelity and validation

Increase fidelity only when it can change the engineering decision.

| Question | Minimum useful model | Add imagery when... | Add closed-loop 6-DOF when... |
|---|---|---|---|
| Is geometry observable? | analytic LOS/apparent-radius measurements with controlled noise | detector bias or illumination may dominate | attitude/actuation changes access or image quality |
| Does a filter converge? | measurement generator plus correct time tags and truth | feature dropouts/bias must be learned from pixels | pointing and estimator errors materially couple |
| Can a detector find the body? | labelled static or scripted image set | already required | only if motion blur, control jitter, or acquisition matters |
| Can the mission navigate autonomously? | validated estimator plus explicit validity/modes | required if imagery is the sensor | required when commands affect future observability/safety |

Validation should progress across seams:

1. Check camera intrinsics/extrinsics and pixel sign conventions with analytic projections.
2. Feed perfect circle/limb features into geometry modules and compare with analytic range/LOS.
3. Run filters on generated measurements with known noise and evaluate normalized innovation and estimation error, not only convergence plots.
4. Test perception on labelled images across phase, range, target size, noise, blur, cosmic rays, partial target, and no-target cases.
5. Measure end-to-end timestamps and latency from exposure to command.
6. Close pointing only after open-loop measurement and estimator tests pass.
7. Introduce common-mode failures and verify safe behaviour, not merely a fault flag.

The local renderer chain is geometric/synthetic. The scenarios do not establish calibrated radiometry, detailed point-spread function, lens distortion, rolling shutter, exposure control, saturation/blooming, thermal behaviour, radiation-induced detector physics, real feature-map errors, or flight-processor timing. Treat those as explicit missing fidelity, not implicit Vizard capability.

## 11. Local compatibility and correctness audit

These findings are important enough to check before any OpNav execution.

### Confirmed package/source mismatch

- [`requirements.txt`](../requirements.txt) pins `bsk[all,examples]==2.11.1`.
- [`BSK_OpNavFsw.py`](../examples/OpNavScenarios/modelsOpNav/BSK_OpNavFsw.py) imports `hasBuildFeature` from `Basilisk`. That symbol is absent from the installed 2.11.1 package, so importing the local FSW master fails before scenario construction.
- Direct imports of `centerRadiusCNN`, `houghCircles`, `limbFinding`, `relativeODuKF`, `pixelLineBiasUKF`, and `headingSuKF` succeeded in the installed environment during this audit. The immediate blocker is therefore the copied build-feature guard, not proof that all OpNav runtime dependencies are healthy.
- The source tree appears to mix development-era code with a 2.11.1 runtime. Select and record one compatible upstream revision before adapting it; do not simply delete a guard and call the system validated.

### Missing CNN asset

`SetCNNOpNav` resolves the network as `BASILISK-X/src/fswAlgorithms/imageProcessing/centerRadiusCNN/CAD.onnx` because its repository-root search stops at this project's `pyproject.toml`. No ONNX file exists anywhere in BASILISK-X. The CNN scenario and CNN fault branch cannot perform intended inference until the exact trained model, preprocessing contract, provenance, licence, and checksum are supplied.

### Vizard path, port, and lifecycle

- [`BSK_OpNav.py`](../examples/OpNavScenarios/BSK_OpNav.py) hard-codes `/Applications/Vizard.app/Contents/MacOS/Vizard`. That executable existed on the audit machine, but its version is not pinned in the repository and the path is not portable.
- Every launcher uses `tcp://localhost:5556`. Concurrent scenarios or parallel Monte Carlo workers would contend for the same endpoint. The supplied MC configurations use one worker, but changing `PROCESSES` is not safe without per-run port/process management.
- Failure to connect can block the image loop. Cleanup uses direct `kill()` calls rather than an exception-safe context/finally discipline, and `end_scenario` exits the interpreter when Vizard was not launched.
- Vizard, its rendered body/texture set, and its version are not repository-contained simulation assets. Archive them as configuration for retained results.

### Mode/task defects and ambiguities

- The bias branches in [`scenario_OpNavAttOD.py`](../examples/OpNavScenarios/scenariosOpNav/scenario_OpNavAttOD.py) and [`scenario_CNNAttOD.py`](../examples/OpNavScenarios/scenariosOpNav/scenario_CNNAttOD.py) request `OpNavAttODB`; no event with that condition exists. The defined bias-OD mode is `OpNavODB`. Defaults use `relOD`, so the defect is latent until `filterUse == "bias"`.
- The `cnnFaultDet` task registers `pixelLine` twice and schedules both CNN and Hough producers against the same circle gateway. This may cause two feature writes and two conversions in one task; the intended primary/secondary semantics are not made explicit. Treat this branch as unverified until its author ordering and payload provenance are tested.
- The same module instances are inserted into multiple mutually exclusive tasks. That is economical, but enabling an unintended combination can update one stateful estimator more than once per tick or create multiple gateway authors.
- Gateways are zeroed on mode changes. Consumers must use `valid` and time tags rather than interpret a zero vector as a real observation.

### Monte Carlo/image-generation defects

- In [`CNN_ImageGen/OpNavMonteCarlo.py`](../examples/OpNavScenarios/scenariosOpNav/CNN_ImageGen/OpNavMonteCarlo.py), CSV generation is outside the loop that loads retained runs; only the last successfully loaded run is written, using the final loop index.
- [`CNN_ImageGen/scenario_CNNImages.py`](../examples/OpNavScenarios/scenariosOpNav/CNN_ImageGen/scenario_CNNImages.py) refers to nonexistent `self.scRecmsgRecList` in `pull_outputs`; the recorded dictionary is `msgRecList`.
- The same file comments out deterministic spacecraft position/velocity initialization and relies on Monte Carlo dispersion to set them. Standalone use therefore has a different initialization contract from the ordinary scenarios.
- Its teardown refers to `get_DynModel().spiceObject`; the dynamics model stores SPICE under `gravFactory.spiceObject`. [`OpNavMC/scenario_OpNavAttODMC.py`](../examples/OpNavScenarios/scenariosOpNav/OpNavMC/scenario_OpNavAttODMC.py) has the same incorrect attribute. [`OpNavMC/scenario_LimbAttOD.py`](../examples/OpNavScenarios/scenariosOpNav/OpNavMC/scenario_LimbAttOD.py) uses the gravity-factory path.

### Model limitations that can look like success

- Mars constants, target identifiers, and plot radius are hard-coded in several locations.
- Camera images arrive nominally every 60 s while the processing/filter/control tasks run every 0.5 s.
- Attitude knowledge is effectively perfect.
- The Hough setup expects one circle; the examples do not demonstrate multiple luminous bodies or target association.
- The fault example compares two algorithms sharing one corrupted image and much of the same geometry.
- The examples measure a resolved planetary disk/limb; they do not establish angles-only point-target or relative-spacecraft navigation.
- Position estimates are frequently analysed but not fed into mission guidance. A converged plotted filter is not yet an autonomous-navigation architecture.

## Engineering recommendation for BASILISK-X

Preserve the layer boundaries, but do not copy the monolithic OpNav master unchanged. A maintainable BASILISK-X OpNav experiment should make these contracts explicit:

```text
truth provider -> renderer adapter -> camera sensor -> feature detector
               -> measurement adapter -> estimator -> validity/mode manager
               -> guidance/control
```

Each arrow should name payload type, frame, units, timestamp/exposure epoch, covariance meaning, validity rules, and update rate. External Vizard lifecycle and port allocation belong in a reusable adapter only after one deterministic end-to-end baseline is version-matched and tested. Detector thresholds, target model, filter tuning, mission modes, and success metrics should remain scenario configuration until repeated use proves a stable abstraction.

Before new OpNav research, first create a compatibility manifest containing the Basilisk and `bsk-opnav` versions, Vizard version/path, endpoint, SPICE kernel set, target assets, CNN weight checksum, and an import/one-frame smoke test. That is the minimum foundation on which deeper Earth-limb, rendezvous-camera, landmark, or autonomy work can be trusted.

## Source index

- Architecture: [`BSK_OpNav.py`](../examples/OpNavScenarios/BSK_OpNav.py)
- Truth, camera, and Vizard wiring: [`modelsOpNav/BSK_OpNavDynamics.py`](../examples/OpNavScenarios/modelsOpNav/BSK_OpNavDynamics.py)
- Perception, filters, GNC, tasks, events, and gateways: [`modelsOpNav/BSK_OpNavFsw.py`](../examples/OpNavScenarios/modelsOpNav/BSK_OpNavFsw.py)
- Output analysis: [`plottingOpNav/OpNav_Plotting.py`](../examples/OpNavScenarios/plottingOpNav/OpNav_Plotting.py)
- Deterministic scenarios: [`scenariosOpNav`](../examples/OpNavScenarios/scenariosOpNav/)
- OpNav ensemble: [`scenariosOpNav/OpNavMC`](../examples/OpNavScenarios/scenariosOpNav/OpNavMC/)
- CNN image generation: [`scenariosOpNav/CNN_ImageGen`](../examples/OpNavScenarios/scenariosOpNav/CNN_ImageGen/)
- Official references: [Basilisk integrated examples](https://avslab.github.io/basilisk/examples/index.html), [installation and optional components](https://avslab.github.io/basilisk/Install.html), and [Vizard live communication](https://avslab.github.io/basilisk/Vizard/vizardAdvanced/vizardLiveComm.html)
