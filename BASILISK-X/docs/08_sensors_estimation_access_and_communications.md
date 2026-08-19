> **Draft status — AI-generated:** This page was drafted by OpenAI Codex on 2026-08-19. It has not yet been technically vetted by the BASILISK-X repository owner. The repository owner will review, correct, and maintain it. Verify APIs, units, frames, assumptions, and version compatibility before engineering use.

# Sensors, Estimation, Access, and Communications

This chapter separates physical truth, sensor measurements, navigation estimates, geometric access, and communications. The local examples contain strong building blocks, but many deliberately bypass one or more of these layers.

## 1. Do not collapse the information chain

```text
physical truth and environment
          ↓
sensor geometry/physics
          ↓
sampled, corrupted measurement
          ↓
calibration and measurement processing
          ↓
estimator/filter + covariance
          ↓
navigation solution
          ↓
guidance/control/mission logic
```

| Layer | Meaning | Typical Basilisk artifact |
|---|---|---|
| Truth | Simulated reality under the chosen physical model | `SCStatesMsg`, ephemeris/environment outputs |
| Sensor | Physical transduction and corruption | CSS, magnetometer, IMU, camera modules |
| Measurement | What the estimator is allowed to observe | `CSSArraySensorMsg`, `TAMSensorMsg`, pixels, ranges, etc. |
| Estimate | Inferred state and uncertainty | Filter output, often `NavAttMsg`, `NavTransMsg`, or a specialized state/covariance message |
| FSW use | Decisions based on the estimate and its validity | Guidance, control, mission logic |

A message called “navigation” is not proof that a real measurement/estimator chain exists. Identify its producer.

## 2. `SimpleNav`: useful, but easy to overclaim

`simpleNav.SimpleNav` subscribes to spacecraft truth and publishes standard attitude and translational navigation messages:

```text
Spacecraft.scStateOutMsg
          ↓
SimpleNav
    ├── attOutMsg   (NavAttMsg)
    └── transOutMsg (NavTransMsg)
```

It can impose bounded Gauss-Markov-style navigation errors through configuration such as `PMatrix` and `walkBounds`. With default or zero-error configuration it is essentially a truth-to-navigation adapter.

Use cases:

- isolate guidance/control behavior before estimation is introduced;
- provide a controlled navigation error model for system trades;
- exercise standard `NavAttMsg`/`NavTransMsg` interfaces.

It does not, by itself, establish sensor observability, estimator consistency, calibration, measurement outages, or filter convergence. Label results “truth-like navigation” or “navigation emulation” unless a real filter is in the loop.

[`scenarioGaussMarkovRandomWalk.py`](../examples/scenarioGaussMarkovRandomWalk.py) is useful for understanding bounded correlated-error configuration, although it demonstrates IMU gyro random walks rather than `SimpleNav` specifically. The small-body examples configure `SimpleNav.PMatrix` and `walkBounds` as synthetic measurements for their filters.

## 3. Coarse Sun sensors and sunline estimation

### Sensor model

[`scenarioCSS.py`](../examples/scenarioCSS.py) demonstrates individual `CoarseSunSensor` objects and `CSSConstellation` aggregation.

```text
Sun ephemeris + spacecraft truth + optional eclipse
                       ↓
              CoarseSunSensor(s)
                       ↓
            CSSArraySensorMsg
```

Important CSS parameters include:

- `nHat_B` or platform/azimuth/elevation geometry;
- field of view and scale factor;
- Kelly-factor response distortion;
- normalized bias and Gaussian noise;
- minimum/maximum output behavior;
- optional eclipse input.

The sensor's body-frame normal is a physical mounting definition. A FSW `CSSConfigMsg` separately tells the estimator its assumed normals and calibration biases. Using the same factory data for both is convenient; deliberately dispersing them is how mounting/calibration error is studied.

### Filters

[`scenarioCSSFilters.py`](../examples/scenarioCSSFilters.py) compares `sunlineUKF`, `sunlineEKF`, `okeefeEKF`, `sunlineSEKF`, and `sunlineSuKF`:

```text
CSSArraySensorMsg + CSSConfigMsg
                 ↓
      chosen sunline filter
          ├── navStateOutMsg
          └── filtDataOutMsg
                state, covariance, observations, post-fit residuals
```

The example configures different state dimensions, process/observation noise, covariance, sigma-point parameters, measurement thresholds, and EKF/CKF switching. It is a comparison harness, not evidence that one filter is universally preferred.

For a meaningful filter assessment, examine:

- sunline angular error, not only state-vector components;
- number and geometry of illuminated sensors;
- post-fit residual distributions;
- covariance bounds and consistency;
- convergence after initialization error;
- eclipse/outage behavior and recovery;
- sensitivity to mounting, scale, bias, and noise mismatch.

## 4. Magnetic field and magnetometers

The TAM architecture explicitly separates environment from sensor:

```text
planet position/time/epoch
          ↓
magneticFieldWMM or centered-dipole environment
          ↓ MagneticFieldMsg at spacecraft
Magnetometer + SCStatesMsg
          ↓ TAMSensorMsg expressed in sensor frame
```

[`scenarioTAM.py`](../examples/scenarioTAM.py) configures `Magnetometer.scaleFactor`, per-axis `senNoiseStd`, `senBias`, and saturation bounds. It subscribes both to spacecraft truth and a magnetic-environment output. [`scenarioTAMcomparison.py`](../examples/scenarioTAMcomparison.py) compares WMM and centered-dipole environment choices while independently changing sensor bias/bounds.

This distinction matters: an incorrect field model is environment/model error, while bias, scale, alignment, noise, and saturation are sensor errors. A navigation or magnetic-unloading study may require both.

The examples do not turn the TAM alone into a full attitude determination system. Magnetic measurements provide a direction tied to a time/location-dependent model; attitude observability generally requires motion, another vector sensor, or an estimator model.

## 5. IMU and stochastic errors

[`scenarioGaussMarkovRandomWalk.py`](../examples/scenarioGaussMarkovRandomWalk.py) uses two `ImuSensor` configurations to demonstrate process-noise matrices, dynamics matrices, and random-walk bounds. The key lesson is that sensor error is usually time-correlated:

```text
truth rate/acceleration
        ↓
scale/alignment/bias + stochastic state + white noise + bounds
        ↓
sampled IMU output
```

Engineering checks should include power spectral density/Allan-like behavior where relevant, steady-state variance, correlation time, bound behavior, fixed seed reproducibility, and units. A bounded Gauss-Markov process is a modelling choice; do not use `walkBounds` as a substitute for deriving a sensor error model.

## 6. Small-body navigation

The local small-body estimators are valuable architectural studies with explicit limitations.

| Example | Estimator/problem | Inputs | Important caveat stated by the source |
|---|---|---|---|
| [`scenarioSmallBodyNav.py`](../examples/scenarioSmallBodyNav.py) | `smallBodyNavEKF`: relative position/velocity plus small-body attitude/rate in a proximity-operations loop | Synthetic `SimpleNav` spacecraft measurements, `planetNav`, Sun ephemeris, commanded force | Intended as a representative autonomous-navigation solution for POMDP work; realistic supporting measurement modules are absent and not every uncertainty is estimated |
| [`scenarioSmallBodyNavUKF.py`](../examples/scenarioSmallBodyNavUKF.py) | `smallBodyNavUKF`: relative translation plus non-Keplerian acceleration | Synthetic spacecraft and body ephemeris measurements | Demonstrates acceleration estimation; not a complete flight measurement architecture |
| [`scenarioSmallBodyLandmarks.py`](../examples/scenarioSmallBodyLandmarks.py) | Pinhole-camera landmark measurement generation | Truth geometry, asteroid shape/landmarks, lighting, camera pose | Produces visible landmark pixels; it is a measurement generator, not an end-to-end landmark navigation filter |

The landmark example is especially useful for separation of concerns. `pinholeCamera` evaluates field of view, illumination, projected pixels, and visibility from the spacecraft and rotating-body geometry. Its batch method can process precomputed truth trajectories without rerunning dynamics, which is useful for image-geometry trades. To make it navigation, add data association, pixel uncertainty/outliers, a measurement model, estimator state/covariance, and closed-loop consumption of the estimate.

For OpNav imagery, use the dedicated optical-navigation chapter rather than treating landmark pixels, Vizard rendering, camera degradation, perception, and filtering as one module.

## 7. Access is geometry, not communication

### Ground access

`groundLocation.GroundLocation` represents a body-fixed site and evaluates visibility to subscribed spacecraft:

```text
ground location + body radius/orientation + spacecraft truth
                         ↓
                    AccessMsg
          hasAccess, elevation, range, relative geometry
```

Key configuration includes geodetic/location specification, `minimumElevation`, and `maximumRange`. [`scenarioGroundDownlink.py`](../examples/scenarioGroundDownlink.py) records `hasAccess`, elevation, and slant range. [`scenarioGroundMapping.py`](../examples/scenarioGroundMapping.py) uses many surface points and an instrument mapping module. [`scenarioGroundLocationImaging.py`](../examples/scenarioGroundLocationImaging.py) couples target access with pointing tolerance to command an imager.

Access is usually necessary but not sufficient. It says geometry permits interaction under simple thresholds; it does not establish antenna pointing, link margin, atmosphere/weather, coding, interference, scheduling, or packet delivery.

### Interspacecraft access

`spacecraftLocation.SpacecraftLocation` evaluates geometric access from a primary spacecraft to one or more secondary spacecraft. [`scenarioSpacecraftLocation.py`](../examples/scenarioSpacecraftLocation.py) demonstrates a maximum-range access condition and records an `AccessMsg`.

Directly subscribing one spacecraft's FSW to another spacecraft's truth or navigation message is an ideal centralized information link. `SpacecraftLocation` adds geometry, but it still does not add propagation delay, bandwidth, packetization, clock error, dropout, or a relative sensor estimate.

## 8. Simplified data and downlink models

[`scenarioGroundDownlink.py`](../examples/scenarioGroundDownlink.py) connects:

```text
instrument data generation → storage status
ground-station AccessMsg ──┐
storage status ────────────┴→ SpaceToGroundTransmitter
                              ↓ negative DataNodeUsage rate
                           storage depletion
```

The transmitter uses values such as `nodeBaudRate`, `packetSize`, buffer count, and access messages. This is useful for data-volume and contact-window analysis. Negative data rate represents depletion from onboard storage.

It is not a link-budget or network simulator. The local examples do not establish:

- transmit power, antenna patterns, gain-to-noise temperature, or received \(E_b/N_0\);
- free-space/path/atmospheric losses and modulation/coding;
- packet errors, retransmission, queuing latency, protocol overhead, routing, or contention;
- Doppler/acquisition, ground-network conflicts, or regulatory constraints;
- realistic intersatellite communications.

Add those only if they affect the mission metric. For an early contact-plan trade, precomputed rate-versus-geometry or a margin threshold may be sufficient and much cheaper than waveform-level simulation.

## 9. Navigation performance metrics

Plotting estimate and truth is a start, not a filter validation. Use timestamp-aligned quantities:

| Metric | Purpose |
|---|---|
| Error \(\tilde x=\hat x-x\) in the declared frame | Accuracy and bias |
| RMS/percentile/maximum error | Mission requirement comparison |
| Post-fit residual \(r=y-h(\hat x^-)\) | Measurement/model compatibility and outliers |
| Innovation covariance and normalized innovation | Whether residual size matches predicted uncertainty |
| NEES-like state consistency | Whether truth error matches state covariance in Monte Carlo |
| Covariance trace/eigenvalues/condition | Uncertainty growth, observability, and numerical health |
| Valid-measurement count and outage duration | Geometry/cadence robustness |
| Convergence and reacquisition time | Operational readiness after initialization/outage |
| Downstream pointing/targeting error using the estimate | Mission consequence of navigation performance |

NEES/NIS statistical thresholds require correct degrees of freedom, independent/repeated trials, and careful treatment of constrained attitude errors. They are engineering recommendations; the local examples mainly plot component errors, covariance envelopes, and post-fit residuals rather than performing a complete statistical consistency campaign.

## 10. Building a reusable sensor/navigation adapter

A reusable BASILISK-X adapter is justified when several scenarios need the same external sensor or estimator boundary. A good adapter should:

```text
source payload/external sample
          ↓
validate timestamp, validity, units, frame, covariance
          ↓
perform one explicit conversion
          ↓
publish standard Basilisk measurement or Nav message
```

Design checklist:

1. State whether the input represents truth, raw measurement, calibrated measurement, or estimate.
2. Preserve source timestamp and expose age/validity; do not silently substitute current simulation time.
3. Declare position point/origin, coordinate frame, attitude direction, rate expression frame, and units.
4. Map covariance with the same state ordering and frame transformation as the estimate.
5. Define behavior for missing, stale, out-of-sequence, or invalid data.
6. Avoid reading `Spacecraft` internal state when an explicit truth message is the intended sensor input.
7. Keep measurement noise generation separate from estimator tuning unless the study deliberately assumes perfect knowledge.
8. Unit-test known transformations, timestamp behavior, zero/default payloads, and invalid-message paths.

Do not write a generic adapter merely to rename fields. Reuse is earned when the semantics and tests are stable across multiple consumers.

## 11. Minimum-fidelity selection

| Engineering question | Minimum useful information model |
|---|---|
| Does a guidance law work with perfect knowledge? | Truth-like `SimpleNav` |
| How sensitive is control to bounded nav error? | Configured navigation-error emulator and real sample rates |
| Can a sensor see the target/vector? | Truth geometry, FOV/occlusion/illumination, measurement validity |
| What accuracy does the sensor provide? | Bias/noise/scale/alignment/saturation/cadence model |
| Does the estimator converge consistently? | Measurement model, dynamics, covariance, residuals, dispersed MC truth |
| Can a ground or intersatellite contact occur? | Access geometry and constraints |
| How much data can be returned? | Access windows, rate model, storage, scheduling |
| Will the RF link close? | Link budget and geometry-dependent availability/rate |
| Will packets arrive on time? | Network/protocol/queue/delay/loss model |

The discipline is to stop at the row that supports the metric. An RF model is unnecessary for pure visibility geometry; an `AccessMsg` alone is inadequate for communications reliability.

## 12. Recommended study sequence and caveats

1. [`scenarioCSS.py`](../examples/scenarioCSS.py): physical sensor geometry and corruptions.
2. [`scenarioCSSFilters.py`](../examples/scenarioCSSFilters.py): measurement/configuration messages, filters, covariance, and residuals.
3. [`scenarioTAM.py`](../examples/scenarioTAM.py): environment-versus-sensor separation.
4. [`scenarioGaussMarkovRandomWalk.py`](../examples/scenarioGaussMarkovRandomWalk.py): correlated stochastic error states.
5. [`scenarioGroundDownlink.py`](../examples/scenarioGroundDownlink.py): access versus data transfer.
6. [`scenarioGroundLocationImaging.py`](../examples/scenarioGroundLocationImaging.py): access and attitude gating of payload commands.
7. [`scenarioSmallBodyLandmarks.py`](../examples/scenarioSmallBodyLandmarks.py): geometric measurement generation.
8. Small-body EKF/UKF examples: estimator architecture and explicit limitations.

**Observed source behavior:** many control and formation examples consume `SimpleNav`; access examples use truth geometry; downlink models operate on simplified signed data rates; small-body estimator examples explicitly state that realistic supporting measurements are absent. **Engineering recommendation:** retain these shortcuts when isolating one subsystem, but label them and replace them in the same order as the uncertainty enters the mission requirement.
