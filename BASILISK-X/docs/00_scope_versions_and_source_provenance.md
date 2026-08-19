> **Draft status — AI-generated:** This page was drafted by OpenAI Codex on 2026-08-19. It has not yet been technically vetted by the BASILISK-X repository owner. The repository owner will review, correct, and maintain it. Verify APIs, units, frames, assumptions, and version compatibility before engineering use.

# Scope, Versions, and Source Provenance

## Purpose

This chapter records the evidence base and compatibility limits for the field manual. It exists because spacecraft simulation results are not reproducible from equations and scenario code alone: the exact Basilisk build, optional modules, external applications, kernels, model weights, assets, and copied-example revision can all affect whether a scenario runs and what it means.

This manual currently studies:

- the copied examples under [`../examples/`](../examples/);
- BASILISK-X scenarios under [`../scenarios/`](../scenarios/);
- reusable code under [`../src/basiliskx/`](../src/basiliskx/);
- tests under [`../tests/`](../tests/);
- the Basilisk package installed in `basiliskx_env`;
- current official AVS Lab documentation when local code does not establish a claim.

It does not claim that the copied examples are flight software, qualified models, or a complete catalogue of Basilisk capabilities.

## Current local baseline

| Item | Current repository evidence |
|---|---|
| BASILISK-X package version | `0.1.0` in [`../pyproject.toml`](../pyproject.toml) |
| Declared Python version | `>=3.11` |
| Basilisk dependency | `bsk[all,examples]==2.11.1` in [`../requirements.txt`](../requirements.txt) |
| Installed Basilisk version observed during the audit | `2.11.1` |
| Copied example provenance | Exact upstream tag/commit not recorded |
| Local BSK-RL | Not present |
| Local reusable subsystem | Vizard process/lifecycle utilities under `src/basiliskx/visualization` |

The installed package is the execution authority for this workspace. A copied example should be considered **unverified** until it imports and executes against that package with its required optional assets.

## Known source/package drift

The local example tree contains APIs not supplied by the installed 2.11.1 package.

### Build-feature API

The following copied sources import `Basilisk.hasBuildFeature`:

- [`../examples/OpNavScenarios/modelsOpNav/BSK_OpNavFsw.py`](../examples/OpNavScenarios/modelsOpNav/BSK_OpNavFsw.py)
- [`../examples/scenarioSpiceReconstruction.py`](../examples/scenarioSpiceReconstruction.py)
- [`../examples/scenarioVestaOrientation.py`](../examples/scenarioVestaOrientation.py)

That symbol is absent from the installed Basilisk 2.11.1 package. It is documented in the Basilisk 2.12.0 development build documentation. These examples therefore cannot be treated as version-matched 2.11.1 references without adaptation or a matching runtime.

### Optical-navigation assets and external software

The copied OpNav architecture additionally depends on:

- a compatible OpNav-enabled Basilisk installation;
- a Vizard executable capable of returning rendered camera images;
- a configured TCP endpoint;
- SPICE data;
- an ONNX model for the CNN branch.

The expected `CAD.onnx` model is not present in this repository, and the copied master contains machine-specific Vizard assumptions. The OpNav files remain valuable architecture studies, but are not presently a turnkey local subsystem.

### MuJoCo drift

The basic MuJoCo reaction-wheel example executed during the audit, but not all copied examples match the installed API. For example, one stochastic-drag source requests an unavailable model and the Earth–Moon example uses a property absent from the installed class. MuJoCo support is also described upstream as beta/work in progress, so each scenario requires version-specific verification.

## Evidence labels for this manual

Use the following labels when documenting a non-obvious claim:

| Label | Meaning |
|---|---|
| **Observed locally** | Read from or exercised against this repository and its installed environment |
| **Upstream documented** | Stated in official AVS Lab documentation for a named version |
| **Inferred** | Reasoned from wiring or behavior but not directly confirmed by authoritative text or a test |
| **Recommended** | Engineering guidance for BASILISK-X, not a Basilisk API guarantee |
| **Unverified** | Plausible or copied behavior still requiring source/test confirmation |

## Scenario provenance record

For a serious study or retained result, record at least:

```text
BASILISK-X commit:
Basilisk package version:
Basilisk source/build revision, if available:
Python version:
Operating system and architecture:
Optional Basilisk packages/features:
Scenario path and configuration:
Random seeds:
SPICE kernels and epochs:
External model/data files and checksums:
Vizard version and mode:
Simulation task rates and integrator settings:
Known warnings or compatibility patches:
```

For Monte Carlo work, retain this information with every ensemble, along with dispersed parameters, per-run seeds, failures, and the precise success-metric definition.

## Compatibility policy for the manual

Until the owner has vetted a page:

1. Treat code fragments as patterns, not paste-ready guarantees.
2. Check the installed class and message interfaces before use.
3. Run import and initialization smoke tests before a long simulation.
4. Verify task execution order and recorder timestamps.
5. Compare at least one result with an analytic solution, conservation law, trusted reference, or independent implementation.
6. Never resolve a version mismatch by silently changing the engineering model.

When a page is vetted, it should record:

- reviewer and date;
- Basilisk version;
- scenarios or tests used for verification;
- remaining assumptions and unresolved issues.

## Relationship to authoritative documentation

The primary upstream references are:

- [Basilisk 2.11.1 documentation](https://avslab.github.io/basilisk/)
- [Basilisk process and task concepts](https://avslab.github.io/basilisk/Learn/bskPrinciples/bskPrinciples-1.html)
- [Basilisk message connections](https://avslab.github.io/basilisk/Learn/bskPrinciples/bskPrinciples-3.html)
- [Basilisk development build information](https://avslab.github.io/basilisk/develop/Build/installBuild.html)
- [BSK-RL documentation](https://avslab.github.io/bsk_rl/)

If this manual and version-matched upstream source disagree, update the manual.

## Next reading

- [Quick Start](QUICK_START.md)
- [Architecture, execution, and lifecycle](01_architecture_execution_and_lifecycle.md)
- [Messages, time ordering, frames, and units](02_messages_time_ordering_frames_and_units.md)
