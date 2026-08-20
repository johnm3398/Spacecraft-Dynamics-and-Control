"""ORBITAL DUET - Shared LEO environment.

Builds Earth gravity and orientation, Sun/Moon ephemerides, atmosphere and
space weather, atmosphere co-rotation, drag, eclipse/SRP, and WMM magnetic
field.  It owns environmental physics but not vehicles, AOCS, propulsion,
mission sequencing, or analysis.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from Basilisk.architecture import messaging
from Basilisk.simulation import (
    dragDynamicEffector,
    eclipse,
    exponentialAtmosphere,
    magneticFieldWMM,
    msisAtmosphere,
    radiationPressure,
    zeroWindModel,
)
from Basilisk.simulation import spacecraft as spacecraft_module
from Basilisk.utilities import simHelpers, simIncludeGravBody, simSetPlanetEnvironment
from Basilisk.utilities.supportDataTools.dataFetcher import DataFile, get_path

from config import ScenarioConfig
from spacecraft_model import SpacecraftPair


@dataclass
class VehicleEnvironment:
    """Per-spacecraft effectors and environment output messages."""

    drag: Any | None
    srp: Any | None
    density_msg: Any
    magnetic_msg: Any
    wind_msg: Any | None
    eclipse_msg: Any | None


@dataclass
class EnvironmentHandles:
    """Shared environmental modules, data provenance, and per-vehicle links."""

    gravity_factory: Any
    earth: Any
    sun_state_msg: Any
    epoch_msg: Any
    atmosphere: Any
    magnetic_field: Any
    spice: Any | None = None
    wind: Any | None = None
    eclipse: Any | None = None
    gravity_file: Path | None = None
    gravity_file_max_degree: int = 0
    active_gravity_degree: int = 0
    static_messages: list[Any] = field(default_factory=list)
    vehicles: dict[str, VehicleEnvironment] = field(default_factory=dict)


def _gravity_file_degree(path: Path) -> int:
    """Read the coefficient-file header rather than assuming its supported degree."""

    with path.open(encoding="utf-8") as stream:
        header = stream.readline().strip().split(",")
    if len(header) < 5:
        raise ValueError(f"Unrecognized spherical-harmonic header in {path}")
    return int(float(header[3]))


def _write_msis_space_weather(atmosphere: Any, config: ScenarioConfig) -> list[Any]:
    """Publish all 23 scalar inputs required by Basilisk 2.11.1 MSIS."""

    ap = config.environment.ap_daily
    history = config.environment.ap_history
    values = {
        "ap_24_0": ap,
        "ap_3_0": history[0],
        "ap_3_-3": history[1],
        "ap_3_-6": history[2],
        "ap_3_-9": history[3],
        "ap_3_-12": history[4],
        "ap_3_-15": history[5],
        "ap_3_-18": history[6],
        "ap_3_-21": ap,
        "ap_3_-24": ap,
        "ap_3_-27": ap,
        "ap_3_-30": ap,
        "ap_3_-33": ap,
        "ap_3_-36": ap,
        "ap_3_-39": ap,
        "ap_3_-42": ap,
        "ap_3_-45": ap,
        "ap_3_-48": ap,
        "ap_3_-51": ap,
        "ap_3_-54": ap,
        "ap_3_-57": ap,
        "f107_1944_0": config.environment.f107_81day_sfu,
        "f107_24_-24": config.environment.f107_daily_sfu,
    }
    messages: list[Any] = []
    if len(atmosphere.swDataInMsgs) != len(values):
        raise RuntimeError(
            f"Installed MSIS expects {len(atmosphere.swDataInMsgs)} space-weather "
            f"inputs; ORBITAL DUET supplies {len(values)}."
        )
    for index, value in enumerate(values.values()):
        message = messaging.SwDataMsg().write(
            messaging.SwDataMsgPayload(dataValue=float(value))
        )
        # Index the SWIG vector directly; iterating yields temporary reader copies.
        atmosphere.swDataInMsgs[index].subscribeTo(message)
        messages.append(message)
    return messages


def build_environment(simulation: Any, dynamics_task: str, config: ScenarioConfig) -> EnvironmentHandles:
    """Create shared modules before spacecraft-specific messages are attached."""

    gravity_factory = simIncludeGravBody.gravBodyFactory()
    epoch_msg = simHelpers.timeStringToGregorianUTCMsg(config.mission.utc_epoch)
    static_messages: list[Any] = [epoch_msg]

    if config.environment.fidelity == "high":
        bodies = gravity_factory.createBodies(["sun", "earth", "moon"])
        sun = bodies["sun"]
        earth = bodies["earth"]
        earth.isCentralBody = True

        gravity_file = Path(get_path(DataFile.LocalGravData.GGM03S))
        file_degree = _gravity_file_degree(gravity_file)
        if config.environment.gravity_degree > file_degree:
            raise ValueError(
                f"Requested gravity degree {config.environment.gravity_degree} exceeds "
                f"{gravity_file.name}'s degree {file_degree}."
            )
        earth.useSphericalHarmonicsGravityModel(
            str(gravity_file), config.environment.gravity_degree
        )

        # zeroBase keeps Earth at the inertial origin; IAU_EARTH orientation in
        # the planet message makes tesseral/sectoral coefficients rotate physically.
        spice = gravity_factory.createSpiceInterface(time=config.mission.utc_epoch)
        spice.zeroBase = "Earth"
        simulation.AddModelToTask(dynamics_task, spice, 400)
        sun_state_msg = spice.planetStateOutMsgs[0]

        atmosphere = msisAtmosphere.MsisAtmosphere()
        atmosphere.ModelTag = "SharedMSISAtmosphere"
        atmosphere.epochInMsg.subscribeTo(epoch_msg)
        static_messages.extend(_write_msis_space_weather(atmosphere, config))

        wind = zeroWindModel.ZeroWindModel()
        wind.ModelTag = "SharedCorotatingAtmosphere"
        wind.planetPosInMsg.subscribeTo(spice.planetStateOutMsgs[1])

        shadow = eclipse.Eclipse()
        shadow.ModelTag = "SharedEarthEclipse"
        shadow.sunInMsg.subscribeTo(sun_state_msg)
        shadow.addPlanetToModel(spice.planetStateOutMsgs[1])

        gravity_file_path = gravity_file
        active_degree = config.environment.gravity_degree
    else:
        earth = gravity_factory.createEarth()
        earth.isCentralBody = True
        gravity_file_path = None
        file_degree = 0
        active_degree = 0
        spice = None
        wind = None
        shadow = None

        # Fast mode retains a physical density/drag path but avoids SPICE and MSIS.
        atmosphere = exponentialAtmosphere.ExponentialAtmosphere()
        atmosphere.ModelTag = "SharedExponentialAtmosphere"
        simSetPlanetEnvironment.exponentialAtmosphere(atmosphere, "earth")

        # A fixed inertial Sun is adequate only for debugging.  High mode uses SPICE.
        sun_payload = messaging.SpicePlanetStateMsgPayload()
        sun_payload.PositionVector = [149_597_870_700.0, 0.0, 0.0]
        sun_state_msg = messaging.SpicePlanetStateMsg().write(sun_payload)
        static_messages.append(sun_state_msg)

    simulation.AddModelToTask(dynamics_task, atmosphere, 300)

    magnetic = magneticFieldWMM.MagneticFieldWMM()
    magnetic.ModelTag = "SharedWorldMagneticModel"
    magnetic.configureWMMFile(str(get_path(DataFile.MagneticFieldData.WMM)))
    magnetic.epochInMsg.subscribeTo(epoch_msg)
    simulation.AddModelToTask(dynamics_task, magnetic, 280)

    if wind is not None:
        simulation.AddModelToTask(dynamics_task, wind, 290)
    if shadow is not None:
        simulation.AddModelToTask(dynamics_task, shadow, 270)

    return EnvironmentHandles(
        gravity_factory=gravity_factory,
        earth=earth,
        sun_state_msg=sun_state_msg,
        epoch_msg=epoch_msg,
        atmosphere=atmosphere,
        magnetic_field=magnetic,
        spice=spice,
        wind=wind,
        eclipse=shadow,
        gravity_file=gravity_file_path,
        gravity_file_max_degree=file_degree,
        active_gravity_degree=active_degree,
        static_messages=static_messages,
    )


def attach_environment_to_spacecraft(
    simulation: Any,
    dynamics_task: str,
    environment: EnvironmentHandles,
    pair: SpacecraftPair,
    config: ScenarioConfig,
) -> None:
    """Wire shared environment messages into each spacecraft's physical effectors."""

    for label, vehicle in (("chief", pair.chief), ("deputy", pair.deputy)):
        # In fast mode this factory contains only Earth.  High mode normally
        # attaches Earth, Sun, and Moon, but the explicit trade-study switch can
        # retain SPICE/Sun geometry while disabling third-body accelerations.
        if (
            config.environment.fidelity == "high"
            and not config.environment.enable_third_bodies
        ):
            vehicle.gravField.gravBodies = spacecraft_module.GravBodyVector(
                [environment.earth]
            )
        else:
            environment.gravity_factory.addBodiesTo(vehicle)

        environment.atmosphere.addSpacecraftToModel(vehicle.scStateOutMsg)
        density_msg = environment.atmosphere.envOutMsgs[-1]
        environment.magnetic_field.addSpacecraftToModel(vehicle.scStateOutMsg)
        magnetic_msg = environment.magnetic_field.envOutMsgs[-1]

        wind_msg = None
        if environment.wind is not None:
            environment.wind.addSpacecraftToModel(vehicle.scStateOutMsg)
            wind_msg = environment.wind.envOutMsgs[-1]

        eclipse_msg = None
        if environment.eclipse is not None:
            environment.eclipse.addSpacecraftToModel(vehicle.scStateOutMsg)
            eclipse_msg = environment.eclipse.eclipseOutMsgs[-1]

        drag = None
        if config.environment.enable_drag:
            drag = dragDynamicEffector.DragDynamicEffector()
            drag.ModelTag = f"{vehicle.ModelTag}-CannonballDrag"
            drag.coreParams.projectedArea = config.spacecraft.drag_area_m2
            drag.coreParams.dragCoeff = config.spacecraft.drag_coefficient
            drag.atmoDensInMsg.subscribeTo(density_msg)
            if wind_msg is not None:
                # ZeroWindModel supplies Earth co-rotation, not forecast thermospheric winds.
                drag.windVelInMsg.subscribeTo(wind_msg)
            vehicle.addDynamicEffector(drag)
            simulation.AddModelToTask(dynamics_task, drag, 210)

        srp = None
        if config.environment.fidelity == "high" and config.environment.enable_srp:
            srp = radiationPressure.RadiationPressure()
            srp.ModelTag = f"{vehicle.ModelTag}-CannonballSRP"
            srp.area = config.spacecraft.srp_area_m2
            srp.coefficientReflection = config.spacecraft.srp_reflection_coefficient
            srp.sunEphmInMsg.subscribeTo(environment.sun_state_msg)
            if eclipse_msg is not None:
                srp.sunEclipseInMsg.subscribeTo(eclipse_msg)
            vehicle.addDynamicEffector(srp)
            simulation.AddModelToTask(dynamics_task, srp, 205)

        environment.vehicles[label] = VehicleEnvironment(
            drag=drag,
            srp=srp,
            density_msg=density_msg,
            magnetic_msg=magnetic_msg,
            wind_msg=wind_msg,
            eclipse_msg=eclipse_msg,
        )


def print_environment_summary(environment: EnvironmentHandles, config: ScenarioConfig) -> None:
    """Report the models actually enabled, including known fidelity boundaries."""

    print("ENVIRONMENT ACTUALLY CONFIGURED")
    if config.environment.fidelity == "high":
        print(f"  Gravity model       : {config.environment.gravity_model}")
        print(f"  Coefficient file    : {environment.gravity_file}")
        print(f"  File capability     : degree/order {environment.gravity_file_max_degree}")
        print(f"  Active field        : degree/order {environment.active_gravity_degree}")
        print("  Earth orientation   : SPICE IAU_EARTH (Earth-centred inertial zero base)")
        print("  Third-body gravity  : Sun and Moon")
        print("  Atmosphere          : MSIS with 23 configured placeholder SW inputs")
        print("  Atmosphere velocity : Earth co-rotation from ZeroWindModel")
        print("  SRP / shadow        : cannonball SRP with Earth eclipse")
    else:
        print("  Gravity model       : point-mass Earth")
        print("  Atmosphere          : exponential debug model")
        print("  Atmosphere velocity : inertial-velocity fallback (no co-rotation)")
        print("  Sun                 : fixed inertial debug ephemeris")
        print("  SRP / shadow        : disabled")
    print("  Magnetic field      : WMM at configured mission epoch")
    print("  Drag geometry       : constant projected-area DragDynamicEffector")
    print("  Space weather       : demonstration placeholders; replace for analysis\n")
