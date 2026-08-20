"""ORBITAL DUET - Representative attitude determination and control.

Builds a visible Basilisk chain from truth-like SimpleNav through velocity
guidance, tracking error, MRP feedback, wheel allocation, reaction wheels, TAM,
and physical magnetic torque bars.  IMU, star tracker, TAM, and CSS outputs are
recorded; SimpleNav is the explicitly declared attitude-estimation fallback.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import (
    attTrackingError,
    mrpFeedback,
    mtbMomentumManagement,
    rwMotorTorque,
    tamComm,
    velocityPoint,
)
from Basilisk.simulation import (
    MtbEffector,
    coarseSunSensor,
    imuSensor,
    magnetometer,
    reactionWheelStateEffector,
    simpleNav,
    starTracker,
)
from Basilisk.utilities import macros, simIncludeRW

from config import ScenarioConfig
from environment import EnvironmentHandles
from spacecraft_model import SpacecraftPair


@dataclass
class AocsSystem:
    """Modules and recorders for one spacecraft AOCS chain."""

    navigation: Any
    guidance: Any
    tracking_error: Any
    controller: Any
    wheel_allocator: Any
    wheels: Any
    imu: Any
    star_tracker: Any
    tam: Any
    css: Any
    mtb: Any
    momentum_manager: Any
    recorders: dict[str, Any]
    persistent_messages: list[Any]


@dataclass
class AocsPair:
    """AOCS systems for the chief and deputy."""

    chief: AocsSystem
    deputy: AocsSystem


def _rw_axes() -> np.ndarray:
    """Return a four-wheel pyramid whose columns are body-frame spin axes."""

    elevation = 52.0 * macros.D2R
    return np.array(
        [
            [0.0, 0.0, np.cos(elevation), -np.cos(elevation)],
            [np.cos(elevation), np.sin(elevation), -np.sin(elevation), -np.sin(elevation)],
            [np.sin(elevation), -np.cos(elevation), 0.0, 0.0],
        ]
    )


def _build_system(
    simulation: Any,
    dynamics_task: str,
    fsw_task: str,
    vehicle: Any,
    vehicle_environment: Any,
    sun_state_msg: Any,
    earth_mu: float,
    label: str,
    config: ScenarioConfig,
) -> AocsSystem:
    """Create, connect, schedule, and record one readable AOCS chain."""

    # ------------------------------------------------------------------
    # Physical actuators: reaction wheels and magnetic torque bars
    # ------------------------------------------------------------------
    rw_factory = simIncludeRW.rwFactory()
    axes = _rw_axes()
    for index in range(4):
        rw_factory.create(
            config.aocs.rw_model,
            axes[:, index],
            Omega=config.aocs.rw_initial_speed_rpm * (1.0 if index % 2 == 0 else -1.0),
            RWModel=messaging.BalancedWheels,
        )
    wheels = reactionWheelStateEffector.ReactionWheelStateEffector()
    wheels.ModelTag = f"{vehicle.ModelTag}-ReactionWheels"
    rw_factory.addToSpacecraft(wheels.ModelTag, wheels, vehicle)
    simulation.AddModelToTask(dynamics_task, wheels, 230)
    rw_config_msg = rw_factory.getConfigMessage()

    mtb = MtbEffector.MtbEffector()
    mtb.ModelTag = f"{vehicle.ModelTag}-MagneticTorqueBars"
    vehicle.addDynamicEffector(mtb)
    simulation.AddModelToTask(dynamics_task, mtb, 225)

    mtb_payload = messaging.MTBArrayConfigMsgPayload()
    mtb_payload.numMTB = 4
    mtb_payload.GtMatrix_B = [
        1.0, 0.0, 0.0, 0.70710678,
        0.0, 1.0, 0.0, 0.70710678,
        0.0, 0.0, 1.0, 0.0,
    ]
    mtb_payload.maxMtbDipoles = [config.aocs.mtq_max_dipole_Am2] * 4
    mtb_config_msg = messaging.MTBArrayConfigMsg().write(mtb_payload)

    # ------------------------------------------------------------------
    # Truth-like/emulated navigation and operational velocity pointing
    # ------------------------------------------------------------------
    navigation = simpleNav.SimpleNav()
    navigation.ModelTag = f"{vehicle.ModelTag}-TruthLikeSimpleNav"
    navigation.scStateInMsg.subscribeTo(vehicle.scStateOutMsg)
    simulation.AddModelToTask(fsw_task, navigation, 120)

    guidance = velocityPoint.velocityPoint()
    guidance.ModelTag = f"{vehicle.ModelTag}-VelocityPoint"
    guidance.mu = earth_mu
    guidance.transNavInMsg.subscribeTo(navigation.transOutMsg)
    simulation.AddModelToTask(fsw_task, guidance, 110)

    tracking = attTrackingError.attTrackingError()
    tracking.ModelTag = f"{vehicle.ModelTag}-TrackingError"
    # velocityPoint's orbit frame plus this verified 90-degree b1 rotation
    # aligns spacecraft +b3 with the velocity direction (scenarioOrbitManeuverTH).
    tracking.sigma_R0R = [np.tan(np.pi / 8.0), 0.0, 0.0]
    tracking.attRefInMsg.subscribeTo(guidance.attRefOutMsg)
    tracking.attNavInMsg.subscribeTo(navigation.attOutMsg)
    simulation.AddModelToTask(fsw_task, tracking, 100)

    controller = mrpFeedback.mrpFeedback()
    controller.ModelTag = f"{vehicle.ModelTag}-MRPFeedback"
    controller.K = config.aocs.controller_K
    controller.P = config.aocs.controller_P
    controller.Ki = config.aocs.controller_Ki
    controller.guidInMsg.subscribeTo(tracking.attGuidOutMsg)
    controller.rwParamsInMsg.subscribeTo(rw_config_msg)
    controller.rwSpeedsInMsg.subscribeTo(wheels.rwSpeedOutMsg)
    inertia = np.diag(config.spacecraft.principal_inertia_kg_m2).reshape(9).tolist()
    vehicle_config_msg = messaging.VehicleConfigMsg().write(
        messaging.VehicleConfigMsgPayload(ISCPntB_B=inertia)
    )
    controller.vehConfigInMsg.subscribeTo(vehicle_config_msg)
    simulation.AddModelToTask(fsw_task, controller, 90)

    allocator = rwMotorTorque.rwMotorTorque()
    allocator.ModelTag = f"{vehicle.ModelTag}-WheelTorqueAllocation"
    allocator.controlAxes_B = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    allocator.rwParamsInMsg.subscribeTo(rw_config_msg)
    allocator.vehControlInMsg.subscribeTo(controller.cmdTorqueOutMsg)
    simulation.AddModelToTask(fsw_task, allocator, 80)

    # ------------------------------------------------------------------
    # Sensors: each consumes truth/environment and publishes a measurement
    # ------------------------------------------------------------------
    imu = imuSensor.ImuSensor()
    imu.ModelTag = f"{vehicle.ModelTag}-{config.aocs.gyro_type.upper()}-IMU"
    gyro_noise = (
        config.aocs.imu_mems_noise_rad_s
        if config.aocs.gyro_type == "mems"
        else config.aocs.imu_fog_noise_rad_s
    )
    gyro_bias = (
        config.aocs.imu_mems_bias_rad_s
        if config.aocs.gyro_type == "mems"
        else config.aocs.imu_fog_bias_rad_s
    )
    imu.PMatrixGyro = np.diag([gyro_noise] * 3).tolist()
    imu.AMatrixGyro = np.diag(
        [-1.0 / config.aocs.imu_error_correlation_time_s] * 3
    ).tolist()
    imu.senRotBias = list(gyro_bias)
    imu.applySensorErrors = True
    imu.scStateInMsg.subscribeTo(vehicle.scStateOutMsg)
    simulation.AddModelToTask(fsw_task, imu, 120)

    tracker = starTracker.StarTracker()
    tracker.ModelTag = f"{vehicle.ModelTag}-StarTracker"
    tracker.PMatrix = np.diag([config.aocs.star_tracker_noise_rad] * 3).tolist()
    tracker.setWalkBounds(np.array([1e-3, 1e-3, 1e-3], dtype=np.float64))
    tracker.scStateInMsg.subscribeTo(vehicle.scStateOutMsg)
    simulation.AddModelToTask(fsw_task, tracker, 120)

    tam = magnetometer.Magnetometer()
    tam.ModelTag = f"{vehicle.ModelTag}-TAM"
    tam.scaleFactor = 1.0
    tam.senNoiseStd = [config.aocs.tam_noise_T] * 3
    tam.senBias = list(config.aocs.tam_bias_T)
    tam.minOutput = -config.aocs.tam_saturation_T
    tam.maxOutput = config.aocs.tam_saturation_T
    tam.dcm_SB = np.asarray(config.aocs.tam_dcm_SB, dtype=float).reshape(3, 3).tolist()
    tam.stateInMsg.subscribeTo(vehicle.scStateOutMsg)
    tam.magInMsg.subscribeTo(vehicle_environment.magnetic_msg)
    simulation.AddModelToTask(fsw_task, tam, 120)

    css = coarseSunSensor.CoarseSunSensor()
    css.ModelTag = f"{vehicle.ModelTag}-CoarseSunSensor"
    css.nHat_B = np.array([1.0, 0.0, 0.0])
    css.senNoiseStd = config.aocs.css_noise
    css.sunInMsg.subscribeTo(sun_state_msg)
    css.stateInMsg.subscribeTo(vehicle.scStateOutMsg)
    if vehicle_environment.eclipse_msg is not None:
        css.sunEclipseInMsg.subscribeTo(vehicle_environment.eclipse_msg)
    simulation.AddModelToTask(fsw_task, css, 120)

    # ------------------------------------------------------------------
    # Magnetic momentum management retains tau_MTQ = m x B physics
    # ------------------------------------------------------------------
    tam_converter = tamComm.tamComm()
    tam_converter.ModelTag = f"{vehicle.ModelTag}-TAM-BodyTransform"
    tam_converter.dcm_BS = list(config.aocs.tam_dcm_SB)
    tam_converter.tamInMsg.subscribeTo(tam.tamDataOutMsg)
    simulation.AddModelToTask(fsw_task, tam_converter, 75)

    momentum_manager = mtbMomentumManagement.mtbMomentumManagement()
    momentum_manager.ModelTag = f"{vehicle.ModelTag}-MomentumManagement"
    momentum_manager.wheelSpeedBiases = [0.0] * 4
    momentum_manager.cGain = config.aocs.momentum_management_gain
    momentum_manager.rwParamsInMsg.subscribeTo(rw_config_msg)
    momentum_manager.mtbParamsInMsg.subscribeTo(mtb_config_msg)
    momentum_manager.tamSensorBodyInMsg.subscribeTo(tam_converter.tamOutMsg)
    momentum_manager.rwSpeedsInMsg.subscribeTo(wheels.rwSpeedOutMsg)
    momentum_manager.rwMotorTorqueInMsg.subscribeTo(allocator.rwMotorTorqueOutMsg)
    simulation.AddModelToTask(fsw_task, momentum_manager, 70)

    wheels.rwMotorCmdInMsg.subscribeTo(momentum_manager.rwMotorTorqueOutMsg)
    mtb.mtbCmdInMsg.subscribeTo(momentum_manager.mtbCmdOutMsg)
    mtb.mtbParamsInMsg.subscribeTo(mtb_config_msg)
    mtb.magInMsg.subscribeTo(vehicle_environment.magnetic_msg)

    sampling_ns = macros.sec2nano(config.mission.record_step_s)
    recorder_sources = {
        "attitude_error": tracking.attGuidOutMsg,
        "wheel_speeds": wheels.rwSpeedOutMsg,
        "wheel_torque_command": momentum_manager.rwMotorTorqueOutMsg,
        "mtb_command": momentum_manager.mtbCmdOutMsg,
        "imu": imu.sensorOutMsg,
        "star_tracker": tracker.sensorOutMsg,
        "tam": tam.tamDataOutMsg,
        "css": css.cssDataOutMsg,
        "nav_translation": navigation.transOutMsg,
        "nav_attitude": navigation.attOutMsg,
    }
    recorders: dict[str, Any] = {}
    for name, message in recorder_sources.items():
        recorder = message.recorder(sampling_ns)
        simulation.AddModelToTask(fsw_task, recorder, 5)
        recorders[name] = recorder

    return AocsSystem(
        navigation=navigation,
        guidance=guidance,
        tracking_error=tracking,
        controller=controller,
        wheel_allocator=allocator,
        wheels=wheels,
        imu=imu,
        star_tracker=tracker,
        tam=tam,
        css=css,
        mtb=mtb,
        momentum_manager=momentum_manager,
        recorders=recorders,
        persistent_messages=[rw_config_msg, mtb_config_msg, vehicle_config_msg],
    )


def build_aocs(
    simulation: Any,
    dynamics_task: str,
    fsw_task: str,
    pair: SpacecraftPair,
    environment: EnvironmentHandles,
    config: ScenarioConfig,
) -> AocsPair:
    """Build equivalent, independently wired AOCS chains for both spacecraft."""

    return AocsPair(
        chief=_build_system(
            simulation,
            dynamics_task,
            fsw_task,
            pair.chief,
            environment.vehicles["chief"],
            environment.sun_state_msg,
            environment.earth.mu,
            "chief",
            config,
        ),
        deputy=_build_system(
            simulation,
            dynamics_task,
            fsw_task,
            pair.deputy,
            environment.vehicles["deputy"],
            environment.sun_state_msg,
            environment.earth.mu,
            "deputy",
            config,
        ),
    )
