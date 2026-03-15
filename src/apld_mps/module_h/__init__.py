"""Module H: Command & Control Dashboard.

Immersive, web-ready control surface for the Aethel V1 sapphire monolith.
Integrates all simulation modules into a unified real-time dashboard with
six sections: Digital Twin viewport, Physics control, Kreislauf thermal
monitor, Petabit I/O, Stress test resilience center, and RL optimizer status.
"""

from .dashboard import AethelDashboard, DashboardState, DashboardMode, PhysicsLoopState
from .viewport import (
    DigitalTwinViewport,
    ViewportSnapshot,
    RenderMode,
    CameraState,
    EmissiveWaveguide,
    PhaseJitterAlarm,
    PhaseJitterAlarmLevel,
)
from .physics_panel import PhysicsControlPanel, SubstrateSelection, TMDConfiguration
from .kreislauf_monitor import (
    KreislaufMonitor,
    ThermalHeatmapData,
    TPVStatus,
    ClockGovernorState,
    ThermalAlertLevel,
)
from .io_panel import IOPanel, ThroughputMeterData, BottleneckRadarData, FibreCouplingStatus
from .resilience_center import ResilienceCenter, SelfHealingStatus, ResilienceStatus
from .rl_status import RLOptimizerPanel, TrainingProgress, OptimiserState

__all__ = [
    "AethelDashboard",
    "DashboardState",
    "DashboardMode",
    "PhysicsLoopState",
    "DigitalTwinViewport",
    "ViewportSnapshot",
    "RenderMode",
    "CameraState",
    "EmissiveWaveguide",
    "PhaseJitterAlarm",
    "PhaseJitterAlarmLevel",
    "PhysicsControlPanel",
    "SubstrateSelection",
    "TMDConfiguration",
    "KreislaufMonitor",
    "ThermalHeatmapData",
    "TPVStatus",
    "ClockGovernorState",
    "ThermalAlertLevel",
    "IOPanel",
    "ThroughputMeterData",
    "BottleneckRadarData",
    "FibreCouplingStatus",
    "ResilienceCenter",
    "SelfHealingStatus",
    "ResilienceStatus",
    "RLOptimizerPanel",
    "TrainingProgress",
    "OptimiserState",
]
