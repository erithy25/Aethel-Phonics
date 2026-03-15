"""Aethel Phonics Command & Control Dashboard — Unified Orchestrator.

Combines all six dashboard sections into a single controller that
manages the complete state of the Aethel V1 simulation and provides
serialisable snapshots for the frontend (React + Three.js).

Sections:
    1. Digital Twin 3D Viewport (centre)
    2. Physics & Material Control (left)
    3. Aethel-Kreislauf Monitor (right top)
    4. Petabit I/O & Fibre Dashboard (right bottom)
    5. Stress Test & Resilience Center (bottom)
    6. RL Optimizer Status (popup/sidebar)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from .viewport import DigitalTwinViewport, ViewportSnapshot, RenderMode
from .physics_panel import PhysicsControlPanel, SubstrateSelection, TMDConfiguration
from .kreislauf_monitor import (
    KreislaufMonitor,
    ThermalHeatmapData,
    TPVStatus,
    ClockGovernorState,
    ThermalAlertLevel,
)
from .io_panel import IOPanel, ThroughputMeterData, BottleneckRadarData, FibreCouplingStatus
from .resilience_center import (
    ResilienceCenter,
    SelfHealingStatus,
    VibrationAnalyserData,
    CosmicRayTickerEntry,
    ResilienceStatus,
)
from .rl_status import RLOptimizerPanel, TrainingProgress, OptimiserState

from ..constants import EV_TO_J, FS_TO_S, NM_TO_M
from ..module_c.laser_source import LaserSource, LaserPulse
from ..module_d.mapper_3d import VolumetricLayout
from ..module_f.heat_optics import ThermoOpticConfig
from ..module_f.tpv_recycler import TPVConfig
from ..module_g.phase_stability import VibrationProfile


class DashboardMode(Enum):
    """Dashboard operating mode."""
    DESIGN = auto()      # Layout and material configuration
    SIMULATION = auto()  # Running physics simulation
    ANALYSIS = auto()    # Post-simulation analysis
    STRESS_TEST = auto() # Running cosmic/vibration stress tests


@dataclass
class DashboardState:
    """Complete serialisable state of all dashboard sections.

    This is the top-level object sent to the frontend on each
    refresh cycle.

    Attributes:
        mode: Current operating mode.
        viewport: 3D viewport snapshot.
        substrate: Current substrate info.
        tmd: Current TMD configuration.
        thermal: Thermal heatmap data.
        tpv: TPV recycling status.
        clock: Clock governor state.
        throughput: I/O throughput meters.
        bottleneck: Bottleneck radar data.
        coupling: Fibre coupling status.
        cosmic_ticker: Recent cosmic ray events.
        self_healing: Self-healing router status.
        vibration: Vibration analysis data.
        rl_progress: RL optimiser progress.
        resilience: Overall resilience status.
        thermal_alert: Current thermal alert level.
    """

    mode: DashboardMode
    viewport: ViewportSnapshot
    substrate: SubstrateSelection | None
    tmd: TMDConfiguration
    thermal: ThermalHeatmapData | None
    tpv: TPVStatus
    clock: ClockGovernorState
    throughput: ThroughputMeterData
    bottleneck: BottleneckRadarData
    coupling: FibreCouplingStatus
    cosmic_ticker: list[CosmicRayTickerEntry]
    self_healing: SelfHealingStatus
    vibration: VibrationAnalyserData
    rl_progress: TrainingProgress
    resilience: ResilienceStatus
    thermal_alert: ThermalAlertLevel


class AethelDashboard:
    """Unified Command & Control Dashboard for the Aethel V1.

    Orchestrates all six dashboard sections and provides a single
    `get_state()` method that returns the complete dashboard state
    for frontend rendering.

    Usage:
        dashboard = AethelDashboard()
        dashboard.initialise()

        # Configure physics
        dashboard.physics.select_substrate("Sapphire (Al₂O₃)")
        dashboard.physics.select_tmd("WSe2")

        # Load a layout
        dashboard.load_layout(layout)

        # Get full state snapshot
        state = dashboard.get_state()
    """

    def __init__(
        self,
        thermo_config: ThermoOpticConfig | None = None,
        tpv_config: TPVConfig | None = None,
        volume_nm: tuple[float, float, float] = (
            400_000_000.0, 400_000_000.0, 400_000_000.0
        ),
        memory_read_rate_Gbps: float = 100_000.0,
        network_bandwidth_Gbps: float = 10_000.0,
    ) -> None:
        self._mode = DashboardMode.DESIGN

        # Section 1: Digital Twin Viewport
        self.viewport = DigitalTwinViewport()

        # Section 2: Physics & Material Control
        self.physics = PhysicsControlPanel()

        # Section 3: Kreislauf Monitor
        self.kreislauf = KreislaufMonitor(thermo_config, tpv_config)

        # Section 4: I/O Panel
        self.io = IOPanel(memory_read_rate_Gbps, network_bandwidth_Gbps)

        # Section 5: Resilience Center
        self.resilience = ResilienceCenter(volume_nm)

        # Section 6: RL Optimizer
        self.rl = RLOptimizerPanel(volume_nm=volume_nm)

        self._initialised = False

    def initialise(self) -> None:
        """Initialise all subsystems that require setup."""
        self.kreislauf.initialise()
        self._initialised = True

    @property
    def is_initialised(self) -> bool:
        return self._initialised

    @property
    def mode(self) -> DashboardMode:
        return self._mode

    def set_mode(self, mode: DashboardMode) -> None:
        self._mode = mode

    # --- Layout management ---

    def load_layout(self, layout: VolumetricLayout) -> None:
        """Load a volumetric layout into the viewport."""
        self.viewport.set_layout(layout)

    # --- Simulation step ---

    def simulation_step(
        self,
        density_3d: np.ndarray | None = None,
        time_ps: float = 0.0,
        n_operations: int = 0,
        laser_source: LaserSource | None = None,
        f_clk_GHz: float = 0.0,
    ) -> None:
        """Advance one simulation cycle across all subsystems.

        This is the main tick function called by the simulation loop:
        1. Update field overlay in viewport.
        2. Feed heat into Kreislauf and advance thermal.
        3. Compute laser input power: ``Σ E_pulse × f_clk``.
        4. Pass total power as ``source_q`` into the thermal solver so
           that the breakdown algorithm can trigger at ~237 GHz.
        5. Record I/O snapshot & capture animation frame.

        Parameters:
            density_3d: 3D polariton density field for viewport overlay.
            time_ps: Current simulation time [ps].
            n_operations: Number of gate operations (Landauer heat).
            laser_source: :class:`LaserSource` containing active pulses
                whose energy is summed and converted to input power.
            f_clk_GHz: Clock rate [GHz].  Multiplied with total pulse
                energy to obtain the input power ``P = Σ E_pulse × f_clk``.
        """
        self._mode = DashboardMode.SIMULATION

        if density_3d is not None:
            self.viewport.update_field_overlay(density_3d)

            # Compute intensity from density for thermal feedback
            intensity = np.abs(density_3d) ** 2 if np.iscomplexobj(density_3d) else density_3d
            self.kreislauf.inject_heat(intensity_field=intensity, n_operations=n_operations)

        # ---------------------------------------------------------------
        # Module C → Module F coupling: laser energy → thermal power
        # ---------------------------------------------------------------
        # Sum the energy of every active LaserPulse:
        #   E_pulse [J] = intensity [W/cm²] × 1e4 [→ W/m²]
        #                 × π × waist² [m²] × duration [s]
        # Total input power:
        #   P_total [W] = Σ E_pulse × f_clk [Hz]
        # ---------------------------------------------------------------
        source_q_W = 0.0
        if laser_source is not None and f_clk_GHz > 0.0:
            f_clk_Hz = f_clk_GHz * 1e9
            total_pulse_energy_J = 0.0
            for pulse in laser_source.pulses:
                intensity_W_m2 = pulse.intensity_W_cm2 * 1e4
                beam_area_m2 = np.pi * (pulse.waist_nm * NM_TO_M) ** 2
                duration_s = pulse.duration_fs * FS_TO_S
                total_pulse_energy_J += intensity_W_m2 * beam_area_m2 * duration_s
            source_q_W = total_pulse_energy_J * f_clk_Hz

        self.kreislauf.advance_thermal(source_q_W=source_q_W)

        # Feed thermal back to viewport
        if self._initialised:
            self.viewport.update_thermal_overlay(self.kreislauf.temperature_3d)

        # Capture frame
        self.viewport.capture_frame(time_ps)

        # Record I/O snapshot
        self.io.record_snapshot()

    # --- Stress test ---

    def run_stress_test(
        self,
        cosmic_duration_s: float = 1.0,
        vibration_profile: VibrationProfile | None = None,
        seed: int = 42,
    ) -> None:
        """Run a combined cosmic ray + vibration stress test."""
        self._mode = DashboardMode.STRESS_TEST
        self.resilience.simulate_cosmic_rays(cosmic_duration_s, seed)
        if vibration_profile is not None:
            self.resilience.analyse_vibration(vibration_profile)

    # --- Full state snapshot ---

    def get_state(self) -> DashboardState:
        """Produce the complete dashboard state for frontend rendering.

        This is the primary interface between the Python backend and
        the web-based UI (React + Three.js).
        """
        # Thermal data (only if initialised)
        thermal = None
        thermal_alert = ThermalAlertLevel.NOMINAL
        if self._initialised:
            thermal = self.kreislauf.get_thermal_heatmap()
            thermal_alert = thermal.alert_level

        return DashboardState(
            mode=self._mode,
            viewport=self.viewport.get_snapshot(),
            substrate=self.physics.current_substrate,
            tmd=self.physics.current_tmd,
            thermal=thermal,
            tpv=self.kreislauf.get_tpv_status(),
            clock=self.kreislauf.get_clock_governor_state(),
            throughput=self.io.get_throughput_data(),
            bottleneck=self.io.get_bottleneck_radar(),
            coupling=self.io.get_coupling_status(),
            cosmic_ticker=self.resilience.get_ticker(),
            self_healing=self.resilience.get_self_healing_status(),
            vibration=self.resilience.get_vibration_data(),
            rl_progress=self.rl.get_progress(),
            resilience=self.resilience.overall_status(),
            thermal_alert=thermal_alert,
        )
