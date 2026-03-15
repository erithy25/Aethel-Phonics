"""Aethel Phonics Command & Control Dashboard — Unified Orchestrator.

Combines all six dashboard sections into a single controller that
manages the complete state of the Aethel V1 simulation and provides
serialisable snapshots for the frontend (React + Three.js).

The closed-loop physics pipeline executed on every simulation tick:

    LaserPulse energy [C] → clock-rate scaling → thermal injection [F]
    → refractive-index correction (Δn) → signal degradation / phase
    jitter [G] → AI-inference noise [J]

Sections:
    1. Digital Twin 3D Viewport (centre)
    2. Physics & Material Control (left)
    3. Aethel-Kreislauf Monitor (right top)
    4. Petabit I/O & Fibre Dashboard (right bottom)
    5. Stress Test & Resilience Center (bottom)
    6. RL Optimizer Status (popup/sidebar)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from .viewport import (
    DigitalTwinViewport,
    ViewportSnapshot,
    RenderMode,
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
from .resilience_center import (
    ResilienceCenter,
    SelfHealingStatus,
    VibrationAnalyserData,
    CosmicRayTickerEntry,
    CosmicImpactLog,
    ResilienceStatus,
)
from .rl_status import RLOptimizerPanel, TrainingProgress, OptimiserState

from ..constants import EV_TO_J, FS_TO_S, NM_TO_M
from ..module_c.laser_source import LaserSource, LaserPulse
from ..module_d.mapper_3d import VolumetricLayout
from ..module_f.heat_optics import ThermoOpticConfig
from ..module_f.tpv_recycler import TPVConfig
from ..module_g.phase_stability import VibrationProfile, PhaseStabilityAnalyser
from ..module_j.engine import AFEE, AFEEConfig, InferenceMode


class DashboardMode(Enum):
    """Dashboard operating mode."""
    DESIGN = auto()      # Layout and material configuration
    SIMULATION = auto()  # Running physics simulation
    ANALYSIS = auto()    # Post-simulation analysis
    STRESS_TEST = auto() # Running cosmic/vibration stress tests


@dataclass
class PhysicsLoopState:
    """Telemetry snapshot of the closed-loop physics pipeline.

    Captures the output of every stage so that each link in the
    chain can be inspected and verified.

    Attributes:
        laser_power_W: Total optical pump power from laser pulses [W].
        f_clk_GHz: Clock rate used for power scaling [GHz].
        peak_temperature_K: Peak chip temperature after thermal step [K].
        delta_n_max: Maximum thermo-optic Δn in the lattice.
        phase_error_rad: Phase error derived from Δn [rad].
        signal_degradation: MZI contrast loss [0–1], 1.0 = no loss.
        cosmic_coherence: Mean sector coherence from cosmic ray map [0–1].
        afee_noise_sigma: Effective noise σ injected into AFEE ops.
        afee_fidelity: Overall AFEE output fidelity [0–1].
        pipeline_complete: Whether all six stages fired this tick.
    """

    laser_power_W: float = 0.0
    f_clk_GHz: float = 0.0
    peak_temperature_K: float = 300.0
    delta_n_max: float = 0.0
    phase_error_rad: float = 0.0
    signal_degradation: float = 1.0
    cosmic_coherence: float = 1.0
    afee_noise_sigma: float = 0.0
    afee_fidelity: float = 1.0
    pipeline_complete: bool = False


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
        cosmic_impact_log: Recent impact log entries with gate resolution.
        self_healing: Self-healing router status.
        vibration: Vibration analysis data.
        rl_progress: RL optimiser progress.
        resilience: Overall resilience status.
        thermal_alert: Current thermal alert level.
        physics_loop: Telemetry of the closed-loop physics pipeline.
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
    cosmic_impact_log: list[CosmicImpactLog]
    self_healing: SelfHealingStatus
    vibration: VibrationAnalyserData
    rl_progress: TrainingProgress
    resilience: ResilienceStatus
    thermal_alert: ThermalAlertLevel
    physics_loop: PhysicsLoopState


class AethelDashboard:
    """Unified Command & Control Dashboard for the Aethel V1.

    Orchestrates all dashboard sections **and** the closed-loop physics
    pipeline that connects every module into a single causal chain:

    .. code-block:: text

        ┌─────────────────────────────────────────────────────────────┐
        │  LaserPulse [C]                                            │
        │       ↓  P = Σ E_pulse × f_clk                            │
        │  Thermal Injection [F]  (Kreislauf)                        │
        │       ↓  ΔT → Δn = (dn/dT) · ΔT                          │
        │  Refractive-Index Correction                               │
        │       ↓  Δφ = (2π/λ) · Δn · L → Phase Jitter alarm        │
        │  Signal Degradation [G]  (PhaseStability + CosmicRays)     │
        │       ↓  coherence map + phase error                       │
        │  AI-Inference Noise [J]  (AFEE tensor core + drift layer)  │
        └─────────────────────────────────────────────────────────────┘

    Usage:
        dashboard = AethelDashboard()
        dashboard.initialise()
        dashboard.physics.select_substrate("Sapphire (Al₂O₃)")
        dashboard.physics.select_tmd("WSe2")
        dashboard.load_layout(layout)
        state = dashboard.get_state()
    """

    # Sapphire baseline refractive index at 720 nm
    _SAPPHIRE_N: float = 1.76
    _WAVELENGTH_NM: float = 720.0
    _WAVEGUIDE_LENGTH_NM: float = 10_000.0

    def __init__(
        self,
        thermo_config: ThermoOpticConfig | None = None,
        tpv_config: TPVConfig | None = None,
        volume_nm: tuple[float, float, float] = (
            400_000_000.0, 400_000_000.0, 400_000_000.0
        ),
        memory_read_rate_Gbps: float = 100_000.0,
        network_bandwidth_Gbps: float = 10_000.0,
        afee_config: AFEEConfig | None = None,
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

        # Module J: AFEE inference engine (physics-coupled)
        self._afee_cfg = afee_config or AFEEConfig(
            n_layers=2, d_model=64, d_ff=256, n_heads=4, vocab_size=256,
            mode=InferenceMode.PHYSICS_FULL, seed=42,
        )
        self.afee = AFEE(self._afee_cfg)

        # Phase stability analyser for Δn → phase-jitter conversion
        self._phase_analyser = PhaseStabilityAnalyser(
            waveguide_length_nm=self._WAVEGUIDE_LENGTH_NM,
            refractive_index=self._SAPPHIRE_N,
            wavelength_nm=self._WAVELENGTH_NM,
        )

        # Pipeline telemetry
        self._loop_state = PhysicsLoopState()

        self._initialised = False

    def initialise(self) -> None:
        """Initialise all subsystems that require setup."""
        self.kreislauf.initialise()
        if not self.afee.is_loaded:
            self.afee.load_weights()
        self._initialised = True

    @property
    def is_initialised(self) -> bool:
        return self._initialised

    @property
    def mode(self) -> DashboardMode:
        return self._mode

    def set_mode(self, mode: DashboardMode) -> None:
        self._mode = mode

    @property
    def physics_loop(self) -> PhysicsLoopState:
        """Most recent closed-loop physics pipeline telemetry."""
        return self._loop_state

    # --- Layout management ---

    def load_layout(self, layout: VolumetricLayout) -> None:
        """Load a volumetric layout into the viewport and resilience center."""
        self.viewport.set_layout(layout)
        self.resilience.set_layout(layout)

    def set_cosmic_intensity(self, multiplier: float) -> None:
        """Set the cosmic-ray intensity slider (1.0 = natural flux)."""
        self.resilience.set_intensity_multiplier(multiplier)

    # --- Simulation step ---

    def simulation_step(
        self,
        density_3d: np.ndarray | None = None,
        time_ps: float = 0.0,
        dt_ps: float = 1.0,
        n_operations: int = 0,
        laser_source: LaserSource | None = None,
        f_clk_GHz: float = 0.0,
    ) -> list[CosmicImpactLog]:
        """Advance one simulation cycle through the full physics pipeline.

        Closed-loop pipeline executed each tick:

        1. **Laser → Power** [Module C]:  ``P = Σ E_pulse × f_clk``
        2. **Power → Heat** [Module F]:  Inject *P* into thermal solver.
        3. **Heat → Δn** [Module F]:  Thermo-optic refractive-index shift.
        4. **Δn → Phase Jitter** [Module G]:  Convert Δn to Δφ via
           ``PhaseStabilityAnalyser``; fire viewport Phase Jitter alarm
           when BER thresholds are exceeded.
        5. **Phase Jitter + Cosmic → Signal Degradation** [Module G]:
           Cosmic ray events degrade the coherence map; combined with
           the thermal phase error this gives the total signal loss.
        6. **Signal Degradation → AFEE Noise** [Module J]:  Feed the
           coherence map into ``PolaritonicTensorCore`` and inject the
           thermal phase error into ``ThermalDriftLayer``, so the next
           AI inference call sees physically correct noise.

        Parameters:
            density_3d: 3D polariton density field for viewport overlay.
            time_ps: Current simulation time [ps].
            dt_ps: Duration of this simulation step [ps].
            n_operations: Number of gate operations (Landauer heat).
            laser_source: :class:`LaserSource` containing active pulses
                whose energy is summed and converted to input power.
            f_clk_GHz: Clock rate [GHz].

        Returns:
            List of :class:`CosmicImpactLog` entries generated this step.
        """
        self._mode = DashboardMode.SIMULATION

        # ===================================================================
        # Stage 0: Density field → viewport + Landauer heat
        # ===================================================================
        if density_3d is not None:
            self.viewport.update_field_overlay(density_3d)
            intensity = (
                np.abs(density_3d) ** 2
                if np.iscomplexobj(density_3d)
                else density_3d
            )
            self.kreislauf.inject_heat(
                intensity_field=intensity, n_operations=n_operations,
            )

        # ===================================================================
        # Stage 1: LaserPulse energy [C] → clock-rate scaling → power
        # ===================================================================
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

            # Also light up the waveguides in the viewport
            self.viewport.configure_laser_pulses(laser_source.pulses)
        else:
            self.viewport.clear_emissive()

        # ===================================================================
        # Stage 2: Thermal injection [F]
        # ===================================================================
        self.kreislauf.advance_thermal(source_q_W=source_q_W)

        # Read thermal state
        peak_T = 300.0
        delta_n_max = 0.0
        if self._initialised:
            peak_T = float(self.kreislauf.temperature_3d.max())
            # Δn from thermo-optic coefficient: dn/dT ≈ 1.3e-5 K⁻¹ (sapphire)
            delta_T = peak_T - 300.0
            dn_dT = 1.3e-5
            delta_n_max = dn_dT * delta_T

        # ===================================================================
        # Stage 3: Refractive-index correction → phase error
        # ===================================================================
        # Δφ = (2π / λ) · Δn · L
        sensitivity = self._phase_analyser.phase_sensitivity()  # rad/nm
        phase_error_rad = sensitivity * (delta_n_max / self._SAPPHIRE_N) * self._WAVEGUIDE_LENGTH_NM

        # Signal degradation (MZI contrast): cos²(Δφ/2)
        signal_degradation = float(np.cos(phase_error_rad / 2.0) ** 2)

        # ===================================================================
        # Stage 4: Signal degradation [G] — cosmic rays + phase jitter
        # ===================================================================
        impact_logs = self.resilience.step_cosmic_rays(dt_ps=dt_ps)

        # Build a synthetic VibrationProfile from the thermal phase error
        # so the viewport Phase Jitter alarm fires when Δn is significant.
        # We express the thermal displacement as an equivalent vibration
        # amplitude: A_eq = Δφ / sensitivity
        equivalent_amplitude_nm = phase_error_rad / sensitivity if sensitivity > 0 else 0.0
        if equivalent_amplitude_nm > 0:
            thermal_vib = VibrationProfile(
                frequencies_Hz=np.array([1.0]),  # DC thermal drift
                amplitudes_nm=np.array([equivalent_amplitude_nm]),
            )
            self.viewport.update_vibration(thermal_vib)
        else:
            self.viewport.clear_phase_jitter_alarm()

        # Cosmic ray coherence map → scalar coherence
        sh_status = self.resilience.get_self_healing_status()
        cosmic_coherence = 1.0 - sh_status.disrupted_fraction

        # ===================================================================
        # Stage 5: AI-Inference noise [J] — feed physics into AFEE
        # ===================================================================
        # (a) Inject cosmic coherence into the tensor core
        self.afee.set_coherence(cosmic_coherence)

        # (b) Synchronise AFEE thermal state from the Kreislauf solver.
        #     We inject the same source_q_W into the AFEE drift layer's
        #     internal solver so it tracks the dashboard temperature.
        if self._initialised and source_q_W > 0.0:
            drift_solver = self.afee.drift.solver
            vol_m3 = np.prod(np.array(drift_solver.cfg.grid_size_nm)) * NM_TO_M ** 3
            if vol_m3 > 0:
                drift_solver._heat_source += source_q_W / vol_m3
                drift_solver.advance(1)

        # Compute effective noise sigma for telemetry
        afee_noise_sigma = 0.0
        if cosmic_coherence < self.afee.core.coherence_threshold:
            afee_noise_sigma = self.afee.core.noise_scale * (1.0 - cosmic_coherence)

        afee_fidelity = self.afee._compute_fidelity().overall_fidelity

        # ===================================================================
        # Pipeline telemetry
        # ===================================================================
        self._loop_state = PhysicsLoopState(
            laser_power_W=source_q_W,
            f_clk_GHz=f_clk_GHz,
            peak_temperature_K=peak_T,
            delta_n_max=delta_n_max,
            phase_error_rad=phase_error_rad,
            signal_degradation=signal_degradation,
            cosmic_coherence=cosmic_coherence,
            afee_noise_sigma=afee_noise_sigma,
            afee_fidelity=afee_fidelity,
            pipeline_complete=True,
        )

        # ===================================================================
        # Viewport overlays & frame capture
        # ===================================================================
        if self._initialised:
            self.viewport.update_thermal_overlay(self.kreislauf.temperature_3d)

        self.viewport.capture_frame(time_ps)
        self.io.record_snapshot()

        return impact_logs

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
            cosmic_impact_log=self.resilience.get_impact_log(),
            self_healing=self.resilience.get_self_healing_status(),
            vibration=self.resilience.get_vibration_data(),
            rl_progress=self.rl.get_progress(),
            resilience=self.resilience.overall_status(),
            thermal_alert=thermal_alert,
            physics_loop=self._loop_state,
        )
