"""Sektion 5 — Stress Test & Resilience Center.

Cosmic ray live ticker, self-healing router status map, and vibration
analyser — integrating Module G's stress test suite into a monitoring
dashboard panel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from ..module_d.mapper_3d import PlacedGate3D, VolumetricLayout
from ..module_g.cosmic_ray import CosmicRaySimulator, CosmicRayEvent, SelfHealingRouter
from ..module_g.phase_stability import (
    PhaseStabilityAnalyser,
    VibrationProfile,
    StabilityReport,
)


class ResilienceStatus(Enum):
    """Overall system resilience health."""
    HEALTHY = auto()
    DEGRADED = auto()
    CRITICAL = auto()


@dataclass
class CosmicRayTickerEntry:
    """Single entry in the cosmic ray live ticker.

    Attributes:
        timestamp_ps: Simulation time of impact [ps].
        position_nm: (x, y, z) impact location.
        energy_MeV: Particle energy [MeV].
        affected_radius_nm: Disruption zone radius [nm].
        coherence_loss: Fractional coherence loss [0–1].
    """

    timestamp_ps: float
    position_nm: tuple[float, float, float]
    energy_MeV: float
    affected_radius_nm: float
    coherence_loss: float


@dataclass
class SelfHealingStatus:
    """Self-healing router status display.

    Attributes:
        disrupted_fraction: Fraction of volume below 50% coherence.
        active_reroutes: Number of paths currently rerouted.
        availability_slice: 2D availability map at mid-plane.
        total_events: Total cosmic ray events processed.
    """

    disrupted_fraction: float
    active_reroutes: int
    availability_slice: np.ndarray
    total_events: int


@dataclass
class VibrationAnalyserData:
    """Vibration analyser display data.

    Attributes:
        rms_phase_jitter_rad: Current RMS phase jitter [rad].
        bit_error_probability: Current BER from vibration.
        coherence_length_nm: Effective coherence length [nm].
        max_tolerable_amplitude_nm: Max safe vibration amplitude [nm].
        isolation_required: Whether active isolation is needed.
        stability_report: Full analysis report.
    """

    rms_phase_jitter_rad: float
    bit_error_probability: float
    coherence_length_nm: float
    max_tolerable_amplitude_nm: float
    isolation_required: bool
    stability_report: StabilityReport | None


@dataclass
class CosmicImpactLog:
    """Log entry for a cosmic ray impact near a gate.

    Attributes:
        timestamp_ps: Simulation time of impact [ps].
        gate_id: Identifier of the nearest gate (e.g. ``"XOR_1"``).
        distance_nm: Distance from the impact to the gate centre [nm].
        energy_MeV: Particle energy [MeV].
        message: Human-readable log line for the dashboard.
    """

    timestamp_ps: float
    gate_id: str
    distance_nm: float
    energy_MeV: float
    message: str


class ResilienceCenter:
    """Stress test and resilience monitoring center.

    Orchestrates cosmic ray simulation, self-healing routing, and
    vibration phase-stability analysis into a unified dashboard panel.
    """

    def __init__(
        self,
        volume_nm: tuple[float, float, float] = (
            300_000_000.0, 300_000_000.0, 10_000_000.0
        ),
        router_resolution_nm: float = 1_000_000.0,
        waveguide_length_nm: float = 10_000.0,
        refractive_index: float = 1.76,
        wavelength_nm: float = 720.0,
        ber_warning_threshold: float = 1e-6,
        disruption_warning_threshold: float = 0.01,
        stress_mode: bool = True,
    ) -> None:
        self.volume_nm = volume_nm
        self._cosmic_sim = CosmicRaySimulator(volume_nm, stress_mode=stress_mode)
        self._router = SelfHealingRouter(volume_nm, router_resolution_nm)
        self._phase_analyser = PhaseStabilityAnalyser(
            waveguide_length_nm, refractive_index, wavelength_nm
        )
        self._ber_warn = ber_warning_threshold
        self._disrupt_warn = disruption_warning_threshold

        self._ticker: list[CosmicRayTickerEntry] = []
        self._impact_log: list[CosmicImpactLog] = []
        self._active_reroutes: int = 0
        self._latest_vib_report: StabilityReport | None = None
        self._layout: VolumetricLayout | None = None
        self._step_seed: int = 0

    # --- Layout / intensity slider ---

    def set_layout(self, layout: VolumetricLayout) -> None:
        """Register the gate layout for nearest-gate resolution in logs."""
        self._layout = layout

    def set_intensity_multiplier(self, value: float) -> None:
        """Set the cosmic-ray flux intensity multiplier (dashboard slider).

        A value of 1.0 corresponds to natural sea-level muon flux.
        Higher values artificially increase the rate for stress testing.
        """
        self._cosmic_sim.intensity_multiplier = value

    @property
    def intensity_multiplier(self) -> float:
        return self._cosmic_sim.intensity_multiplier

    def set_stress_mode(self, enabled: bool) -> None:
        """Enable/disable cosmic-ray stress mode (100× flux)."""
        self._cosmic_sim.stress_mode = enabled

    # --- Per-step cosmic ray generation ---

    def step_cosmic_rays(
        self, dt_ps: float = 1.0,
    ) -> list[CosmicImpactLog]:
        """Generate cosmic ray events for a single simulation time step.

        Called every ``simulation_step`` in the dashboard.  Returns new
        impact-log entries with nearest-gate resolution.
        """
        duration_s = dt_ps * 1e-12
        self._step_seed += 1
        events = self._cosmic_sim.generate_events(
            duration_s, seed=self._step_seed,
        )

        entries: list[CosmicRayTickerEntry] = []
        logs: list[CosmicImpactLog] = []
        for ev in events:
            entry = CosmicRayTickerEntry(
                timestamp_ps=ev.timestamp_ps,
                position_nm=ev.position_nm,
                energy_MeV=ev.energy_MeV,
                affected_radius_nm=ev.affected_radius_nm,
                coherence_loss=ev.coherence_loss,
            )
            entries.append(entry)
            self._ticker.append(entry)
            self._router.mark_disruption(ev)

            # Resolve nearest gate
            gate_id, dist = self._nearest_gate(ev.position_nm)
            msg = f"Einschlag detektiert bei Gatter {gate_id}"
            log = CosmicImpactLog(
                timestamp_ps=ev.timestamp_ps,
                gate_id=gate_id,
                distance_nm=dist,
                energy_MeV=ev.energy_MeV,
                message=msg,
            )
            logs.append(log)
            self._impact_log.append(log)

        self._ticker.sort(key=lambda e: e.timestamp_ps)
        return logs

    def get_impact_log(self, last_n: int = 50) -> list[CosmicImpactLog]:
        """Return the most recent impact log entries."""
        return self._impact_log[-last_n:]

    def _nearest_gate(
        self, position_nm: tuple[float, float, float],
    ) -> tuple[str, float]:
        """Find the gate closest to *position_nm*.

        Returns ``(gate_id, distance_nm)``.  If no layout is loaded,
        returns ``("UNKNOWN", inf)``.
        """
        if self._layout is None or not self._layout.placed_gates:
            return ("UNKNOWN", float("inf"))

        best_id = "UNKNOWN"
        best_dist = float("inf")
        pos = np.array(position_nm)
        for pg in self._layout.placed_gates:
            centre = np.array(pg.origin_nm) + np.array(pg.footprint_nm) / 2.0
            dist = float(np.linalg.norm(pos - centre))
            if dist < best_dist:
                best_dist = dist
                best_id = pg.gate_id
        return best_id, best_dist

    # --- Cosmic ray ticker ---

    def simulate_cosmic_rays(
        self, duration_s: float = 1.0, seed: int = 42
    ) -> list[CosmicRayTickerEntry]:
        """Run cosmic ray simulation and produce ticker entries."""
        events = self._cosmic_sim.generate_events(duration_s, seed)

        entries = []
        for ev in events:
            entry = CosmicRayTickerEntry(
                timestamp_ps=ev.timestamp_ps,
                position_nm=ev.position_nm,
                energy_MeV=ev.energy_MeV,
                affected_radius_nm=ev.affected_radius_nm,
                coherence_loss=ev.coherence_loss,
            )
            entries.append(entry)
            self._ticker.append(entry)

            # Feed into self-healing router
            self._router.mark_disruption(ev)

        # Sort ticker by timestamp
        self._ticker.sort(key=lambda e: e.timestamp_ps)
        return entries

    def get_ticker(self, last_n: int = 20) -> list[CosmicRayTickerEntry]:
        """Get the most recent ticker entries."""
        return self._ticker[-last_n:]

    @property
    def total_events(self) -> int:
        return len(self._ticker)

    def error_rate(self, total_operations: int) -> float:
        """Estimated bit-error rate from cosmic ray damage."""
        return self._cosmic_sim.error_rate(total_operations)

    # --- Self-healing router ---

    def attempt_reroute(
        self,
        start_nm: tuple[float, float, float],
        end_nm: tuple[float, float, float],
    ) -> bool:
        """Attempt to find a path avoiding disrupted zones.

        Returns True if a valid path exists, False otherwise.
        Increments the active reroute counter on success.
        """
        path = self._router.find_path(start_nm, end_nm)
        if path is not None:
            self._active_reroutes += 1
            return True
        return False

    def get_self_healing_status(self) -> SelfHealingStatus:
        """Produce self-healing router status display."""
        avail = self._router.availability
        mid_z = avail.shape[2] // 2

        return SelfHealingStatus(
            disrupted_fraction=self._router.disrupted_fraction,
            active_reroutes=self._active_reroutes,
            availability_slice=avail[:, :, mid_z],
            total_events=self.total_events,
        )

    # --- Vibration analyser ---

    def analyse_vibration(self, profile: VibrationProfile) -> VibrationAnalyserData:
        """Run phase stability analysis for a vibration environment."""
        report = self._phase_analyser.analyse(profile)
        self._latest_vib_report = report

        return VibrationAnalyserData(
            rms_phase_jitter_rad=report.rms_phase_jitter_rad,
            bit_error_probability=report.bit_error_probability,
            coherence_length_nm=report.coherence_length_nm,
            max_tolerable_amplitude_nm=report.max_tolerable_amplitude_nm,
            isolation_required=report.isolation_required,
            stability_report=report,
        )

    def get_vibration_data(self) -> VibrationAnalyserData:
        """Get the latest vibration analysis data."""
        if self._latest_vib_report is None:
            return VibrationAnalyserData(
                rms_phase_jitter_rad=0.0,
                bit_error_probability=0.0,
                coherence_length_nm=self._phase_analyser.L,
                max_tolerable_amplitude_nm=self._phase_analyser.max_tolerable_amplitude(),
                isolation_required=False,
                stability_report=None,
            )
        r = self._latest_vib_report
        return VibrationAnalyserData(
            rms_phase_jitter_rad=r.rms_phase_jitter_rad,
            bit_error_probability=r.bit_error_probability,
            coherence_length_nm=r.coherence_length_nm,
            max_tolerable_amplitude_nm=r.max_tolerable_amplitude_nm,
            isolation_required=r.isolation_required,
            stability_report=r,
        )

    # --- Overall status ---

    def overall_status(self) -> ResilienceStatus:
        """Compute overall resilience status from all subsystems."""
        disrupted = self._router.disrupted_fraction
        ber = 0.0
        if self._latest_vib_report:
            ber = self._latest_vib_report.bit_error_probability

        if disrupted > 0.05 or ber > 1e-3:
            return ResilienceStatus.CRITICAL
        if disrupted > self._disrupt_warn or ber > self._ber_warn:
            return ResilienceStatus.DEGRADED
        return ResilienceStatus.HEALTHY
