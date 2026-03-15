"""Sektion 1 — Digital Twin 3D Viewport.

Provides the data model and controller for an interactive 3D rendering
of the sapphire monolith, including volumetric layout visualisation,
layer toggling, live field overlays from the polariton density solver,
emissive waveguide glow for active laser pulses, and a Phase Jitter
alarm that couples vibration amplitude to live BER via PhaseStability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from ..module_c.laser_source import LaserPulse
from ..module_d.mapper_3d import VolumetricLayout, PlacedGate3D, Wire3D
from ..module_e.visualiser import FieldVisualiser, FrameData
from ..module_g.phase_stability import (
    PhaseStabilityAnalyser,
    VibrationProfile,
)


class RenderMode(Enum):
    """Rendering mode for the 3D viewport."""
    SOLID = auto()
    WIREFRAME = auto()
    XRAY = auto()
    HEATMAP = auto()


@dataclass
class CameraState:
    """Camera position and orientation for the 3D viewport.

    Attributes:
        position: (x, y, z) eye position [nm].
        target: (x, y, z) look-at point [nm].
        up: (ux, uy, uz) up vector.
        fov_deg: Vertical field of view [degrees].
        zoom: Zoom multiplier (1.0 = default).
    """

    position: tuple[float, float, float] = (0.0, 0.0, 1e9)
    target: tuple[float, float, float] = (2e8, 2e8, 2e8)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov_deg: float = 45.0
    zoom: float = 1.0


@dataclass
class LayerVisibility:
    """Toggle visibility for individual z-layers and overlay types."""

    visible_layers: set[int] = field(default_factory=set)
    show_gates: bool = True
    show_wires: bool = True
    show_field_overlay: bool = True
    show_thermal_overlay: bool = False
    render_mode: RenderMode = RenderMode.SOLID
    field_opacity: float = 0.6


@dataclass
class EmissiveWaveguide:
    """A waveguide that glows in the 3D viewport because a LaserPulse is
    injecting light into it.

    Attributes:
        gate_id: Identifier of the gate whose waveguide is illuminated.
        position_nm: (x, y) injection point on the chip [nm].
        wavelength_nm: Emission wavelength [nm] — determines glow colour.
        intensity: Normalised emissive intensity [0–1].
        emissive_color_rgb: (R, G, B) float triplet in [0, 1] derived from
            the pulse wavelength (warm red-orange for ~720 nm).
    """

    gate_id: str
    position_nm: tuple[float, float]
    wavelength_nm: float
    intensity: float
    emissive_color_rgb: tuple[float, float, float]


class PhaseJitterAlarmLevel(Enum):
    """Severity level for the Phase Jitter alarm."""
    NONE = auto()
    WARNING = auto()
    CRITICAL = auto()


@dataclass
class PhaseJitterAlarm:
    """Live Phase Jitter alarm driven by vibration amplitude.

    Couples the vibration-induced phase jitter from
    :class:`PhaseStabilityAnalyser` to a dashboard alarm that reports
    the instantaneous BER (Bit Error Rate).

    Attributes:
        active: Whether the alarm is currently raised.
        level: Severity level (NONE / WARNING / CRITICAL).
        rms_phase_jitter_rad: Current RMS phase jitter [rad].
        bit_error_rate: Live BER computed from the jitter.
        vibration_amplitude_nm: RMS vibration amplitude that caused
            this alarm [nm].
        message: Human-readable alarm text for the dashboard.
    """

    active: bool
    level: PhaseJitterAlarmLevel
    rms_phase_jitter_rad: float
    bit_error_rate: float
    vibration_amplitude_nm: float
    message: str


@dataclass
class ViewportSnapshot:
    """A serialisable snapshot of the 3D viewport state.

    Suitable for sending to a React/Three.js frontend via JSON.

    Attributes:
        gates: List of gate render descriptors.
        wires: List of wire path descriptors.
        field_slices: Dict mapping axis label to 2D field arrays.
        thermal_slice: Optional 2D temperature slice for heatmap overlay.
        camera: Current camera state.
        visibility: Current layer visibility settings.
        emissive_waveguides: Waveguides currently glowing from laser input.
        phase_jitter_alarm: Current Phase Jitter alarm state.
    """

    gates: list[dict]
    wires: list[dict]
    field_slices: dict[str, np.ndarray]
    thermal_slice: np.ndarray | None
    camera: CameraState
    visibility: LayerVisibility
    emissive_waveguides: list[EmissiveWaveguide]
    phase_jitter_alarm: PhaseJitterAlarm


class DigitalTwinViewport:
    """Controller for the Digital Twin 3D viewport.

    Manages the volumetric layout, camera state, layer visibility,
    live field overlay integration, emissive waveguide glow for active
    laser pulses, and a Phase Jitter alarm driven by vibration amplitude.
    Produces ViewportSnapshot objects that can be serialised for a
    web-based renderer.
    """

    # Phase Jitter alarm thresholds (BER)
    JITTER_WARNING_BER: float = 1e-9
    JITTER_CRITICAL_BER: float = 1e-6

    def __init__(
        self,
        layout: VolumetricLayout | None = None,
        waveguide_length_nm: float = 10_000.0,
        refractive_index: float = 1.76,
        wavelength_nm: float = 720.0,
    ) -> None:
        self._layout = layout
        self.camera = CameraState()
        self.visibility = LayerVisibility()
        self._field_visualiser = FieldVisualiser(title="Digital Twin")
        self._current_density: np.ndarray | None = None
        self._current_temperature: np.ndarray | None = None

        # Emissive waveguide state
        self._emissive_waveguides: list[EmissiveWaveguide] = []

        # Phase Jitter alarm state
        self._phase_analyser = PhaseStabilityAnalyser(
            waveguide_length_nm, refractive_index, wavelength_nm,
        )
        self._current_alarm = PhaseJitterAlarm(
            active=False,
            level=PhaseJitterAlarmLevel.NONE,
            rms_phase_jitter_rad=0.0,
            bit_error_rate=0.0,
            vibration_amplitude_nm=0.0,
            message="",
        )

        if layout:
            self._init_layers(layout)

    def _init_layers(self, layout: VolumetricLayout) -> None:
        """Discover all z-layers and make them visible by default."""
        z_indices: set[int] = set()
        for pg in layout.placed_gates:
            z_idx = int(pg.origin_nm[2] / max(layout.volume_nm[2] / 100, 1))
            z_indices.add(z_idx)
        self.visibility.visible_layers = z_indices

    def set_layout(self, layout: VolumetricLayout) -> None:
        self._layout = layout
        self._init_layers(layout)

    @property
    def layout(self) -> VolumetricLayout | None:
        return self._layout

    def update_field_overlay(self, density_3d: np.ndarray) -> None:
        """Update the live polariton density |ψ(r,t)|² overlay."""
        self._current_density = density_3d

    def update_thermal_overlay(self, temperature_3d: np.ndarray) -> None:
        """Update the thermal heatmap overlay from ThermoOpticSolver."""
        self._current_temperature = temperature_3d

    def capture_frame(self, time_ps: float) -> None:
        """Capture the current density as an animation frame."""
        if self._current_density is not None:
            mid_z = self._current_density.shape[2] // 2
            slice_2d = self._current_density[:, :, mid_z]
            self._field_visualiser.capture_frame(slice_2d, time_ps, label="|ψ|²")

    @property
    def frame_count(self) -> int:
        return self._field_visualiser.n_frames

    def toggle_layer(self, layer_idx: int) -> None:
        """Toggle visibility of a specific z-layer."""
        if layer_idx in self.visibility.visible_layers:
            self.visibility.visible_layers.discard(layer_idx)
        else:
            self.visibility.visible_layers.add(layer_idx)

    def set_render_mode(self, mode: RenderMode) -> None:
        self.visibility.render_mode = mode

    # --- Emissive waveguide glow ---

    @staticmethod
    def _wavelength_to_rgb(wavelength_nm: float) -> tuple[float, float, float]:
        """Convert a visible wavelength to an approximate (R, G, B) triplet.

        Uses a simple piecewise linear approximation for the visible range
        380–780 nm.  Values outside the range are clamped.
        """
        wl = max(380.0, min(780.0, wavelength_nm))
        r = g = b = 0.0
        if wl < 440:
            r = -(wl - 440.0) / (440.0 - 380.0)
            b = 1.0
        elif wl < 490:
            g = (wl - 440.0) / (490.0 - 440.0)
            b = 1.0
        elif wl < 510:
            g = 1.0
            b = -(wl - 510.0) / (510.0 - 490.0)
        elif wl < 580:
            r = (wl - 510.0) / (580.0 - 510.0)
            g = 1.0
        elif wl < 645:
            r = 1.0
            g = -(wl - 645.0) / (645.0 - 580.0)
        else:
            r = 1.0
        return (round(r, 4), round(g, 4), round(b, 4))

    def configure_laser_pulses(self, pulses: list[LaserPulse]) -> list[EmissiveWaveguide]:
        """Configure which waveguides glow based on active laser pulses.

        For each pulse, the nearest gate (resolved by injection position)
        is marked with an emissive material whose colour is derived from
        the pulse wavelength and whose intensity is proportional to the
        pulse intensity (normalised to 1e4 W/cm²).

        Returns the list of :class:`EmissiveWaveguide` descriptors.
        """
        self._emissive_waveguides.clear()
        if not pulses:
            return []

        for pulse in pulses:
            gate_id = self._resolve_nearest_gate(pulse.position_nm)
            wl = pulse.wavelength_nm
            intensity = min(1.0, pulse.intensity_W_cm2 / 1e4)
            rgb = self._wavelength_to_rgb(wl)

            ew = EmissiveWaveguide(
                gate_id=gate_id,
                position_nm=pulse.position_nm,
                wavelength_nm=wl,
                intensity=intensity,
                emissive_color_rgb=rgb,
            )
            self._emissive_waveguides.append(ew)

        return list(self._emissive_waveguides)

    def clear_emissive(self) -> None:
        """Remove all emissive waveguide highlights."""
        self._emissive_waveguides.clear()

    @property
    def emissive_waveguides(self) -> list[EmissiveWaveguide]:
        return list(self._emissive_waveguides)

    def _resolve_nearest_gate(
        self, position_nm: tuple[float, float],
    ) -> str:
        """Find the gate nearest to a 2D injection point."""
        if self._layout is None or not self._layout.placed_gates:
            return "UNRESOLVED"
        pos = np.array([position_nm[0], position_nm[1]])
        best_id = "UNRESOLVED"
        best_dist = float("inf")
        for pg in self._layout.placed_gates:
            centre_2d = np.array([pg.centre_nm[0], pg.centre_nm[1]])
            dist = float(np.linalg.norm(pos - centre_2d))
            if dist < best_dist:
                best_dist = dist
                best_id = pg.gate_id
        return best_id

    # --- Phase Jitter alarm ---

    def update_vibration(self, profile: VibrationProfile) -> PhaseJitterAlarm:
        """Recompute the Phase Jitter alarm from a vibration profile.

        Couples the vibration amplitude directly to the PhaseStability
        calculation.  An increase in vibration amplitude drives the BER
        upward and triggers the alarm when thresholds are exceeded.

        Returns the updated :class:`PhaseJitterAlarm`.
        """
        rms_amp = float(np.sqrt(np.sum(profile.amplitudes_nm ** 2)))
        jitter = self._phase_analyser.compute_phase_jitter(profile)
        ber = self._phase_analyser.bit_error_probability(jitter)

        if ber >= self.JITTER_CRITICAL_BER:
            level = PhaseJitterAlarmLevel.CRITICAL
            msg = (
                f"CRITICAL Phase Jitter — BER {ber:.2e} "
                f"(σ_φ = {jitter:.4f} rad, A_rms = {rms_amp:.2f} nm)"
            )
        elif ber >= self.JITTER_WARNING_BER:
            level = PhaseJitterAlarmLevel.WARNING
            msg = (
                f"WARNING Phase Jitter — BER {ber:.2e} "
                f"(σ_φ = {jitter:.4f} rad, A_rms = {rms_amp:.2f} nm)"
            )
        else:
            level = PhaseJitterAlarmLevel.NONE
            msg = ""

        self._current_alarm = PhaseJitterAlarm(
            active=level is not PhaseJitterAlarmLevel.NONE,
            level=level,
            rms_phase_jitter_rad=jitter,
            bit_error_rate=ber,
            vibration_amplitude_nm=rms_amp,
            message=msg,
        )
        return self._current_alarm

    def clear_phase_jitter_alarm(self) -> None:
        """Reset the Phase Jitter alarm to inactive."""
        self._current_alarm = PhaseJitterAlarm(
            active=False,
            level=PhaseJitterAlarmLevel.NONE,
            rms_phase_jitter_rad=0.0,
            bit_error_rate=0.0,
            vibration_amplitude_nm=0.0,
            message="",
        )

    @property
    def phase_jitter_alarm(self) -> PhaseJitterAlarm:
        return self._current_alarm

    def _gate_to_render_dict(self, pg: PlacedGate3D) -> dict:
        """Convert a PlacedGate3D to a frontend-friendly dict."""
        return {
            "gate_id": pg.gate_id,
            "gate_type": pg.gate.gate_type.name,
            "origin_nm": list(pg.origin_nm),
            "centre_nm": list(pg.centre_nm),
            "footprint_nm": list(pg.footprint_nm),
        }

    def _wire_to_render_dict(self, wire: Wire3D) -> dict:
        """Convert a Wire3D to a frontend-friendly dict."""
        return {
            "from_gate_id": wire.from_gate_id,
            "to_gate_id": wire.to_gate_id,
            "waypoints_nm": [list(wp) for wp in wire.waypoints_nm],
            "length_nm": wire.length_nm,
            "min_bend_radius_nm": wire.min_bend_radius_nm,
        }

    def get_snapshot(self) -> ViewportSnapshot:
        """Produce a complete snapshot of the current viewport state."""
        gates = []
        wires = []

        if self._layout:
            if self.visibility.show_gates:
                gates = [self._gate_to_render_dict(pg) for pg in self._layout.placed_gates]
            if self.visibility.show_wires:
                wires = [self._wire_to_render_dict(w) for w in self._layout.wires]

        # Field slices at midpoints
        field_slices: dict[str, np.ndarray] = {}
        if self._current_density is not None and self.visibility.show_field_overlay:
            d = self._current_density
            field_slices["xy"] = d[:, :, d.shape[2] // 2]
            field_slices["xz"] = d[:, d.shape[1] // 2, :]
            field_slices["yz"] = d[d.shape[0] // 2, :, :]

        thermal_slice = None
        if self._current_temperature is not None and self.visibility.show_thermal_overlay:
            t = self._current_temperature
            thermal_slice = t[:, :, t.shape[2] // 2]

        return ViewportSnapshot(
            gates=gates,
            wires=wires,
            field_slices=field_slices,
            thermal_slice=thermal_slice,
            camera=self.camera,
            visibility=self.visibility,
            emissive_waveguides=list(self._emissive_waveguides),
            phase_jitter_alarm=self._current_alarm,
        )

    def gate_count(self) -> int:
        if self._layout is None:
            return 0
        return self._layout.gate_count

    def wire_count(self) -> int:
        if self._layout is None:
            return 0
        return len(self._layout.wires)

    def layer_count(self) -> int:
        if self._layout is None:
            return 0
        return self._layout.layer_count
