"""Sektion 1 — Digital Twin 3D Viewport.

Provides the data model and controller for an interactive 3D rendering
of the sapphire monolith, including volumetric layout visualisation,
layer toggling, and live field overlays from the polariton density solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from ..module_d.mapper_3d import VolumetricLayout, PlacedGate3D, Wire3D
from ..module_e.visualiser import FieldVisualiser, FrameData


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
    """

    gates: list[dict]
    wires: list[dict]
    field_slices: dict[str, np.ndarray]
    thermal_slice: np.ndarray | None
    camera: CameraState
    visibility: LayerVisibility


class DigitalTwinViewport:
    """Controller for the Digital Twin 3D viewport.

    Manages the volumetric layout, camera state, layer visibility, and
    live field overlay integration. Produces ViewportSnapshot objects
    that can be serialised for a web-based renderer.
    """

    def __init__(self, layout: VolumetricLayout | None = None) -> None:
        self._layout = layout
        self.camera = CameraState()
        self.visibility = LayerVisibility()
        self._field_visualiser = FieldVisualiser(title="Digital Twin")
        self._current_density: np.ndarray | None = None
        self._current_temperature: np.ndarray | None = None

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
