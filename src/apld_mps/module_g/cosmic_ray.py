"""Detail 5.1 — Cosmic ray impact simulation and self-healing routing.

Models high-energy particle strikes in the sapphire crystal, computes
the resulting local disruption to optical coherence, estimates bit-error
rates, and provides self-healing algorithms that reroute data paths
around damaged regions in real time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..constants import EV_TO_J, NM_TO_M


@dataclass
class CosmicRayEvent:
    """A single cosmic-ray particle impact.

    Attributes:
        position_nm: (x, y, z) impact point in the crystal [nm].
        energy_MeV: Kinetic energy of the incident particle [MeV].
        direction: Unit vector (dx, dy, dz) of the particle trajectory.
        affected_radius_nm: Radius of the coherence-disrupted zone [nm].
        coherence_loss: Fractional loss of optical coherence in the zone [0–1].
        timestamp_ps: Simulation time of the strike [ps].
    """

    position_nm: tuple[float, float, float]
    energy_MeV: float
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    affected_radius_nm: float = 0.0
    coherence_loss: float = 0.0
    timestamp_ps: float = 0.0


class CosmicRaySimulator:
    """Generate and model cosmic-ray events in the sapphire volume.

    Models the energy deposition via the Bethe-Bloch formula (simplified),
    computing the disruption zone size and coherence loss.
    """

    def __init__(
        self,
        volume_nm: tuple[float, float, float] = (
            400_000_000.0, 400_000_000.0, 400_000_000.0
        ),
        flux_per_cm2_s: float = 1.0,  # ~1 muon/cm²/s at sea level
    ) -> None:
        self.volume_nm = volume_nm
        self.flux = flux_per_cm2_s
        self._events: list[CosmicRayEvent] = []

    def generate_events(
        self,
        duration_s: float = 1.0,
        seed: int = 42,
    ) -> list[CosmicRayEvent]:
        """Generate Poisson-distributed cosmic ray events.

        Returns the list of events that occur within *duration_s* seconds.
        """
        rng = np.random.default_rng(seed)

        # Surface area in cm²
        area_cm2 = (
            self.volume_nm[0] * self.volume_nm[1] * (NM_TO_M * 100) ** 2
        )
        expected = self.flux * area_cm2 * duration_s
        n_events = rng.poisson(expected)

        events = []
        for _ in range(n_events):
            pos = tuple(
                rng.uniform(0, s) for s in self.volume_nm
            )
            energy = rng.exponential(scale=200.0)  # MeV, rough muon spectrum

            # Affected radius: ~ sqrt(energy) × 100 nm (simplified scaling)
            radius = 100.0 * np.sqrt(energy)

            # Coherence loss: higher energy → more disruption, capped at 1
            loss = min(1.0, energy / 1000.0)

            # Random direction (isotropic)
            theta = rng.uniform(0, np.pi)
            phi = rng.uniform(0, 2 * np.pi)
            direction = (
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            )

            event = CosmicRayEvent(
                position_nm=pos,  # type: ignore[arg-type]
                energy_MeV=energy,
                direction=direction,
                affected_radius_nm=radius,
                coherence_loss=loss,
                timestamp_ps=rng.uniform(0, duration_s * 1e12),
            )
            events.append(event)

        self._events.extend(events)
        return events

    @property
    def events(self) -> list[CosmicRayEvent]:
        return list(self._events)

    def error_rate(self, total_operations: int) -> float:
        """Estimate bit-error rate from accumulated events.

        Simplified: each event disrupts a spherical volume; operations
        passing through that volume are corrupted.
        """
        if total_operations == 0:
            return 0.0
        total_volume = np.prod(np.array(self.volume_nm))
        disrupted_volume = 0.0
        for ev in self._events:
            disrupted_volume += (4 / 3) * np.pi * ev.affected_radius_nm**3 * ev.coherence_loss
        # Fraction of volume disrupted
        frac = min(disrupted_volume / total_volume, 1.0)
        return frac

    def apply_to_field(
        self, field: np.ndarray, grid_spacing_nm: float, events: list[CosmicRayEvent] | None = None
    ) -> np.ndarray:
        """Apply coherence disruption from events to a 3D complex field array.

        Multiplies the field by (1 - loss) within the affected radius of each event.
        """
        if events is None:
            events = self._events
        result = field.copy()
        shape = field.shape

        for ev in events:
            # Compute grid indices of affected region
            cx = int(ev.position_nm[0] / grid_spacing_nm)
            cy = int(ev.position_nm[1] / grid_spacing_nm)
            cz = int(ev.position_nm[2] / grid_spacing_nm) if len(shape) > 2 else 0
            r_cells = int(np.ceil(ev.affected_radius_nm / grid_spacing_nm))

            if len(shape) == 3:
                for ix in range(max(0, cx - r_cells), min(shape[0], cx + r_cells + 1)):
                    for iy in range(max(0, cy - r_cells), min(shape[1], cy + r_cells + 1)):
                        for iz in range(max(0, cz - r_cells), min(shape[2], cz + r_cells + 1)):
                            dx = (ix - cx) * grid_spacing_nm
                            dy = (iy - cy) * grid_spacing_nm
                            dz = (iz - cz) * grid_spacing_nm
                            dist = np.sqrt(dx**2 + dy**2 + dz**2)
                            if dist <= ev.affected_radius_nm:
                                result[ix, iy, iz] *= (1.0 - ev.coherence_loss)
            elif len(shape) == 2:
                for ix in range(max(0, cx - r_cells), min(shape[0], cx + r_cells + 1)):
                    for iy in range(max(0, cy - r_cells), min(shape[1], cy + r_cells + 1)):
                        dx = (ix - cx) * grid_spacing_nm
                        dy = (iy - cy) * grid_spacing_nm
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist <= ev.affected_radius_nm:
                            result[ix, iy] *= (1.0 - ev.coherence_loss)
        return result


class SelfHealingRouter:
    """Real-time data-path rerouting around disrupted regions.

    When a cosmic ray strike disables a waveguide segment, this router
    finds alternative paths through the 3D crystal lattice.
    """

    def __init__(
        self,
        volume_nm: tuple[float, float, float],
        grid_resolution_nm: float = 1000.0,
    ) -> None:
        self.volume_nm = volume_nm
        self.resolution = grid_resolution_nm
        self.grid_shape = tuple(
            int(np.ceil(s / grid_resolution_nm)) for s in volume_nm
        )
        # Availability mask: 1.0 = fully functional, 0.0 = disrupted
        self._availability = np.ones(self.grid_shape, dtype=np.float64)

    def mark_disruption(self, event: CosmicRayEvent) -> None:
        """Mark a region as disrupted following a cosmic ray event."""
        cx = int(event.position_nm[0] / self.resolution)
        cy = int(event.position_nm[1] / self.resolution)
        cz = int(event.position_nm[2] / self.resolution)
        r_cells = int(np.ceil(event.affected_radius_nm / self.resolution))

        nx, ny, nz = self.grid_shape
        for ix in range(max(0, cx - r_cells), min(nx, cx + r_cells + 1)):
            for iy in range(max(0, cy - r_cells), min(ny, cy + r_cells + 1)):
                for iz in range(max(0, cz - r_cells), min(nz, cz + r_cells + 1)):
                    dx = (ix - cx) * self.resolution
                    dy = (iy - cy) * self.resolution
                    dz = (iz - cz) * self.resolution
                    if np.sqrt(dx**2 + dy**2 + dz**2) <= event.affected_radius_nm:
                        self._availability[ix, iy, iz] *= (1.0 - event.coherence_loss)

    def find_path(
        self,
        start_nm: tuple[float, float, float],
        end_nm: tuple[float, float, float],
        coherence_threshold: float = 0.5,
    ) -> list[tuple[int, int, int]] | None:
        """Find a path avoiding disrupted zones using greedy A*-like search.

        Returns list of grid-cell indices from start to end, or None if no
        path exists above the coherence threshold.
        """
        def to_grid(pos: tuple[float, float, float]) -> tuple[int, int, int]:
            return tuple(
                min(int(p / self.resolution), s - 1)
                for p, s in zip(pos, self.grid_shape)
            )  # type: ignore[return-value]

        start = to_grid(start_nm)
        end = to_grid(end_nm)

        if self._availability[start] < coherence_threshold:
            return None
        if self._availability[end] < coherence_threshold:
            return None

        # BFS with coherence gating
        from collections import deque
        visited: set[tuple[int, int, int]] = {start}
        queue: deque[list[tuple[int, int, int]]] = deque([[start]])

        nx, ny, nz = self.grid_shape
        directions = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ]

        while queue:
            path = queue.popleft()
            current = path[-1]

            if current == end:
                return path

            for dx, dy, dz in directions:
                nx_, ny_, nz_ = current[0] + dx, current[1] + dy, current[2] + dz
                neighbour = (nx_, ny_, nz_)
                if (
                    0 <= nx_ < nx
                    and 0 <= ny_ < ny
                    and 0 <= nz_ < nz
                    and neighbour not in visited
                    and self._availability[neighbour] >= coherence_threshold
                ):
                    visited.add(neighbour)
                    queue.append(path + [neighbour])

        return None  # no path found

    @property
    def availability(self) -> np.ndarray:
        return self._availability.copy()

    @property
    def disrupted_fraction(self) -> float:
        """Fraction of the volume with coherence below 50%."""
        return float(np.mean(self._availability < 0.5))
