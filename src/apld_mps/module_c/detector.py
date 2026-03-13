"""Detail 4.2 — Virtual detectors: measurement points within the structure.

Place measurement probes anywhere in the simulation domain to record light
intensity, spectral composition, and phase with picosecond time resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..constants import PS_TO_S, HBAR, EV_TO_J


@dataclass
class DetectorReading:
    """Time-resolved measurement captured by a virtual detector.

    Attributes:
        times_ps: Time array [ps].
        intensity: Light intensity vs time [arb. units].
        phase: Phase of the complex field vs time [rad].
        spectrum_energies_eV: Energy axis for spectral data [eV] (populated after FFT).
        spectrum_amplitude: Spectral amplitude (populated after FFT).
    """

    times_ps: np.ndarray
    intensity: np.ndarray
    phase: np.ndarray
    spectrum_energies_eV: np.ndarray = field(default_factory=lambda: np.array([]))
    spectrum_amplitude: np.ndarray = field(default_factory=lambda: np.array([]))

    def compute_spectrum(self) -> None:
        """Compute energy spectrum via FFT of the complex field history."""
        dt = (self.times_ps[1] - self.times_ps[0]) * PS_TO_S
        complex_field = np.sqrt(self.intensity) * np.exp(1j * self.phase)
        spectrum = np.fft.fft(complex_field)
        freqs = np.fft.fftfreq(len(complex_field), d=dt)

        # Convert angular frequency to eV and keep positive half
        n_pos = len(freqs) // 2
        self.spectrum_energies_eV = HBAR * 2 * np.pi * freqs[:n_pos] / EV_TO_J
        self.spectrum_amplitude = np.abs(spectrum[:n_pos])

    @property
    def peak_intensity(self) -> float:
        return float(np.max(self.intensity))

    @property
    def arrival_time_ps(self) -> float:
        """Time at which intensity first exceeds 10% of peak (rising edge)."""
        threshold = 0.1 * self.peak_intensity
        above = np.where(self.intensity > threshold)[0]
        if len(above) == 0:
            return float("inf")
        return float(self.times_ps[above[0]])


class VirtualDetector:
    """Measurement probe placed at a fixed position in the simulation domain.

    Records the complex field at its grid position at every time step, then
    provides intensity, phase, and spectral analysis.
    """

    def __init__(
        self,
        name: str,
        grid_position: tuple[int, int],
        dt_ps: float = 0.01,
    ) -> None:
        self.name = name
        self.grid_position = grid_position
        self.dt_ps = dt_ps
        self._field_history: list[complex] = []
        self._times: list[float] = []
        self._current_time_ps: float = 0.0

    def record(self, field_value: complex) -> None:
        """Record a single field sample."""
        self._field_history.append(field_value)
        self._times.append(self._current_time_ps)
        self._current_time_ps += self.dt_ps

    def record_from_array(self, field_2d: np.ndarray) -> None:
        """Sample the 2D field at our grid position and record."""
        i, j = self.grid_position
        self.record(complex(field_2d[i, j]))

    def get_reading(self) -> DetectorReading:
        """Produce a DetectorReading from all recorded samples."""
        fields = np.array(self._field_history, dtype=np.complex128)
        times = np.array(self._times)
        reading = DetectorReading(
            times_ps=times,
            intensity=np.abs(fields) ** 2,
            phase=np.angle(fields),
        )
        reading.compute_spectrum()
        return reading

    def reset(self) -> None:
        self._field_history.clear()
        self._times.clear()
        self._current_time_ps = 0.0
