"""Detail 4.1 — Laser source editor: definition of input light pulses.

Supports specification of frequency (tuned to exciton resonance), power/intensity,
polarisation (linear/circular), pulse duration (fs–ps), spatial profile (Gaussian),
and multi-source control with precise time delays for stimulating logic operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence

import numpy as np

from ..constants import HBAR, C_LIGHT, EV_TO_J, FS_TO_S, PS_TO_S, NM_TO_M


class PolarisationState(Enum):
    LINEAR_H = auto()
    LINEAR_V = auto()
    CIRCULAR_LEFT = auto()
    CIRCULAR_RIGHT = auto()


@dataclass
class LaserPulse:
    """Single laser pulse specification.

    Attributes:
        energy_eV: Photon energy [eV] (should be tuned to exciton resonance).
        intensity_W_cm2: Peak intensity [W/cm²].
        polarisation: Polarisation state.
        duration_fs: Pulse duration (FWHM) [fs].
        delay_ps: Time delay relative to t = 0 [ps].
        waist_nm: Gaussian beam waist (1/e² radius) [nm].
        position_nm: (x, y) injection point on the chip [nm].
    """

    energy_eV: float
    intensity_W_cm2: float = 1e4
    polarisation: PolarisationState = PolarisationState.LINEAR_H
    duration_fs: float = 100.0
    delay_ps: float = 0.0
    waist_nm: float = 500.0
    position_nm: tuple[float, float] = (0.0, 0.0)

    @property
    def wavelength_nm(self) -> float:
        return HBAR * C_LIGHT / (self.energy_eV * EV_TO_J) * 2 * np.pi / NM_TO_M

    @property
    def angular_frequency(self) -> float:
        """ω [rad/s]."""
        return self.energy_eV * EV_TO_J / HBAR

    def temporal_envelope(self, t_ps: np.ndarray) -> np.ndarray:
        """Gaussian temporal envelope centred at *delay_ps*.

        Returns amplitude envelope (normalised to 1 at peak).
        """
        sigma_ps = self.duration_fs * 1e-3 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        return np.exp(-((t_ps - self.delay_ps) ** 2) / (2.0 * sigma_ps**2))

    def spatial_profile(
        self, x_nm: np.ndarray, y_nm: np.ndarray
    ) -> np.ndarray:
        """2D Gaussian spatial profile centred at *position_nm*."""
        x0, y0 = self.position_nm
        w = self.waist_nm
        r2 = (x_nm - x0) ** 2 + (y_nm - y0) ** 2
        return np.exp(-r2 / w**2)


class LaserSource:
    """Multi-source laser controller with precise timing.

    Manages one or more laser pulses that can be fired simultaneously or with
    controlled time delays to stimulate XOR, AND, or other logic operations.
    """

    def __init__(self) -> None:
        self._pulses: list[LaserPulse] = []

    def add_pulse(self, pulse: LaserPulse) -> None:
        self._pulses.append(pulse)

    @property
    def pulses(self) -> list[LaserPulse]:
        return list(self._pulses)

    @property
    def n_sources(self) -> int:
        return len(self._pulses)

    def combined_field(
        self,
        t_ps: float,
        x_nm: np.ndarray,
        y_nm: np.ndarray,
    ) -> np.ndarray:
        """Superposition of all pulse fields at time t [ps] on the (x, y) grid.

        Returns a complex field amplitude array.
        """
        total = np.zeros_like(x_nm, dtype=np.complex128)
        t_arr = np.array([t_ps])
        for pulse in self._pulses:
            env = float(pulse.temporal_envelope(t_arr)[0])
            spatial = pulse.spatial_profile(x_nm, y_nm)
            omega = pulse.angular_frequency
            phase = omega * t_ps * PS_TO_S
            total += env * spatial * np.exp(1j * phase) * np.sqrt(
                pulse.intensity_W_cm2
            )
        return total

    @staticmethod
    def xor_pair(
        energy_eV: float,
        position_a: tuple[float, float],
        position_b: tuple[float, float],
        intensity: float = 1e4,
        duration_fs: float = 100.0,
    ) -> LaserSource:
        """Create a pair of simultaneous pulses for XOR gate stimulation."""
        src = LaserSource()
        src.add_pulse(
            LaserPulse(
                energy_eV=energy_eV,
                intensity_W_cm2=intensity,
                duration_fs=duration_fs,
                position_nm=position_a,
            )
        )
        src.add_pulse(
            LaserPulse(
                energy_eV=energy_eV,
                intensity_W_cm2=intensity,
                duration_fs=duration_fs,
                position_nm=position_b,
            )
        )
        return src
