"""Detail 6.2 — Performance dashboard: automatic computation of hardware metrics.

Computes switching latency, switching energy, thermal budget, and
signal-to-noise ratio from simulation data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..module_c.detector import DetectorReading
from ..constants import PS_TO_S, EV_TO_J


@dataclass
class GateMetrics:
    """Key performance indicators for a single gate operation.

    Attributes:
        switching_latency_ps: Time from laser input to stable output [ps].
        switching_energy_aJ: Integrated laser pulse energy per operation [aJ].
        thermal_budget_aJ: Residual heat per operation from absorption losses [aJ].
        snr_dB: Signal-to-noise ratio — contrast between logic "0" and "1" [dB].
    """

    switching_latency_ps: float
    switching_energy_aJ: float
    thermal_budget_aJ: float
    snr_dB: float

    def summary(self) -> str:
        return (
            f"Latency:       {self.switching_latency_ps:.2f} ps\n"
            f"Switch energy: {self.switching_energy_aJ:.2f} aJ\n"
            f"Thermal load:  {self.thermal_budget_aJ:.2f} aJ\n"
            f"SNR:           {self.snr_dB:.1f} dB"
        )


class PerformanceDashboard:
    """Analyse simulation output to produce hardware performance metrics."""

    def __init__(
        self,
        absorption_coefficient: float = 0.01,
    ) -> None:
        self.alpha = absorption_coefficient

    def compute_latency(
        self,
        output_reading: DetectorReading,
        input_time_ps: float = 0.0,
        threshold_frac: float = 0.1,
    ) -> float:
        """Switching latency: time from input pulse to output rising edge [ps]."""
        arrival = output_reading.arrival_time_ps
        return arrival - input_time_ps

    def compute_switching_energy(
        self,
        pulse_energies_eV: list[float],
        pulse_durations_fs: list[float],
        peak_intensities_W_cm2: list[float],
        spot_areas_um2: list[float],
    ) -> float:
        """Integrated pulse energy per operation [attojoules].

        E = Σ I_peak · A_spot · τ_pulse  (approximate for Gaussian pulses).
        """
        total_J = 0.0
        for I, A, tau in zip(
            peak_intensities_W_cm2, spot_areas_um2, pulse_durations_fs
        ):
            # I [W/cm²] → W/m², A [µm²] → m², τ [fs] → s
            total_J += (I * 1e4) * (A * 1e-12) * (tau * 1e-15)
        return total_J * 1e18  # J → aJ

    def compute_thermal_budget(self, switching_energy_aJ: float) -> float:
        """Residual heat per operation [aJ] based on absorption losses."""
        return switching_energy_aJ * self.alpha

    def compute_snr(
        self,
        intensity_logic_1: float,
        intensity_logic_0: float,
    ) -> float:
        """Signal-to-noise ratio [dB] between logic levels."""
        if intensity_logic_0 <= 0:
            return float("inf")
        return 10.0 * np.log10(intensity_logic_1 / intensity_logic_0)

    def evaluate_gate(
        self,
        output_reading_1: DetectorReading,
        output_reading_0: DetectorReading,
        pulse_energies_eV: list[float],
        pulse_durations_fs: list[float],
        peak_intensities: list[float],
        spot_areas_um2: list[float],
        input_time_ps: float = 0.0,
    ) -> GateMetrics:
        """Full performance evaluation of a single gate.

        Args:
            output_reading_1: Detector reading for the logic-"1" output case.
            output_reading_0: Detector reading for the logic-"0" output case.
            pulse_*: Laser pulse parameters.
            input_time_ps: Time of the input pulse.
        """
        latency = self.compute_latency(output_reading_1, input_time_ps)
        energy = self.compute_switching_energy(
            pulse_energies_eV, pulse_durations_fs, peak_intensities, spot_areas_um2
        )
        thermal = self.compute_thermal_budget(energy)
        snr = self.compute_snr(
            output_reading_1.peak_intensity,
            output_reading_0.peak_intensity,
        )
        return GateMetrics(
            switching_latency_ps=latency,
            switching_energy_aJ=energy,
            thermal_budget_aJ=thermal,
            snr_dB=snr,
        )
