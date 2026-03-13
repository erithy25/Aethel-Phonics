"""Detail 5.2 — Phase stability analysis under mechanical vibrations.

Models the influence of external vibrations on holographic memory addressing
and waveguide phase coherence. Computes phase jitter, bit-error probability,
and recommends vibration isolation requirements.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import C_LIGHT, NM_TO_M


@dataclass
class VibrationProfile:
    """Mechanical vibration specification.

    Attributes:
        frequencies_Hz: Array of vibration frequency components [Hz].
        amplitudes_nm: Peak displacement amplitude per component [nm].
        psd_g2_Hz: Power spectral density [g²/Hz] (optional, for broadband).
    """

    frequencies_Hz: np.ndarray
    amplitudes_nm: np.ndarray
    psd_g2_Hz: np.ndarray | None = None


@dataclass
class StabilityReport:
    """Result of a phase-stability analysis.

    Attributes:
        rms_phase_jitter_rad: RMS phase jitter across the waveguide [rad].
        max_phase_deviation_rad: Peak phase deviation [rad].
        bit_error_probability: Estimated BER from phase noise.
        coherence_length_nm: Effective coherence length under vibration [nm].
        max_tolerable_amplitude_nm: Maximum vibration amplitude for < 1% BER [nm].
        isolation_required: Whether active vibration isolation is needed.
    """

    rms_phase_jitter_rad: float
    max_phase_deviation_rad: float
    bit_error_probability: float
    coherence_length_nm: float
    max_tolerable_amplitude_nm: float
    isolation_required: bool


class PhaseStabilityAnalyser:
    """Analyse phase coherence under mechanical vibrations.

    Vibrations cause path-length fluctuations δL(t) in waveguides,
    producing phase noise δφ = (2π n / λ) · δL. For holographic addressing,
    the accumulated phase error must stay well below π to avoid bit flips.
    """

    def __init__(
        self,
        waveguide_length_nm: float = 10_000.0,
        refractive_index: float = 1.76,
        wavelength_nm: float = 720.0,
    ) -> None:
        self.L = waveguide_length_nm
        self.n = refractive_index
        self.lam = wavelength_nm

    def phase_sensitivity(self) -> float:
        """Phase change per nm of path-length fluctuation [rad/nm].

        δφ/δL = 2π n / λ
        """
        return 2.0 * np.pi * self.n / self.lam

    def compute_phase_jitter(self, profile: VibrationProfile) -> float:
        """RMS phase jitter from the vibration profile [rad].

        Each vibration mode contributes δL ≈ amplitude (for axial modes).
        Total RMS jitter = sensitivity × √(Σ A_i²)
        """
        rms_displacement = np.sqrt(np.sum(profile.amplitudes_nm**2))
        return self.phase_sensitivity() * rms_displacement

    def bit_error_probability(self, rms_jitter_rad: float) -> float:
        """Estimate BER from phase jitter.

        Using Gaussian error function: BER ≈ 0.5 · erfc(π / (2√2 · σ_φ))
        where σ_φ is the RMS phase jitter.
        """
        from scipy.special import erfc
        if rms_jitter_rad <= 0:
            return 0.0
        return 0.5 * float(erfc(np.pi / (2.0 * np.sqrt(2.0) * rms_jitter_rad)))

    def coherence_length(self, rms_jitter_rad: float) -> float:
        """Effective coherence length under phase noise [nm].

        L_coh = L × exp(-σ_φ²/2)  — Gaussian dephasing model.
        """
        return self.L * np.exp(-rms_jitter_rad**2 / 2.0)

    def max_tolerable_amplitude(self, target_ber: float = 0.01) -> float:
        """Maximum single-mode vibration amplitude [nm] for target BER."""
        from scipy.special import erfcinv
        if target_ber >= 0.5:
            return float("inf")
        sigma_max = np.pi / (2.0 * np.sqrt(2.0) * erfcinv(2.0 * target_ber))
        return sigma_max / self.phase_sensitivity()

    def analyse(self, profile: VibrationProfile) -> StabilityReport:
        """Full phase-stability analysis for a given vibration environment."""
        rms_jitter = self.compute_phase_jitter(profile)
        max_deviation = self.phase_sensitivity() * np.max(profile.amplitudes_nm)
        ber = self.bit_error_probability(rms_jitter)
        coh_len = self.coherence_length(rms_jitter)
        max_amp = self.max_tolerable_amplitude()

        return StabilityReport(
            rms_phase_jitter_rad=rms_jitter,
            max_phase_deviation_rad=max_deviation,
            bit_error_probability=ber,
            coherence_length_nm=coh_len,
            max_tolerable_amplitude_nm=max_amp,
            isolation_required=ber > 0.001,
        )
