"""Detail 2.4 — Thermal Drifting Inference.

Couples AI inference to the ``ThermoOpticSolver`` from Module F.

As the monolith heats up from cumulative gate switching, the refractive index
shifts by Δn = (dn/dT) · ΔT.  In a polariton interference gate this manifests
as a *phase error* Δφ = (2π/λ) · Δn · L that corrupts every complex-valued
multiplication.

Implementation:
    After each tensor operation the ``ThermalDriftLayer``:
    1. Injects Landauer heat proportional to the operation's gate count.
    2. Advances the thermal solver by one time step.
    3. Reads the current Δn from the solver.
    4. Converts Δn to a phase error Δφ.
    5. Applies the phase error as a complex rotation to the result.
    6. If T_chip > breakdown temperature (~409 K), outputs become NaN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import K_BOLTZMANN, T_ROOM, NM_TO_M
from ..module_f.heat_optics import ThermoOpticSolver, ThermoOpticConfig


@dataclass
class DriftSnapshot:
    """Snapshot of the thermal state after one operation.

    Attributes:
        peak_temperature_K: Maximum temperature in the lattice.
        mean_delta_T_K: Mean temperature rise above ambient.
        peak_delta_n: Maximum refractive-index perturbation.
        phase_error_rad: Applied phase error [rad].
        is_breakdown: Whether the system has exceeded breakdown temperature.
    """

    peak_temperature_K: float
    mean_delta_T_K: float
    peak_delta_n: float
    phase_error_rad: float
    is_breakdown: bool


class ThermalDriftLayer:
    """Physics layer that injects thermo-optic phase errors into tensor results.

    Parameters:
        thermo_config: Configuration for the thermal solver.
        wavelength_nm: Operating wavelength of the polariton gates.
        interaction_length_nm: Effective gate cascade path [nm].
        breakdown_temperature_K: Temperature above which outputs are NaN.
        energy_per_gate_aJ: Energy dissipated per polariton gate [aJ].
        gates_per_flop: Number of polariton gates per floating-point op.
        reversible_fraction: Fraction of ops that are reversible (0 heat).
    """

    def __init__(
        self,
        thermo_config: ThermoOpticConfig | None = None,
        *,
        wavelength_nm: float = 721.0,
        interaction_length_nm: float = 50_000.0,
        breakdown_temperature_K: float = 409.0,
        energy_per_gate_aJ: float = 0.5,
        gates_per_flop: int = 3,
        reversible_fraction: float = 0.0,
    ) -> None:
        cfg = thermo_config or ThermoOpticConfig()
        self._solver = ThermoOpticSolver(cfg)
        self._solver.initialise()

        self._wavelength_m = wavelength_nm * NM_TO_M
        self._interaction_m = interaction_length_nm * NM_TO_M
        self._breakdown_K = breakdown_temperature_K
        self._energy_per_gate_J = energy_per_gate_aJ * 1e-18
        self._gates_per_flop = gates_per_flop
        self._reversible_fraction = max(0.0, min(1.0, reversible_fraction))

        self._history: list[DriftSnapshot] = []
        self._total_ops: int = 0
        self._in_breakdown: bool = False

    @property
    def solver(self) -> ThermoOpticSolver:
        return self._solver

    @property
    def history(self) -> list[DriftSnapshot]:
        return list(self._history)

    @property
    def is_breakdown(self) -> bool:
        return self._in_breakdown

    @property
    def peak_temperature_K(self) -> float:
        return float(self._solver.peak_temperature_K)

    @property
    def current_phase_error_rad(self) -> float:
        """Current peak thermo-optic phase error [rad]."""
        peak_dn = float(np.max(self._solver.delta_n))
        return 2.0 * np.pi * peak_dn * self._interaction_m / self._wavelength_m

    @property
    def fidelity(self) -> float:
        """Inference fidelity [0, 1] based on phase error.

        Maps the phase error to a fidelity score:
        fidelity = cos²(Δφ/2) — the MZI contrast.
        At Δφ = 0 → 1.0, at Δφ = π/2 → 0.5, at breakdown → 0.0.
        """
        if self._in_breakdown:
            return 0.0
        phi = self.current_phase_error_rad
        return float(np.cos(phi / 2) ** 2)

    def apply(
        self, result: np.ndarray, n_flops: int | None = None,
    ) -> np.ndarray:
        """Apply thermal drift to a tensor operation result.

        Parameters:
            result: Output tensor from a compute operation.
            n_flops: Number of FLOPs in the operation.  If ``None``,
                estimated as ``result.size``.

        Returns:
            The tensor with thermo-optic phase error applied.
            Returns NaN-filled array if in breakdown state.
        """
        if n_flops is None:
            n_flops = result.size

        self._total_ops += n_flops

        # 1. Inject Landauer heat for irreversible fraction
        irreversible_gates = int(
            n_flops * self._gates_per_flop * (1.0 - self._reversible_fraction)
        )
        if irreversible_gates > 0:
            self._solver.add_landauer_heat(irreversible_gates)

        # 2. Advance thermal solver
        self._solver.advance(1)

        # 3. Read current state
        peak_T = float(self._solver.peak_temperature_K)
        mean_dT = float(np.mean(self._solver.delta_T))
        peak_dn = float(np.max(self._solver.delta_n))
        phase_error = 2.0 * np.pi * peak_dn * self._interaction_m / self._wavelength_m

        # 4. Check breakdown
        if peak_T >= self._breakdown_K:
            self._in_breakdown = True
            snap = DriftSnapshot(
                peak_temperature_K=peak_T,
                mean_delta_T_K=mean_dT,
                peak_delta_n=peak_dn,
                phase_error_rad=phase_error,
                is_breakdown=True,
            )
            self._history.append(snap)
            return np.full_like(result, np.nan)

        # 5. Apply phase error as a perturbation.
        #    Model: each element is rotated by a phase Δφ in the complex plane.
        #    For real-valued tensors this manifests as a multiplicative factor
        #    cos(Δφ) plus additive noise sin(Δφ) · |x|.
        if phase_error > 1e-12:
            cos_phi = np.cos(phase_error)
            sin_phi = np.sin(phase_error)
            result = result * cos_phi + np.abs(result) * sin_phi * 0.1

        snap = DriftSnapshot(
            peak_temperature_K=peak_T,
            mean_delta_T_K=mean_dT,
            peak_delta_n=peak_dn,
            phase_error_rad=phase_error,
            is_breakdown=False,
        )
        self._history.append(snap)
        return result

    def reset(self) -> None:
        """Re-initialise the thermal solver (cool down)."""
        self._solver.initialise()
        self._history.clear()
        self._total_ops = 0
        self._in_breakdown = False
