"""Detail 3.2 — Near-field thermophotovoltaic (TPV) energy recycling model.

Models the Aethel V1's unique ability to recapture waste heat as electrical
energy via the enclosure walls, forming a closed energy loop:

    Computation → Heat → IR photons → TPV cell → Electricity → Laser emitters

Includes efficiency calculation and maximum stable clock-rate analysis
under passive radiative cooling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import K_BOLTZMANN, HBAR, C_LIGHT, T_ROOM


# Stefan-Boltzmann constant [W/(m²·K⁴)]
SIGMA_SB = 5.670374419e-8


@dataclass
class TPVConfig:
    """Configuration for TPV energy recycling analysis.

    Attributes:
        chip_surface_area_cm2: Total radiating surface area [cm²].
        tpv_cell_area_cm2: Total TPV cell collection area [cm²].
        tpv_bandgap_eV: Bandgap of the TPV cell material [eV].
        tpv_quantum_efficiency: Internal quantum efficiency of TPV cells.
        view_factor: Geometric view factor (fraction of emitted photons
            reaching TPV cells).
        emissivity_sapphire: IR emissivity of sapphire surface.
        ambient_temperature_K: External ambient temperature [K].
        energy_per_operation_aJ: Energy consumed per logic operation [aJ].
        thermal_resistance_K_W: Thermal resistance chip-to-enclosure [K/W].
    """

    chip_surface_area_cm2: float = 960.0  # 40 cm monolith: ~6 faces
    tpv_cell_area_cm2: float = 800.0
    tpv_bandgap_eV: float = 0.55  # InGaAsSb for mid-IR
    tpv_quantum_efficiency: float = 0.85
    view_factor: float = 0.92  # near-field, close enclosure walls
    emissivity_sapphire: float = 0.9
    ambient_temperature_K: float = T_ROOM
    energy_per_operation_aJ: float = 0.5
    thermal_resistance_K_W: float = 0.1  # K/W


@dataclass
class StabilityResult:
    """Result of the maximum clock-rate stability analysis.

    Attributes:
        max_clock_rate_GHz: Maximum clock rate for thermal stability [GHz].
        equilibrium_temperature_K: Steady-state chip temperature at max clock.
        radiated_power_W: Total radiated IR power at equilibrium [W].
        tpv_recovered_power_W: Electrical power recovered by TPV cells [W].
        recycling_efficiency: Fraction of waste heat recovered as electricity.
        net_power_W: Net external power still needed [W].
    """

    max_clock_rate_GHz: float
    equilibrium_temperature_K: float
    radiated_power_W: float
    tpv_recovered_power_W: float
    recycling_efficiency: float
    net_power_W: float


class TPVRecycler:
    """Near-field thermophotovoltaic energy recycling model."""

    def __init__(self, config: TPVConfig) -> None:
        self.cfg = config

    def radiated_power(self, temperature_K: float) -> float:
        """Total radiative power from the chip surface [W].

        P_rad = ε · σ · A · (T_chip⁴ - T_amb⁴)
        """
        A = self.cfg.chip_surface_area_cm2 * 1e-4  # to m²
        return (
            self.cfg.emissivity_sapphire
            * SIGMA_SB
            * A
            * (temperature_K**4 - self.cfg.ambient_temperature_K**4)
        )

    def tpv_electrical_power(self, temperature_K: float) -> float:
        """Electrical power recovered by TPV cells [W].

        Uses a simplified spectral model: fraction of blackbody radiation
        above the TPV bandgap, multiplied by quantum efficiency and view factor.
        """
        E_bg = self.cfg.tpv_bandgap_eV * 1.602176634e-19  # to J
        kT = K_BOLTZMANN * temperature_K

        # Fraction of blackbody spectrum above bandgap (Wien approximation)
        x = E_bg / kT
        # Integral fraction ≈ e^{-x}(x² + 2x + 2) / 2  (asymptotic)
        if x > 50:
            above_gap_fraction = 0.0
        else:
            above_gap_fraction = np.exp(-x) * (x**2 + 2 * x + 2) / 2.0

        # Carnot-limited conversion efficiency
        eta_carnot = 1.0 - self.cfg.ambient_temperature_K / max(temperature_K, 1.0)
        # Practical efficiency is lower — use ~40% of Carnot
        eta_conversion = 0.4 * eta_carnot

        P_rad = self.radiated_power(temperature_K)
        P_above_gap = P_rad * above_gap_fraction * self.cfg.view_factor

        return P_above_gap * self.cfg.tpv_quantum_efficiency * eta_conversion

    def dissipated_power(self, clock_rate_GHz: float) -> float:
        """Total power dissipated at a given clock rate [W].

        P_diss = clock_rate · energy_per_op
        """
        ops_per_second = clock_rate_GHz * 1e9
        energy_per_op_J = self.cfg.energy_per_operation_aJ * 1e-18
        return ops_per_second * energy_per_op_J

    def equilibrium_temperature(
        self, clock_rate_GHz: float, max_iter: int = 200, tol: float = 0.01
    ) -> float:
        """Find steady-state temperature where radiated power = dissipated power.

        Uses simple bisection on the thermal balance equation:
            P_dissipated = P_radiated(T) + conductive losses
        """
        P_diss = self.dissipated_power(clock_rate_GHz)
        T_lo = self.cfg.ambient_temperature_K
        T_hi = 2000.0  # sapphire melts ~2050°C, limit search

        for _ in range(max_iter):
            T_mid = (T_lo + T_hi) / 2.0
            P_rad = self.radiated_power(T_mid)
            P_cond = (T_mid - self.cfg.ambient_temperature_K) / self.cfg.thermal_resistance_K_W
            P_total_cooling = P_rad + P_cond

            if P_total_cooling < P_diss:
                T_lo = T_mid
            else:
                T_hi = T_mid

            if T_hi - T_lo < tol:
                break

        return (T_lo + T_hi) / 2.0

    def analyse_max_clock_rate(
        self,
        max_temperature_K: float = 400.0,
        search_resolution_GHz: float = 0.1,
        max_clock_GHz: float = 1000.0,
    ) -> StabilityResult:
        """Find the maximum clock rate where the chip stays below max_temperature_K
        through passive radiative cooling alone.

        Sweeps clock rate upward until equilibrium temperature exceeds the limit.
        """
        best_clock = 0.0

        clock = search_resolution_GHz
        while clock <= max_clock_GHz:
            T_eq = self.equilibrium_temperature(clock)
            if T_eq > max_temperature_K:
                break
            best_clock = clock
            clock += search_resolution_GHz

        T_eq = self.equilibrium_temperature(best_clock)
        P_rad = self.radiated_power(T_eq)
        P_tpv = self.tpv_electrical_power(T_eq)
        P_diss = self.dissipated_power(best_clock)

        recycling_eff = P_tpv / P_diss if P_diss > 0 else 0.0

        return StabilityResult(
            max_clock_rate_GHz=best_clock,
            equilibrium_temperature_K=T_eq,
            radiated_power_W=P_rad,
            tpv_recovered_power_W=P_tpv,
            recycling_efficiency=recycling_eff,
            net_power_W=max(P_diss - P_tpv, 0.0),
        )
