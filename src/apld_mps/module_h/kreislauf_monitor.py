"""Sektion 3 — Aethel-Kreislauf Monitor.

Real-time thermal heatmap, TPV recycling statistics, and clock-rate
governor linked to the thermodynamic feedback module (Module F).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from ..constants import NM_TO_M
from ..module_f.heat_optics import ThermoOpticSolver, ThermoOpticConfig
from ..module_f.tpv_recycler import TPVRecycler, TPVConfig, StabilityResult


class ThermalAlertLevel(Enum):
    """Thermal alert severity."""
    NOMINAL = auto()
    WARNING = auto()
    CRITICAL = auto()


@dataclass
class ThermalHeatmapData:
    """Thermal heatmap data for rendering.

    Attributes:
        temperature_slice: 2D temperature array [K] (mid-plane).
        peak_temperature_K: Maximum temperature in the volume.
        mean_temperature_K: Mean temperature in the volume.
        delta_n_slice: 2D thermo-optic Δn at mid-plane.
        alert_level: Current thermal alert.
    """

    temperature_slice: np.ndarray
    peak_temperature_K: float
    mean_temperature_K: float
    delta_n_slice: np.ndarray
    alert_level: ThermalAlertLevel


@dataclass
class TPVStatus:
    """TPV energy recycling status display.

    Attributes:
        recycling_efficiency: Fraction of waste heat recovered [0–1].
        recovered_power_W: Electrical power from TPV cells [W].
        radiated_power_W: Total IR power radiated [W].
        net_power_W: External power still required [W].
        equilibrium_temperature_K: Current steady-state temperature [K].
    """

    recycling_efficiency: float
    recovered_power_W: float
    radiated_power_W: float
    net_power_W: float
    equilibrium_temperature_K: float


@dataclass
class ClockGovernorState:
    """Clock rate governor status.

    Attributes:
        current_clock_GHz: Currently set clock rate [GHz].
        max_stable_clock_GHz: Maximum stable clock rate [GHz].
        headroom_percent: Remaining thermal headroom [%].
        stability: Result of the latest stability analysis.
    """

    current_clock_GHz: float
    max_stable_clock_GHz: float
    headroom_percent: float
    stability: StabilityResult | None


class KreislaufMonitor:
    """Aethel-Kreislauf: closed-loop thermal management monitor.

    Integrates the ThermoOpticSolver and TPVRecycler into a real-time
    dashboard panel with thermal heatmap, TPV statistics, and clock
    rate governor.
    """

    def __init__(
        self,
        thermo_config: ThermoOpticConfig | None = None,
        tpv_config: TPVConfig | None = None,
        warning_temperature_K: float = 350.0,
        critical_temperature_K: float = 400.0,
    ) -> None:
        self._thermo_cfg = thermo_config or ThermoOpticConfig()
        self._tpv_cfg = tpv_config or TPVConfig()

        self._thermo_solver = ThermoOpticSolver(self._thermo_cfg)
        self._tpv_recycler = TPVRecycler(self._tpv_cfg)

        self._warning_K = warning_temperature_K
        self._critical_K = critical_temperature_K
        self._current_clock_GHz = 0.0
        self._latest_stability: StabilityResult | None = None
        self._initialised = False

    def initialise(self) -> None:
        """Initialise the thermal solver."""
        self._thermo_solver.initialise()
        self._initialised = True

    @property
    def is_initialised(self) -> bool:
        return self._initialised

    @property
    def solver(self) -> ThermoOpticSolver:
        """Access the underlying thermal solver."""
        return self._thermo_solver

    # --- Thermal heatmap ---

    def _alert_level(self, peak_K: float) -> ThermalAlertLevel:
        if peak_K >= self._critical_K:
            return ThermalAlertLevel.CRITICAL
        if peak_K >= self._warning_K:
            return ThermalAlertLevel.WARNING
        return ThermalAlertLevel.NOMINAL

    def get_thermal_heatmap(self) -> ThermalHeatmapData:
        """Produce thermal heatmap data for the dashboard."""
        if not self._initialised:
            raise RuntimeError("Call initialise() first.")

        temp = self._thermo_solver.temperature
        mid_z = temp.shape[2] // 2
        temp_slice = temp[:, :, mid_z]
        dn_slice = self._thermo_solver.delta_n[:, :, mid_z]
        peak = self._thermo_solver.peak_temperature_K
        mean = self._thermo_solver.mean_temperature_K

        return ThermalHeatmapData(
            temperature_slice=temp_slice,
            peak_temperature_K=peak,
            mean_temperature_K=mean,
            delta_n_slice=dn_slice,
            alert_level=self._alert_level(peak),
        )

    def inject_heat(
        self,
        intensity_field: np.ndarray | None = None,
        n_operations: int = 0,
    ) -> None:
        """Feed heat sources into the thermal solver."""
        if not self._initialised:
            raise RuntimeError("Call initialise() first.")
        if intensity_field is not None:
            self._thermo_solver.set_heat_from_intensity(intensity_field)
        if n_operations > 0:
            self._thermo_solver.add_landauer_heat(n_operations)

    def advance_thermal(
        self, n_steps: int = 1, source_q_W: float = 0.0,
    ) -> None:
        """Advance the thermal simulation.

        Parameters:
            n_steps: Number of thermal time steps to advance.
            source_q_W: Total input power [W] to inject as a uniform
                volumetric heat source before advancing.  This is typically
                the laser pump power: ``Σ E_pulse × f_clk``.
        """
        if not self._initialised:
            raise RuntimeError("Call initialise() first.")
        if source_q_W > 0.0:
            # Distribute power uniformly as volumetric heat [W/m³]
            vol_m3 = (
                np.prod(np.array(self._thermo_cfg.grid_size_nm)) * NM_TO_M ** 3
            )
            self._thermo_solver._heat_source += source_q_W / vol_m3
        self._thermo_solver.advance(n_steps)

    @property
    def temperature_3d(self) -> np.ndarray:
        """Full 3D temperature array for viewport overlay."""
        if not self._initialised:
            raise RuntimeError("Call initialise() first.")
        return self._thermo_solver.temperature

    @property
    def eps_r_correction(self) -> np.ndarray:
        """Permittivity correction for FDTD feedback loop."""
        if not self._initialised:
            raise RuntimeError("Call initialise() first.")
        sapphire_n = 1.76
        return self._thermo_solver.eps_r_correction(sapphire_n)

    # --- TPV recycling ---

    def get_tpv_status(self) -> TPVStatus:
        """Compute TPV recycling statistics at current temperature."""
        peak_T = self._thermo_solver.peak_temperature_K if self._initialised else 300.0
        P_rad = self._tpv_recycler.radiated_power(peak_T)
        P_tpv = self._tpv_recycler.tpv_electrical_power(peak_T)
        P_diss = self._tpv_recycler.dissipated_power(self._current_clock_GHz)
        eff = P_tpv / P_diss if P_diss > 0 else 0.0
        net = max(P_diss - P_tpv, 0.0)

        return TPVStatus(
            recycling_efficiency=eff,
            recovered_power_W=P_tpv,
            radiated_power_W=P_rad,
            net_power_W=net,
            equilibrium_temperature_K=peak_T,
        )

    # --- Clock rate governor ---

    def set_clock_rate(self, clock_GHz: float) -> ClockGovernorState:
        """Set the current clock rate and recompute stability."""
        self._current_clock_GHz = clock_GHz
        return self.get_clock_governor_state()

    def run_stability_analysis(
        self,
        max_temperature_K: float | None = None,
        resolution_GHz: float = 1.0,
    ) -> StabilityResult:
        """Run full max-clock-rate stability analysis."""
        max_T = max_temperature_K or self._critical_K
        self._latest_stability = self._tpv_recycler.analyse_max_clock_rate(
            max_temperature_K=max_T,
            search_resolution_GHz=resolution_GHz,
        )
        return self._latest_stability

    def get_clock_governor_state(self) -> ClockGovernorState:
        """Current clock governor display state."""
        max_clock = 0.0
        if self._latest_stability:
            max_clock = self._latest_stability.max_clock_rate_GHz

        headroom = 0.0
        if max_clock > 0:
            headroom = max(0.0, (1.0 - self._current_clock_GHz / max_clock) * 100)

        return ClockGovernorState(
            current_clock_GHz=self._current_clock_GHz,
            max_stable_clock_GHz=max_clock,
            headroom_percent=headroom,
            stability=self._latest_stability,
        )
