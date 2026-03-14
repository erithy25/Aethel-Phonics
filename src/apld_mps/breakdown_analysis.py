"""Coupled Thermo-FDTD feedback loop and breakdown clock-rate analysis.

Implements the missing link between Module B (FDTD) and Module F (Thermal):
a bidirectional runtime coupling where computed ΔT continuously modifies
the permittivity map ε_r(r,t) that the FDTD solver operates on.

The key question answered: at what clock rate does thermo-optic self-heating
destroy the interference contrast of polariton logic gates?

Physics:
    1. Each clock cycle, N_ops irreversible bit erasures deposit
       Q_Landauer = N_ops · k_B T ln(2) into the sapphire lattice.
    2. Heat diffuses (spectral solver) and raises ΔT(r).
    3. ΔT shifts the refractive index: Δn = (dn/dT) · ΔT.
    4. The FDTD permittivity updates: ε_r(r) = ε_r_base + 2n·Δn(r).
    5. The gate's output extinction ratio (ER) degrades.
    6. Breakdown := ER < threshold  →  gate can no longer switch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.constants import c as C_LIGHT

from .constants import K_BOLTZMANN, NM_TO_M, T_ROOM
from .module_b.fdtd_solver import FDTDSolver, FDTDConfig
from .module_f.heat_optics import ThermoOpticSolver, ThermoOpticConfig


@dataclass
class CoupledConfig:
    """Configuration for coupled thermo-FDTD simulation.

    Attributes:
        fdtd_config: Electromagnetic solver configuration.
        thermo_config: Thermal solver configuration.
        base_refractive_index: Refractive index of sapphire substrate.
        substrate_eps_r: Base relative permittivity (n²).
        coupling_interval: Re-couple thermal→FDTD every N EM steps.
        n_ops_per_cycle: Irreversible bit erasures per clock cycle.
        source_position: Grid index for the input source.
        probe_position: Grid index for the output probe.
        source_amplitude: Amplitude of injected E-field source.
    """

    fdtd_config: FDTDConfig = field(default_factory=lambda: FDTDConfig(
        grid_spacing_nm=50.0,
        domain_size_nm=(10_000.0, 5_000.0),
        courant_factor=0.5,
        n_steps=200,
        pml_layers=10,
    ))
    thermo_config: ThermoOpticConfig = field(default_factory=ThermoOpticConfig)
    base_refractive_index: float = 1.76  # sapphire at visible
    substrate_eps_r: float = 1.76 ** 2  # n²
    coupling_interval: int = 50  # re-couple every 50 EM steps
    n_ops_per_cycle: int = 1_000_000  # ops per clock tick
    source_position: tuple[int, int] | None = None
    probe_position: tuple[int, int] | None = None
    source_amplitude: float = 1.0

    def __post_init__(self) -> None:
        """Validate and fix source/probe positions for grid bounds."""
        shape = self.fdtd_config.grid_shape
        pml = self.fdtd_config.pml_layers
        # Ensure source/probe positions are within valid grid range
        if self.source_position is not None:
            sx = min(self.source_position[0], shape[0] - 1)
            sy = min(self.source_position[1], shape[1] - 1)
            self.source_position = (sx, sy)
        if self.probe_position is not None:
            px = min(self.probe_position[0], shape[0] - 1)
            py = min(self.probe_position[1], shape[1] - 1)
            self.probe_position = (px, py)


@dataclass
class CoupledStepResult:
    """Snapshot from one coupling iteration."""

    em_step: int
    peak_delta_T_K: float
    peak_delta_n: float
    peak_eps_r_correction: float
    source_field: float
    probe_field: float
    field_energy_J: float


class CoupledThermoFDTDSimulator:
    """Bidirectional FDTD ↔ thermal feedback simulator.

    Each coupling cycle:
    1. Runs the FDTD for ``coupling_interval`` steps.
    2. Extracts |Ez|² as optical intensity.
    3. Feeds intensity + Landauer heat into the thermal solver.
    4. Advances thermal diffusion.
    5. Computes Δε_r from ΔT and patches the FDTD permittivity map.
    """

    def __init__(self, config: CoupledConfig) -> None:
        self.cfg = config
        self.fdtd = FDTDSolver(config.fdtd_config)
        self.thermo = ThermoOpticSolver(config.thermo_config)
        self._history: list[CoupledStepResult] = []
        self._base_eps_r: np.ndarray | None = None
        self._initialised = False

    def initialise(self) -> None:
        """Set up both solvers and establish the permittivity baseline."""
        shape = self.cfg.fdtd_config.grid_shape
        self._base_eps_r = np.full(shape, self.cfg.substrate_eps_r, dtype=np.float64)
        self.fdtd.initialise(eps_r_map=self._base_eps_r.copy())
        self.thermo.initialise()

        # Default source/probe positions if not set — keep within grid
        nx, ny = shape
        pml = self.cfg.fdtd_config.pml_layers
        if self.cfg.source_position is None:
            sx = min(pml + 2, nx - 1)
            self.cfg.source_position = (sx, ny // 2)
        if self.cfg.probe_position is None:
            px = max(0, min(nx - pml - 2, nx - 1))
            self.cfg.probe_position = (px, ny // 2)
        self._history = []
        self._initialised = True

    def _intensity_2d(self) -> np.ndarray:
        """Extract |Ez|² as a proxy for optical intensity [arb. units]."""
        return self.fdtd.ez ** 2

    def _resample_correction_to_2d(self, correction_3d: np.ndarray) -> np.ndarray:
        """Take a central z-slice of the 3D thermal correction for 2D FDTD."""
        nz = correction_3d.shape[2]
        mid_z = nz // 2
        slice_2d = correction_3d[:, :, mid_z]

        # Resample if thermal and FDTD grids differ in x,y
        target = self.cfg.fdtd_config.grid_shape
        if slice_2d.shape != target:
            from scipy.ndimage import zoom
            factors = (target[0] / slice_2d.shape[0], target[1] / slice_2d.shape[1])
            slice_2d = zoom(slice_2d, factors, order=1)
        return slice_2d

    def run(self, n_coupling_cycles: int) -> list[CoupledStepResult]:
        """Run the coupled simulation for the given number of coupling cycles.

        Returns a list of snapshots, one per coupling cycle.
        """
        if not self._initialised:
            raise RuntimeError("Call initialise() first.")

        assert self._base_eps_r is not None
        src = self.cfg.source_position
        prb = self.cfg.probe_position
        assert src is not None and prb is not None

        for _ in range(n_coupling_cycles):
            # 1. Inject source and advance FDTD
            self.fdtd.inject_source(src, self.cfg.source_amplitude)
            self.fdtd.advance(self.cfg.coupling_interval)

            # 2. Extract intensity → thermal source
            intensity = self._intensity_2d()

            # Expand 2D intensity to 3D for thermo solver (uniform in z)
            thermo_shape = self.cfg.thermo_config.grid_shape
            from scipy.ndimage import zoom
            factors_2d = (
                thermo_shape[0] / intensity.shape[0],
                thermo_shape[1] / intensity.shape[1],
            )
            intensity_resampled = zoom(intensity, factors_2d, order=1)
            intensity_3d = np.repeat(
                intensity_resampled[:, :, np.newaxis], thermo_shape[2], axis=2
            )
            self.thermo.set_heat_from_intensity(intensity_3d)

            # 3. Add Landauer heat
            self.thermo.add_landauer_heat(self.cfg.n_ops_per_cycle)

            # 4. Advance thermal solver
            #    Map EM time to thermal time: coupling_interval EM steps
            em_dt_s = self.cfg.fdtd_config.dt_s
            em_time_ps = self.cfg.coupling_interval * em_dt_s * 1e12
            thermo_steps = max(1, int(em_time_ps / self.cfg.thermo_config.dt_ps))
            self.thermo.advance(thermo_steps)

            # 5. Feed Δε_r back into FDTD
            correction_3d = self.thermo.eps_r_correction(self.cfg.base_refractive_index)
            correction_2d = self._resample_correction_to_2d(correction_3d)
            self.fdtd._eps_r = self._base_eps_r + correction_2d

            # 6. Record snapshot
            snap = CoupledStepResult(
                em_step=self.fdtd.step,
                peak_delta_T_K=float(np.max(self.thermo.delta_T)),
                peak_delta_n=float(np.max(self.thermo.delta_n)),
                peak_eps_r_correction=float(np.max(correction_2d)),
                source_field=float(self.fdtd.ez[src]),
                probe_field=float(self.fdtd.ez[prb]),
                field_energy_J=self.fdtd.field_energy,
            )
            self._history.append(snap)

        return self._history

    @property
    def history(self) -> list[CoupledStepResult]:
        return list(self._history)

    @property
    def peak_delta_T(self) -> float:
        """Peak temperature rise across the whole simulation."""
        if not self._history:
            return 0.0
        return max(s.peak_delta_T_K for s in self._history)


@dataclass
class BreakdownResult:
    """Result of the breakdown clock-rate analysis.

    Attributes:
        breakdown_clock_GHz: Clock rate at which gate function is lost.
        max_safe_clock_GHz: Highest clock rate with acceptable extinction ratio.
        breakdown_delta_T_K: ΔT at breakdown.
        breakdown_delta_n: Δn at breakdown.
        breakdown_phase_shift_rad: Thermo-optic phase shift at breakdown.
        extinction_ratio_dB_at_breakdown: Gate contrast at breakdown.
        sweep_clock_rates_GHz: Full sweep data — clock rates.
        sweep_delta_T_K: Full sweep data — peak ΔT.
        sweep_extinction_ratio_dB: Full sweep data — extinction ratios.
        sweep_phase_shift_rad: Full sweep data — phase shifts.
    """

    breakdown_clock_GHz: float
    max_safe_clock_GHz: float
    breakdown_delta_T_K: float
    breakdown_delta_n: float
    breakdown_phase_shift_rad: float
    extinction_ratio_dB_at_breakdown: float
    sweep_clock_rates_GHz: list[float]
    sweep_delta_T_K: list[float]
    sweep_extinction_ratio_dB: list[float]
    sweep_phase_shift_rad: list[float]


class BreakdownAnalyser:
    """Finds the clock rate at which thermo-optic self-heating breaks gate function.

    Strategy:
        For each candidate clock rate f_clk:
        1. Compute equilibrium ΔT from Landauer heat at f_clk using the
           lumped thermal balance (TPV recycler).
        2. Compute Δn = (dn/dT) · ΔT.
        3. Compute thermo-optic phase shift:
           Δφ = (2π/λ) · Δn · L_interaction
        4. Compute extinction ratio degradation from phase error.
        5. Breakdown when ER < min_extinction_ratio_dB.

    The phase-shift model is the critical insight: an XOR interference gate
    relies on destructive interference at its output. A parasitic Δφ from
    thermal index drift shifts the interference fringe, reducing contrast.

    ER(Δφ) = 10·log₁₀[ cos²(Δφ/2) / sin²(Δφ/2) ]  (for ideal MZI)

    Reversible logic:
        When ``reversible_fraction > 0`` a portion of the operations are
        implemented with reversible gates (Toffoli / Fredkin) that erase no
        bits and therefore produce *zero* Landauer entropy heat.  Only the
        irreversible fraction contributes to P_diss:

            P_diss = f_clk · N_ops · E_op · (1 - reversible_fraction)

        At ``reversible_fraction = 1.0`` all computation is information-
        preserving and the thermodynamic heat floor vanishes entirely.
    """

    def __init__(
        self,
        *,
        dn_dT: float = 1.3e-5,  # sapphire
        interaction_length_nm: float = 50_000.0,  # effective cascade path (~4 gates)
        wavelength_nm: float = 721.0,  # λ = hc/E for 1.72 eV
        base_refractive_index: float = 1.76,
        energy_per_op_aJ: float = 0.5,
        n_ops_per_cycle: int = 10_000_000_000,  # 10 billion ops/cycle (realistic chip)
        chip_surface_area_cm2: float = 960.0,
        thermal_resistance_K_W: float = 0.1,
        min_extinction_ratio_dB: float = 3.0,
        ambient_temperature_K: float = T_ROOM,
        reversible_fraction: float = 0.0,
    ) -> None:
        if not 0.0 <= reversible_fraction <= 1.0:
            raise ValueError("reversible_fraction must be in [0, 1]")
        self.dn_dT = dn_dT
        self.interaction_length_m = interaction_length_nm * NM_TO_M
        self.wavelength_m = wavelength_nm * NM_TO_M
        self.base_n = base_refractive_index
        self.energy_per_op_J = energy_per_op_aJ * 1e-18
        self.n_ops_per_cycle = n_ops_per_cycle
        self.chip_area_m2 = chip_surface_area_cm2 * 1e-4
        self.thermal_resistance = thermal_resistance_K_W
        self.min_er_dB = min_extinction_ratio_dB
        self.T_amb = ambient_temperature_K
        self.reversible_fraction = reversible_fraction

        # Stefan-Boltzmann for radiative cooling
        self._sigma = 5.670374419e-8
        self._emissivity = 0.9

    def _equilibrium_delta_T(self, clock_GHz: float) -> float:
        """Find ΔT where dissipated power = radiative + conductive cooling.

        P_diss = f_clk · N_ops · E_op · (1 - reversible_fraction)
        P_cool = ε·σ·A·((T_amb+ΔT)⁴ - T_amb⁴) + ΔT/R_th

        Reversible gates erase no bits and therefore contribute zero
        Landauer heat.  Only the irreversible fraction dissipates.

        Solved by bisection.
        """
        irreversible = 1.0 - self.reversible_fraction
        P_diss = clock_GHz * 1e9 * self.n_ops_per_cycle * self.energy_per_op_J * irreversible

        dT_lo = 0.0
        dT_hi = 2000.0

        for _ in range(200):
            dT_mid = (dT_lo + dT_hi) / 2.0
            T_chip = self.T_amb + dT_mid
            P_rad = (
                self._emissivity
                * self._sigma
                * self.chip_area_m2
                * (T_chip ** 4 - self.T_amb ** 4)
            )
            P_cond = dT_mid / self.thermal_resistance
            P_cool = P_rad + P_cond

            if P_cool < P_diss:
                dT_lo = dT_mid
            else:
                dT_hi = dT_mid

            if dT_hi - dT_lo < 0.001:
                break

        return (dT_lo + dT_hi) / 2.0

    def _phase_shift(self, delta_n: float) -> float:
        """Thermo-optic phase shift in the interaction zone [rad].

        Δφ = (2π / λ) · Δn · L
        """
        return 2.0 * np.pi * delta_n * self.interaction_length_m / self.wavelength_m

    def _extinction_ratio_dB(self, phase_shift_rad: float) -> float:
        """Extinction ratio of an MZI gate with parasitic phase error.

        For an ideal Mach-Zehnder interferometer with a thermal phase error Δφ:
        ER = 10 · log₁₀[ cos²(Δφ) / sin²(Δφ) ]

        Returns ER in dB (clamped to avoid singularities).
        """
        cos2 = np.cos(phase_shift_rad) ** 2
        sin2 = np.sin(phase_shift_rad) ** 2
        # Clamp to avoid log(0)
        sin2 = max(sin2, 1e-30)
        cos2 = max(cos2, 1e-30)
        return float(10.0 * np.log10(cos2 / sin2))

    def sweep(
        self,
        clock_min_GHz: float = 1.0,
        clock_max_GHz: float = 500.0,
        n_points: int = 200,
    ) -> BreakdownResult:
        """Sweep clock rate and find the breakdown point.

        Returns a BreakdownResult with the critical clock rate and full sweep data.
        """
        clock_rates = np.linspace(clock_min_GHz, clock_max_GHz, n_points).tolist()
        delta_Ts: list[float] = []
        ers: list[float] = []
        phase_shifts: list[float] = []

        breakdown_idx: int | None = None

        for i, f_clk in enumerate(clock_rates):
            dT = self._equilibrium_delta_T(f_clk)
            dn = self.dn_dT * dT
            dphi = self._phase_shift(dn)
            er = self._extinction_ratio_dB(dphi)

            delta_Ts.append(dT)
            phase_shifts.append(dphi)
            ers.append(er)

            if breakdown_idx is None and er < self.min_er_dB:
                breakdown_idx = i

        if breakdown_idx is None:
            # No breakdown found in range
            breakdown_idx = len(clock_rates) - 1

        safe_idx = max(0, breakdown_idx - 1)

        return BreakdownResult(
            breakdown_clock_GHz=clock_rates[breakdown_idx],
            max_safe_clock_GHz=clock_rates[safe_idx],
            breakdown_delta_T_K=delta_Ts[breakdown_idx],
            breakdown_delta_n=self.dn_dT * delta_Ts[breakdown_idx],
            breakdown_phase_shift_rad=phase_shifts[breakdown_idx],
            extinction_ratio_dB_at_breakdown=ers[breakdown_idx],
            sweep_clock_rates_GHz=clock_rates,
            sweep_delta_T_K=delta_Ts,
            sweep_extinction_ratio_dB=ers,
            sweep_phase_shift_rad=phase_shifts,
        )

    def coupled_verification(
        self,
        clock_GHz: float,
        n_coupling_cycles: int = 20,
    ) -> tuple[BreakdownResult, list[CoupledStepResult]]:
        """Run both analytical sweep AND full coupled FDTD simulation.

        Verifies the analytical model against the numerical coupled solver
        at a specific clock rate. Returns both the sweep result and the
        coupled simulation history.
        """
        # Analytical sweep
        result = self.sweep(
            clock_min_GHz=max(0.1, clock_GHz * 0.5),
            clock_max_GHz=clock_GHz * 1.5,
            n_points=50,
        )

        # Coupled numerical verification
        dT_eq = self._equilibrium_delta_T(clock_GHz)
        ops_per_step = int(
            clock_GHz * 1e9
            * self.energy_per_op_J
            / (K_BOLTZMANN * self.T_amb * np.log(2))
        )

        coupled_cfg = CoupledConfig(
            n_ops_per_cycle=max(1, ops_per_step),
        )
        sim = CoupledThermoFDTDSimulator(coupled_cfg)
        sim.initialise()
        history = sim.run(n_coupling_cycles)

        return result, history
