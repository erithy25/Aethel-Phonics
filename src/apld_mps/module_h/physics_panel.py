"""Sektion 2 — Physics & Material Control panel.

Provides interactive control over substrate selection, TMD layer tuning,
and laser pulse generation. Acts as the bridge between user interface
controls and the simulation engine parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..module_a.substrate import SubstrateDatabase, SubstrateMaterial
from ..module_a.tmd_layer import TMDMonolayer, ExcitonProperties, WSe2, MoSe2
from ..module_c.laser_source import LaserSource, LaserPulse, PolarisationState


@dataclass
class SubstrateSelection:
    """Current substrate configuration state.

    Attributes:
        material_name: Selected substrate from the database.
        refractive_index_at_720nm: n(720 nm) for quick display.
        thermal_conductivity: κ [W/(m·K)].
        dispersion_curve: (wavelengths_nm, n_values) for plotting.
    """

    material_name: str
    refractive_index_at_720nm: float
    thermal_conductivity: float
    dispersion_curve: tuple[np.ndarray, np.ndarray]


@dataclass
class TMDConfiguration:
    """Current TMD layer tuning state.

    Attributes:
        material_name: TMD material identifier.
        resonance_energy_eV: Current exciton resonance [eV].
        binding_energy_eV: Current binding energy [eV].
        oscillator_strength: Current oscillator strength.
        effective_mass_ratio: Current effective mass ratio.
    """

    material_name: str
    resonance_energy_eV: float
    binding_energy_eV: float
    oscillator_strength: float
    effective_mass_ratio: float


@dataclass
class PulseGeneratorState:
    """Current state of the laser pulse generator console.

    Attributes:
        n_pulses: Number of configured pulses.
        pulses: Configured pulse parameters.
        total_energy_aJ: Total pulse energy [aJ].
    """

    n_pulses: int
    pulses: list[dict]
    total_energy_aJ: float


class PhysicsControlPanel:
    """Interactive control for physics parameters and material configuration.

    Integrates SubstrateDatabase, TMD layer properties, and LaserSource
    into a unified control surface.
    """

    def __init__(self) -> None:
        self._substrate_db = SubstrateDatabase()
        self._current_substrate: SubstrateMaterial | None = None
        self._current_tmd: TMDMonolayer = WSe2
        self._laser_source = LaserSource()
        self._available_tmds: dict[str, TMDMonolayer] = {
            "WSe2": WSe2,
            "MoSe2": MoSe2,
        }

        # Default substrate
        self.select_substrate("Sapphire (Al₂O₃)")

    # --- Substrate control ---

    @property
    def available_substrates(self) -> list[str]:
        return self._substrate_db.list_materials()

    def select_substrate(self, name: str) -> SubstrateSelection:
        """Select a substrate material and compute display properties."""
        self._current_substrate = self._substrate_db.get(name)
        wavelengths = np.linspace(400, 1200, 200)
        n_values = np.array([float(self._current_substrate.n(w)) for w in wavelengths])

        return SubstrateSelection(
            material_name=name,
            refractive_index_at_720nm=float(self._current_substrate.n(720.0)),
            thermal_conductivity=self._current_substrate.thermal_conductivity,
            dispersion_curve=(wavelengths, n_values),
        )

    @property
    def current_substrate(self) -> SubstrateSelection | None:
        if self._current_substrate is None:
            return None
        wavelengths = np.linspace(400, 1200, 200)
        n_values = np.array([float(self._current_substrate.n(w)) for w in wavelengths])
        return SubstrateSelection(
            material_name=self._current_substrate.name,
            refractive_index_at_720nm=float(self._current_substrate.n(720.0)),
            thermal_conductivity=self._current_substrate.thermal_conductivity,
            dispersion_curve=(wavelengths, n_values),
        )

    # --- TMD layer control ---

    @property
    def available_tmd_materials(self) -> list[str]:
        return list(self._available_tmds.keys())

    def select_tmd(self, name: str) -> TMDConfiguration:
        """Select a pre-defined TMD material."""
        if name not in self._available_tmds:
            raise KeyError(f"TMD '{name}' not found. Available: {list(self._available_tmds)}")
        self._current_tmd = self._available_tmds[name]
        return self._tmd_config()

    def tune_tmd(
        self,
        resonance_energy_eV: float | None = None,
        binding_energy_eV: float | None = None,
        oscillator_strength: float | None = None,
        effective_mass_ratio: float | None = None,
    ) -> TMDConfiguration:
        """Fine-tune TMD exciton parameters via sliders."""
        ex = self._current_tmd.exciton
        self._current_tmd = TMDMonolayer(
            name=self._current_tmd.name + " (tuned)",
            exciton=ExcitonProperties(
                binding_energy_eV=binding_energy_eV or ex.binding_energy_eV,
                oscillator_strength=oscillator_strength or ex.oscillator_strength,
                resonance_energy_eV=resonance_energy_eV or ex.resonance_energy_eV,
                effective_mass_ratio=effective_mass_ratio or ex.effective_mass_ratio,
            ),
            gamma_nr=self._current_tmd.gamma_nr,
            gamma_r=self._current_tmd.gamma_r,
        )
        return self._tmd_config()

    @property
    def current_tmd(self) -> TMDConfiguration:
        return self._tmd_config()

    def _tmd_config(self) -> TMDConfiguration:
        ex = self._current_tmd.exciton
        return TMDConfiguration(
            material_name=self._current_tmd.name,
            resonance_energy_eV=ex.resonance_energy_eV,
            binding_energy_eV=ex.binding_energy_eV,
            oscillator_strength=ex.oscillator_strength,
            effective_mass_ratio=ex.effective_mass_ratio,
        )

    # --- Laser pulse generator ---

    def add_pulse(
        self,
        energy_eV: float,
        intensity_W_cm2: float = 1e4,
        duration_fs: float = 100.0,
        delay_ps: float = 0.0,
        position_nm: tuple[float, float] = (0.0, 0.0),
        polarisation: PolarisationState = PolarisationState.LINEAR_H,
    ) -> PulseGeneratorState:
        """Add a laser pulse to the generator."""
        pulse = LaserPulse(
            energy_eV=energy_eV,
            intensity_W_cm2=intensity_W_cm2,
            duration_fs=duration_fs,
            delay_ps=delay_ps,
            position_nm=position_nm,
            polarisation=polarisation,
        )
        self._laser_source.add_pulse(pulse)
        return self._pulse_state()

    def clear_pulses(self) -> PulseGeneratorState:
        """Clear all configured pulses."""
        self._laser_source = LaserSource()
        return self._pulse_state()

    @property
    def laser_source(self) -> LaserSource:
        return self._laser_source

    @property
    def pulse_generator_state(self) -> PulseGeneratorState:
        return self._pulse_state()

    def _pulse_state(self) -> PulseGeneratorState:
        pulses = []
        total_energy = 0.0
        for p in self._laser_source.pulses:
            # Approximate energy: I × A × τ  (Gaussian spot)
            spot_area_m2 = np.pi * (p.waist_nm * 1e-9) ** 2
            energy_J = p.intensity_W_cm2 * 1e4 * spot_area_m2 * p.duration_fs * 1e-15
            energy_aJ = energy_J * 1e18
            total_energy += energy_aJ
            pulses.append({
                "energy_eV": p.energy_eV,
                "wavelength_nm": p.wavelength_nm,
                "intensity_W_cm2": p.intensity_W_cm2,
                "duration_fs": p.duration_fs,
                "delay_ps": p.delay_ps,
                "position_nm": list(p.position_nm),
                "polarisation": p.polarisation.name,
                "energy_aJ": energy_aJ,
            })
        return PulseGeneratorState(
            n_pulses=len(pulses),
            pulses=pulses,
            total_energy_aJ=total_energy,
        )
