"""Tests for Module H: Command & Control Dashboard."""

import numpy as np
import pytest

from apld_mps.module_h.viewport import (
    DigitalTwinViewport,
    RenderMode,
    CameraState,
    LayerVisibility,
    EmissiveWaveguide,
    PhaseJitterAlarm,
    PhaseJitterAlarmLevel,
)
from apld_mps.module_h.physics_panel import PhysicsControlPanel
from apld_mps.module_h.kreislauf_monitor import (
    KreislaufMonitor,
    ThermalAlertLevel,
)
from apld_mps.module_h.io_panel import IOPanel, BottleneckType
from apld_mps.module_h.resilience_center import (
    ResilienceCenter,
    ResilienceStatus,
    CosmicImpactLog,
)
from apld_mps.module_h.rl_status import RLOptimizerPanel, OptimiserState
from apld_mps.module_h.dashboard import AethelDashboard, DashboardMode

from apld_mps.module_c.laser_source import LaserSource, LaserPulse
from apld_mps.module_d.mapper_3d import AutoMapper3D, VolumetricLayout
from apld_mps.module_f.heat_optics import ThermoOpticConfig
from apld_mps.module_g.phase_stability import VibrationProfile


# ---------------------------------------------------------------------------
# Sektion 1: Digital Twin Viewport
# ---------------------------------------------------------------------------

class TestDigitalTwinViewport:
    def test_empty_viewport(self):
        vp = DigitalTwinViewport()
        assert vp.gate_count() == 0
        assert vp.wire_count() == 0
        assert vp.layer_count() == 0

    def test_load_layout(self):
        mapper = AutoMapper3D(volume_nm=(1e6, 1e6, 1e6))
        layout = mapper.map_half_adder_3d(n_episodes=10)
        vp = DigitalTwinViewport(layout)
        assert vp.gate_count() == 2
        assert vp.wire_count() == 0  # half adder has no internal wires (XOR/AND from ext)
        assert vp.layer_count() >= 1

    def test_snapshot_structure(self):
        mapper = AutoMapper3D(volume_nm=(1e6, 1e6, 1e6))
        layout = mapper.map_full_adder_3d(n_episodes=10)
        vp = DigitalTwinViewport(layout)
        snap = vp.get_snapshot()
        assert len(snap.gates) == 5
        assert isinstance(snap.camera, CameraState)

    def test_toggle_layer(self):
        vp = DigitalTwinViewport()
        vp.visibility.visible_layers = {0, 1, 2}
        vp.toggle_layer(1)
        assert 1 not in vp.visibility.visible_layers
        vp.toggle_layer(1)
        assert 1 in vp.visibility.visible_layers

    def test_field_overlay(self):
        vp = DigitalTwinViewport()
        density = np.random.rand(10, 10, 10)
        vp.update_field_overlay(density)
        snap = vp.get_snapshot()
        assert "xy" in snap.field_slices

    def test_thermal_overlay(self):
        vp = DigitalTwinViewport()
        vp.visibility.show_thermal_overlay = True
        temp = np.full((10, 10, 10), 300.0)
        vp.update_thermal_overlay(temp)
        snap = vp.get_snapshot()
        assert snap.thermal_slice is not None

    def test_render_mode(self):
        vp = DigitalTwinViewport()
        vp.set_render_mode(RenderMode.XRAY)
        assert vp.visibility.render_mode == RenderMode.XRAY

    def test_capture_frame(self):
        vp = DigitalTwinViewport()
        density = np.random.rand(5, 5, 4)
        vp.update_field_overlay(density)
        vp.capture_frame(1.0)
        vp.capture_frame(2.0)
        assert vp.frame_count == 2

    def test_gate_render_dict(self):
        mapper = AutoMapper3D(volume_nm=(1e6, 1e6, 1e6))
        layout = mapper.map_half_adder_3d(n_episodes=10)
        vp = DigitalTwinViewport(layout)
        snap = vp.get_snapshot()
        gate = snap.gates[0]
        assert "gate_id" in gate
        assert "gate_type" in gate
        assert "origin_nm" in gate


# ---------------------------------------------------------------------------
# Sektion 2: Physics & Material Control
# ---------------------------------------------------------------------------

class TestPhysicsControlPanel:
    def test_default_substrate(self):
        panel = PhysicsControlPanel()
        sub = panel.current_substrate
        assert sub is not None
        assert "Sapphire" in sub.material_name
        assert sub.refractive_index_at_720nm > 1.7

    def test_available_substrates(self):
        panel = PhysicsControlPanel()
        assert len(panel.available_substrates) >= 1

    def test_select_substrate(self):
        panel = PhysicsControlPanel()
        sel = panel.select_substrate("Sapphire (Al₂O₃)")
        assert sel.thermal_conductivity == pytest.approx(35.0)
        wl, n = sel.dispersion_curve
        assert len(wl) == 200
        assert np.all(n > 1.5)

    def test_default_tmd(self):
        panel = PhysicsControlPanel()
        tmd = panel.current_tmd
        assert tmd.material_name == "WSe2"
        assert tmd.resonance_energy_eV == pytest.approx(1.72)

    def test_select_tmd(self):
        panel = PhysicsControlPanel()
        tmd = panel.select_tmd("MoSe2")
        assert tmd.material_name == "MoSe2"
        assert tmd.resonance_energy_eV == pytest.approx(1.66)

    def test_tune_tmd(self):
        panel = PhysicsControlPanel()
        tmd = panel.tune_tmd(resonance_energy_eV=1.80)
        assert tmd.resonance_energy_eV == pytest.approx(1.80)
        assert "(tuned)" in tmd.material_name

    def test_add_pulse(self):
        panel = PhysicsControlPanel()
        state = panel.add_pulse(energy_eV=1.72, intensity_W_cm2=1e4)
        assert state.n_pulses == 1
        assert state.total_energy_aJ > 0

    def test_clear_pulses(self):
        panel = PhysicsControlPanel()
        panel.add_pulse(energy_eV=1.72)
        panel.add_pulse(energy_eV=1.72)
        state = panel.clear_pulses()
        assert state.n_pulses == 0

    def test_pulse_details(self):
        panel = PhysicsControlPanel()
        panel.add_pulse(energy_eV=1.72, duration_fs=200.0)
        state = panel.pulse_generator_state
        assert state.pulses[0]["duration_fs"] == 200.0
        assert state.pulses[0]["energy_eV"] == 1.72


# ---------------------------------------------------------------------------
# Sektion 3: Kreislauf Monitor
# ---------------------------------------------------------------------------

class TestKreislaufMonitor:
    def test_initialise(self):
        cfg = ThermoOpticConfig(
            grid_size_nm=(500, 500, 200),
            grid_spacing_nm=100,
        )
        mon = KreislaufMonitor(thermo_config=cfg)
        mon.initialise()
        assert mon.is_initialised

    def test_thermal_heatmap_baseline(self):
        cfg = ThermoOpticConfig(
            grid_size_nm=(500, 500, 200),
            grid_spacing_nm=100,
        )
        mon = KreislaufMonitor(thermo_config=cfg)
        mon.initialise()
        hm = mon.get_thermal_heatmap()
        assert hm.alert_level == ThermalAlertLevel.NOMINAL
        assert hm.peak_temperature_K == pytest.approx(300.0)

    def test_heat_injection_raises_alert(self):
        cfg = ThermoOpticConfig(
            grid_size_nm=(500, 500, 200),
            grid_spacing_nm=100,
            dt_ps=100.0,
        )
        mon = KreislaufMonitor(
            thermo_config=cfg,
            warning_temperature_K=301.0,
        )
        mon.initialise()
        intensity = np.ones(cfg.grid_shape) * 1e14
        mon.inject_heat(intensity_field=intensity, n_operations=1_000_000)
        mon.advance_thermal(200)
        hm = mon.get_thermal_heatmap()
        assert hm.peak_temperature_K > 300.0

    def test_tpv_status(self):
        mon = KreislaufMonitor()
        mon.initialise()
        mon.set_clock_rate(100.0)
        status = mon.get_tpv_status()
        assert status.equilibrium_temperature_K >= 300.0

    def test_clock_governor(self):
        mon = KreislaufMonitor()
        mon.initialise()
        result = mon.run_stability_analysis(max_temperature_K=350.0, resolution_GHz=10.0)
        assert result.max_clock_rate_GHz > 0
        gov = mon.set_clock_rate(result.max_clock_rate_GHz / 2)
        assert gov.headroom_percent > 0

    def test_eps_r_correction(self):
        cfg = ThermoOpticConfig(
            grid_size_nm=(500, 500, 200),
            grid_spacing_nm=100,
        )
        mon = KreislaufMonitor(thermo_config=cfg)
        mon.initialise()
        correction = mon.eps_r_correction
        assert correction.shape == cfg.grid_shape


# ---------------------------------------------------------------------------
# Sektion 4: I/O Panel
# ---------------------------------------------------------------------------

class TestIOPanel:
    def test_add_channels(self):
        panel = IOPanel()
        panel.add_input_channel("fibre-1", 100.0, 400.0)
        panel.add_output_channel("fibre-2", 200.0, 400.0)
        data = panel.get_throughput_data()
        assert data.input_Gbps == pytest.approx(100.0)
        assert data.output_Gbps == pytest.approx(200.0)
        assert len(data.input_channels) == 1
        assert len(data.output_channels) == 1

    def test_bottleneck_radar(self):
        panel = IOPanel(network_bandwidth_Gbps=50.0)
        panel.add_input_channel("in", 10.0, 100.0)
        radar = panel.get_bottleneck_radar()
        assert radar.bottleneck_type == BottleneckType.NETWORK_BOUND

    def test_fibre_interface(self):
        panel = IOPanel()
        result = panel.add_fibre_interface("port-1")
        assert result.coupling_efficiency > 0
        status = panel.get_coupling_status()
        assert len(status.interfaces) == 1
        assert status.worst_insertion_loss_dB > 0

    def test_snapshot_recording(self):
        panel = IOPanel()
        panel.add_input_channel("in", 50.0, 100.0)
        panel.record_snapshot()
        data = panel.get_throughput_data()
        assert len(data.history) == 1

    def test_petabit_conversion(self):
        panel = IOPanel()
        panel.add_input_channel("big", 1_000_000.0, 2_000_000.0)
        data = panel.get_throughput_data()
        assert data.input_Pbps == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Sektion 5: Resilience Center
# ---------------------------------------------------------------------------

class TestResilienceCenter:
    def test_cosmic_ray_simulation(self):
        rc = ResilienceCenter(
            volume_nm=(1e6, 1e6, 1e6),
            router_resolution_nm=200_000,
        )
        entries = rc.simulate_cosmic_rays(duration_s=1.0, seed=42)
        assert rc.total_events >= 0
        ticker = rc.get_ticker()
        assert len(ticker) == len(entries)

    def test_self_healing_status(self):
        rc = ResilienceCenter(
            volume_nm=(1e6, 1e6, 1e6),
            router_resolution_nm=200_000,
        )
        status = rc.get_self_healing_status()
        assert status.disrupted_fraction == 0.0
        assert status.availability_slice.ndim == 2

    def test_vibration_analysis(self):
        rc = ResilienceCenter()
        profile = VibrationProfile(
            frequencies_Hz=np.array([50.0, 120.0]),
            amplitudes_nm=np.array([0.5, 0.2]),
        )
        data = rc.analyse_vibration(profile)
        assert data.rms_phase_jitter_rad > 0
        assert data.stability_report is not None

    def test_default_vibration_data(self):
        rc = ResilienceCenter()
        data = rc.get_vibration_data()
        assert data.rms_phase_jitter_rad == 0.0
        assert data.stability_report is None

    def test_overall_status_healthy(self):
        rc = ResilienceCenter(volume_nm=(1e6, 1e6, 1e6), router_resolution_nm=200_000)
        assert rc.overall_status() == ResilienceStatus.HEALTHY

    def test_reroute_succeeds_clean_grid(self):
        rc = ResilienceCenter(
            volume_nm=(10_000, 10_000, 10_000),
            router_resolution_nm=2_000,
        )
        result = rc.attempt_reroute(
            start_nm=(0, 0, 0),
            end_nm=(9_000, 9_000, 9_000),
        )
        assert result is True
        status = rc.get_self_healing_status()
        assert status.active_reroutes == 1

    def test_error_rate(self):
        rc = ResilienceCenter(volume_nm=(1e6, 1e6, 1e6), router_resolution_nm=200_000)
        rc.simulate_cosmic_rays(duration_s=10.0, seed=99)
        rate = rc.error_rate(total_operations=1_000_000)
        assert 0 <= rate <= 1


# ---------------------------------------------------------------------------
# Sektion 6: RL Optimizer Status
# ---------------------------------------------------------------------------

class TestRLOptimizerPanel:
    def test_initial_state(self):
        panel = RLOptimizerPanel(volume_nm=(1e6, 1e6, 1e6))
        assert panel.state == OptimiserState.IDLE
        progress = panel.get_progress()
        assert progress.current_episode == 0

    def test_run_half_adder(self):
        panel = RLOptimizerPanel(volume_nm=(1e6, 1e6, 1e6))
        layout = panel.run_half_adder(n_episodes=20)
        assert layout.gate_count == 2
        assert panel.state == OptimiserState.CONVERGED
        progress = panel.get_progress()
        assert progress.current_episode == 20
        assert len(progress.reward_history) == 20

    def test_run_full_adder(self):
        panel = RLOptimizerPanel(volume_nm=(1e6, 1e6, 1e6))
        layout = panel.run_full_adder(n_episodes=30)
        assert layout.gate_count == 5
        assert panel.state == OptimiserState.CONVERGED

    def test_training_log(self):
        panel = RLOptimizerPanel(volume_nm=(1e6, 1e6, 1e6))
        panel.run_half_adder(n_episodes=15)
        log = panel.training_log
        assert len(log) == 15
        for entry in log:
            assert entry.episode > 0
            assert entry.wire_length_nm >= 0

    def test_last_log_entries(self):
        panel = RLOptimizerPanel(volume_nm=(1e6, 1e6, 1e6))
        panel.run_half_adder(n_episodes=25)
        last = panel.get_last_log_entries(5)
        assert len(last) == 5
        assert last[-1].episode == 25

    def test_convergence_rate(self):
        panel = RLOptimizerPanel(volume_nm=(1e6, 1e6, 1e6))
        panel.run_full_adder(n_episodes=50)
        progress = panel.get_progress()
        # convergence_rate is a float — could be positive or negative
        assert isinstance(progress.convergence_rate, float)


# ---------------------------------------------------------------------------
# Unified Dashboard Orchestrator
# ---------------------------------------------------------------------------

class TestAethelDashboard:
    def test_create_and_initialise(self):
        cfg = ThermoOpticConfig(
            grid_size_nm=(500, 500, 200),
            grid_spacing_nm=100,
        )
        dash = AethelDashboard(thermo_config=cfg)
        dash.initialise()
        assert dash.is_initialised
        assert dash.mode == DashboardMode.DESIGN

    def test_full_state_snapshot(self):
        cfg = ThermoOpticConfig(
            grid_size_nm=(500, 500, 200),
            grid_spacing_nm=100,
        )
        dash = AethelDashboard(thermo_config=cfg)
        dash.initialise()
        state = dash.get_state()
        assert state.mode == DashboardMode.DESIGN
        assert state.substrate is not None
        assert state.tmd.material_name == "WSe2"
        assert state.thermal is not None
        assert state.resilience == ResilienceStatus.HEALTHY

    def test_load_layout(self):
        cfg = ThermoOpticConfig(
            grid_size_nm=(500, 500, 200),
            grid_spacing_nm=100,
        )
        dash = AethelDashboard(thermo_config=cfg)
        dash.initialise()

        mapper = AutoMapper3D(volume_nm=(1e6, 1e6, 1e6))
        layout = mapper.map_half_adder_3d(n_episodes=10)
        dash.load_layout(layout)

        state = dash.get_state()
        assert len(state.viewport.gates) == 2

    def test_simulation_step(self):
        cfg = ThermoOpticConfig(
            grid_size_nm=(500, 500, 200),
            grid_spacing_nm=100,
        )
        dash = AethelDashboard(thermo_config=cfg)
        dash.initialise()

        density = np.random.rand(5, 5, 2) * 1e6
        dash.simulation_step(density_3d=density, time_ps=1.0, n_operations=100)
        assert dash.mode == DashboardMode.SIMULATION
        assert dash.viewport.frame_count == 1

    def test_stress_test_mode(self):
        cfg = ThermoOpticConfig(
            grid_size_nm=(500, 500, 200),
            grid_spacing_nm=100,
        )
        dash = AethelDashboard(
            thermo_config=cfg,
            volume_nm=(1e6, 1e6, 1e6),
        )
        dash.initialise()

        profile = VibrationProfile(
            frequencies_Hz=np.array([100.0]),
            amplitudes_nm=np.array([0.1]),
        )
        dash.run_stress_test(cosmic_duration_s=0.1, vibration_profile=profile)
        assert dash.mode == DashboardMode.STRESS_TEST

        state = dash.get_state()
        assert state.vibration.rms_phase_jitter_rad > 0

    def test_set_mode(self):
        dash = AethelDashboard()
        dash.set_mode(DashboardMode.ANALYSIS)
        assert dash.mode == DashboardMode.ANALYSIS

    def test_physics_integration(self):
        dash = AethelDashboard()
        dash.physics.select_tmd("MoSe2")
        assert dash.physics.current_tmd.material_name == "MoSe2"

    def test_io_integration(self):
        dash = AethelDashboard()
        dash.io.add_input_channel("port-1", 100.0, 400.0)
        dash.io.add_fibre_interface("fibre-1")
        state = dash.get_state()
        assert state.throughput.input_Gbps == pytest.approx(100.0)
        assert len(state.coupling.interfaces) == 1


# ---------------------------------------------------------------------------
# Laser → Thermal coupling (Module C ↔ F ↔ H)
# ---------------------------------------------------------------------------

class TestLaserThermalCoupling:
    """Verify that laser pulse energy flows into the thermal solver."""

    def _make_dashboard(self) -> AethelDashboard:
        cfg = ThermoOpticConfig(
            grid_size_nm=(500, 500, 200),
            grid_spacing_nm=100,
            dt_ps=10.0,
        )
        dash = AethelDashboard(thermo_config=cfg)
        dash.initialise()
        return dash

    def _make_laser(self, n_pulses: int = 2) -> LaserSource:
        src = LaserSource()
        for _ in range(n_pulses):
            src.add_pulse(LaserPulse(
                energy_eV=1.72,
                intensity_W_cm2=1e4,
                duration_fs=100.0,
                waist_nm=500.0,
            ))
        return src

    def test_no_laser_no_heating(self):
        """Without laser source, temperature stays at ambient."""
        dash = self._make_dashboard()
        dash.simulation_step(time_ps=1.0)
        assert dash.kreislauf.temperature_3d.max() == pytest.approx(300.0)

    def test_laser_injects_heat(self):
        """With a laser source and clock rate, temperature must rise."""
        dash = self._make_dashboard()
        laser = self._make_laser()
        # Run several steps at a moderate clock rate
        for t in range(50):
            dash.simulation_step(
                time_ps=float(t),
                laser_source=laser,
                f_clk_GHz=100.0,
            )
        peak_T = dash.kreislauf.temperature_3d.max()
        assert peak_T > 300.0, "Laser energy must heat the chip"

    def test_higher_clock_more_heat(self):
        """Doubling f_clk must produce more heating."""
        dash_lo = self._make_dashboard()
        dash_hi = self._make_dashboard()
        laser = self._make_laser()
        for t in range(30):
            dash_lo.simulation_step(
                time_ps=float(t), laser_source=laser, f_clk_GHz=50.0,
            )
            dash_hi.simulation_step(
                time_ps=float(t), laser_source=laser, f_clk_GHz=200.0,
            )
        assert dash_hi.kreislauf.temperature_3d.max() > dash_lo.kreislauf.temperature_3d.max()

    def test_more_pulses_more_heat(self):
        """More active pulses → more total energy → hotter."""
        dash_1 = self._make_dashboard()
        dash_4 = self._make_dashboard()
        laser_1 = self._make_laser(n_pulses=1)
        laser_4 = self._make_laser(n_pulses=4)
        for t in range(30):
            dash_1.simulation_step(
                time_ps=float(t), laser_source=laser_1, f_clk_GHz=100.0,
            )
            dash_4.simulation_step(
                time_ps=float(t), laser_source=laser_4, f_clk_GHz=100.0,
            )
        assert dash_4.kreislauf.temperature_3d.max() > dash_1.kreislauf.temperature_3d.max()

    def test_zero_clock_no_heating(self):
        """Laser present but f_clk=0 must not inject heat."""
        dash = self._make_dashboard()
        laser = self._make_laser()
        for t in range(20):
            dash.simulation_step(
                time_ps=float(t), laser_source=laser, f_clk_GHz=0.0,
            )
        assert dash.kreislauf.temperature_3d.max() == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# Cosmic Ray Intensity Slider & Impact Log (Module G ↔ H)
# ---------------------------------------------------------------------------

class TestCosmicRayIntegration:
    """Verify cosmic ray slider, per-step generation, and gate-impact log."""

    def _make_dashboard_with_layout(self) -> AethelDashboard:
        cfg = ThermoOpticConfig(
            grid_size_nm=(500, 500, 200),
            grid_spacing_nm=100,
        )
        dash = AethelDashboard(
            thermo_config=cfg,
            volume_nm=(1_000_000.0, 1_000_000.0, 1_000_000.0),
        )
        dash.initialise()

        mapper = AutoMapper3D(volume_nm=(1_000_000.0, 1_000_000.0, 1_000_000.0))
        layout = mapper.map_half_adder_3d(n_episodes=5)
        dash.load_layout(layout)
        return dash

    def test_intensity_slider_default(self):
        """Default intensity multiplier is 1.0."""
        dash = self._make_dashboard_with_layout()
        assert dash.resilience.intensity_multiplier == 1.0

    def test_set_intensity_slider(self):
        """Dashboard.set_cosmic_intensity forwards to ResilienceCenter."""
        dash = self._make_dashboard_with_layout()
        dash.set_cosmic_intensity(1000.0)
        assert dash.resilience.intensity_multiplier == 1000.0

    def test_high_intensity_produces_events(self):
        """With a very high multiplier, events must appear during steps."""
        dash = self._make_dashboard_with_layout()
        # area ≈ 0.01 cm², dt=1000 ps=1e-9 s → expected ≈ mult × 1e-11
        # mult=1e13 → ~100 events per step
        dash.set_cosmic_intensity(1e13)
        all_logs: list[CosmicImpactLog] = []
        for t in range(5):
            logs = dash.simulation_step(time_ps=float(t), dt_ps=1000.0)
            all_logs.extend(logs)
        assert len(all_logs) > 0, "High intensity must produce impacts"

    def test_impact_log_has_gate_id(self):
        """Impact log entries must resolve to nearest gate_id."""
        dash = self._make_dashboard_with_layout()
        dash.set_cosmic_intensity(1e13)
        all_logs: list[CosmicImpactLog] = []
        for t in range(5):
            all_logs.extend(dash.simulation_step(time_ps=float(t), dt_ps=1000.0))
        assert len(all_logs) > 0
        for log in all_logs:
            assert log.gate_id != "UNKNOWN"
            assert "Einschlag detektiert bei Gatter" in log.message

    def test_impact_log_in_dashboard_state(self):
        """DashboardState must contain cosmic_impact_log."""
        dash = self._make_dashboard_with_layout()
        dash.set_cosmic_intensity(1e12)
        for t in range(5):
            dash.simulation_step(time_ps=float(t), dt_ps=1.0)
        state = dash.get_state()
        assert hasattr(state, "cosmic_impact_log")
        assert isinstance(state.cosmic_impact_log, list)

    def test_natural_flux_few_events(self):
        """At natural flux (multiplier=1), almost no events in tiny volume."""
        dash = self._make_dashboard_with_layout()
        all_logs: list[CosmicImpactLog] = []
        for t in range(10):
            all_logs.extend(dash.simulation_step(time_ps=float(t), dt_ps=1.0))
        # 1 cm² area, 1e-12 s per step × 10 steps → ~1e-11 expected events
        assert len(all_logs) == 0

    def test_simulation_step_returns_logs(self):
        """simulation_step must return a list (even if empty)."""
        dash = self._make_dashboard_with_layout()
        result = dash.simulation_step(time_ps=0.0, dt_ps=1.0)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Emissive Waveguide Glow (Laser → 3D Viewport)
# ---------------------------------------------------------------------------

class TestEmissiveWaveguideGlow:
    """Verify that configuring LaserPulses makes waveguides glow."""

    def _make_viewport_with_layout(self) -> DigitalTwinViewport:
        mapper = AutoMapper3D(volume_nm=(1e6, 1e6, 1e6))
        layout = mapper.map_half_adder_3d(n_episodes=10)
        return DigitalTwinViewport(layout)

    def test_no_pulses_no_glow(self):
        """Without laser pulses, no emissive waveguides."""
        vp = DigitalTwinViewport()
        result = vp.configure_laser_pulses([])
        assert result == []
        assert vp.emissive_waveguides == []

    def test_single_pulse_creates_emissive(self):
        """A single LaserPulse must produce one EmissiveWaveguide."""
        vp = self._make_viewport_with_layout()
        pulse = LaserPulse(energy_eV=1.72, intensity_W_cm2=1e4, waist_nm=500.0)
        result = vp.configure_laser_pulses([pulse])
        assert len(result) == 1
        ew = result[0]
        assert isinstance(ew, EmissiveWaveguide)
        assert ew.intensity == pytest.approx(1.0)
        assert ew.wavelength_nm > 0

    def test_multiple_pulses(self):
        """Multiple pulses produce multiple emissive waveguides."""
        vp = self._make_viewport_with_layout()
        pulses = [
            LaserPulse(energy_eV=1.72, position_nm=(100.0, 100.0)),
            LaserPulse(energy_eV=1.72, position_nm=(500.0, 500.0)),
        ]
        result = vp.configure_laser_pulses(pulses)
        assert len(result) == 2

    def test_emissive_color_is_valid_rgb(self):
        """Emissive colour must be a valid (R, G, B) triplet in [0, 1]."""
        vp = self._make_viewport_with_layout()
        pulse = LaserPulse(energy_eV=1.72)
        result = vp.configure_laser_pulses([pulse])
        r, g, b = result[0].emissive_color_rgb
        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0

    def test_intensity_clamped_at_one(self):
        """Very high laser intensity should clamp emissive intensity to 1.0."""
        vp = self._make_viewport_with_layout()
        pulse = LaserPulse(energy_eV=1.72, intensity_W_cm2=1e8)
        result = vp.configure_laser_pulses([pulse])
        assert result[0].intensity == pytest.approx(1.0)

    def test_low_intensity_scales(self):
        """Low laser intensity should produce proportionally lower glow."""
        vp = self._make_viewport_with_layout()
        pulse = LaserPulse(energy_eV=1.72, intensity_W_cm2=5e3)
        result = vp.configure_laser_pulses([pulse])
        assert result[0].intensity == pytest.approx(0.5)

    def test_clear_emissive(self):
        """clear_emissive must remove all highlights."""
        vp = self._make_viewport_with_layout()
        vp.configure_laser_pulses([LaserPulse(energy_eV=1.72)])
        assert len(vp.emissive_waveguides) == 1
        vp.clear_emissive()
        assert vp.emissive_waveguides == []

    def test_snapshot_contains_emissive(self):
        """ViewportSnapshot must include emissive_waveguides."""
        vp = self._make_viewport_with_layout()
        vp.configure_laser_pulses([LaserPulse(energy_eV=1.72)])
        snap = vp.get_snapshot()
        assert len(snap.emissive_waveguides) == 1

    def test_gate_resolution_with_layout(self):
        """With a layout loaded, emissive gate_id must not be UNRESOLVED."""
        vp = self._make_viewport_with_layout()
        pulse = LaserPulse(energy_eV=1.72, position_nm=(0.0, 0.0))
        result = vp.configure_laser_pulses([pulse])
        assert result[0].gate_id != "UNRESOLVED"

    def test_gate_resolution_without_layout(self):
        """Without a layout, gate_id should be UNRESOLVED."""
        vp = DigitalTwinViewport()
        pulse = LaserPulse(energy_eV=1.72)
        result = vp.configure_laser_pulses([pulse])
        assert result[0].gate_id == "UNRESOLVED"

    def test_wavelength_to_rgb_red(self):
        """720 nm should produce a red-dominant colour."""
        r, g, b = DigitalTwinViewport._wavelength_to_rgb(720.0)
        assert r > g and r > b

    def test_wavelength_to_rgb_blue(self):
        """450 nm should produce a blue-dominant colour."""
        r, g, b = DigitalTwinViewport._wavelength_to_rgb(450.0)
        assert b > g


# ---------------------------------------------------------------------------
# Phase Jitter Alarm (Vibration → BER)
# ---------------------------------------------------------------------------

class TestPhaseJitterAlarm:
    """Verify vibration-driven Phase Jitter alarm with live BER."""

    def test_no_vibration_no_alarm(self):
        """Without vibration, alarm is inactive."""
        vp = DigitalTwinViewport()
        assert vp.phase_jitter_alarm.active is False
        assert vp.phase_jitter_alarm.level == PhaseJitterAlarmLevel.NONE

    def test_small_vibration_no_alarm(self):
        """Tiny vibration stays below alarm thresholds."""
        vp = DigitalTwinViewport()
        profile = VibrationProfile(
            frequencies_Hz=np.array([50.0]),
            amplitudes_nm=np.array([0.001]),
        )
        alarm = vp.update_vibration(profile)
        assert alarm.active is False
        assert alarm.bit_error_rate < vp.JITTER_WARNING_BER

    def test_large_vibration_triggers_alarm(self):
        """Large vibration amplitude must trigger the alarm."""
        vp = DigitalTwinViewport()
        profile = VibrationProfile(
            frequencies_Hz=np.array([100.0, 200.0]),
            amplitudes_nm=np.array([50.0, 30.0]),
        )
        alarm = vp.update_vibration(profile)
        assert alarm.active is True
        assert alarm.level in (
            PhaseJitterAlarmLevel.WARNING,
            PhaseJitterAlarmLevel.CRITICAL,
        )
        assert alarm.bit_error_rate > 0

    def test_increasing_vibration_increases_ber(self):
        """Higher vibration amplitude must drive BER upward."""
        vp = DigitalTwinViewport()
        profile_lo = VibrationProfile(
            frequencies_Hz=np.array([100.0]),
            amplitudes_nm=np.array([1.0]),
        )
        profile_hi = VibrationProfile(
            frequencies_Hz=np.array([100.0]),
            amplitudes_nm=np.array([10.0]),
        )
        alarm_lo = vp.update_vibration(profile_lo)
        alarm_hi = vp.update_vibration(profile_hi)
        assert alarm_hi.bit_error_rate > alarm_lo.bit_error_rate

    def test_alarm_message_contains_ber(self):
        """Active alarm message must mention BER value."""
        vp = DigitalTwinViewport()
        profile = VibrationProfile(
            frequencies_Hz=np.array([100.0]),
            amplitudes_nm=np.array([50.0]),
        )
        alarm = vp.update_vibration(profile)
        assert alarm.active is True
        assert "BER" in alarm.message

    def test_alarm_tracks_vibration_amplitude(self):
        """Alarm must report the RMS vibration amplitude."""
        vp = DigitalTwinViewport()
        profile = VibrationProfile(
            frequencies_Hz=np.array([50.0, 120.0]),
            amplitudes_nm=np.array([3.0, 4.0]),
        )
        alarm = vp.update_vibration(profile)
        expected_rms = np.sqrt(3.0**2 + 4.0**2)
        assert alarm.vibration_amplitude_nm == pytest.approx(expected_rms)

    def test_clear_alarm(self):
        """clear_phase_jitter_alarm resets state."""
        vp = DigitalTwinViewport()
        profile = VibrationProfile(
            frequencies_Hz=np.array([100.0]),
            amplitudes_nm=np.array([50.0]),
        )
        vp.update_vibration(profile)
        assert vp.phase_jitter_alarm.active is True
        vp.clear_phase_jitter_alarm()
        assert vp.phase_jitter_alarm.active is False
        assert vp.phase_jitter_alarm.bit_error_rate == 0.0

    def test_snapshot_contains_alarm(self):
        """ViewportSnapshot must include phase_jitter_alarm."""
        vp = DigitalTwinViewport()
        snap = vp.get_snapshot()
        assert isinstance(snap.phase_jitter_alarm, PhaseJitterAlarm)

    def test_critical_threshold(self):
        """BER above JITTER_CRITICAL_BER must produce CRITICAL level."""
        vp = DigitalTwinViewport()
        # Very large amplitude to guarantee critical
        profile = VibrationProfile(
            frequencies_Hz=np.array([100.0]),
            amplitudes_nm=np.array([200.0]),
        )
        alarm = vp.update_vibration(profile)
        assert alarm.level == PhaseJitterAlarmLevel.CRITICAL

    def test_phase_jitter_matches_analyser(self):
        """Phase jitter value must match PhaseStabilityAnalyser output."""
        from apld_mps.module_g.phase_stability import PhaseStabilityAnalyser
        vp = DigitalTwinViewport(
            waveguide_length_nm=10_000.0,
            refractive_index=1.76,
            wavelength_nm=720.0,
        )
        analyser = PhaseStabilityAnalyser(10_000.0, 1.76, 720.0)
        profile = VibrationProfile(
            frequencies_Hz=np.array([100.0]),
            amplitudes_nm=np.array([5.0]),
        )
        alarm = vp.update_vibration(profile)
        expected_jitter = analyser.compute_phase_jitter(profile)
        assert alarm.rms_phase_jitter_rad == pytest.approx(expected_jitter)
