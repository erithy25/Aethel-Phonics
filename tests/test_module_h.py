"""Tests for Module H: Command & Control Dashboard."""

import numpy as np
import pytest

from apld_mps.module_h.viewport import (
    DigitalTwinViewport,
    RenderMode,
    CameraState,
    LayerVisibility,
)
from apld_mps.module_h.physics_panel import PhysicsControlPanel
from apld_mps.module_h.kreislauf_monitor import (
    KreislaufMonitor,
    ThermalAlertLevel,
)
from apld_mps.module_h.io_panel import IOPanel, BottleneckType
from apld_mps.module_h.resilience_center import ResilienceCenter, ResilienceStatus
from apld_mps.module_h.rl_status import RLOptimizerPanel, OptimiserState
from apld_mps.module_h.dashboard import AethelDashboard, DashboardMode

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
