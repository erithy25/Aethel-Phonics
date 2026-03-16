"""Aethel Phonics — Command & Control Dashboard (Streamlit).

Launch with:
    streamlit run src/apld_mps/module_h/streamlit_app.py
"""

from __future__ import annotations

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from apld_mps import __version__
from apld_mps.module_h.dashboard import AethelDashboard, DashboardMode
from apld_mps.module_h.kreislauf_monitor import ThermalAlertLevel
from apld_mps.module_h.io_panel import BottleneckType
from apld_mps.module_h.resilience_center import ResilienceStatus
from apld_mps.module_h.rl_status import OptimiserState
from apld_mps.module_f.heat_optics import ThermoOpticConfig
from apld_mps.module_f.tpv_recycler import TPVConfig
from apld_mps.module_d.mapper_3d import AutoMapper3D
from apld_mps.module_g.phase_stability import VibrationProfile
from apld_mps.module_i.model_parser import ModelParser
from apld_mps.module_i.polaritonic_mapping import PolaritonicMapper
from apld_mps.module_i.resource_estimator import ResourceEstimator
from apld_mps.module_i.latency_analyser import LatencyAnalyser
from apld_mps.module_j.engine import AFEE, AFEEConfig, InferenceMode

# ---------------------------------------------------------------------------
# Plotly template — readable on both light and dark Streamlit themes
# ---------------------------------------------------------------------------
_PLOTLY_TEMPLATE = "plotly_white"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Aethel V1 \u2014 Command & Control",
    page_icon="\u2b22",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_dashboard() -> AethelDashboard:
    thermo_cfg = ThermoOpticConfig(
        grid_size_nm=(1000, 1000, 500),
        grid_spacing_nm=100,
        dt_ps=10.0,
    )
    tpv_cfg = TPVConfig()
    dash = AethelDashboard(
        thermo_config=thermo_cfg,
        tpv_config=tpv_cfg,
        volume_nm=(1_000_000, 1_000_000, 1_000_000),
    )
    dash.initialise()
    return dash


if "dashboard" not in st.session_state:
    st.session_state.dashboard = _init_dashboard()
    st.session_state.sim_step = 0

dash: AethelDashboard = st.session_state.dashboard

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>\u2b22 Aethel V1 &mdash; Command &amp; Control Dashboard</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:gray;'>"
    f"APLD-MPS v{__version__} &mdash; Massively Parallel Multiphysics Simulation Suite"
    "</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: Sektion 2 — Physics & Material Control
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Physics & Material Control")

    # Substrate
    st.subheader("Substrat")
    substrates = dash.physics.available_substrates
    selected_sub = st.selectbox("Material", substrates, index=0)
    sub_info = dash.physics.select_substrate(selected_sub)
    col1, col2 = st.columns(2)
    col1.metric("n(720 nm)", f"{sub_info.refractive_index_at_720nm:.4f}")
    col2.metric("\u03ba [W/(m\u00b7K)]", f"{sub_info.thermal_conductivity:.1f}")

    # Dispersion curve
    wl, n_vals = sub_info.dispersion_curve
    fig_disp = go.Figure()
    fig_disp.add_trace(go.Scatter(
        x=wl, y=n_vals, mode="lines", name="n(\u03bb)",
        line=dict(color="#0088cc"),
    ))
    fig_disp.update_layout(
        height=200, margin=dict(l=40, r=10, t=10, b=30),
        xaxis_title="\u03bb [nm]", yaxis_title="n",
        template=_PLOTLY_TEMPLATE,
    )
    st.plotly_chart(fig_disp, use_container_width=True)

    # TMD layer
    st.subheader("TMD-Layer")
    tmd_mats = dash.physics.available_tmd_materials
    selected_tmd = st.selectbox("TMD Material", tmd_mats, index=0)
    dash.physics.select_tmd(selected_tmd)

    res_eV = st.slider("Resonanz [eV]", 1.4, 2.0, dash.physics.current_tmd.resonance_energy_eV, 0.01)
    bind_eV = st.slider("Bindungsenergie [eV]", 0.1, 0.8, dash.physics.current_tmd.binding_energy_eV, 0.01)
    osc = st.slider("Oszillatorst\u00e4rke", 0.05, 0.30, dash.physics.current_tmd.oscillator_strength, 0.01)
    dash.physics.tune_tmd(resonance_energy_eV=res_eV, binding_energy_eV=bind_eV, oscillator_strength=osc)
    tmd_cfg = dash.physics.current_tmd
    st.caption(f"{tmd_cfg.material_name} \u2014 E_res={tmd_cfg.resonance_energy_eV:.2f} eV")

    # Laser pulse generator
    st.subheader("Laser-Puls-Generator")
    pulse_energy = st.number_input("Energie [eV]", value=1.72, step=0.01)
    pulse_intensity = st.number_input("Intensit\u00e4t [W/cm\u00b2]", value=1e4, format="%.0e")
    pulse_dur = st.number_input("Dauer [fs]", value=100.0, step=10.0)
    if st.button("Puls hinzuf\u00fcgen"):
        dash.physics.add_pulse(
            energy_eV=pulse_energy,
            intensity_W_cm2=pulse_intensity,
            duration_fs=pulse_dur,
        )
    if st.button("Alle l\u00f6schen"):
        dash.physics.clear_pulses()
    ps = dash.physics.pulse_generator_state
    st.caption(f"{ps.n_pulses} Pulse konfiguriert \u2014 {ps.total_energy_aJ:.2f} aJ gesamt")

# ---------------------------------------------------------------------------
# Main area: 3 columns
# ---------------------------------------------------------------------------

left_col, center_col, right_col = st.columns([1, 2, 1])

# ---------------------------------------------------------------------------
# Centre: Sektion 1 — Digital Twin 3D Viewport
# ---------------------------------------------------------------------------
with center_col:
    st.subheader("\u2b22 Digital Twin 3D Viewport")

    # Generate / load layout
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Half-Adder generieren"):
            mapper = AutoMapper3D(volume_nm=(1e6, 1e6, 1e6))
            layout = mapper.map_half_adder_3d(n_episodes=100)
            dash.load_layout(layout)
    with col_btn2:
        if st.button("Full-Adder generieren"):
            mapper = AutoMapper3D(volume_nm=(1e6, 1e6, 1e6))
            layout = mapper.map_full_adder_3d(n_episodes=100)
            dash.load_layout(layout)

    # 3D scatter of gates + wire paths
    snap = dash.viewport.get_snapshot()
    if snap.gates:
        fig3d = go.Figure()

        # Gates as markers
        gx = [g["centre_nm"][0] for g in snap.gates]
        gy = [g["centre_nm"][1] for g in snap.gates]
        gz = [g["centre_nm"][2] for g in snap.gates]
        labels = [f'{g["gate_id"]} ({g["gate_type"]})' for g in snap.gates]
        fig3d.add_trace(go.Scatter3d(
            x=gx, y=gy, z=gz,
            mode="markers+text",
            marker=dict(size=8, color="#00bcd4", symbol="diamond"),
            text=labels,
            textposition="top center",
            textfont=dict(size=11, color="#333333"),
            name="Gates",
        ))

        # Wires as lines
        for w in snap.wires:
            wps = w["waypoints_nm"]
            if len(wps) >= 2:
                fig3d.add_trace(go.Scatter3d(
                    x=[p[0] for p in wps],
                    y=[p[1] for p in wps],
                    z=[p[2] for p in wps],
                    mode="lines",
                    line=dict(color="#ff7043", width=3),
                    showlegend=False,
                ))

        fig3d.update_layout(
            height=500,
            scene=dict(
                xaxis_title="X [nm]", yaxis_title="Y [nm]", zaxis_title="Z [nm]",
                bgcolor="rgb(240,245,250)",
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            template=_PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig3d, use_container_width=True)
        st.caption(
            f"{dash.viewport.gate_count()} Gates \u00b7 "
            f"{dash.viewport.wire_count()} Wires \u00b7 "
            f"{dash.viewport.layer_count()} Layers"
        )
    else:
        st.info("Kein Layout geladen. Generiere einen Adder oben.")

    # Field overlay (if density exists)
    if snap.field_slices:
        st.subheader("|\u03c8(r,t)|\u00b2 \u2014 Polariton-Dichte")
        fig_field = px.imshow(
            snap.field_slices.get("xy", np.zeros((5, 5))),
            color_continuous_scale="inferno",
            labels=dict(color="\u03c1"),
        )
        fig_field.update_layout(
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            template=_PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig_field, use_container_width=True)


# ---------------------------------------------------------------------------
# Right top: Sektion 3 — Aethel-Kreislauf Monitor
# ---------------------------------------------------------------------------
with right_col:
    st.subheader("Aethel-Kreislauf")

    # Thermal heatmap
    hm = dash.kreislauf.get_thermal_heatmap()
    alert = hm.alert_level
    alert_color = {"NOMINAL": "green", "WARNING": "orange", "CRITICAL": "red"}
    st.markdown(
        f"**Thermal Status:** "
        f"<span style='color:{alert_color[alert.name]}; font-weight:bold;'>"
        f"\u25cf {alert.name}</span>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    c1.metric("T_peak", f"{hm.peak_temperature_K:.1f} K")
    c2.metric("T_mean", f"{hm.mean_temperature_K:.1f} K")

    fig_thermal = px.imshow(
        hm.temperature_slice,
        color_continuous_scale="hot",
        labels=dict(color="T [K]"),
    )
    fig_thermal.update_layout(
        height=200, margin=dict(l=10, r=10, t=10, b=10),
        template=_PLOTLY_TEMPLATE,
    )
    st.plotly_chart(fig_thermal, use_container_width=True)

    # TPV recycling
    st.markdown("**TPV-Recycling**")
    clock_rate = st.slider("Taktrate [GHz]", 0.0, 500.0, 100.0, 1.0)
    dash.kreislauf.set_clock_rate(clock_rate)
    tpv = dash.kreislauf.get_tpv_status()
    c3, c4 = st.columns(2)
    c3.metric("Recycling-Eff.", f"{tpv.recycling_efficiency * 100:.2f}%")
    c4.metric("Netto-Leistung", f"{tpv.net_power_W:.3f} W")

    # Stability analysis
    if st.button("Max-Taktrate berechnen"):
        result = dash.kreislauf.run_stability_analysis(
            max_temperature_K=400.0, resolution_GHz=5.0,
        )
        st.success(f"Max stabile Taktrate: **{result.max_clock_rate_GHz:.0f} GHz**")
        st.caption(
            f"T_eq = {result.equilibrium_temperature_K:.1f} K \u00b7 "
            f"P_rad = {result.radiated_power_W:.2f} W \u00b7 "
            f"P_tpv = {result.tpv_recovered_power_W:.4f} W"
        )

    gov = dash.kreislauf.get_clock_governor_state()
    if gov.max_stable_clock_GHz > 0:
        st.progress(
            min(clock_rate / gov.max_stable_clock_GHz, 1.0),
            text=f"Headroom: {gov.headroom_percent:.0f}%",
        )

# ---------------------------------------------------------------------------
# Right bottom: Sektion 4 — Petabit I/O
# ---------------------------------------------------------------------------
with right_col:
    st.subheader("Petabit I/O")

    # Add demo channels if none exist
    throughput = dash.io.get_throughput_data()
    if not throughput.input_channels and not throughput.output_channels:
        for i in range(4):
            dash.io.add_input_channel(f"fibre-in-{i}", 250.0 * (i + 1), 2000.0)
            dash.io.add_output_channel(f"fibre-out-{i}", 200.0 * (i + 1), 2000.0)
        dash.io.add_fibre_interface("port-0")
        throughput = dash.io.get_throughput_data()

    c5, c6 = st.columns(2)
    c5.metric("Input", f"{throughput.input_Gbps:.0f} Gbps")
    c6.metric("Output", f"{throughput.output_Gbps:.0f} Gbps")

    # Bottleneck radar
    radar = dash.io.get_bottleneck_radar()
    bn_colors = {
        BottleneckType.NONE: "green",
        BottleneckType.MEMORY_BOUND: "red",
        BottleneckType.NETWORK_BOUND: "orange",
        BottleneckType.INPUT_BOUND: "#cc9900",
        BottleneckType.OUTPUT_BOUND: "#cc9900",
    }
    st.markdown(
        f"**Bottleneck:** "
        f"<span style='color:{bn_colors[radar.bottleneck_type]}; font-weight:bold;'>"
        f"{radar.bottleneck_name}</span> "
        f"({radar.headroom_percent:.0f}% Headroom)",
        unsafe_allow_html=True,
    )

    # Coupling status
    coupling = dash.io.get_coupling_status()
    if coupling.interfaces:
        iface = coupling.interfaces[0]
        st.caption(
            f"Fibre IL: {iface['insertion_loss_dB']:.2f} dB \u00b7 "
            f"\u03b7 = {iface['coupling_efficiency'] * 100:.1f}%"
        )


# ---------------------------------------------------------------------------
# Left: Sektion 6 — RL Optimizer
# ---------------------------------------------------------------------------
with left_col:
    st.subheader("RL-Optimizer")

    n_episodes = st.number_input("Episoden", value=200, step=50, min_value=10)
    if st.button("Full-Adder optimieren (RL)"):
        with st.spinner("RL-Training l\u00e4uft..."):
            layout = dash.rl.run_full_adder(n_episodes=int(n_episodes))
            dash.load_layout(layout)
        st.success(f"Konvergiert nach {dash.rl.get_progress().best_episode} Episoden")

    progress = dash.rl.get_progress()
    if progress.reward_history:
        st.markdown(f"**Status:** {progress.state.name}")
        st.caption(
            f"Best Reward: {progress.best_reward:.2f} @ Episode {progress.best_episode}"
        )

        fig_rl = go.Figure()
        fig_rl.add_trace(go.Scatter(
            y=progress.reward_history,
            mode="lines",
            name="Reward",
            line=dict(color="#4caf50", width=2),
        ))
        fig_rl.update_layout(
            height=220,
            margin=dict(l=40, r=10, t=10, b=30),
            xaxis_title="Episode",
            yaxis_title="Reward",
            template=_PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig_rl, use_container_width=True)

        # Training log
        last_entries = dash.rl.get_last_log_entries(5)
        log_data = [
            {"Ep": e.episode, "Reward": f"{e.reward:.2f}",
             "Wire [nm]": f"{e.wire_length_nm:.0f}", "Layers": e.n_layers}
            for e in last_entries
        ]
        st.dataframe(log_data, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Bottom: Tabbed sections — Resilience / AI Tensor Compiler / AFEE
# ---------------------------------------------------------------------------
st.markdown("---")

tab_resilience, tab_tensor, tab_afee = st.tabs([
    "Stress-Test & Resilience (Module G)",
    "AI Tensor Compiler (Module I)",
    "Inferenz-Interface (Module J)",
])

# ---------------------------------------------------------------------------
# Tab 1: Sektion 5 — Stress Test & Resilience Center
# ---------------------------------------------------------------------------
with tab_resilience:
    st.subheader("Stress-Test & Resilience Center")

    bot_left, bot_mid, bot_right = st.columns(3)

    with bot_left:
        st.markdown("**Cosmic Ray Ticker**")
        if st.button("1s Simulation starten"):
            entries = dash.resilience.simulate_cosmic_rays(duration_s=1.0)
            st.info(f"{len(entries)} Events generiert")

        ticker = dash.resilience.get_ticker(10)
        if ticker:
            ticker_data = [
                {
                    "t [ps]": f"{e.timestamp_ps:.0f}",
                    "E [MeV]": f"{e.energy_MeV:.1f}",
                    "r [nm]": f"{e.affected_radius_nm:.0f}",
                    "Loss": f"{e.coherence_loss:.2f}",
                }
                for e in ticker[-5:]
            ]
            st.dataframe(ticker_data, use_container_width=True, hide_index=True)
        else:
            st.caption("Keine Events \u2014 Simulation starten.")

    with bot_mid:
        st.markdown("**Self-Healing Status**")
        sh = dash.resilience.get_self_healing_status()
        c7, c8 = st.columns(2)
        c7.metric("Disrupted", f"{sh.disrupted_fraction * 100:.3f}%")
        c8.metric("Reroutes", sh.active_reroutes)

        if sh.availability_slice.size > 1:
            fig_avail = px.imshow(
                sh.availability_slice,
                color_continuous_scale="RdYlGn",
                labels=dict(color="Availability"),
                zmin=0, zmax=1,
            )
            fig_avail.update_layout(
                height=200, margin=dict(l=10, r=10, t=10, b=10),
                template=_PLOTLY_TEMPLATE,
            )
            st.plotly_chart(fig_avail, use_container_width=True)

    with bot_right:
        st.markdown("**Vibrations-Analyse**")
        vib_amp = st.slider("Vibrationsamplitude [nm]", 0.01, 5.0, 0.5, 0.01)
        vib_freq = st.slider("Frequenz [Hz]", 10.0, 1000.0, 100.0, 10.0)

        if st.button("Analyse starten"):
            profile = VibrationProfile(
                frequencies_Hz=np.array([vib_freq]),
                amplitudes_nm=np.array([vib_amp]),
            )
            data = dash.resilience.analyse_vibration(profile)

        vib_data = dash.resilience.get_vibration_data()
        c9, c10 = st.columns(2)
        c9.metric("Phase Jitter", f"{vib_data.rms_phase_jitter_rad:.4f} rad")
        c10.metric("BER", f"{vib_data.bit_error_probability:.2e}")
        if vib_data.isolation_required:
            st.warning("Aktive Vibrationsisolation empfohlen!")
        else:
            st.success("Isolation nicht erforderlich")

# ---------------------------------------------------------------------------
# Tab 2: AI Tensor Compiler (Module I)
# ---------------------------------------------------------------------------
with tab_tensor:
    st.subheader("AI Tensor Compiler")

    tc_left, tc_right = st.columns([1, 1])

    with tc_left:
        st.markdown("**Transformer-Spezifikation**")
        tc_name = st.text_input("Modellname", value="GPT-Aethel", key="tc_name")
        tc_layers = st.number_input("Transformer-Layers", value=12, min_value=1, max_value=128, step=1, key="tc_layers")
        tc_d_model = st.number_input("d_model", value=768, min_value=64, step=64, key="tc_dmodel")
        tc_n_heads = st.number_input("Attention-Heads", value=12, min_value=1, step=1, key="tc_nheads")
        tc_d_ff = st.number_input("d_ff (0 = auto 4\u00d7d_model)", value=0, min_value=0, step=64, key="tc_dff")
        tc_vocab = st.number_input("Vokabular", value=32000, min_value=256, step=1000, key="tc_vocab")
        tc_seq = st.number_input("Sequenzl\u00e4nge", value=2048, min_value=64, step=64, key="tc_seq")

        if st.button("Kompilieren & Analysieren", key="tc_compile"):
            parser = ModelParser()
            d_ff_val = int(tc_d_ff) if int(tc_d_ff) > 0 else None
            graph = parser.from_transformer_spec(
                name=tc_name,
                n_layers=int(tc_layers),
                d_model=int(tc_d_model),
                n_heads=int(tc_n_heads),
                d_ff=d_ff_val,
                vocab_size=int(tc_vocab),
                seq_len=int(tc_seq),
            )

            mapper = PolaritonicMapper()
            mapping = mapper.map(graph)

            estimator = ResourceEstimator()
            estimate = estimator.estimate(graph, mapping)

            analyser = LatencyAnalyser()
            latency = analyser.analyse(graph, mapping)

            st.session_state.tc_estimate = estimate
            st.session_state.tc_latency = latency
            st.session_state.tc_mapping = mapping

    with tc_right:
        if "tc_estimate" in st.session_state:
            est = st.session_state.tc_estimate
            lat = st.session_state.tc_latency
            mapping = st.session_state.tc_mapping

            st.markdown(f"**Modell:** {est.model_name}")

            r1, r2, r3 = st.columns(3)
            r1.metric("Parameter", f"{est.total_parameters:,}")
            r2.metric("FLOPs/Token", f"{est.total_flops_per_token:,}")
            r3.metric("Polariton-Gates", f"{est.total_gate_count:,}")

            r4, r5, r6 = st.columns(3)
            r4.metric("Z-Layers", f"{est.layer_count}")
            r5.metric("Speicher-Regionen", f"{est.memory_regions}")
            r6.metric("Optische Leistung", f"{est.total_power_w:.2f} W")

            st.markdown(
                f"**Monolith:** {est.monolith_volume_mm[0]:.1f} \u00d7 "
                f"{est.monolith_volume_mm[1]:.1f} \u00d7 "
                f"{est.monolith_volume_mm[2]:.1f} mm"
            )
            st.markdown(
                f"**Gewicht-Speicher:** {est.weight_storage_bytes:,} Bytes "
                f"({est.weight_bits}-bit Quantisierung)"
            )

            st.markdown("---")
            st.markdown("**Latenz-Analyse**")

            l1, l2, l3 = st.columns(3)
            l1.metric("Gesamtlatenz", f"{lat.total_latency_ps:.1f} ps")
            l2.metric("Latenz (ns)", f"{lat.total_latency_ns:.4f} ns")
            l3.metric("Tokens/s", f"{lat.tokens_per_second:,.0f}")

            if lat.layer_latencies:
                layer_data = [
                    {
                        "Layer": ll.layer_index,
                        "Attn [ps]": f"{ll.attention_ps:.2f}",
                        "FFN [ps]": f"{ll.ffn_ps:.2f}",
                        "Mem [ps]": f"{ll.memory_read_ps:.2f}",
                        "Prop [ps]": f"{ll.propagation_ps:.2f}",
                        "Total [ps]": f"{ll.total_ps:.2f}",
                    }
                    for ll in lat.layer_latencies[:10]
                ]
                st.dataframe(layer_data, use_container_width=True, hide_index=True)
                if len(lat.layer_latencies) > 10:
                    st.caption(f"... und {len(lat.layer_latencies) - 10} weitere Layers")

            # Cluster distribution bar chart
            from collections import Counter
            cluster_counts = Counter(c.cluster_type.name for c in mapping.clusters)
            if cluster_counts:
                fig_clusters = go.Figure(go.Bar(
                    x=list(cluster_counts.keys()),
                    y=list(cluster_counts.values()),
                    marker_color="#0088cc",
                ))
                fig_clusters.update_layout(
                    title="Gate-Cluster-Verteilung",
                    height=280,
                    margin=dict(l=40, r=10, t=40, b=30),
                    xaxis_title="Cluster-Typ",
                    yaxis_title="Anzahl",
                    template=_PLOTLY_TEMPLATE,
                )
                st.plotly_chart(fig_clusters, use_container_width=True)
        else:
            st.info(
                "Transformer-Spezifikation links eingeben und "
                "'Kompilieren & Analysieren' klicken."
            )

# ---------------------------------------------------------------------------
# Tab 3: Inferenz-Interface (Module J) — AFEE
# ---------------------------------------------------------------------------
with tab_afee:
    st.subheader("AFEE \u2014 Physics-Aware Inference Engine")

    # AFEE session state
    if "afee" not in st.session_state:
        afee_cfg = AFEEConfig(
            n_layers=2, d_model=64, d_ff=256, n_heads=4, vocab_size=256,
            mode=InferenceMode.PHYSICS_FULL, seed=42,
        )
        afee = AFEE(afee_cfg)
        afee.load_weights()
        st.session_state.afee = afee
        st.session_state.chat_history = []

    afee: AFEE = st.session_state.afee

    afee_left, afee_right = st.columns([2, 1])

    with afee_left:
        st.markdown("**Chat-Konsole**")

        # Mode selector
        mode_choice = st.radio(
            "Inference-Modus",
            ["Physics Full", "Reversibel", "Standard"],
            horizontal=True,
        )
        mode_map = {
            "Physics Full": InferenceMode.PHYSICS_FULL,
            "Reversibel": InferenceMode.REVERSIBLE,
            "Standard": InferenceMode.STANDARD,
        }
        afee.cfg.mode = mode_map[mode_choice]

        # Coherence slider (simulates Module G)
        coherence = st.slider(
            "Sektor-Koharenz (Module G)", 0.0, 1.0, 1.0, 0.01,
            help="Unter 0.8: kosmische Strahlung injiziert Rauschen in die Inferenz",
        )
        afee.set_coherence(coherence)

        # Chat input
        user_input = st.text_input("Eingabe (wird als Token-IDs kodiert):", key="afee_chat_input")
        if st.button("Senden") and user_input:
            # Encode input as byte values (simple tokenisation)
            token_ids = [min(b, afee.cfg.vocab_size - 1) for b in user_input.encode("utf-8")]

            # Reset thermal state for fresh inference
            afee.reset_thermal()

            # Run generation
            generated = afee.generate(token_ids, max_tokens=20, temperature=0.8)

            # Decode output (skip prompt tokens)
            output_ids = generated[len(token_ids):]
            output_bytes = bytes([min(t, 127) for t in output_ids])
            try:
                output_text = output_bytes.decode("utf-8", errors="replace")
            except Exception:
                output_text = repr(output_bytes)

            # Get fidelity from last forward pass
            last_result = afee.forward(generated[-min(len(generated), 10):])
            fidelity = last_result.fidelity

            st.session_state.chat_history.append({
                "user": user_input,
                "response": output_text,
                "fidelity": fidelity.overall_fidelity,
                "temperature_K": fidelity.peak_temperature_K,
                "is_breakdown": fidelity.is_breakdown,
            })

        # Display chat history
        for entry in st.session_state.chat_history[-5:]:
            st.markdown(f"> **User:** {entry['user']}")
            if entry["is_breakdown"]:
                st.error(f"AFEE: [THERMAL BREAKDOWN - NaN] T={entry['temperature_K']:.1f} K")
            else:
                fid_color = "green" if entry["fidelity"] > 0.9 else "orange" if entry["fidelity"] > 0.5 else "red"
                st.markdown(
                    f"**AFEE:** {entry['response']} "
                    f"<span style='color:{fid_color}; font-size:0.9em; font-weight:bold;'>"
                    f"[Fidelity: {entry['fidelity']:.1%}]</span>",
                    unsafe_allow_html=True,
                )

    with afee_right:
        st.markdown("### Fidelity-Meter")

        # Get current fidelity
        fid = afee._compute_fidelity()

        # Overall fidelity gauge — large, readable, theme-compatible
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=fid.overall_fidelity * 100,
            number={"suffix": "%", "font": {"size": 36}},
            title={"text": "Gesamt-Fidelity", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 2, "tickfont": {"size": 12}},
                "bar": {"color": "#0088cc", "thickness": 0.6},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#cccccc",
                "steps": [
                    {"range": [0, 50], "color": "#ffcdd2"},
                    {"range": [50, 80], "color": "#fff9c4"},
                    {"range": [80, 100], "color": "#c8e6c9"},
                ],
                "threshold": {
                    "line": {"color": "#d32f2f", "width": 4},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
        ))
        fig_gauge.update_layout(
            height=280,
            margin=dict(l=30, r=30, t=60, b=20),
            template=_PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Sub-fidelity component bars with clear labels
        st.markdown("**Komponenten:**")

        # Thermal fidelity
        st.markdown(f"Thermal: **{fid.thermal_fidelity:.1%}**")
        st.progress(fid.thermal_fidelity)

        # Coherence fidelity
        st.markdown(f"Koh\u00e4renz: **{fid.coherence_fidelity:.1%}**")
        st.progress(fid.coherence_fidelity)

        # Reversibility fidelity
        st.markdown(f"Reversibel: **{fid.reversibility_fidelity:.1%}**")
        st.progress(fid.reversibility_fidelity)

        # Temperature & phase error as clear metrics
        st.markdown("---")
        m1, m2 = st.columns(2)
        m1.metric("T_chip", f"{fid.peak_temperature_K:.1f} K")
        m2.metric("Phase Error", f"{fid.phase_error_rad:.4f} rad")

        if fid.is_breakdown:
            st.error("THERMAL BREAKDOWN!")

        # Reversibility stats
        rev = afee._rev.report
        if rev.total_ops > 0:
            st.caption(
                f"Ops: {rev.total_ops} | "
                f"Reversibel: {rev.reversible_ops} | "
                f"Bits erased: {rev.bits_erased} | "
                f"Entropy: {rev.entropy_J:.2e} J"
            )

# ---------------------------------------------------------------------------
# Footer: Overall status
# ---------------------------------------------------------------------------
st.markdown("---")
overall = dash.resilience.overall_status()
status_emoji = {
    ResilienceStatus.HEALTHY: "\u2705",
    ResilienceStatus.DEGRADED: "\u26a0\ufe0f",
    ResilienceStatus.CRITICAL: "\ud83d\udea8",
}
st.markdown(
    f"<h3 style='text-align:center;'>"
    f"System-Status: {status_emoji.get(overall, '')} {overall.name}"
    f"</h3>",
    unsafe_allow_html=True,
)
