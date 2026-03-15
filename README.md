# Aethel APLD-MPS Suite

**A**ethel **P**olariton **L**ogic **D**evice — **M**ulti**p**hysics **S**imulation Suite

An integrated development environment and simulation platform for the mathematical
and physical verification of exciton-polariton-based logic gates and waveguide
structures at room temperature.

## Architecture

| Module | Purpose |
|--------|---------|
| **A** — Material & Geometry Editor | Substrate database, TMD layer specification, 2D/3D CAD geometry |
| **B** — Multiphysics Simulation Core | 2D/3D FDTD Maxwell (GPU-accelerated, multi-GPU), coupled-oscillator model, 2D/3D Gross-Pitaevskii with polariton-bullet solitons |
| **C** — Input/Output & Detection | Laser sources, virtual detectors, hollow-core fibre interface, petabit I/O dashboard |
| **D** — Logic Compiler & Mapping | Gate library, 2D flat & 3D volumetric placement with RL-based thermal-aware optimisation |
| **E** — Visualization & Metrics | Real-time field visualization, performance dashboard |
| **F** — Thermodynamic Feedback | Coupled heat-optics (Landauer + thermo-optic), TPV energy recycling, max clock-rate analysis |
| **G** — Quantum-Cosmic Stress Tests | Cosmic ray impact simulation, self-healing routing, phase stability under vibration |
| **H** — Command & Control Dashboard | Digital Twin 3D viewport, physics control, Kreislauf thermal monitor, petabit I/O radar, resilience center, RL optimizer status |

## Quick Start

### Dashboard lokal starten (empfohlen)

```bash
# Einzeiler — macht alles automatisch:
chmod +x run_local.sh && ./run_local.sh
```

Das Skript erstellt ein Virtual Environment, installiert alle Abhängigkeiten
und startet das vollständige Command & Control Dashboard unter
**http://localhost:8501**.

### Manuelles Setup

```bash
# 1. Virtual Environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Abhängigkeiten (inkl. Dashboard)
pip install -e ".[dev,dashboard]"

# 3. Tests
pytest

# 4. Dashboard starten
streamlit run src/apld_mps/module_h/streamlit_app.py
```

### Voraussetzungen

- Python 3.10+
- Kein GPU nötig — alle Simulationen laufen auf CPU
