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

## Quick Start

```bash
pip install -e ".[dev]"
pytest
```
