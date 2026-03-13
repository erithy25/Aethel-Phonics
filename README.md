# Aethel APLD-MPS Suite

**A**ethel **P**olariton **L**ogic **D**evice — **M**ulti**p**hysics **S**imulation Suite

An integrated development environment and simulation platform for the mathematical
and physical verification of exciton-polariton-based logic gates and waveguide
structures at room temperature.

## Architecture

| Module | Purpose |
|--------|---------|
| **A** — Material & Geometry Editor | Substrate database, TMD layer specification, 2D/3D CAD geometry |
| **B** — Multiphysics Simulation Core | FDTD Maxwell solver, coupled-oscillator model, Gross-Pitaevskii nonlinear solver |
| **C** — Input/Output & Detection | Laser source editor, virtual detectors with ps-resolution |
| **D** — Logic Compiler & Mapping | Gate library (AND/OR/NOT/XOR/NAND), automatic placement & routing |
| **E** — Visualization & Metrics | Real-time field visualization, performance dashboard |

## Quick Start

```bash
pip install -e ".[dev]"
pytest
```
