"""Module J — Aethel Functional Emulation Engine (AFEE).

Physics-aware AI inference engine that couples real tensor computation to the
polariton multiphysics simulation stack.  Every MatMul, every activation, every
weight read is modulated by the physical state of Modules B, F, G, and I:

* **Cosmic-ray noise** (Module G): degraded-coherence sectors inject Gaussian
  noise into tensor results.
* **Thermal drift** (Module F): thermo-optic phase errors distort complex-valued
  matrix multiplications; above the breakdown temperature outputs become NaN.
* **Holographic memory** (Module I): weight reads consume simulated time
  proportional to the speed of light in sapphire.
* **Reversible inference** (Module D): Toffoli/Fredkin gate emulation with
  entropy tracking and bijection verification.
"""

from .tensor_core import PolaritonicTensorCore, CoherenceSector
from .reversible_inference import ReversibleInferenceEngine, ReversibilityReport
from .weight_loader import WeightLoader, WeightTensor, LoadReport
from .thermal_drift import ThermalDriftLayer, DriftSnapshot
from .engine import AFEE, AFEEConfig, InferenceResult, FidelityReport

__all__ = [
    "PolaritonicTensorCore",
    "CoherenceSector",
    "ReversibleInferenceEngine",
    "ReversibilityReport",
    "WeightLoader",
    "WeightTensor",
    "LoadReport",
    "ThermalDriftLayer",
    "DriftSnapshot",
    "AFEE",
    "AFEEConfig",
    "InferenceResult",
    "FidelityReport",
]
