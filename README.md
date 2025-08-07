# FLWRS Simulator v2.0
GPU-accelerated implementation of Field of Lumina: Warp-Resonated Symmetry theory

## Key Features
- Solves Fundamental Equation of Dynamics (FED)
- Implements all 5 quantum operators:
  - Ĥₗ (Fractal Laplacian)
  - Ĥₚ (Nonlinear self-interaction)
  - Ĥₗ (Decoherence)
  - Ĥₗₘ (Dark matter phase transition)
  - Ĥᵣ (Rectification)
- GPU acceleration via CuPy (NVIDIA CUDA)
- Models CDMO formation and coherence dynamics

## Installation
```bash
git clone https://github.com/SarvajJa/FLWRS-Simulator
cd FLWRS-Simulator
pip install -r requirements.txt