# cubic-membrane-photonics

**A modular Python simulation suite for curvature-driven cubic membrane templating and photonic crystal formation in beetle scales.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This repository implements the full six-phase computational pipeline for modelling how protein-induced spontaneous curvature drives the self-assembly of cubic membrane architectures — specifically gyroid and double-diamond minimal surfaces — that serve as biological photonic crystal templates in the wing scales of *Callophrys rubi* and related beetles.

The causal chain modelled here is:

```
Membrane protein density (P)
  → Spontaneous curvature bias H₀(P)
  → Cubic membrane phase (gyroid / diamond TPMS)
  → Chitin infiltration & template removal
  → Photonic crystal → visible stop band
```

Each phase is implemented as an independent, well-documented Python module in `src/`, and a complete interactive demonstration is provided as a Jupyter Notebook (`Demo.ipynb`).

---

## Repository Structure

```
cubic-membrane-photonics/
├── src/
│   ├── __init__.py
│   ├── phase1_baseline.py       # Allen-Cahn phase-field model (no proteins)
│   ├── phase2_curvature.py      # Spontaneous curvature term (protein loading P)
│   ├── phase3_symmetry.py       # Symmetry identification (gyroid vs diamond)
│   ├── phase4_scaling.py        # Lattice scaling to visible-light regime
│   ├── phase5_photonics.py      # Photonic band structure (plane-wave expansion)
│   └── phase6_polycrystal.py    # Polycrystalline domain assembly
├── Demo.ipynb                   # Interactive Jupyter Notebook walkthrough
├── run_demo.py                  # Fast command-line demo (all 6 phases, ~2 min)
├── environment.yml              # Conda environment specification
├── figures/                     # Output figures (generated; git-ignored)
├── data/                        # Output data arrays (generated; git-ignored)
├── .gitignore
└── README.md
```

---

## Installation

This project uses **Conda** for dependency management.

```bash
# Clone the repository
git clone https://github.com/your-username/cubic-membrane-photonics.git
cd cubic-membrane-photonics

# Create and activate the Conda environment
conda env create -f environment.yml
conda activate cubic-membrane-photonics

# Register the Jupyter kernel (optional)
python -m ipykernel install --user --name cubic-membrane-photonics

# Launch the interactive notebook
jupyter lab Demo.ipynb
```

---

## Quick Start

### Command-line demo (all 6 phases, ~2 minutes on a laptop)

```bash
python run_demo.py
```

Figures are saved to `figures/` and data arrays to `data/`. The key output is `figures/summary_all_phases.png`, which shows the complete pipeline in a single panel.

### Interactive Jupyter Notebook

```bash
jupyter notebook Demo.ipynb
```

### Individual modules

Each module can be run independently as a script:

```bash
python src/phase1_baseline.py    # Baseline simulation (N=64, 1000 steps)
python src/phase2_curvature.py   # Curvature sweep (P = 0, 0.25, 0.5, 0.75, 1.0)
python src/phase3_symmetry.py    # Symmetry classification
python src/phase4_scaling.py     # Lattice scaling analysis
python src/phase5_photonics.py   # Photonic band structure
python src/phase6_polycrystal.py # Polycrystalline assembly
```

---

## Development Phases

| Phase | Module | Description |
|-------|--------|-------------|
| 1 | `phase1_baseline.py` | Allen-Cahn phase-field model without proteins (baseline) |
| 2 | `phase2_curvature.py` | Helfrich spontaneous curvature term coupled to protein loading P |
| 3 | `phase3_symmetry.py` | Symmetry identification: Lamellar → Gyroid → Diamond transitions |
| 4 | `phase4_scaling.py` | Lattice constant measurement and scaling to visible-light regime |
| 5 | `phase5_photonics.py` | Dielectric mapping and photonic band structure via plane-wave expansion |
| 6 | `phase6_polycrystal.py` | Polycrystalline domain assembly with spatially varying P(r) |

---

## Physical Model

### Phase 1 — Baseline Allen-Cahn Model

The baseline model evolves a scalar order parameter `phi(r, t)` representing the local membrane density according to the Allen-Cahn gradient flow:

```
d phi/dt = lam * nabla^2 phi + phi - phi^3
```

derived from the free energy `F[phi] = integral [ (lam/2)|nabla phi|^2 + (phi^2-1)^2/4 ] dV`. The double-well potential drives phase separation into bulk phases `phi = ±1`, while the gradient term regularises the interface width. The equation is solved spectrally on a periodic 3-D grid using a semi-implicit Fourier-space time-stepper:

```
phi_hat^{n+1} = (phi_hat^n + dt * (-phi^3)_hat^n) / (1 - dt*(lam*k2 + 1))
```

where `k2 = -|k|^2` is the Laplacian eigenvalue. The denominator is always positive, giving unconditional stability of the linear part.

### Phase 2 — Spontaneous Curvature

Protein loading `P ∈ [0,1]` introduces a spontaneous curvature `H0(P) = alpha * P` into the Helfrich free energy. In the phase-field representation this modifies the effective interface parameter:

```
lam_eff(P) = lam + kappa(P) * H0(P)
kappa(P)   = kappa0 * (1 + beta * P)
```

As `P` increases, `lam_eff` grows, shifting the dominant wavenumber and driving morphology transitions from lamellar to bicontinuous cubic phases.

### Phase 3 — Symmetry Identification

Morphology is classified from the Euler characteristic `chi` of the thresholded field (computed via the Gauss-Bonnet theorem on the discrete surface) and the peak structure of the spherically-averaged power spectrum `S(|k|)`:

| chi | Dominant peaks | Morphology |
|-----|---------------|------------|
| 0 | Single | Lamellar |
| -4 | √6 ratio | Gyroid (Ia3d) |
| -8 | √3 ratio | Double-diamond (Pn3m) |
| < -8 | Broad | Disordered bicontinuous |

### Phase 4 — Lattice Scaling

The simulation lattice constant `a_sim` is calibrated to physical units via the Helfrich membrane length scale `xi(P) = sqrt(kappa(P) / sigma)`. The physical lattice constant `a_nm(P)` is calibrated so that `P=1` gives `a = 350 nm`, placing the stop band in the visible range (221–350 nm across the full P sweep).

### Phase 5 — Photonic Band Structure

The thresholded phase field is mapped to a dielectric function `epsilon(r)` (chitin: `eps=2.56`, air: `eps=1`). The photonic band structure is computed using a plane-wave expansion (PWE) method, solving the eigenvalue problem:

```
sum_{G'} M(G,G') H(G') = (omega/c)^2 H(G)
M(G,G') = (k+G)·(k+G') * epsilon_inv(G-G')
```

along the high-symmetry path `Gamma → X → M → Gamma → R` of the simple cubic Brillouin zone.

> **Note:** The demo uses `n_pw=2` (125 plane waves) for speed. For quantitatively accurate band gaps, increase `n_pw` to 5–7 (≥ 2744 plane waves) and use a denser k-path.

### Phase 6 — Polycrystalline Domains

A Voronoi tessellation of `n_domains` randomly placed seed points partitions the simulation box into independent domains, each assigned a random protein loading `P_i ~ U[P_min, P_max]`. Independent phase-field simulations are run per domain and assembled into a global polycrystalline field with smooth boundary blending. Domain sizes are calibrated to the 3–15 μm range observed in beetle wing scales.

---

## Key Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Grid size | N | 64 | Voxels per dimension (N³ total) |
| Protein loading | P | [0, 1] | Effective membrane protein crowding |
| Spontaneous curvature | H₀(P) | alpha·P | Curvature bias (alpha = 0.5) |
| Bending rigidity | kappa(P) | kappa0·(1+beta·P) | Stiffening (kappa0=0.2, beta=1.0) |
| Interface parameter | lam | 0.1 | Allen-Cahn gradient penalty |
| Chitin dielectric | eps_h | 2.56 | n ≈ 1.6 |
| Target lattice | a | 200–500 nm | Visible-light photonic regime |

---

## Demo Output

Running `run_demo.py` produces the following figures in `figures/`:

| File | Description |
|------|-------------|
| `phase1_baseline_snapshots.png` | Phase-field evolution at t=0, 50, 100, 200 |
| `phase1_baseline_slices.png` | Three orthogonal mid-plane slices |
| `phase2_curvature_sweep.png` | Morphology comparison across P values |
| `phase3_power_spectra.png` | Spherically averaged power spectra |
| `phase3_phase_diagram.png` | Morphology classification bar chart |
| `phase4_lattice_scaling.png` | Lattice constant and Helfrich length vs P |
| `phase5_dielectric_*.png` | Dielectric structure slices |
| `phase5_band_structure_*.png` | Photonic band diagram and wavelength map |
| `phase6_polycrystal_slices.png` | Polycrystalline field and domain map |
| `phase6_domain_analysis.png` | Domain size distribution |
| `summary_all_phases.png` | Full pipeline summary panel |

---

## Production Parameters

The demo uses a coarse grid (N=32) and reduced step counts for speed. For publication-quality results, use:

| Parameter | Demo | Production |
|-----------|------|------------|
| Grid size N | 32 | 128–256 |
| Time steps | 200 | 5000–20000 |
| n_pw (Phase 5) | 2 | 5–7 |
| n_kpoints (Phase 5) | 8 | 40–60 |

---

## Dependencies

All dependencies are specified in `environment.yml`. Key packages:

- `numpy`, `scipy` — numerical computation and FFT
- `matplotlib` — visualisation
- `jupyter`, `ipykernel` — interactive notebooks

---

## Citation

If you use this code in your research, please cite:

- Michielsen, K. & Stavenga, D. G. (2008). Gyroid cuticular structures in butterfly wing scales. *J. R. Soc. Interface*, 5, 85–94.
- Saranathan, V. et al. (2010). Structure, function, and self-assembly of single network gyroid (I4132) photonic crystals in butterfly wing scales. *PNAS*, 107(26), 11676–11681.
- Galusha, J. W. et al. (2008). Discovery of a diamond-based photonic crystal structure in beetle scales. *Phys. Rev. E*, 77, 050904(R).

---

## License

MIT License — see [LICENSE](LICENSE) for details.
