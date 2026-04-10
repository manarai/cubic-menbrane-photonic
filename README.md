# cubic-membrane-photonics

**A computational pipeline for testing the cubic membrane templating hypothesis for beetle photonic crystals.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository provides the full computational framework accompanying the manuscript:

> *"Cubic Membrane Templating as a Mechanism for the Formation of Diamond-Based Photonic Crystals in Beetle Scales"*

---

## What This Repository Does

The pipeline tests the hypothesis that intracellular bicontinuous cubic membranes act as transient templates for photonic crystal formation in beetle wing scales. It does this by:

1. **Deriving** the photonic crystal geometry from membrane physics (Allen-Cahn phase-field + Helfrich curvature), not by assuming it.
2. **Mapping** all simulation parameters to explicit biological variables with testable predictions.
3. **Computing** full photonic band structures and broadband reflectance spectra.
4. **Simulating** polycrystalline domain formation with grain boundaries and orientation mismatch statistics.
5. **Testing** three explicit falsifiability conditions against published experimental data.

---

## Repository Structure

```
cubic-membrane-photonics/
│
├── membrane_sim/               # Layer 1: Membrane physics
│   ├── phase_field.py          # Allen-Cahn baseline model
│   ├── curvature.py            # Helfrich spontaneous curvature (protein loading P)
│   ├── polycrystal.py          # Basic polycrystalline assembly
│   └── domain_formation.py     # ★ NEW: Grain boundaries, orientation mismatch, domain stats
│
├── geometry/                   # Geometry analysis
│   ├── symmetry.py             # Symmetry identification (gyroid vs diamond)
│   └── scaling.py              # Lattice scaling to visible-light regime
│
├── photonics/                  # Layer 2: Photonic simulation
│   └── band_structure.py       # PWE band structure (SC + FCC Brillouin zones)
│
├── chitin_mapping/             # ★ NEW: Biological parameter mapping
│   └── bio_mapping.py          # P→colour, t→volume fraction, ε→hydration, reflectance spectra
│
├── falsifiability.py           # ★ NEW: Three explicit falsification tests
├── run_demo.py                 # Fast command-line demo (~2 min)
├── manuscript_revised.md       # ★ NEW: Revised manuscript
├── Demo.ipynb                  # Interactive Jupyter Notebook
├── Validation.ipynb            # Validation vs Michielsen et al. 2010
├── environment.yml             # Conda environment
├── figures/                    # Generated figures (git-ignored)
├── data/                       # Generated data (git-ignored)
└── README.md
```

---

## Quick Start

```bash
# Install dependencies
conda env create -f environment.yml
conda activate cubic-membrane-photonics

# Run the full pipeline (~2 min)
python run_demo.py

# Run biological parameter mapping
python chitin_mapping/bio_mapping.py

# Run domain formation simulation
python membrane_sim/domain_formation.py

# Run falsifiability tests
python falsifiability.py
```

---

## Key Results

### 1. Geometry from Physics, Not Assumption

The phase-field model with protein loading $P$ spontaneously produces gyroid and diamond topologies. The Euler characteristic $\chi$ shows discrete jumps at the lamellar→gyroid ($P \approx 0.25$) and gyroid→diamond ($P \approx 0.65$) transitions.

### 2. Biological Parameter Mapping

| Simulation Parameter | Biological Meaning | Testable Prediction |
|----------------------|--------------------|---------------------|
| $P$ (protein loading) | Curvature-inducing protein density | Higher P → smaller $a$ → bluer color |
| $t$ (threshold) | Chitin volume fraction | Synthase inhibition → weaker stop band |
| $a$ (lattice constant) | Membrane spacing | Scales with cell size across species |
| $\varepsilon_h$ (chitin) | Refractive index (crystallinity) | Dehydration → blueshifted stop band |
| $N_{\text{domains}}$ (seeds) | Nucleation site density | More seeds → smaller domains |

### 3. Constraint Region Analysis

Only a narrow region of $(a, f_{\text{chitin}})$ parameter space satisfies both the optical ($\lambda_{\text{peak}} \in [495, 595]$ nm) and structural ($f \in [0.14, 0.20]$) constraints of *C. rubi*. The model correctly places *C. rubi* ($a = 363$ nm, $f = 0.17$) within this region.

### 4. Domain Formation

Simulated polycrystalline domains with 8 nucleation sites yield a mean domain size of $\sim 6$ μm (experimental: 3–7 μm) and an isotropic orientation mismatch distribution, consistent with TEM observations.

### 5. Falsifiability Tests

| Test | Prediction | Result |
|------|-----------|--------|
| Scaling law | $a \propto \sqrt{\kappa/\sigma}$ across species | $R^2 = 0.998$ — **PASSED** |
| Topology transition | Discrete jumps in $\chi(P)$ | Detected — **PASSED** |
| Uniqueness | Only cubic geometry satisfies both constraints | Lamellar also satisfies $\lambda$ constraint — requires TEM to distinguish |

---

## Reproducing Manuscript Figures

Each figure in the manuscript can be reproduced by running the corresponding script:

| Figure | Script | Description |
|--------|--------|-------------|
| Fig. 1 | `membrane_sim/curvature.py` | Phase-field morphology sweep |
| Fig. 2 | `chitin_mapping/bio_mapping.py` | Protein loading → colour chain |
| Fig. 3 | `chitin_mapping/bio_mapping.py` | Parameter constraint region |
| Fig. 4 | `chitin_mapping/bio_mapping.py` | Broadband reflectance spectra |
| Fig. 5 | `photonics/band_structure.py` | Photonic band structure (FCC BZ) |
| Fig. 6 | `membrane_sim/domain_formation.py` | Polycrystalline domain formation |
| Fig. 7 | `falsifiability.py` | Falsifiability tests |

All figures are also reproduced in `Demo.ipynb` and `Validation.ipynb`.

---

## Performance Benchmarks

| Phase | Grid N | Steps | Time (s) |
|-------|--------|-------|----------|
| Phase-field (Phase 1–2) | 64 | 2000 | ~12 |
| Phase-field (Phase 1–2) | 128 | 5000 | ~180 |
| Band structure (n_pw=4) | — | 15 k-pts | ~30 |
| Domain formation (8 domains) | 64 | — | ~15 |
| Falsifiability tests | — | — | ~60 |

---

## Citation

If you use this code, please cite:

- Michielsen, K., De Raedt, H. & Stavenga, D. G. (2010). *J. R. Soc. Interface*, 7(46), 765–771.
- Saranathan, V. et al. (2010). *PNAS*, 107(26), 11676–11681.
- Galusha, J. W. et al. (2008). *Phys. Rev. E*, 77, 050904(R).

---

## License

MIT License — see [LICENSE](LICENSE) for details.
