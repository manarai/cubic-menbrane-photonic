# cubic-membrane-photonics

**A modular Python simulation suite for curvature-driven cubic membrane templating and photonic crystal formation in beetle scales.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This repository implements a six-phase computational pipeline for modelling how protein-induced spontaneous curvature drives the self-assembly of cubic membrane architectures—specifically gyroid and double-diamond triply periodic minimal surfaces (TPMS)—that later template biological photonic crystals in the wing scales of *Callophrys rubi* and related beetles.

The causal chain modelled here is:

```text
Membrane protein density P
  → Spontaneous curvature bias H_0(P)
  → Cubic membrane phase (gyroid / diamond)
  → Chitin infiltration & template removal
  → Photonic crystal → visible stop band
```

Each phase is implemented as an independent, well-documented Python module in `src/`, and a complete interactive demonstration is provided as a Jupyter Notebook (`Demo.ipynb`).

---

## Repository Structure

```text
cubic-membrane-photonics/
├── src/
│   ├── __init__.py
│   ├── phase1_baseline.py       # Allen–Cahn phase-field model (no proteins)
│   ├── phase2_curvature.py      # Spontaneous curvature term (protein loading P)
│   ├── phase3_symmetry.py       # Symmetry identification (gyroid vs diamond)
│   ├── phase4_scaling.py        # Lattice scaling to visible-light regime
│   ├── phase5_photonics.py      # Photonic band structure (plane-wave expansion)
│   └── phase6_polycrystal.py    # Polycrystalline domain assembly
├── Demo.ipynb
├── run_demo.py
├── environment.yml
├── figures/
├── data/
├── .gitignore
└── README.md
```

---

## Physical Model

### Phase 1 — Baseline Allen–Cahn Model

A scalar order parameter $\phi(\mathbf{r},t)$ represents the local membrane density. Its evolution follows the Allen–Cahn gradient flow

$$
\frac{\partial \phi}{\partial t}
= \lambda\,\nabla^2\phi + \phi - \phi^3,
$$

which is obtained from the free energy functional

$$
F[\phi] = \int \left[
\frac{\lambda}{2}|\nabla \phi|^2 + \frac{(\phi^2-1)^2}{4}
\right] dV.
$$

The double-well potential enforces phase separation ($\phi = \pm 1$), while the gradient term regularises the interface width. The equation is solved spectrally on a periodic three-dimensional grid using a semi-implicit Fourier-space time stepper:

$$
\hat\phi^{n+1}(\mathbf{k}) =
\frac{\hat\phi^{n}(\mathbf{k}) - \Delta t\, \widehat{\phi^3}^{\,n}(\mathbf{k})}
{1 + \Delta t\,[\lambda |\mathbf{k}|^2 - 1]}.
$$

Here $|\mathbf{k}|^2$ is the squared wavevector magnitude (the Laplacian eigenvalue in Fourier space). The linear part is treated implicitly, ensuring unconditional linear stability.

---

### Phase 2 — Spontaneous Curvature

Protein loading $P \in [0,1]$ introduces a spontaneous curvature

$$
H_0(P) = \alpha P.
$$

In the phase-field representation this modifies the effective gradient penalty via a protein-dependent bending rigidity

$$
\kappa(P) = \kappa_0(1 + \beta P),
$$

leading to an effective interface parameter

$$
\lambda_{\mathrm{eff}}(P) = \lambda + \kappa(P) H_0(P).
$$

Increasing $P$ increases the preferred curvature, shifting the dominant wavelength and inducing morphology transitions from lamellar to bicontinuous cubic phases.

---

### Phase 3 — Symmetry Identification

Morphology is classified using two independent measures:

1. The Euler characteristic $\chi$ of the thresholded interface, computed via a discrete Gauss–Bonnet estimator.
2. The peak structure of the spherically averaged power spectrum $S(|\mathbf{k}|)$.

| $\chi$ | Dominant peak ratios | Morphology |
|------:|---------------------|------------|
| 0 | single peak | Lamellar |
| −4 | $\sqrt{6}$ | Gyroid (Ia$\bar{3}$d) |
| −8 | $\sqrt{3}$ | Double-diamond (Pn$\bar{3}$m) |
| $< -8$ | broad | Disordered bicontinuous |

---

### Phase 4 — Lattice Scaling

The simulation lattice constant $a_{\mathrm{sim}}$ is converted to physical units using the Helfrich length scale

$$
\xi(P) = \sqrt{\frac{\kappa(P)}{\sigma}},
$$

where $\sigma$ is the membrane tension. The physical lattice constant $a(P)$ is calibrated so that $P=1$ yields $a = 350\,\mathrm{nm}$, placing the photonic stop band in the visible range.

---

### Phase 5 — Photonic Band Structure

The binary phase field is mapped onto a dielectric function

$$
\varepsilon(\mathbf{r}) =
\begin{cases}
\varepsilon_{\mathrm{chitin}} = 2.56, & \phi > 0 \\
\varepsilon_{\mathrm{air}} = 1.0, & \phi < 0.
\end{cases}
$$

The magnetic-field eigenproblem solved via plane-wave expansion (PWE) is

$$
\sum_{\mathbf{G}'} \mathbf{M}(\mathbf{G},\mathbf{G}')\,\mathbf{H}(\mathbf{G}')
= \left(\frac{\omega}{c}\right)^2 \mathbf{H}(\mathbf{G}),
$$

with

$$
\mathbf{M}(\mathbf{G},\mathbf{G}') =
(\mathbf{k}+\mathbf{G})\cdot(\mathbf{k}+\mathbf{G}')\,\varepsilon^{-1}(\mathbf{G}-\mathbf{G}').
$$

Band structures are computed along the path

$$
\Gamma \rightarrow X \rightarrow M \rightarrow \Gamma \rightarrow R
$$

of the simple-cubic Brillouin zone.

---

### Phase 6 — Polycrystalline Domains

A Voronoi tessellation partitions the simulation volume into domains of characteristic size 3–15\,µm, consistent with experimental observations. Each domain is assigned an independent protein loading

$$
P_i \sim \mathcal{U}(P_{\min}, P_{\max}),
$$

and evolved independently before being smoothly blended into a global polycrystalline field.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
