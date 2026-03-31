
# Cubic‑Membrane‑Photonics  
### A Virtual Prototyping Tool for Biomimetic Structural Colour Engineering

**cubic‑membrane‑photonics** is a six‑phase computational pipeline for designing, simulating, and evaluating **bicontinuous photonic crystals** inspired by biological cubic membranes (e.g. beetle scales).

The tool enables **colour‑on‑demand design**, **geometry selection**, **materials screening**, and **polycrystalline appearance prediction** *before* any physical fabrication.

It is intended for **biomimetic engineers, materials scientists, photonics researchers, and fabrication chemists** developing pigment‑free structural colour materials.

---

## Core Concept

The six simulation phases map directly onto the **real engineering workflow** used to fabricate synthetic structural‑colour materials:

> **Target colour → crystal geometry → chemistry → dielectric materials → film appearance**

This makes the tool a **virtual prototyping environment**, replacing costly trial‑and‑error synthesis with fast computational screening.

---

## Six‑Phase Engineering Workflow

### Phase Overview

| Phase | Purpose |
|-----|------|
| Phase 2 | Self‑assembly & spontaneous curvature |
| Phase 3 | Crystal symmetry classification |
| Phase 4 | Lattice constant prediction |
| Phase 5 | Photonic band structure calculation |
| Phase 6 | Polycrystalline appearance simulation |

---

## 1. Designing the Target Colour  
### *(Phase 4 + Phase 5)*

The most direct engineering application is **colour‑on‑demand design**.

By adjusting the **protein loading parameter `P`**, the tool predicts the resulting **lattice constant `a_nm`**, which sets the **photonic stop band** (reflected colour).

### Example Mapping

| Protein Loading `P` | Lattice Constant (nm) | Approximate Colour |
|--------------------|----------------------|------------------|
| 0.00 | 221 | Near‑UV / Violet |
| 0.25 | 260 | Violet / Blue |
| 0.50 | 293 | Blue / Cyan |
| 0.75 | 323 | Cyan / Green |
| 1.00 | 350 | Green |

**Engineering Use Case**

An engineer targeting a pigment‑free blue coating can:

1. Sweep `P` in **Phase 4** to identify the target lattice constant  
2. Use **Phase 5** to verify the photonic stop band  
3. Proceed to synthesis only after optical validation  

---

## 2. Selecting the Crystal Geometry  
### *(Phase 3)*

The tool distinguishes between two bicontinuous cubic geometries.

### Gyroid — *Ia3̄d*
- Single continuous network  
- Easier to fabricate  
- Compatible with block‑copolymer self‑assembly and two‑photon lithography  

### Double‑Diamond — *Pn3̄m*
- Two interpenetrating networks  
- Wider photonic bandgap  
- Higher reflectivity and colour saturation  

By tuning the **interface width (`lam`)** and **protein loading (`P`)**, Phase 3 predicts which geometry is likely to emerge for a given fabrication chemistry.

---

## 3. Optimising Dielectric Contrast  
### *(Phase 5)*

Photonic bandgap width depends critically on **dielectric contrast**.

Natural reference system:
- Chitin: ε = 2.56  
- Air: ε = 1.0  

Synthetic material pairs can be explored by modifying `eps_high` in `phase5_photonics.py`.

### Example Applications

- **Titania / air** (ε ≈ 6.25)  
  → Strong reflectivity, wide bandgaps  

- **Silicon / polymer**  
  → Near‑infrared photonic devices  

- **Hydrogel / water**  
  → Stimuli‑responsive structural colour sensors  

The plane‑wave expansion solver immediately returns the revised band structure, enabling rapid material screening.

---

## 4. Predicting Polycrystalline Appearance  
### *(Phase 6)*

Real films are **polycrystalline**, not perfect single crystals.

Phase 6 simulates:
- Multiple misoriented domains  
- Local lattice‑constant variation  
- Resulting colour non‑uniformity and iridescence  

### Engineering Applications

**Quality Control**
- Narrow `P` spread → uniform colour  
- Wide `P` spread → shimmering, angle‑dependent appearance  

**Intentional Design**
- Cosmetics  
- Security inks  
- Decorative and architectural coatings  

Domain size and disorder are controlled using:
- `n_domains`
- `P_min` and `P_max`

---

## 5. Guiding Self‑Assembly Protocols  
### *(Phase 2)*

Phase 2 directly links simulation outputs to **experimental chemistry**.

Morphology transitions are controlled by the **effective spontaneous curvature `H₀(P)`**, which can be tuned experimentally via:

- **Block‑copolymer composition**  
  (hydrophilic vs hydrophobic block lengths)

- **Lipid mixtures**  
  (e.g. DOPE vs DPPC)

- **Additives and surfactants**  
  (salts, co‑solvents, interfacial modifiers)

By reading transition points from the `lam_eff(P)` curve, chemists can translate simulation parameters into **target molecular compositions**.

---

## Practical Workflow Summary

A complete biomimetic engineering workflow using this tool:

1. **Specify the target colour**  
   → Phase 4 → lattice constant `a_nm`

2. **Select the crystal geometry**  
   → Phase 3 → gyroid or double‑diamond

3. **Translate to chemistry**  
   → Phase 2 → block‑copolymer ratio or lipid composition

4. **Screen dielectric materials**  
   → Phase 5 → maximise bandgap and reflectivity

5. **Predict final film appearance**  
   → Phase 6 → colour uniformity or intentional disorder

---

## Why This Tool Matters

- Eliminates many costly synthesis‑and‑characterisation loops  
- Enables **inverse photonic design**  
- Bridges:
  - Soft‑matter physics  
  - Photonic band‑structure theory  
  - Fabrication chemistry  
  - Real‑world visual appearance  

**cubic‑membrane‑photonics** is not just a biological model —  
it is a **general design engine for bicontinuous photonic materials**.

---

## Repository

GitHub:  
https://github.com/manarai/cubic-menbrane-photonic
