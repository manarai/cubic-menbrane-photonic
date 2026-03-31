Cubic‑Membrane‑Photonics
A Virtual Prototyping Tool for Biomimetic Structural Colour Engineering
cubic‑membrane‑photonics is a six‑phase computational pipeline for designing, simulating, and evaluating bicontinuous photonic crystals inspired by biological cubic membranes (e.g. beetle scales).
It enables colour‑on‑demand design, geometry selection, materials screening, and appearance prediction before any physical fabrication.
The tool is intended for biomimetic engineers, materials scientists, photonics researchers, and fabrication chemists developing pigment‑free structural colour materials.

Key Idea
The six simulation phases map directly onto the real engineering workflow used to fabricate synthetic structural‑colour materials:

Target colour → crystal geometry → chemistry → dielectric materials → film appearance

This makes the tool a virtual prototyping environment, replacing many costly trial‑and‑error synthesis cycles with fast computational screening.

Six‑Phase Engineering Workflow
Phase 1–2: Morphology & Self‑Assembly Physics
(Foundation for fabrication realism)

Models the transition from lamellar sheets to bicontinuous cubic phases
Controlled by the effective spontaneous curvature H0(P)H_0(P)H0​(P)
Directly translatable to experimental parameters such as:

Block‑copolymer composition
Lipid mixture ratios
Additives, salts, or co‑solvents




1. Designing the Target Colour
(Phase 4 + Phase 5)
The most direct engineering application is colour‑on‑demand design.
By adjusting the protein loading parameter PPP, the tool predicts the lattice constant anma_{\text{nm}}anm​ of the nanostructure, which determines the photonic stop band (reflected colour).



































Protein Loading PPPLattice Constant (nm)Approximate Colour0.00221Near‑UV / Violet0.25260Violet / Blue0.50293Blue / Cyan0.75323Cyan / Green1.00350Green
Engineering use case
An engineer targeting a specific colour (e.g. vivid blue for a pigment‑free coating):

Sweeps PPP in Phase 4 to find the required lattice constant
Uses Phase 5 to verify the photonic stop band
Proceeds to fabrication only after optical validation


2. Selecting the Crystal Geometry
(Phase 3)
The tool distinguishes between two bicontinuous cubic geometries:
Gyroid — Ia3̄d

Single continuous network
Easier to fabricate
Compatible with:

Block‑copolymer self‑assembly
Two‑photon lithography



Double‑Diamond — Pn3̄m

Two interpenetrating networks
Wider photonic bandgap
Higher reflectivity and colour saturation

By tuning interface width (lam) and protein loading (P), Phase 3 predicts which geometry a given fabrication chemistry will produce — allowing engineers to adjust parameters before synthesis.

3. Optimising Dielectric Contrast
(Phase 5)
Photonic bandgap width depends critically on dielectric contrast.
Natural reference:

Chitin: ε = 2.56
Air: ε = 1.0

Synthetic alternatives can be explored by modifying eps_high in phase5_photonics.py:

Titania / air (ε ≈ 6.25)
→ Strong reflectivity, wide bandgaps
Silicon / polymer
→ Near‑infrared photonic devices
Hydrogel / water
→ Stimuli‑responsive structural colour sensors

The plane‑wave expansion solver immediately returns the revised band structure, enabling rapid material screening.

4. Predicting Polycrystalline Appearance
(Phase 6)
Real films are polycrystalline, not perfect single crystals.
Phase 6 simulates:

Misoriented domains
Lattice‑constant variation
Resulting colour non‑uniformity

Engineering Applications
Quality Control

Narrow PPP distribution → uniform colour
Wide PPP distribution → increased iridescence

Intentional Aesthetic Design

Cosmetics
Security inks
Decorative and architectural coatings

Engineers can tune:

n_domains
P_min, P_max

to achieve a target visual effect.

5. Guiding Self‑Assembly Protocols
(Phase 2)
Phase 2 directly links simulation parameters to experimental chemistry.
Morphology transitions are controlled by effective spontaneous curvature H0(P)H_0(P)H0​(P), which can be tuned experimentally via:

Block‑copolymer ratios
(hydrophilic vs hydrophobic block lengths)
Lipid mixtures
(e.g. DOPE vs DPPC)
Additives and surfactants

By reading transition points from the lam_eff(P) curve, chemists can translate simulation outputs into target molecular compositions.

Practical Workflow Summary
A complete biomimetic engineering workflow using this tool:


Specify target colour
→ Phase 4 → lattice constant anma_{\text{nm}}anm​


Choose crystal geometry
→ Phase 3 → gyroid or double‑diamond


Translate to chemistry
→ Phase 2 → block‑copolymer or lipid composition


Screen dielectric materials
→ Phase 5 → maximise bandgap and reflectivity


Predict film appearance
→ Phase 6 → colour uniformity or intentional disorder



Why This Tool Matters

Replaces costly synthesis‑and‑characterisation loops
Enables inverse photonic design
Bridges:

Soft‑matter physics
Photonic band theory
Fabrication chemistry
Real‑world visual appearance



cubic‑membrane‑photonics is not just a biological model —
it is a general design engine for bicontinuous photonic materials.
