"""
run_demo.py
===========
Runs a fast simulated example of the full 6-phase pipeline.

Grid size N=32 and reduced step counts are used so the demo
completes in a reasonable time on a standard CPU.
For production runs, increase N to 64-128 and n_steps to 2000+.
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
FIG  = ROOT / "figures"
DATA = ROOT / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

# ── shared demo parameters ───────────────────────────────────────────────────
N       = 32      # grid size (32^3 = 32768 voxels)
N_STEPS = 200     # time steps per simulation (Allen-Cahn converges fast)
DT      = 0.05    # time step
LAM     = 0.1     # interface width parameter
SEED    = 42

# ════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Baseline
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 1 — Baseline Phase-Field Model")
print("="*60)

from phase1_baseline import run_baseline, plot_isosurface_slice

phi_baseline = run_baseline(N=N, lam=LAM, dt=DT, n_steps=N_STEPS,
                             seed=SEED, save_snapshots=True, out_dir=FIG)
plot_isosurface_slice(phi_baseline, FIG, tag="baseline")
np.save(DATA / "phi_baseline.npy", phi_baseline)
print("[Phase 1] Complete.")

# ════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Spontaneous Curvature Sweep
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 2 — Spontaneous Curvature (Protein Loading Sweep)")
print("="*60)

from phase2_curvature import (run_with_curvature, sweep_protein_loading,
                               spontaneous_curvature, effective_kappa)

P_sweep = [0.0, 0.25, 0.5, 0.75, 1.0]
results_p2 = {}
for P in P_sweep:
    print(f"  P = {P:.2f} ...", end=" ", flush=True)
    phi = run_with_curvature(P=P, N=N, lam=LAM, dt=DT,
                              n_steps=N_STEPS, seed=SEED)
    results_p2[P] = phi
    np.save(DATA / f"phi_P{int(P*100):03d}.npy", phi)
    print("done")

# Comparison figure
fig, axes = plt.subplots(1, len(P_sweep), figsize=(4*len(P_sweep), 4))
for ax, P in zip(axes, P_sweep):
    mid = results_p2[P].shape[2] // 2
    im = ax.imshow(results_p2[P][:, :, mid], cmap='RdBu_r',
                   vmin=-1.2, vmax=1.2, origin='lower')
    ax.set_title(f"P={P:.2f}\nH₀={spontaneous_curvature(P):.2f}", fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("Phase 2 — Membrane morphology vs protein loading P",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG / "phase2_curvature_sweep.png", dpi=150, bbox_inches='tight')
plt.close()
print("[Phase 2] Complete.")

# ════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Symmetry Identification
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 3 — Symmetry Identification")
print("="*60)

from phase3_symmetry import (classify_morphology, power_spectrum_1d,
                              euler_characteristic, MORPHOLOGY_COLORS)

morphologies = {}
for P, phi in results_p2.items():
    morph = classify_morphology(phi)
    chi   = euler_characteristic(phi)
    morphologies[P] = morph
    print(f"  P={P:.2f}  →  {morph:12s}  (χ = {chi})")

# Power spectra comparison
fig, ax = plt.subplots(figsize=(7, 4))
colors = plt.cm.viridis(np.linspace(0, 1, len(P_sweep)))
for (P, phi), color in zip(results_p2.items(), colors):
    k, power = power_spectrum_1d(phi)
    ax.semilogy(k, power + 1e-12, label=f"P={P:.2f} ({morphologies[P]})",
                color=color, linewidth=1.8)
ax.set_xlabel("|k| (grid units)", fontsize=11)
ax.set_ylabel("Power (log scale)", fontsize=11)
ax.set_title("Phase 3 — Spherically averaged power spectra", fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "phase3_power_spectra.png", dpi=150, bbox_inches='tight')
plt.close()

# Phase diagram (P vs morphology bar chart)
fig, ax = plt.subplots(figsize=(7, 3))
morph_labels = [morphologies[P] for P in P_sweep]
bar_colors   = [MORPHOLOGY_COLORS.get(m, 'grey') for m in morph_labels]
ax.bar([str(P) for P in P_sweep], [1]*len(P_sweep), color=bar_colors,
       edgecolor='k', linewidth=0.8)
ax.set_xlabel("Protein loading P", fontsize=11)
ax.set_title("Phase 3 — Morphology classification", fontsize=12, fontweight='bold')
ax.set_yticks([])
for i, m in enumerate(morph_labels):
    ax.text(i, 0.5, m, ha='center', va='center', fontsize=10, fontweight='bold',
            color='white' if m in ('Gyroid', 'Diamond') else 'black')
plt.tight_layout()
plt.savefig(FIG / "phase3_phase_diagram.png", dpi=150, bbox_inches='tight')
plt.close()
print("[Phase 3] Complete.")

with open(DATA / "morphology_classification.json", "w") as f:
    json.dump({str(P): m for P, m in morphologies.items()}, f, indent=2)

# ════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Lattice Scaling
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 4 — Lattice Scaling to Visible-Light Regime")
print("="*60)

from phase4_scaling import (measure_lattice_constant, helfrich_length,
                             VISIBLE_MIN_NM, VISIBLE_MAX_NM, TARGET_NM)

raw_scaling = {}
for P, phi in results_p2.items():
    a_sim, k_peak = measure_lattice_constant(phi)
    xi = helfrich_length(P, LAM)
    raw_scaling[P] = {'a_sim': a_sim, 'k_peak': k_peak, 'xi': xi}

# Calibrate: P=1 → TARGET_NM
P_ref = max(P_sweep)
a_nat_ref = raw_scaling[P_ref]['a_sim'] * raw_scaling[P_ref]['xi']
scale_factor = TARGET_NM / (a_nat_ref if a_nat_ref > 1e-9 else 1.0)

scaling_results = {}
for P, d in raw_scaling.items():
    a_nm = d['a_sim'] * d['xi'] * scale_factor
    scaling_results[P] = {**d, 'a_nm': a_nm,
                          'in_visible': VISIBLE_MIN_NM <= a_nm <= VISIBLE_MAX_NM}
    print(f"  P={P:.2f}  a_sim={d['a_sim']:.3f}  ξ={d['xi']:.3f}  "
          f"a_nm={a_nm:.1f} nm  {'✓ visible' if scaling_results[P]['in_visible'] else ''}")

P_arr = np.array(P_sweep)
a_nm_arr = np.array([scaling_results[P]['a_nm'] for P in P_sweep])
xi_arr   = np.array([scaling_results[P]['xi']   for P in P_sweep])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.plot(P_arr, a_nm_arr, 'o-', color='#D65F5F', linewidth=2, markersize=8)
ax.axhspan(VISIBLE_MIN_NM, VISIBLE_MAX_NM, alpha=0.15, color='gold',
           label='Visible range (200–500 nm)')
ax.axhline(TARGET_NM, linestyle='--', color='grey', linewidth=1)
ax.set_xlabel("Protein loading P", fontsize=11)
ax.set_ylabel("Lattice constant a (nm)", fontsize=11)
ax.set_title("Phase 4 — Lattice constant vs P", fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(P_arr, xi_arr, 's-', color='#4878CF', linewidth=2, markersize=8)
ax.set_xlabel("Protein loading P", fontsize=11)
ax.set_ylabel("Helfrich length ξ(P)", fontsize=11)
ax.set_title("Phase 4 — Membrane length scale vs P", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG / "phase4_lattice_scaling.png", dpi=150, bbox_inches='tight')
plt.close()
print("[Phase 4] Complete.")

with open(DATA / "lattice_scaling.json", "w") as f:
    json.dump({str(P): {k: float(v) for k, v in d.items()}
               for P, d in scaling_results.items()}, f, indent=2)

# ════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Photonic Band Structure
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 5 — Photonic Band Structure")
print("="*60)

from phase5_photonics import (compute_band_structure, plot_band_structure,
                               plot_dielectric_slices, find_stop_band)

band_results = {}
for P in [0.5, 1.0]:
    phi = results_p2[P]
    a_nm = scaling_results[P]['a_nm']
    morph = morphologies[P]
    tag = f"P{int(P*100):03d}_{morph.lower()}"
    print(f"  Computing bands for P={P} ({morph}), a={a_nm:.0f} nm ...", flush=True)
    plot_dielectric_slices(phi, FIG, tag=tag)
    result = compute_band_structure(phi, a_nm=a_nm, n_pw=2, n_kpoints=8)
    plot_band_structure(result, FIG, tag=tag)
    band_results[P] = result
    if result['stop_band']:
        sb = result['stop_band']
        lam_lo = a_nm / sb[1]
        lam_hi = a_nm / sb[0]
        print(f"    Stop band: Ω = {sb[0]:.4f}–{sb[1]:.4f}  "
              f"→  λ = {lam_lo:.0f}–{lam_hi:.0f} nm  "
              f"(gap ratio = {result['gap_ratio']:.4f})")
    else:
        print("    No complete stop band found.")

print("[Phase 5] Complete.")

# ════════════════════════════════════════════════════════════════════════════
# PHASE 6 — Polycrystalline Domains
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 6 — Polycrystalline Domains")
print("="*60)

from phase6_polycrystal import (assemble_polycrystal, plot_polycrystal,
                                 plot_domain_analysis, measure_domain_sizes)

phi_poly, domain_map, P_vals, centers = assemble_polycrystal(
    N=N, n_domains=6, P_min=0.4, P_max=1.0,
    lam=LAM, dt=DT, n_steps=N_STEPS, seed=SEED
)

np.save(DATA / "phi_polycrystal.npy", phi_poly)
np.save(DATA / "domain_map.npy", domain_map)

sizes_nm, sizes_um = measure_domain_sizes(domain_map, a_nm=350.0)
plot_polycrystal(phi_poly, domain_map, P_vals, FIG)
plot_domain_analysis(domain_map, P_vals, sizes_um, FIG)

print("\n  Domain sizes:")
for i, (nm, um) in enumerate(zip(sizes_nm.values(), sizes_um.values())):
    print(f"    Domain {i}: P={P_vals[i]:.3f}  size={um:.2f} μm")

print("[Phase 6] Complete.")

# ════════════════════════════════════════════════════════════════════════════
# SUMMARY FIGURE — All phases in one panel
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Generating summary figure ...")
print("="*60)

fig = plt.figure(figsize=(20, 10))
fig.suptitle("Cubic Membrane Templating — Full Pipeline Summary",
             fontsize=16, fontweight='bold', y=1.01)

# Row 1: Phase 1 baseline, Phase 2 P=0.5, Phase 2 P=1.0
ax1 = fig.add_subplot(2, 4, 1)
mid = phi_baseline.shape[2] // 2
im = ax1.imshow(phi_baseline[:, :, mid], cmap='RdBu_r', vmin=-1.2, vmax=1.2, origin='lower')
ax1.set_title("Phase 1\nBaseline (P=0)", fontsize=10, fontweight='bold')
ax1.axis('off')
plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

ax2 = fig.add_subplot(2, 4, 2)
phi_05 = results_p2[0.5]
im = ax2.imshow(phi_05[:, :, mid], cmap='RdBu_r', vmin=-1.2, vmax=1.2, origin='lower')
ax2.set_title(f"Phase 2\nP=0.5 ({morphologies[0.5]})", fontsize=10, fontweight='bold')
ax2.axis('off')
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

ax3 = fig.add_subplot(2, 4, 3)
phi_10 = results_p2[1.0]
im = ax3.imshow(phi_10[:, :, mid], cmap='RdBu_r', vmin=-1.2, vmax=1.2, origin='lower')
ax3.set_title(f"Phase 2\nP=1.0 ({morphologies[1.0]})", fontsize=10, fontweight='bold')
ax3.axis('off')
plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

# Phase 4 lattice scaling
ax4 = fig.add_subplot(2, 4, 4)
ax4.plot(P_arr, a_nm_arr, 'o-', color='#D65F5F', linewidth=2, markersize=7)
ax4.axhspan(VISIBLE_MIN_NM, VISIBLE_MAX_NM, alpha=0.15, color='gold',
            label='Visible')
ax4.set_xlabel("P", fontsize=9)
ax4.set_ylabel("a (nm)", fontsize=9)
ax4.set_title("Phase 4\nLattice Scaling", fontsize=10, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# Phase 5 band structure (P=1.0)
ax5 = fig.add_subplot(2, 4, 5)
bands = band_results[1.0]['bands']
n_k, n_b = bands.shape
x = np.arange(n_k)
colors_b = plt.cm.tab10(np.linspace(0, 1, n_b))
for b in range(n_b):
    ax5.plot(x, bands[:, b], color=colors_b[b], linewidth=1.2)
sb = band_results[1.0]['stop_band']
if sb:
    ax5.axhspan(sb[0], sb[1], alpha=0.2, color='red', label='Stop band')
lp = band_results[1.0]['label_pos']
ll = band_results[1.0]['labels']
ax5.set_xticks(lp); ax5.set_xticklabels(ll, fontsize=8)
ax5.set_ylabel("Ω = ωa/2πc", fontsize=9)
ax5.set_title("Phase 5\nBand Structure (P=1.0)", fontsize=10, fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.2)

# Phase 5 wavelength map (P=1.0)
ax6 = fig.add_subplot(2, 4, 6)
lam_nm = band_results[1.0]['lambda_nm']
for b in range(n_b):
    ax6.plot(x, lam_nm[:, b], color=colors_b[b], linewidth=1.2)
if sb:
    lam_lo = band_results[1.0]['a_nm'] / sb[1]
    lam_hi = band_results[1.0]['a_nm'] / sb[0]
    ax6.axhspan(lam_lo, lam_hi, alpha=0.2, color='red')
ax6.axhspan(380, 700, alpha=0.08, color='gold', label='Visible')
ax6.set_xticks(lp); ax6.set_xticklabels(ll, fontsize=8)
ax6.set_ylabel("λ (nm)", fontsize=9)
ax6.set_ylim(0, 1200)
ax6.set_title("Phase 5\nWavelength Map", fontsize=10, fontweight='bold')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.2)

# Phase 6 polycrystal domain map
ax7 = fig.add_subplot(2, 4, 7)
n_dom = len(P_vals)
cmap_dom = plt.cm.get_cmap('tab20', n_dom)
im = ax7.imshow(domain_map[:, :, domain_map.shape[2]//2],
                cmap=cmap_dom, vmin=0, vmax=n_dom-1,
                origin='lower', interpolation='nearest')
ax7.set_title("Phase 6\nDomain Map", fontsize=10, fontweight='bold')
ax7.axis('off')
plt.colorbar(im, ax=ax7, fraction=0.046, pad=0.04, label='Domain ID')

# Phase 6 polycrystal phi
ax8 = fig.add_subplot(2, 4, 8)
im = ax8.imshow(phi_poly[:, :, phi_poly.shape[2]//2],
                cmap='RdBu_r', vmin=-1.2, vmax=1.2,
                origin='lower', interpolation='nearest')
ax8.set_title("Phase 6\nPolycrystalline φ(r)", fontsize=10, fontweight='bold')
ax8.axis('off')
plt.colorbar(im, ax=ax8, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(FIG / "summary_all_phases.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Summary figure saved: {FIG / 'summary_all_phases.png'}")

print("\n" + "="*60)
print("ALL PHASES COMPLETE")
print(f"Figures saved to: {FIG}")
print(f"Data saved to:    {DATA}")
print("="*60)
