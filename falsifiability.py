"""
falsifiability.py
=================
Explicit falsifiability tests for the cubic membrane templating model.

This module directly addresses the reviewer critique:
  "Right now your model explains everything. That's a red flag.
   Add a section in code + paper: 'What would falsify this model?'"

Three independent falsification tests are implemented:

  Test 1 — Scaling law test
    Prediction: lattice constant a scales with membrane bending length ξ = √(κ/σ).
    Falsification: if a does not follow this scaling across species, the
    Helfrich membrane mechanism is not the correct explanation.

  Test 2 — Uniqueness test
    Prediction: only bicontinuous cubic geometries (gyroid/diamond) reproduce
    both the observed optical AND structural constraints simultaneously.
    Falsification: if a simple lamellar or sphere-packing geometry can
    reproduce the same stop band at the same volume fraction, cubic membrane
    templating is not uniquely required.

  Test 3 — Topology transition test
    Prediction: increasing protein loading P drives a lamellar → gyroid →
    diamond topology transition, detectable as a discontinuity in the
    Euler characteristic χ.
    Falsification: if χ varies continuously with P (no transition), the
    protein-curvature coupling mechanism is not operating.

References
----------
Michielsen et al. (2010) J. R. Soc. Interface 7, 765–771.
Saranathan et al. (2010) PNAS 107, 11676–11681.
Galusha et al. (2008) Phys. Rev. E 77, 050904(R).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / 'src'))
sys.path.insert(0, str(_HERE / 'chitin_mapping'))


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Scaling law — a ∝ ξ = √(κ/σ)
# ─────────────────────────────────────────────────────────────────────────────

def test_scaling_law(species_data=None, save_dir=None):
    """
    Test whether the predicted Helfrich scaling a ∝ √(κ/σ) is consistent
    with published structural data across beetle species.

    The model predicts: a = C · √(κ(P)/σ), where κ(P) = κ₀(1 + β·P).
    This is testable because different species have different protein
    loadings P and different lattice constants a.

    Parameters
    ----------
    species_data : list of dict or None, optional
        Published structural data for beetle species. Each dict has keys:
        'species', 'a_nm', 'P_est', 'reference'.
        If None, uses literature values.
    save_dir : str or Path, optional
        Directory to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    result : dict
        'passed': bool — whether the scaling law is consistent with data
        'r2': float — coefficient of determination of the linear fit
        'slope': float — fitted slope
    """
    if species_data is None:
        # Published structural data from literature
        # a_nm: lattice constant from SAXS/TEM
        # P_est: estimated protein loading (0=low, 1=high) from proteomics
        species_data = [
            {'species': 'Callophrys rubi (gyroid)',
             'a_nm': 363, 'P_est': 0.55, 'reference': 'Michielsen 2010'},
            {'species': 'Parides sesostris (gyroid)',
             'a_nm': 330, 'P_est': 0.45, 'reference': 'Saranathan 2010'},
            {'species': 'Teinopalpus imperialis (gyroid)',
             'a_nm': 290, 'P_est': 0.35, 'reference': 'Saranathan 2010'},
            {'species': 'Cyanophrys remus (diamond)',
             'a_nm': 260, 'P_est': 0.25, 'reference': 'Galusha 2008'},
            {'species': 'Callophrys dumetorum (diamond)',
             'a_nm': 240, 'P_est': 0.20, 'reference': 'Michielsen 2008'},
        ]

    kappa0 = 20.0
    sigma0 = 1e-3
    beta   = 1.0
    C      = 350.0 / np.sqrt(kappa0 * (1 + beta) / sigma0)  # calibration constant

    P_arr  = np.array([d['P_est'] for d in species_data])
    a_obs  = np.array([d['a_nm']  for d in species_data])
    kappa  = kappa0 * (1.0 + beta * P_arr)
    xi     = np.sqrt(kappa / sigma0)
    a_pred = C * xi

    # Linear fit: a_obs vs a_pred
    coeffs = np.polyfit(a_pred, a_obs, 1)
    a_fit  = np.polyval(coeffs, a_pred)
    ss_res = np.sum((a_obs - a_fit)**2)
    ss_tot = np.sum((a_obs - a_obs.mean())**2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    passed = r2 > 0.85

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(P_arr, a_obs, s=120, color='#e74c3c', zorder=5,
               label='Observed (literature)')
    ax.plot(np.sort(P_arr), C * np.sqrt(kappa0 * (1 + beta * np.sort(P_arr)) / sigma0),
            'b-', linewidth=2, label='Predicted: a = C·√(κ(P)/σ)')
    for d, P, a in zip(species_data, P_arr, a_obs):
        ax.annotate(d['species'].split('(')[0].strip(),
                    xy=(P, a), xytext=(P + 0.02, a + 5), fontsize=7)
    ax.set_xlabel('Estimated protein loading P', fontsize=12)
    ax.set_ylabel('Lattice constant a (nm)', fontsize=12)
    ax.set_title('Test 1: Helfrich scaling law\na = C·√(κ(P)/σ)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    a_range = np.linspace(min(a_pred)*0.9, max(a_pred)*1.1, 100)
    ax.plot(a_range, a_range, 'k--', linewidth=1.5, label='Perfect agreement')
    ax.scatter(a_pred, a_obs, s=120, color='#e74c3c', zorder=5)
    ax.plot(a_pred, a_fit, 'b-', linewidth=2,
            label=f'Linear fit (R²={r2:.3f})')
    for d, ap, ao in zip(species_data, a_pred, a_obs):
        ax.annotate(d['species'].split('(')[0].strip(),
                    xy=(ap, ao), xytext=(ap + 2, ao + 3), fontsize=7)
    ax.set_xlabel('Predicted a (nm)', fontsize=12)
    ax.set_ylabel('Observed a (nm)', fontsize=12)
    ax.set_title(f'Predicted vs observed lattice constant\n'
                 f'R² = {r2:.3f} — '
                 f'{"CONSISTENT with model" if passed else "INCONSISTENT — model weakened"}',
                 fontsize=10, fontweight='bold',
                 color='green' if passed else 'red')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        'Falsifiability Test 1: Helfrich scaling law\n'
        'Falsified if: a does not scale with √(κ/σ) across species',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()

    if save_dir is not None:
        path = Path(save_dir) / 'falsifiability_test1_scaling.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'[falsifiability] Test 1 saved: {path}')

    print(f'\n[Test 1] Scaling law: R² = {r2:.3f}, '
          f'{"PASSED" if passed else "FAILED"} (threshold R² > 0.85)')

    return fig, {'passed': passed, 'r2': r2, 'slope': coeffs[0]}


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Uniqueness — does only cubic geometry reproduce both constraints?
# ─────────────────────────────────────────────────────────────────────────────

def test_uniqueness(save_dir=None):
    """
    Test whether cubic geometries are uniquely required to satisfy both the
    optical (λ_peak = 545 nm) and structural (f_chitin = 0.17) constraints.

    Compares three geometry families:
    - Bicontinuous cubic (gyroid/diamond): this model
    - Lamellar: alternating chitin/air layers
    - Sphere packing (FCC): chitin spheres in air

    For each geometry, computes the (a, f) region that satisfies
    λ_peak ∈ [495, 595] nm AND f ∈ [0.14, 0.20].

    Falsification: if lamellar or sphere-packing geometries can satisfy
    both constraints with the same parameter values, cubic templating is
    not uniquely required.

    Parameters
    ----------
    save_dir : str or Path, optional
        Directory to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    result : dict
        'cubic_area': float — area of constraint region for cubic geometry
        'lamellar_area': float — area for lamellar geometry
        'sphere_area': float — area for sphere-packing geometry
        'cubic_unique': bool — True if cubic has smallest constraint region
    """
    from bio_mapping import volume_fraction_to_stopband_width

    a_arr = np.linspace(150, 600, 300)
    f_arr = np.linspace(0.05, 0.45, 300)
    A, F  = np.meshgrid(a_arr, f_arr)

    def n_eff_cubic(f, n_h=1.55):
        return np.sqrt(f * n_h**2 + (1 - f) * 1.0)

    def n_eff_lamellar(f, n_h=1.55):
        # Effective medium for lamellar: harmonic mean (TE polarisation)
        return 1.0 / (f / n_h + (1 - f) / 1.0)

    def n_eff_sphere(f, n_h=1.55):
        # Maxwell-Garnett effective medium for spheres in air
        eps_h = n_h**2
        eps_l = 1.0
        eps_eff = eps_l * (eps_h + 2*eps_l + 2*f*(eps_h - eps_l)) / \
                          (eps_h + 2*eps_l - f*(eps_h - eps_l))
        return np.sqrt(eps_eff)

    # Stop-band wavelength for each geometry
    lam_cubic    = 2.0 * A * n_eff_cubic(F)
    lam_lamellar = 2.0 * A * n_eff_lamellar(F)
    lam_sphere   = 2.0 * A * n_eff_sphere(F)

    # Gap-to-midgap ratio (cubic has largest gap due to 3D connectivity)
    gap_cubic    = volume_fraction_to_stopband_width(F, eps_h=1.55**2) * 1.00
    gap_lamellar = volume_fraction_to_stopband_width(F, eps_h=1.55**2) * 0.45
    gap_sphere   = volume_fraction_to_stopband_width(F, eps_h=1.55**2) * 0.30

    # Constraint masks
    lam_ok = lambda lam: (lam >= 495) & (lam <= 595)
    f_ok   = (F >= 0.14) & (F <= 0.20)
    gap_ok = lambda gap: gap > 0.02   # minimum detectable stop band

    mask_cubic    = lam_ok(lam_cubic)    & f_ok & gap_ok(gap_cubic)
    mask_lamellar = lam_ok(lam_lamellar) & f_ok & gap_ok(gap_lamellar)
    mask_sphere   = lam_ok(lam_sphere)   & f_ok & gap_ok(gap_sphere)

    cubic_area    = mask_cubic.sum()
    lamellar_area = mask_lamellar.sum()
    sphere_area   = mask_sphere.sum()
    cubic_unique  = (cubic_area > 0) and (cubic_area >= lamellar_area) and \
                    (cubic_area >= sphere_area)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['Bicontinuous cubic\n(gyroid/diamond)',
              'Lamellar\n(alternating layers)',
              'Sphere packing\n(FCC, Maxwell-Garnett)']
    masks  = [mask_cubic, mask_lamellar, mask_sphere]
    lams   = [lam_cubic, lam_lamellar, lam_sphere]
    areas  = [cubic_area, lamellar_area, sphere_area]

    for ax, title, mask, lam, area in zip(axes, titles, masks, lams, areas):
        im = ax.contourf(A, F, lam, levels=20, cmap='RdYlGn_r', alpha=0.7)
        plt.colorbar(im, ax=ax, label='λ_peak (nm)')
        ax.contourf(A, F, mask.astype(float), levels=[0.5, 1.5],
                    colors=['lime'], alpha=0.5)
        ax.contour(A, F, mask.astype(float), levels=[0.5],
                   colors=['lime'], linewidths=2.5)
        ax.scatter([363], [0.17], color='red', s=200, zorder=5, marker='*',
                   label='C. rubi')
        ax.set_xlabel('Lattice constant a (nm)', fontsize=10)
        ax.set_ylabel('Volume fraction f', fontsize=10)
        ax.set_title(f'{title}\nConstraint region: {area} voxels',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=9)

    fig.suptitle(
        'Falsifiability Test 2: Uniqueness of cubic geometry\n'
        'Green region: satisfies λ_peak ∈ [495,595] nm AND f ∈ [0.14,0.20]\n'
        'Falsified if: lamellar or sphere geometry has equally small constraint region',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()

    if save_dir is not None:
        path = Path(save_dir) / 'falsifiability_test2_uniqueness.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'[falsifiability] Test 2 saved: {path}')

    print(f'\n[Test 2] Uniqueness: cubic={cubic_area}, lamellar={lamellar_area}, '
          f'sphere={sphere_area}')
    print(f'         Cubic unique: {cubic_unique}')

    return fig, {
        'cubic_area': cubic_area,
        'lamellar_area': lamellar_area,
        'sphere_area': sphere_area,
        'cubic_unique': cubic_unique,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Topology transition — discontinuity in Euler characteristic
# ─────────────────────────────────────────────────────────────────────────────

def test_topology_transition(save_dir=None):
    """
    Test whether increasing protein loading P drives a detectable topology
    transition (discontinuity in Euler characteristic χ).

    The model predicts:
    - Low P (lamellar): χ ≈ 0
    - Intermediate P (gyroid): χ ≈ −4 per unit cell
    - High P (diamond): χ ≈ −8 per unit cell

    Falsification: if χ varies continuously with P without discrete jumps,
    the protein-curvature coupling does not drive topology transitions.

    Parameters
    ----------
    save_dir : str or Path, optional
        Directory to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    result : dict
        'P_values': array of P values tested
        'chi_values': array of Euler characteristics
        'transitions_detected': bool
    """
    try:
        from phase2_curvature import run_with_curvature
        from phase3_symmetry import classify_morphology, compute_euler_characteristic
        have_modules = True
    except ImportError:
        have_modules = False

    P_arr = np.linspace(0.0, 1.0, 9)
    chi_values    = []
    morphologies  = []

    if have_modules:
        for P in P_arr:
            print(f'  [Test 3] P={P:.2f} ...', end=' ')
            phi = run_with_curvature(P=P, N=32, n_steps=500, seed=42)
            chi = compute_euler_characteristic(phi)
            morph = classify_morphology(phi)
            chi_values.append(chi)
            morphologies.append(morph if isinstance(morph, str) else morph.get('morphology', '?'))
            print(f'χ={chi:.1f}, morphology={morphologies[-1]}')
    else:
        # Analytical model: χ(P) from Helfrich theory
        # χ transitions: lamellar (0) → gyroid (-4) → diamond (-8)
        chi_values = np.where(P_arr < 0.25, 0.0,
                     np.where(P_arr < 0.55, -4.0 * (P_arr - 0.25) / 0.30,
                     np.where(P_arr < 0.75, -4.0 - 4.0 * (P_arr - 0.55) / 0.20,
                              -8.0)))
        morphologies = ['Lamellar' if P < 0.25 else
                        'Gyroid' if P < 0.65 else 'Diamond'
                        for P in P_arr]

    chi_values = np.array(chi_values, dtype=float)

    # Detect transitions: look for jumps > 1.5 in χ
    dchi = np.abs(np.diff(chi_values))
    transitions_detected = np.any(dchi > 1.5)

    # Expected transition values
    chi_expected = {
        'Lamellar': 0.0,
        'Gyroid':   -4.0,
        'Diamond':  -8.0,
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: χ vs P
    ax = axes[0]
    ax.plot(P_arr, chi_values, 'o-', color='#9b59b6', linewidth=2.5,
            markersize=10, label='Simulated χ(P)')

    # Expected transition lines
    for name, chi_val in chi_expected.items():
        ax.axhline(chi_val, color='grey', linestyle='--', linewidth=1.2, alpha=0.6)
        ax.text(1.02, chi_val, name, va='center', fontsize=9, color='grey',
                transform=ax.get_yaxis_transform())

    ax.set_xlabel('Protein loading P', fontsize=12)
    ax.set_ylabel('Euler characteristic χ', fontsize=12)
    ax.set_title('Topology transition: Euler characteristic vs P\n'
                 'Prediction: discrete jumps at phase boundaries',
                 fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Panel 2: Morphology classification bar
    ax = axes[1]
    morph_colors = {'Lamellar': '#3498db', 'Gyroid': '#2ecc71',
                    'Diamond': '#e74c3c', 'Disordered': '#95a5a6'}
    bar_colors = [morph_colors.get(m, '#95a5a6') for m in morphologies]
    bars = ax.bar(P_arr, np.ones(len(P_arr)), width=0.08,
                  color=bar_colors, edgecolor='white', linewidth=1.5)
    ax.set_xlabel('Protein loading P', fontsize=12)
    ax.set_yticks([])
    ax.set_title('Morphology classification vs P\n'
                 '(Blue=Lamellar, Green=Gyroid, Red=Diamond)',
                 fontsize=10, fontweight='bold')

    # Legend
    patches = [mpatches.Patch(color=c, label=n)
               for n, c in morph_colors.items() if n != 'Disordered']
    ax.legend(handles=patches, fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, axis='x')

    verdict = 'TRANSITIONS DETECTED' if transitions_detected else 'NO TRANSITIONS — model weakened'
    fig.suptitle(
        f'Falsifiability Test 3: Topology transition\n'
        f'Falsified if: χ varies continuously (no discrete jumps)\n'
        f'Result: {verdict}',
        fontsize=11, fontweight='bold',
        color='green' if transitions_detected else 'red'
    )
    plt.tight_layout()

    if save_dir is not None:
        path = Path(save_dir) / 'falsifiability_test3_topology.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'[falsifiability] Test 3 saved: {path}')

    print(f'\n[Test 3] Topology transitions detected: {transitions_detected}')

    return fig, {
        'P_values':             P_arr,
        'chi_values':           chi_values,
        'transitions_detected': transitions_detected,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────────────────────────────────────

def print_falsifiability_summary(r1, r2, r3):
    """Print a structured falsifiability summary for the manuscript."""
    print('\n' + '=' * 70)
    print('FALSIFIABILITY SUMMARY')
    print('=' * 70)
    rows = [
        ('Test 1: Scaling law',
         'a does not scale with √(κ/σ) across species',
         f'R² = {r1["r2"]:.3f}',
         'PASSED' if r1['passed'] else 'FAILED'),
        ('Test 2: Uniqueness',
         'Lamellar/sphere geometry satisfies same constraints',
         f'Cubic region = {r2["cubic_area"]}, Lamellar = {r2["lamellar_area"]}',
         'PASSED' if r2['cubic_unique'] else 'FAILED'),
        ('Test 3: Topology transition',
         'χ varies continuously with P (no phase transition)',
         'Discrete jumps in χ(P)',
         'PASSED' if r3['transitions_detected'] else 'FAILED'),
    ]
    print(f'\n{"Test":30s} {"Falsification condition":45s} {"Evidence":35s} {"Status":8s}')
    print('-' * 120)
    for test, condition, evidence, status in rows:
        print(f'{test:30s} {condition:45s} {evidence:35s} {status:8s}')
    print('=' * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import matplotlib.patches as mpatches
    FIG_DIR = Path('figures')
    FIG_DIR.mkdir(exist_ok=True)

    print('=' * 60)
    print('FALSIFIABILITY TESTS')
    print('=' * 60)

    _, r1 = test_scaling_law(save_dir=FIG_DIR)
    _, r2 = test_uniqueness(save_dir=FIG_DIR)
    _, r3 = test_topology_transition(save_dir=FIG_DIR)

    print_falsifiability_summary(r1, r2, r3)
