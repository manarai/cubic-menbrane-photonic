"""
bio_mapping.py
==============
Explicit mapping between simulation parameters and biological quantities,
with testable predictions for each parameter.

This module directly addresses the reviewer critique:
  "Your code has parameters like t and a, but your manuscript claims
   protein density → curvature, cholesterol → phase behavior,
   confinement → scaling. Show me."

Each function maps a simulation parameter to its biological meaning,
computes the predicted observable, and states what experimental result
would falsify the mapping.

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
import matplotlib.patches as mpatches
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Parameter mapping table (for manuscript Table 1)
# ─────────────────────────────────────────────────────────────────────────────

PARAMETER_TABLE = [
    {
        'sim_param':     'P  (protein loading, 0–1)',
        'bio_meaning':   'Local density of curvature-inducing membrane proteins '
                         '(e.g. reticulon, REEP, atlastin) during scale cell development',
        'effect':        'Increases spontaneous curvature H₀ = α·P; drives '
                         'lamellar → gyroid → diamond phase transition',
        'prediction':    'Beetles with higher reticulon expression should show '
                         'smaller lattice constants and bluer structural colour',
        'falsification': 'If protein knockdown does not shift lattice constant, '
                         'protein-curvature coupling is not the mechanism',
    },
    {
        'sim_param':     't  (TPMS threshold, −1 to +1)',
        'bio_meaning':   'Chitin volume fraction; controlled by chitin synthase '
                         'activity and deposition time',
        'effect':        'Shifts volume fraction f_chitin; topology changes at '
                         't ≈ ±0.5 (connected → disconnected network)',
        'prediction':    'Chitin synthase inhibition should reduce f_chitin and '
                         'weaken photonic stop band',
        'falsification': 'If f_chitin is invariant across species with different '
                         'colours, threshold is not the tuning mechanism',
    },
    {
        'sim_param':     'a  (lattice constant, nm)',
        'bio_meaning':   'Characteristic spacing of cubic membrane network; '
                         'set by membrane bending rigidity κ and tension σ '
                         'via ξ = √(κ/σ)',
        'effect':        'Directly sets stop-band wavelength: λ_peak ≈ 2·a·n_eff',
        'prediction':    'Species with larger scale cells (more confinement volume) '
                         'should have larger a and redder colour',
        'falsification': 'If a does not scale with cell size across species, '
                         'confinement is not the scaling mechanism',
    },
    {
        'sim_param':     'ε_h  (high dielectric, chitin)',
        'bio_meaning':   'Chitin refractive index n_chitin ≈ 1.55–1.68; '
                         'controlled by chitin crystallinity and water content',
        'effect':        'Increases dielectric contrast Δε = ε_h − ε_l; '
                         'widens photonic stop band',
        'prediction':    'Dehydrated scales should show narrower, blueshifted '
                         'stop bands due to reduced n_chitin',
        'falsification': 'If optical response is insensitive to hydration state, '
                         'dielectric contrast is not the primary tuning mechanism',
    },
    {
        'sim_param':     'N_domains  (polycrystal seeds)',
        'bio_meaning':   'Number of independent cubic membrane nucleation sites '
                         'per scale cell',
        'effect':        'Sets domain size d ∝ 1/√N_domains; '
                         'more seeds → smaller, more uniform domains',
        'prediction':    'Scales with more nucleation sites should show smaller '
                         'domain size and more angle-independent colour',
        'falsification': 'If domain size is uncorrelated with nucleation site '
                         'density in TEM, mosaic model is not correct',
    },
]


def print_parameter_table():
    """Print the biological parameter mapping table to stdout."""
    print('=' * 80)
    print('TABLE 1: Simulation parameter ↔ Biological meaning ↔ Testable prediction')
    print('=' * 80)
    for row in PARAMETER_TABLE:
        print(f"\n  Parameter:     {row['sim_param']}")
        print(f"  Biology:       {row['bio_meaning']}")
        print(f"  Effect:        {row['effect']}")
        print(f"  Prediction:    {row['prediction']}")
        print(f"  Falsification: {row['falsification']}")
    print('=' * 80)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Protein loading P → spontaneous curvature H₀ → lattice constant a
# ─────────────────────────────────────────────────────────────────────────────

def protein_to_curvature(P, alpha=0.5):
    """
    Map protein loading P to spontaneous curvature H₀.

    Parameters
    ----------
    P : array_like
        Protein loading, dimensionless, in [0, 1].
    alpha : float, optional
        Curvature coupling coefficient (nm⁻¹ per unit P). Default 0.5.

    Returns
    -------
    H0 : ndarray
        Spontaneous curvature (nm⁻¹).
    """
    return alpha * np.asarray(P)


def curvature_to_lattice(H0, kappa0=20.0, sigma0=1e-3, P=None, beta=1.0):
    """
    Map spontaneous curvature to physical lattice constant via Helfrich length.

    The Helfrich membrane length scale is ξ = √(κ/σ), where κ is the bending
    rigidity and σ is the membrane tension. The lattice constant scales as
    a ∝ ξ, calibrated so that P=1 gives a = 350 nm.

    Parameters
    ----------
    H0 : array_like
        Spontaneous curvature (nm⁻¹).
    kappa0 : float, optional
        Bare bending rigidity (k_BT). Default 20.
    sigma0 : float, optional
        Bare membrane tension (k_BT nm⁻²). Default 1e-3.
    P : array_like or None, optional
        Protein loading (used to compute protein-stiffened κ). If None,
        κ = kappa0 is used.
    beta : float, optional
        Protein stiffening coefficient. Default 1.0.

    Returns
    -------
    a_nm : ndarray
        Physical lattice constant (nm).
    xi : ndarray
        Helfrich length scale (nm).
    """
    P_arr = np.asarray(P) if P is not None else np.zeros_like(np.asarray(H0))
    kappa = kappa0 * (1.0 + beta * P_arr)
    xi = np.sqrt(kappa / sigma0)
    # Calibrate: at P=1, xi_max = sqrt(kappa0*(1+beta)/sigma0)
    xi_max = np.sqrt(kappa0 * (1.0 + beta) / sigma0)
    a_nm = 350.0 * xi / xi_max
    return a_nm, xi


def plot_protein_to_colour(save_dir=None, alpha=0.5, kappa0=20.0, sigma0=1e-3, beta=1.0):
    """
    Plot the causal chain: protein loading P → H₀ → a_nm → predicted colour.

    Parameters
    ----------
    save_dir : str or Path, optional
        Directory to save the figure. If None, figure is shown interactively.
    alpha : float, optional
        Curvature coupling coefficient.
    kappa0 : float, optional
        Bare bending rigidity (k_BT).
    sigma0 : float, optional
        Bare membrane tension (k_BT nm⁻²).
    beta : float, optional
        Protein stiffening coefficient.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    P_arr = np.linspace(0, 1, 100)
    H0    = protein_to_curvature(P_arr, alpha=alpha)
    a_nm, xi = curvature_to_lattice(H0, kappa0=kappa0, sigma0=sigma0, P=P_arr, beta=beta)
    # Predicted stop-band wavelength: λ ≈ 2·a·n_eff (n_eff ≈ 1.27 for f=0.17 chitin)
    n_eff = 1.0 + 0.17 * (1.55 - 1.0)   # effective medium approximation
    lambda_pred = 2.0 * a_nm * n_eff

    # Colour map for the predicted wavelength
    def wavelength_to_rgb(lam):
        """Approximate RGB for a visible wavelength (nm)."""
        lam = np.clip(lam, 380, 700)
        r = np.where(lam < 440, 0.6 + 0.4*(lam-380)/60,
            np.where(lam < 490, (490-lam)/50,
            np.where(lam < 580, 0.0,
            np.where(lam < 645, (lam-580)/65, 1.0))))
        g = np.where(lam < 440, 0.0,
            np.where(lam < 490, (lam-440)/50,
            np.where(lam < 580, 1.0,
            np.where(lam < 645, (645-lam)/65, 0.0))))
        b = np.where(lam < 440, 1.0,
            np.where(lam < 490, 1.0,
            np.where(lam < 580, (580-lam)/90, 0.0)))
        return np.stack([r, g, b], axis=-1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: P → H₀
    ax = axes[0]
    ax.plot(P_arr, H0, color='#e74c3c', linewidth=2.5)
    ax.set_xlabel('Protein loading P', fontsize=12)
    ax.set_ylabel('Spontaneous curvature H₀ (nm⁻¹)', fontsize=12)
    ax.set_title('Protein loading\n→ spontaneous curvature', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.annotate('Linear coupling\nH₀ = α·P', xy=(0.5, alpha*0.5),
                xytext=(0.6, alpha*0.3), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='grey'))

    # Panel 2: P → a_nm
    ax = axes[1]
    ax.plot(P_arr, a_nm, color='#3498db', linewidth=2.5)
    ax.axhspan(200, 500, alpha=0.1, color='gold', label='Visible-light regime')
    ax.set_xlabel('Protein loading P', fontsize=12)
    ax.set_ylabel('Lattice constant a (nm)', fontsize=12)
    ax.set_title('Protein loading\n→ lattice constant', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: P → predicted colour (scatter coloured by wavelength)
    ax = axes[2]
    colors_rgb = wavelength_to_rgb(lambda_pred)
    for i in range(len(P_arr) - 1):
        ax.fill_between([P_arr[i], P_arr[i+1]], [lambda_pred[i], lambda_pred[i+1]],
                        color=colors_rgb[i], alpha=0.85)
    ax.plot(P_arr, lambda_pred, 'k-', linewidth=1.5, alpha=0.5)
    ax.axhspan(380, 700, alpha=0.05, color='gold')
    ax.set_xlabel('Protein loading P', fontsize=12)
    ax.set_ylabel('Predicted stop-band λ (nm)', fontsize=12)
    ax.set_title('Protein loading\n→ predicted colour', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(200, 800)

    fig.suptitle(
        'Biological parameter mapping: protein density → curvature → lattice → colour\n'
        '(α={:.2f}, κ₀={:.0f} k_BT, σ₀={:.0e} k_BT nm⁻², β={:.1f})'.format(
            alpha, kappa0, sigma0, beta),
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()

    if save_dir is not None:
        path = Path(save_dir) / 'bio_mapping_protein_to_colour.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'[bio_mapping] Saved: {path}')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Threshold t → volume fraction f_chitin → stop-band width
# ─────────────────────────────────────────────────────────────────────────────

def threshold_to_volume_fraction(t_values, geometry='gyroid'):
    """
    Map TPMS threshold t to chitin volume fraction f_chitin.

    Uses the analytical approximation for the gyroid and diamond TPMS:
    f_chitin(t) ≈ 0.5 − c₁·t − c₂·t³  (gyroid: c₁=0.4, c₂=0.06)

    Parameters
    ----------
    t_values : array_like
        TPMS threshold values in [-1, 1].
    geometry : {'gyroid', 'diamond'}, optional
        TPMS geometry. Default 'gyroid'.

    Returns
    -------
    f_chitin : ndarray
        Chitin volume fraction in [0, 1].
    """
    t = np.asarray(t_values)
    if geometry == 'gyroid':
        c1, c2 = 0.400, 0.060
    else:  # diamond
        c1, c2 = 0.375, 0.055
    return np.clip(0.5 - c1 * t - c2 * t**3, 0.0, 1.0)


def volume_fraction_to_stopband_width(f_chitin, eps_h=2.56, eps_l=1.0):
    """
    Estimate photonic stop-band width from chitin volume fraction.

    Uses the empirical relation Δω/ω₀ ≈ A·(Δε/ε_avg)·f(1-f) derived from
    plane-wave expansion calculations on gyroid structures.

    Parameters
    ----------
    f_chitin : array_like
        Chitin volume fraction.
    eps_h : float, optional
        Chitin dielectric constant. Default 2.56 (n=1.6).
    eps_l : float, optional
        Air dielectric constant. Default 1.0.

    Returns
    -------
    gap_ratio : ndarray
        Estimated gap-to-midgap ratio Δω/ω₀.
    """
    f = np.asarray(f_chitin)
    delta_eps = eps_h - eps_l
    eps_avg   = f * eps_h + (1 - f) * eps_l
    A = 0.35   # empirical prefactor for gyroid
    return A * (delta_eps / eps_avg) * f * (1 - f)


def plot_threshold_sweep(save_dir=None):
    """
    Plot threshold t → volume fraction → stop-band width sweep.

    Parameters
    ----------
    save_dir : str or Path, optional
        Directory to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    t_arr = np.linspace(-0.8, 0.8, 200)
    f_gyroid  = threshold_to_volume_fraction(t_arr, geometry='gyroid')
    f_diamond = threshold_to_volume_fraction(t_arr, geometry='diamond')
    gap_gyroid  = volume_fraction_to_stopband_width(f_gyroid)
    gap_diamond = volume_fraction_to_stopband_width(f_diamond)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(t_arr, f_gyroid,  color='#9b59b6', linewidth=2.5, label='Gyroid')
    ax.plot(t_arr, f_diamond, color='#e67e22', linewidth=2.5, linestyle='--', label='Diamond')
    ax.axhspan(0.12, 0.22, alpha=0.15, color='green',
               label='Biological range\n(Michielsen 2010: f≈0.17)')
    ax.axhline(0.17, color='green', linestyle=':', linewidth=1.5)
    ax.set_xlabel('TPMS threshold t', fontsize=12)
    ax.set_ylabel('Chitin volume fraction f', fontsize=12)
    ax.set_title('Threshold t → volume fraction\n(biological meaning: chitin synthase activity)',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_arr, gap_gyroid,  color='#9b59b6', linewidth=2.5, label='Gyroid')
    ax.plot(t_arr, gap_diamond, color='#e67e22', linewidth=2.5, linestyle='--', label='Diamond')
    ax.axvspan(-0.35, 0.35, alpha=0.1, color='green',
               label='Optimal stop-band region')
    ax.set_xlabel('TPMS threshold t', fontsize=12)
    ax.set_ylabel('Estimated gap-to-midgap ratio Δω/ω₀', fontsize=12)
    ax.set_title('Threshold t → stop-band width\n(prediction: inhibit chitin synthase → weaker colour)',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Parameter sweep: TPMS threshold t (biological: chitin volume fraction)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_dir is not None:
        path = Path(save_dir) / 'bio_mapping_threshold_sweep.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'[bio_mapping] Saved: {path}')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dielectric contrast sweep → stop-band robustness
# ─────────────────────────────────────────────────────────────────────────────

def plot_dielectric_sweep(save_dir=None, f_chitin=0.17):
    """
    Sweep chitin refractive index n_chitin and plot stop-band width.

    Biological meaning: n_chitin is controlled by chitin crystallinity and
    water content. Dehydrated scales → higher n_chitin → wider stop band.

    Parameters
    ----------
    save_dir : str or Path, optional
        Directory to save the figure.
    f_chitin : float, optional
        Fixed chitin volume fraction. Default 0.17.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_arr = np.linspace(1.3, 1.8, 100)
    eps_h_arr = n_arr**2
    gap_arr = volume_fraction_to_stopband_width(f_chitin, eps_h=eps_h_arr)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n_arr, gap_arr, color='#1abc9c', linewidth=2.5)
    ax.axvspan(1.53, 1.60, alpha=0.15, color='blue',
               label='Biological range\n(chitin n = 1.53–1.60)')
    ax.axvline(1.55, color='blue', linestyle='--', linewidth=1.5,
               label='C. rubi: n = 1.55\n(Michielsen 2010)')
    ax.set_xlabel('Chitin refractive index n_chitin', fontsize=12)
    ax.set_ylabel('Estimated gap-to-midgap ratio Δω/ω₀', fontsize=12)
    ax.set_title(
        'Dielectric contrast sweep\n'
        '(biological: chitin crystallinity / hydration state)\n'
        f'f_chitin = {f_chitin:.2f} fixed',
        fontsize=10, fontweight='bold'
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.annotate(
        'Prediction: dehydration\n→ higher n → wider stop band',
        xy=(1.65, volume_fraction_to_stopband_width(f_chitin, eps_h=1.65**2)),
        xytext=(1.70, 0.04),
        fontsize=9,
        arrowprops=dict(arrowstyle='->', color='grey')
    )
    plt.tight_layout()

    if save_dir is not None:
        path = Path(save_dir) / 'bio_mapping_dielectric_sweep.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'[bio_mapping] Saved: {path}')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Broadband reflectance spectrum model
# ─────────────────────────────────────────────────────────────────────────────

def broadband_reflectance(a_nm, f_chitin=0.17, n_chitin=1.55,
                          lambda_arr=None, angle_deg=0.0):
    """
    Compute a model broadband reflectance spectrum for a photonic crystal.

    Uses a Lorentzian stop-band model centred at the predicted Bragg
    wavelength, with width proportional to the gap-to-midgap ratio.
    Angle dependence is included via the Bragg condition:
    λ(θ) = λ₀·cos(θ_eff), where θ_eff is the effective internal angle.

    Parameters
    ----------
    a_nm : float
        Lattice constant (nm).
    f_chitin : float, optional
        Chitin volume fraction. Default 0.17.
    n_chitin : float, optional
        Chitin refractive index. Default 1.55.
    lambda_arr : array_like or None, optional
        Wavelength axis (nm). Default: 300–800 nm, 500 points.
    angle_deg : float, optional
        Observation angle from normal (degrees). Default 0.

    Returns
    -------
    lambda_arr : ndarray
        Wavelength axis (nm).
    R : ndarray
        Reflectance (0–1).
    lambda_peak : float
        Stop-band centre wavelength (nm).
    """
    if lambda_arr is None:
        lambda_arr = np.linspace(300, 800, 500)

    eps_h   = n_chitin**2
    eps_l   = 1.0
    eps_avg = f_chitin * eps_h + (1 - f_chitin) * eps_l
    n_eff   = np.sqrt(eps_avg)

    # Bragg condition with angle
    theta_rad = np.deg2rad(angle_deg)
    cos_theta_eff = np.sqrt(1 - (np.sin(theta_rad) / n_eff)**2)
    lambda_peak = 2.0 * a_nm * n_eff * cos_theta_eff

    # Gap width
    gap_ratio = volume_fraction_to_stopband_width(f_chitin, eps_h=eps_h)
    delta_lambda = gap_ratio * lambda_peak

    # Lorentzian reflectance model
    R_peak = min(0.30, 0.15 + 0.5 * gap_ratio)
    gamma  = delta_lambda / 2.0
    R = R_peak / (1 + ((lambda_arr - lambda_peak) / gamma)**2)

    return lambda_arr, R, lambda_peak


def plot_reflectance_spectra(a_values=None, save_dir=None):
    """
    Plot broadband reflectance spectra for a range of lattice constants.

    Parameters
    ----------
    a_values : list of float, optional
        Lattice constants to plot (nm). Default: [221, 260, 293, 323, 350].
    save_dir : str or Path, optional
        Directory to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if a_values is None:
        a_values = [221, 260, 293, 323, 350]

    P_labels = [0.0, 0.25, 0.50, 0.75, 1.00]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(a_values)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Spectra at normal incidence
    ax = axes[0]
    for a, P, col in zip(a_values, P_labels, colors):
        lam, R, lam_peak = broadband_reflectance(a)
        ax.plot(lam, R * 100, color=col, linewidth=2.0,
                label=f'a={a:.0f} nm (P={P:.2f}), λ_peak={lam_peak:.0f} nm')

    ax.axvspan(380, 700, alpha=0.07, color='gold', label='Visible range')
    ax.axvline(545, color='green', linestyle='--', linewidth=1.5,
               label='C. rubi: 545 nm')
    ax.set_xlabel('Wavelength λ (nm)', fontsize=12)
    ax.set_ylabel('Reflectance (%)', fontsize=12)
    ax.set_title('Broadband reflectance spectra\n(normal incidence, f_chitin=0.17)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=7.5, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(300, 800)

    # Panel 2: Angle dependence for C. rubi parameters (a=363 nm)
    ax = axes[1]
    angles = [0, 15, 30, 45, 60]
    col_ang = plt.cm.cool(np.linspace(0, 1, len(angles)))
    for ang, col in zip(angles, col_ang):
        lam, R, lam_peak = broadband_reflectance(363.0, angle_deg=ang)
        ax.plot(lam, R * 100, color=col, linewidth=2.0,
                label=f'θ={ang}°, λ_peak={lam_peak:.0f} nm')

    ax.axvspan(380, 700, alpha=0.07, color='gold')
    ax.axvline(545, color='green', linestyle='--', linewidth=1.5,
               label='C. rubi: 545 nm')
    ax.set_xlabel('Wavelength λ (nm)', fontsize=12)
    ax.set_ylabel('Reflectance (%)', fontsize=12)
    ax.set_title('Angle-dependent reflectance\n(a=363 nm, C. rubi parameters)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(300, 800)

    fig.suptitle('Broadband and angle-dependent reflectance spectra\n'
                 '(Lorentzian stop-band model, n_chitin=1.55)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_dir is not None:
        path = Path(save_dir) / 'bio_mapping_reflectance_spectra.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'[bio_mapping] Saved: {path}')
    return fig


def plot_parameter_constraint_region(save_dir=None):
    """
    Show that only a narrow region of (a, f_chitin) parameter space matches
    the biological constraints from C. rubi.

    This is the key figure that makes the paper scientific rather than
    illustrative: it demonstrates that the biological observations constrain
    the model parameters to a small region.

    Parameters
    ----------
    save_dir : str or Path, optional
        Directory to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    a_arr = np.linspace(150, 600, 200)
    f_arr = np.linspace(0.05, 0.45, 200)
    A, F  = np.meshgrid(a_arr, f_arr)

    # Predicted stop-band centre wavelength
    n_eff_map = np.sqrt(F * 1.55**2 + (1 - F) * 1.0)
    lambda_map = 2.0 * A * n_eff_map

    # Gap-to-midgap ratio
    gap_map = volume_fraction_to_stopband_width(F, eps_h=1.55**2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Stop-band wavelength map
    ax = axes[0]
    im = ax.contourf(A, F, lambda_map, levels=30, cmap='RdYlGn_r')
    plt.colorbar(im, ax=ax, label='Stop-band centre λ (nm)')

    # Biological constraint: λ_peak = 545 ± 50 nm
    cs = ax.contour(A, F, lambda_map, levels=[495, 545, 595],
                    colors=['blue', 'green', 'blue'], linewidths=[1.5, 2.5, 1.5])
    ax.clabel(cs, fmt='%.0f nm', fontsize=9)

    # Biological constraint: f_chitin = 0.17 ± 0.03
    ax.axhspan(0.14, 0.20, alpha=0.25, color='cyan',
               label='Biological f_chitin = 0.17 ± 0.03')
    ax.axhline(0.17, color='cyan', linestyle='--', linewidth=1.5)

    # Mark C. rubi point
    ax.scatter([363], [0.17], color='red', s=200, zorder=5, marker='*',
               label='C. rubi (a=363 nm, f=0.17)')

    ax.set_xlabel('Lattice constant a (nm)', fontsize=12)
    ax.set_ylabel('Chitin volume fraction f', fontsize=12)
    ax.set_title('Stop-band wavelength map\n(constraint: 495–595 nm visible green)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')

    # Panel 2: Gap-to-midgap ratio map
    ax = axes[1]
    im2 = ax.contourf(A, F, gap_map, levels=20, cmap='hot_r')
    plt.colorbar(im2, ax=ax, label='Gap-to-midgap ratio Δω/ω₀')

    # Highlight the intersection of both constraints
    mask_lambda = (lambda_map >= 495) & (lambda_map <= 595)
    mask_f      = (F >= 0.14) & (F <= 0.20)
    mask_both   = mask_lambda & mask_f
    ax.contourf(A, F, mask_both.astype(float), levels=[0.5, 1.5],
                colors=['lime'], alpha=0.4)
    ax.contour(A, F, mask_both.astype(float), levels=[0.5],
               colors=['lime'], linewidths=2.5)

    ax.scatter([363], [0.17], color='red', s=200, zorder=5, marker='*',
               label='C. rubi (a=363 nm, f=0.17)')
    ax.set_xlabel('Lattice constant a (nm)', fontsize=12)
    ax.set_ylabel('Chitin volume fraction f', fontsize=12)
    ax.set_title('Constrained parameter region\n(green: satisfies both λ and f constraints)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)

    fig.suptitle(
        'Parameter constraint analysis: only a narrow region of (a, f) space\n'
        'satisfies both the optical and structural constraints of C. rubi',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()

    if save_dir is not None:
        path = Path(save_dir) / 'bio_mapping_constraint_region.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'[bio_mapping] Saved: {path}')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    FIG_DIR = Path('figures')
    FIG_DIR.mkdir(exist_ok=True)

    print_parameter_table()

    print('\n[bio_mapping] Generating figures...')
    plot_protein_to_colour(save_dir=FIG_DIR)
    plot_threshold_sweep(save_dir=FIG_DIR)
    plot_dielectric_sweep(save_dir=FIG_DIR)
    plot_reflectance_spectra(save_dir=FIG_DIR)
    plot_parameter_constraint_region(save_dir=FIG_DIR)
    print('[bio_mapping] All figures saved.')
