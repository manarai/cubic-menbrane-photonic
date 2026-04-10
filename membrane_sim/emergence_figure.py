"""
emergence_figure.py
===================
The single most convincing figure for the paper: showing spontaneous
emergence of the cubic structure from a random initial condition.

This module directly addresses reviewer critique:
  "You say 'spontaneous emergence of gyroid/diamond'. You must show it
   cleanly. Add a figure with:
     - initial random field
     - intermediate state
     - final cubic structure
     - corresponding structure factor (peaks)"

Four-panel figure layout:
  Panel A: t=0    — random initial condition (noise)
  Panel B: t=t₁   — early coarsening (lamellar precursor)
  Panel C: t=t₂   — intermediate bicontinuous network
  Panel D: t=t_f  — final cubic structure (gyroid/diamond)
  Panel E: Structure factor evolution — power spectra at each time point
  Panel F: Order parameter metrics — mean |φ|, variance, χ(t)

This is the "killer figure" that transforms the paper from a model to
a demonstration of spontaneous self-organisation.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / 'src'))


def run_with_snapshots(P=0.6, N=64, lam=0.1, dt=0.05,
                       snapshot_steps=None, seed=42):
    """
    Run the phase-field simulation and save snapshots at specified time steps.

    Parameters
    ----------
    P : float, optional
        Protein loading. Default 0.6 (gyroid regime).
    N : int, optional
        Grid size. Default 64.
    lam : float, optional
        Interface width parameter. Default 0.1.
    dt : float, optional
        Time step. Default 0.05.
    snapshot_steps : list of int or None, optional
        Time steps at which to save snapshots.
        Default: [0, 100, 300, 800] (initial, early, mid, final).
    seed : int, optional
        Random seed. Default 42.

    Returns
    -------
    snapshots : dict
        {step: phi_array} for each snapshot step.
    """
    from phase2_curvature import effective_lam, _make_k2, step_with_curvature
    from phase1_baseline import gyroid_seed

    if snapshot_steps is None:
        snapshot_steps = [0, 100, 300, 800]

    lam_e = effective_lam(P, lam)
    k2    = _make_k2(N)

    # Pure random initial condition (no gyroid seed — to show true emergence)
    rng = np.random.default_rng(seed)
    phi = rng.uniform(-0.1, 0.1, (N, N, N))
    phi_hat = np.fft.fftn(phi)

    snapshots = {}
    max_step  = max(snapshot_steps)

    for step in range(max_step + 1):
        if step in snapshot_steps:
            snapshots[step] = np.real(np.fft.ifftn(phi_hat)).copy()
        phi_hat = step_with_curvature(phi_hat, k2, lam_e, dt)

    return snapshots


def compute_order_metrics(phi):
    """
    Compute order parameter metrics for a phase-field snapshot.

    Parameters
    ----------
    phi : ndarray, shape (N, N, N)
        Phase-field snapshot.

    Returns
    -------
    metrics : dict
        'mean_abs': float — mean |φ| (order parameter amplitude)
        'variance': float — variance of φ
        'chi': int — Euler characteristic
        'dominant_k': float — dominant wavenumber (peak of power spectrum)
    """
    from phase3_symmetry import euler_characteristic, power_spectrum_1d

    k_c, pwr = power_spectrum_1d(phi)
    # Ignore DC
    dominant_k = k_c[1:][np.argmax(pwr[1:])]

    return {
        'mean_abs':  float(np.mean(np.abs(phi))),
        'variance':  float(np.var(phi)),
        'chi':       euler_characteristic(phi),
        'dominant_k': float(dominant_k),
    }


def plot_emergence_figure(P=0.6, N=64, snapshot_steps=None,
                          seed=42, save_dir=None):
    """
    Generate the emergence figure: spontaneous self-organisation from noise.

    This is the single most convincing figure for the manuscript.
    It shows that the cubic structure emerges spontaneously from a random
    initial condition under the phase-field dynamics, without any imposed
    geometry.

    Parameters
    ----------
    P : float, optional
        Protein loading. Default 0.6 (gyroid regime).
    N : int, optional
        Grid size. Default 64.
    snapshot_steps : list of int or None, optional
        Time steps for snapshots. Default [0, 100, 300, 800].
    seed : int, optional
        Random seed. Default 42.
    save_dir : str or Path, optional
        Directory to save figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    metrics_list : list of dict
        Order metrics at each snapshot.
    """
    from phase3_symmetry import power_spectrum_1d, classify_morphology

    if snapshot_steps is None:
        snapshot_steps = [0, 100, 300, 800]

    print(f'[emergence] Running phase-field with P={P}, N={N}...')
    snapshots = run_with_snapshots(P=P, N=N, snapshot_steps=snapshot_steps,
                                   seed=seed)

    steps = sorted(snapshots.keys())
    n     = len(steps)

    # Compute metrics
    metrics_list = []
    morphologies = []
    for step in steps:
        phi   = snapshots[step]
        m     = compute_order_metrics(phi)
        morph = classify_morphology(phi) if step > 0 else 'Random'
        metrics_list.append(m)
        morphologies.append(morph)
        print(f'  t={step:4d}: |φ|={m["mean_abs"]:.3f}, '
              f'var={m["variance"]:.3f}, χ={m["chi"]:+d}, '
              f'k*={m["dominant_k"]:.2f}, morph={morph}')

    # ── Figure layout: 2 rows × (n+1) columns ─────────────────────────────────
    # Row 1: mid-plane slices
    # Row 2: power spectra
    fig = plt.figure(figsize=(4.5 * (n + 1), 9))
    gs  = fig.add_gridspec(2, n + 1, hspace=0.4, wspace=0.35)

    cmap = plt.cm.RdBu_r
    spec_colors = plt.cm.viridis(np.linspace(0.1, 0.9, n))

    step_labels = [f't = {s}' for s in steps]
    step_labels[0] = 't = 0\n(random noise)'

    for col, (step, label, morph, col_c) in enumerate(
            zip(steps, step_labels, morphologies, spec_colors)):
        phi = snapshots[step]
        mid = N // 2

        # Row 1: mid-plane slice
        ax_img = fig.add_subplot(gs[0, col])
        vmax   = max(0.5, np.abs(phi).max() * 0.9)
        im = ax_img.imshow(phi[:, :, mid], cmap=cmap,
                           vmin=-vmax, vmax=vmax,
                           interpolation='nearest', origin='lower')
        ax_img.set_title(f'{label}\n{morph}',
                         fontsize=10, fontweight='bold')
        ax_img.set_xlabel('x (voxels)', fontsize=9)
        ax_img.set_ylabel('y (voxels)', fontsize=9)
        plt.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

        # Row 2: power spectrum
        ax_sp = fig.add_subplot(gs[1, col])
        k_c, pwr = power_spectrum_1d(phi)
        pwr_norm = pwr / (pwr[1:].max() + 1e-12)
        ax_sp.plot(k_c[1:], pwr_norm[1:], color=col_c, linewidth=2.5)
        ax_sp.set_xlabel('Wavenumber |k|', fontsize=9)
        ax_sp.set_ylabel('Norm. power', fontsize=9)
        ax_sp.set_title(f'Structure factor\n(t={step})', fontsize=9)
        ax_sp.grid(True, alpha=0.3)

    # Last column: order parameter evolution
    ax_ev = fig.add_subplot(gs[0, n])
    mean_abs = [m['mean_abs'] for m in metrics_list]
    variance = [m['variance'] for m in metrics_list]
    ax_ev.plot(steps, mean_abs, 'o-', color='#3498db', linewidth=2,
               markersize=9, label='Mean |φ|')
    ax_ev.plot(steps, variance, 's--', color='#e74c3c', linewidth=2,
               markersize=9, label='Variance')
    ax_ev.set_xlabel('Time step', fontsize=10)
    ax_ev.set_ylabel('Order parameter', fontsize=10)
    ax_ev.set_title('Order parameter\nevolution', fontsize=10, fontweight='bold')
    ax_ev.legend(fontsize=9)
    ax_ev.grid(True, alpha=0.3)

    ax_chi = fig.add_subplot(gs[1, n])
    chi_vals = [m['chi'] for m in metrics_list]
    ax_chi.plot(steps, chi_vals, 'D-', color='#9b59b6', linewidth=2,
                markersize=9)
    ax_chi.axhline(-4, color='green', linestyle='--', linewidth=1.2,
                   label='Gyroid (χ≈−4)')
    ax_chi.axhline(-8, color='red', linestyle=':', linewidth=1.2,
                   label='Diamond (χ≈−8)')
    ax_chi.axhline(0, color='blue', linestyle='-.', linewidth=1.2,
                   label='Lamellar (χ≈0)')
    ax_chi.set_xlabel('Time step', fontsize=10)
    ax_chi.set_ylabel('Euler characteristic χ', fontsize=10)
    ax_chi.set_title('Topology evolution\nχ(t)', fontsize=10, fontweight='bold')
    ax_chi.legend(fontsize=8)
    ax_chi.grid(True, alpha=0.3)

    fig.suptitle(
        f'Spontaneous emergence of cubic photonic structure from random noise\n'
        f'P = {P:.2f} (protein loading), N = {N}³ grid\n'
        f'Final morphology: {morphologies[-1]}',
        fontsize=13, fontweight='bold'
    )

    if save_dir is not None:
        path = Path(save_dir) / 'emergence_spontaneous_selforganisation.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'[emergence] Saved: {path}')

    return fig, metrics_list


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    FIG_DIR = Path('figures')
    FIG_DIR.mkdir(exist_ok=True)

    print('=' * 60)
    print('EMERGENCE FIGURE: SPONTANEOUS SELF-ORGANISATION')
    print('=' * 60)

    fig, metrics = plot_emergence_figure(
        P=0.6, N=64,
        snapshot_steps=[0, 100, 300, 800],
        seed=42,
        save_dir=FIG_DIR
    )

    print('\n[emergence] Done.')
