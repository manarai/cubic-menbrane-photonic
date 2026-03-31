"""
Phase 3: Identify Symmetry Transitions (Gyroid vs Diamond)
===========================================================
Analyses the phase-field output to classify the emergent morphology as:
  - Lamellar (L)
  - Gyroid Ia3d (G)
  - Diamond Pn3m (D)
  - Disordered (X)

Classification strategy
-----------------------
1. Compute the 3-D power spectrum |phi_hat(k)|^2.
2. Identify dominant shells in reciprocal space (peaks at |k| = k_peak).
3. Compare the ratio of first two peak positions to known TPMS ratios:
     Gyroid  (Ia3d):  k1/k2 ≈ sqrt(6)/sqrt(8)  ≈ 0.866
     Diamond (Pn3m):  k1/k2 ≈ sqrt(2)/sqrt(3)  ≈ 0.816
     Lamellar:        single dominant peak

4. Also compute the Euler characteristic (χ) of the thresholded structure
   as a topological discriminator:
     χ < 0  → bicontinuous (gyroid/diamond)
     χ ≈ 0  → lamellar
     χ > 0  → disconnected droplets

5. Build a phase diagram over (P, lam) parameter space.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from itertools import product

# ─────────────────────────────────────────────────────────────────────────────
# Power spectrum helpers
# ─────────────────────────────────────────────────────────────────────────────

def power_spectrum_1d(phi: np.ndarray, n_bins: int = 50):
    """
    Compute the spherically averaged power spectrum of phi.

    Returns
    -------
    k_centers : ndarray
    power : ndarray
    """
    N = phi.shape[0]
    phi_hat = np.fft.fftn(phi)
    power3d = np.abs(phi_hat)**2 / N**3

    k1d = np.fft.fftfreq(N) * N          # integer wave-numbers
    KX, KY, KZ = np.meshgrid(k1d, k1d, k1d, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)

    k_max = K.max()
    bins = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (bins[:-1] + bins[1:])
    power = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (K >= bins[i]) & (K < bins[i + 1])
        if mask.any():
            power[i] = power3d[mask].mean()

    return k_centers, power


def find_peaks_1d(k_centers, power, n_peaks: int = 3, min_sep: float = 1.5):
    """Return indices of the n_peaks largest local maxima."""
    from scipy.signal import find_peaks as sp_find_peaks
    peaks, _ = sp_find_peaks(power, distance=int(min_sep))
    if len(peaks) == 0:
        return np.array([np.argmax(power)])
    # Sort by power
    sorted_peaks = peaks[np.argsort(power[peaks])[::-1]]
    return sorted_peaks[:n_peaks]


# ─────────────────────────────────────────────────────────────────────────────
# Euler characteristic (topological discriminator)
# ─────────────────────────────────────────────────────────────────────────────

def euler_characteristic(phi: np.ndarray, threshold: float = 0.0) -> int:
    """
    Estimate the Euler characteristic of the thresholded binary structure
    using the voxel-counting formula on a periodic grid.

    χ = V - E + F - C   (vertices, edges, faces, cells in cubical complex)

    This is a simplified 3-D cubical complex formula.
    """
    B = (phi > threshold).astype(np.int8)
    N = B.shape[0]

    # Periodic roll
    def roll(a, shift, axis):
        return np.roll(a, shift, axis=axis)

    # Vertices (V): each voxel contributes 1
    V = int(B.sum())

    # Edges (E): pairs of adjacent voxels both = 1
    E = int(
        (B * roll(B, -1, 0)).sum() +
        (B * roll(B, -1, 1)).sum() +
        (B * roll(B, -1, 2)).sum()
    )

    # Faces (F): 2×2 faces all = 1
    F = int(
        (B * roll(B, -1, 0) * roll(B, -1, 1) * roll(roll(B, -1, 0), -1, 1)).sum() +
        (B * roll(B, -1, 0) * roll(B, -1, 2) * roll(roll(B, -1, 0), -1, 2)).sum() +
        (B * roll(B, -1, 1) * roll(B, -1, 2) * roll(roll(B, -1, 1), -1, 2)).sum()
    )

    # Cells (C): 2×2×2 cubes all = 1
    C = int(
        (B *
         roll(B, -1, 0) *
         roll(B, -1, 1) *
         roll(B, -1, 2) *
         roll(roll(B, -1, 0), -1, 1) *
         roll(roll(B, -1, 0), -1, 2) *
         roll(roll(B, -1, 1), -1, 2) *
         roll(roll(roll(B, -1, 0), -1, 1), -1, 2)
        ).sum()
    )

    chi = V - E + F - C
    return chi


# ─────────────────────────────────────────────────────────────────────────────
# Morphology classifier
# ─────────────────────────────────────────────────────────────────────────────

# Known TPMS peak ratios (first / second dominant shell)
GYROID_RATIO  = np.sqrt(6) / np.sqrt(8)   # ≈ 0.866
DIAMOND_RATIO = np.sqrt(2) / np.sqrt(3)   # ≈ 0.816
LAMELLAR_RATIO_MAX = 0.55                  # lamellar has very sharp single peak

MORPHOLOGY_COLORS = {
    'Lamellar': '#4878CF',
    'Gyroid':   '#6ACC65',
    'Diamond':  '#D65F5F',
    'Disordered': '#B47CC7',
}


def classify_morphology(phi: np.ndarray, threshold: float = 0.0) -> str:
    """
    Classify the morphology of the phase-field as L / G / D / X.
    """
    k_centers, power = power_spectrum_1d(phi)
    peak_idx = find_peaks_1d(k_centers, power, n_peaks=3)

    chi = euler_characteristic(phi, threshold)

    if len(peak_idx) < 2:
        return 'Lamellar' if chi >= -5 else 'Disordered'

    k_peaks = sorted(k_centers[peak_idx[:2]])
    if k_peaks[1] < 1e-6:
        return 'Disordered'

    ratio = k_peaks[0] / k_peaks[1]

    # Use Euler characteristic as primary discriminator for bicontinuous
    if chi < -10:
        # Bicontinuous: distinguish gyroid vs diamond by peak ratio
        if abs(ratio - GYROID_RATIO) < abs(ratio - DIAMOND_RATIO):
            return 'Gyroid'
        else:
            return 'Diamond'
    elif -10 <= chi <= 10:
        return 'Lamellar'
    else:
        return 'Disordered'


# ─────────────────────────────────────────────────────────────────────────────
# Phase diagram over (P, lam)
# ─────────────────────────────────────────────────────────────────────────────

def build_phase_diagram(
    P_values: list,
    lam_values: list,
    N: int = 48,
    dt: float = 0.04,
    n_steps: int = 1500,
    seed: int = 42,
    out_dir: Path = Path("figures"),
) -> dict:
    """
    Sweep (P, lam) parameter space and classify each morphology.

    Returns
    -------
    diagram : dict  {(P, lam): morphology_string}
    """
    # Import here to avoid circular dependency
    from phase2_curvature import run_with_curvature

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diagram = {}
    total = len(P_values) * len(lam_values)
    count = 0

    for P, lam in product(P_values, lam_values):
        count += 1
        print(f"  [Phase 3] ({count}/{total}) P={P:.2f}, lam={lam:.2f} ...", end=" ")
        phi = run_with_curvature(P=P, N=N, lam=lam, dt=dt,
                                 n_steps=n_steps, seed=seed)
        morph = classify_morphology(phi)
        diagram[(P, lam)] = morph
        print(morph)

    _plot_phase_diagram(diagram, P_values, lam_values, out_dir)
    return diagram


def _plot_phase_diagram(diagram: dict, P_values: list, lam_values: list,
                        out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 5))

    color_map = {m: MORPHOLOGY_COLORS[m] for m in MORPHOLOGY_COLORS}
    marker_map = {'Lamellar': 's', 'Gyroid': 'o', 'Diamond': '^', 'Disordered': 'x'}

    for (P, lam), morph in diagram.items():
        ax.scatter(P, lam,
                   c=color_map.get(morph, 'grey'),
                   marker=marker_map.get(morph, 'o'),
                   s=120, edgecolors='k', linewidths=0.5, zorder=3)

    patches = [mpatches.Patch(color=color_map[m], label=m)
               for m in color_map if any(v == m for v in diagram.values())]
    ax.legend(handles=patches, loc='upper left', fontsize=9)
    ax.set_xlabel("Protein loading P", fontsize=11)
    ax.set_ylabel("Surface regularisation λ", fontsize=11)
    ax.set_title("Phase diagram: morphology vs (P, λ)", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "phase3_phase_diagram.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Phase 3] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Power spectrum plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_power_spectra(phi_dict: dict, out_dir: Path):
    """
    Plot 1-D power spectra for a dict of {label: phi_array}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.viridis(np.linspace(0, 1, len(phi_dict)))

    for (label, phi), color in zip(phi_dict.items(), colors):
        k, power = power_spectrum_1d(phi)
        ax.semilogy(k, power + 1e-12, label=label, color=color)

    ax.set_xlabel("|k| (grid units)", fontsize=11)
    ax.set_ylabel("Power (log scale)", fontsize=11)
    ax.set_title("Spherically averaged power spectra", fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "phase3_power_spectra.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Phase 3] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    FIG_DIR  = Path("/home/ubuntu/cubic-membrane-photonics/figures")
    DATA_DIR = Path("/home/ubuntu/cubic-membrane-photonics/data")

    print("=== Phase 3: Symmetry identification ===")

    P_vals   = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    lam_vals = [0.5, 1.0, 1.5, 2.0]

    diagram = build_phase_diagram(P_vals, lam_vals, N=48, n_steps=1500,
                                  out_dir=FIG_DIR)

    import json
    with open(DATA_DIR / "phase_diagram.json", "w") as f:
        json.dump({str(k): v for k, v in diagram.items()}, f, indent=2)

    print("[Phase 3] Done.")
