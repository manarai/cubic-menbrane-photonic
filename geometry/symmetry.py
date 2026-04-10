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
    Compute the spherically averaged 1-D power spectrum of a 3-D field.

    The 3-D power spectrum ``|phi_hat(k)|^2 / N^3`` is binned into
    ``n_bins`` shells of equal width in ``|k|`` and averaged within
    each shell.

    Parameters
    ----------
    phi : ndarray, shape (N, N, N)
        Real-valued order-parameter field on a periodic cubic grid.
    n_bins : int, optional
        Number of radial bins for the spherical average.  Default is 50.

    Returns
    -------
    k_centers : ndarray, shape (n_bins,)
        Centre wavenumber of each radial bin (in grid units).
    power : ndarray, shape (n_bins,)
        Mean power in each radial bin.

    Notes
    -----
    The DC component (``k = 0``) is included in the first bin.  For
    morphology classification it is advisable to ignore this bin.
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


def find_peaks_1d(k_centers: np.ndarray, power: np.ndarray,
                  n_peaks: int = 3, min_sep: float = 1.5) -> np.ndarray:
    """
    Return the indices of the ``n_peaks`` largest local maxima in a 1-D spectrum.

    Uses ``scipy.signal.find_peaks`` with a minimum peak separation of
    ``min_sep`` bins to avoid detecting noise fluctuations as peaks.

    Parameters
    ----------
    k_centers : ndarray, shape (n_bins,)
        Wavenumber axis of the power spectrum.
    power : ndarray, shape (n_bins,)
        Power values at each wavenumber bin.
    n_peaks : int, optional
        Maximum number of peaks to return.  Default is 3.
    min_sep : float, optional
        Minimum separation between peaks in units of bins.  Default is 1.5.

    Returns
    -------
    peak_indices : ndarray
        Indices (into ``k_centers`` and ``power``) of the detected peaks,
        sorted by descending power.  If no peaks are found, the index of
        the global maximum is returned.
    """
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
    Estimate the Euler characteristic of the thresholded binary structure.

    Uses the cubical complex formula on a periodic 3-D grid:

        chi = V - E + F - C

    where V, E, F, C are the counts of vertices, edges, faces, and cells
    in the cubical complex formed by the occupied (``phi > threshold``) voxels.

    The Euler characteristic serves as a topological discriminator:

    +------+------------------------------+
    | chi  | Morphology                   |
    +======+==============================+
    | > 0  | Disconnected droplets        |
    +------+------------------------------+
    | ~ 0  | Lamellar                     |
    +------+------------------------------+
    | -4   | Gyroid (Ia3d)                |
    +------+------------------------------+
    | -8   | Double-diamond (Pn3m)        |
    +------+------------------------------+
    | << 0 | Disordered bicontinuous      |
    +------+------------------------------+

    Parameters
    ----------
    phi : ndarray, shape (N, N, N)
        Order-parameter field on a periodic cubic grid.
    threshold : float, optional
        Iso-value used to binarise the field.  Voxels with
        ``phi > threshold`` are considered occupied.  Default is 0.0.

    Returns
    -------
    chi : int
        Euler characteristic of the thresholded structure.

    Notes
    -----
    This is a simplified voxel-counting formula that gives a qualitative
    topological estimate.  For quantitatively accurate results, use a
    proper cubical homology library (e.g., ``gudhi`` or ``scikit-tda``).
    """
    B = (phi > threshold).astype(np.int8)

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

    # Faces (F): 2x2 faces all = 1
    F = int(
        (B * roll(B, -1, 0) * roll(B, -1, 1) * roll(roll(B, -1, 0), -1, 1)).sum() +
        (B * roll(B, -1, 0) * roll(B, -1, 2) * roll(roll(B, -1, 0), -1, 2)).sum() +
        (B * roll(B, -1, 1) * roll(B, -1, 2) * roll(roll(B, -1, 1), -1, 2)).sum()
    )

    # Cells (C): 2x2x2 cubes all = 1
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
    'Lamellar':   '#4878CF',
    'Gyroid':     '#6ACC65',
    'Diamond':    '#D65F5F',
    'Disordered': '#B47CC7',
}


def classify_morphology(phi: np.ndarray, threshold: float = 0.0) -> str:
    """
    Classify the morphology of a phase-field as Lamellar, Gyroid, Diamond,
    or Disordered.

    The classification uses two complementary criteria:

    1. **Euler characteristic** (topological): distinguishes bicontinuous
       (chi < 0) from lamellar (chi ~ 0) and disordered (chi >> 0) phases.
    2. **Peak ratio** (spectral): distinguishes gyroid from diamond by
       comparing the ratio of the two dominant wavenumbers in the spherically
       averaged power spectrum to the known TPMS values.

    Parameters
    ----------
    phi : ndarray, shape (N, N, N)
        Order-parameter field, as returned by :func:`phase2_curvature.run_with_curvature`.
    threshold : float, optional
        Iso-value for binarisation in the Euler characteristic calculation.
        Default is 0.0.

    Returns
    -------
    morphology : str
        One of ``'Lamellar'``, ``'Gyroid'``, ``'Diamond'``, or ``'Disordered'``.

    Notes
    -----
    The thresholds used for the Euler characteristic (chi < -10 for
    bicontinuous, -10 <= chi <= 10 for lamellar) are empirical and may
    need adjustment for grids smaller than N=32 or unusual parameter values.
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
    Sweep the (P, lam) parameter space and classify the morphology at each point.

    Runs one full phase-field simulation per (P, lam) combination and
    classifies the resulting morphology using :func:`classify_morphology`.

    Parameters
    ----------
    P_values : list of float
        Protein loading values to sweep.  Each must be in [0, 1].
    lam_values : list of float
        Allen-Cahn interface width values to sweep.
    N : int, optional
        Number of grid points per spatial dimension.  Default is 48.
    dt : float, optional
        Time step size.  Default is 0.04.
    n_steps : int, optional
        Number of time steps per simulation.  Default is 1500.
    seed : int, optional
        Random seed for all initial conditions.  Default is 42.
    out_dir : Path or str, optional
        Directory in which to save the phase diagram figure.
        Default is ``"figures"``.

    Returns
    -------
    diagram : dict
        Mapping of ``{(P, lam): morphology_string}`` for each parameter
        combination.
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
    """
    Save a scatter-plot phase diagram of morphology vs (P, lam).

    Parameters
    ----------
    diagram : dict
        Mapping of ``{(P, lam): morphology_string}``.
    P_values : list of float
        Protein loading values used in the sweep.
    lam_values : list of float
        Interface width values used in the sweep.
    out_dir : Path
        Output directory for the saved figure.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    color_map  = {m: MORPHOLOGY_COLORS[m] for m in MORPHOLOGY_COLORS}
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
    Plot the 1-D spherically averaged power spectra for a set of phase fields.

    Parameters
    ----------
    phi_dict : dict
        Mapping of ``{label: phi_array}`` where each ``phi_array`` has
        shape (N, N, N).
    out_dir : Path or str
        Output directory for the saved figure.
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
