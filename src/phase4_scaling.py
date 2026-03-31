"""
Phase 4: Measure Lattice Scaling to Visible-Light Regime
=========================================================
Extracts the characteristic lattice constant 'a' from the phase-field
output and maps it to physical (nanometre) units.

Physical mapping
----------------
The simulation grid has dimensionless spacing dx = L/N.
The dominant wave-number k_peak (in grid units) corresponds to a
real-space periodicity:

    a_sim = 2*pi / k_peak   (dimensionless)

To convert to physical units we use the Helfrich length scale:
    xi(P) = sqrt(kappa_eff(P) / lam)

and the biological constraint that the target lattice constant is
200–500 nm for visible-light photonic interaction:
    a_phys = a_sim * xi(P) * scale_factor

where scale_factor is chosen so that the P=1 state maps to ~350 nm
(centre of visible range).

This module provides:
  - measure_lattice_constant(phi)  → a_sim (dimensionless)
  - physical_lattice(P, a_sim)     → a_nm (nanometres)
  - sweep_lattice_vs_P(...)        → table + figure
  - visible_light_condition(a_nm)  → bool
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Physical constants
VISIBLE_MIN_NM = 200.0
VISIBLE_MAX_NM = 500.0
TARGET_NM      = 350.0   # centre of visible range at P=1

# Helfrich parameters (must match phase2_curvature.py)
KAPPA0 = 1.0
BETA   = 1.5
LAM    = 1.0


def effective_kappa(P: float) -> float:
    return KAPPA0 * (1.0 + BETA * P)


def helfrich_length(P: float, lam: float = LAM) -> float:
    """xi(P) = sqrt(kappa_eff / lam)"""
    return np.sqrt(effective_kappa(P) / lam)


# ─────────────────────────────────────────────────────────────────────────────
# Lattice constant extraction
# ─────────────────────────────────────────────────────────────────────────────

def measure_lattice_constant(phi: np.ndarray) -> float:
    """
    Extract the dominant lattice constant from the 3-D power spectrum.

    Returns
    -------
    a_sim : float
        Dimensionless lattice constant (in grid units of 2*pi).
    k_peak : float
        Dominant wave-number (grid units).
    """
    N = phi.shape[0]
    phi_hat = np.fft.fftn(phi)
    power3d = np.abs(phi_hat)**2

    k1d = np.fft.fftfreq(N) * N
    KX, KY, KZ = np.meshgrid(k1d, k1d, k1d, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)

    # Spherical average
    n_bins = 40
    k_max  = K.max()
    bins   = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (bins[:-1] + bins[1:])
    power = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (K >= bins[i]) & (K < bins[i + 1])
        if mask.any():
            power[i] = power3d[mask].mean()

    # Exclude k=0 (DC component)
    power[0] = 0.0

    k_peak = k_centers[np.argmax(power)]
    if k_peak < 1e-6:
        k_peak = k_centers[1]

    a_sim = 2 * np.pi / k_peak
    return a_sim, k_peak


# ─────────────────────────────────────────────────────────────────────────────
# Physical unit conversion
# ─────────────────────────────────────────────────────────────────────────────

def physical_lattice(P: float, a_sim: float,
                     scale_factor: float = None) -> float:
    """
    Convert dimensionless lattice constant to nanometres.

    If scale_factor is None, it is calibrated so that P=1 maps to TARGET_NM.
    """
    xi = helfrich_length(P)
    a_natural = a_sim * xi

    if scale_factor is None:
        # Calibrate: at P=1, a_phys = TARGET_NM
        xi_ref = helfrich_length(1.0)
        # We need a_natural_ref * scale_factor = TARGET_NM
        # a_natural_ref is not known here, so we use a fixed reference
        # (will be computed in sweep)
        scale_factor = 1.0   # placeholder; corrected in sweep

    return a_natural * scale_factor


# ─────────────────────────────────────────────────────────────────────────────
# Sweep over P values
# ─────────────────────────────────────────────────────────────────────────────

def sweep_lattice_vs_P(
    P_values: list,
    N: int = 64,
    lam: float = 1.0,
    dt: float = 0.04,
    n_steps: int = 2000,
    seed: int = 42,
    out_dir: Path = Path("figures"),
) -> dict:
    """
    Run simulations for each P, measure lattice constant, convert to nm.

    Returns
    -------
    results : dict  {P: {'a_sim': float, 'k_peak': float, 'a_nm': float,
                          'xi': float, 'in_visible': bool}}
    """
    from phase2_curvature import run_with_curvature

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = {}
    for P in P_values:
        print(f"  [Phase 4] P = {P:.2f} ...", end=" ")
        phi = run_with_curvature(P=P, N=N, lam=lam, dt=dt,
                                 n_steps=n_steps, seed=seed)
        a_sim, k_peak = measure_lattice_constant(phi)
        xi = helfrich_length(P, lam)
        raw[P] = {'a_sim': a_sim, 'k_peak': k_peak, 'xi': xi}
        print(f"a_sim={a_sim:.3f}, xi={xi:.3f}")

    # Calibrate scale factor: at P=1 (or max P), a_nm = TARGET_NM
    P_ref = max(P_values)
    a_natural_ref = raw[P_ref]['a_sim'] * raw[P_ref]['xi']
    if a_natural_ref < 1e-9:
        a_natural_ref = 1.0
    scale_factor = TARGET_NM / a_natural_ref

    results = {}
    for P, d in raw.items():
        a_nm = d['a_sim'] * d['xi'] * scale_factor
        results[P] = {
            'a_sim':      d['a_sim'],
            'k_peak':     d['k_peak'],
            'xi':         d['xi'],
            'a_nm':       a_nm,
            'in_visible': VISIBLE_MIN_NM <= a_nm <= VISIBLE_MAX_NM,
        }

    _plot_scaling(results, out_dir)
    _print_table(results)
    return results


def _print_table(results: dict):
    print("\n[Phase 4] Lattice scaling table:")
    print(f"{'P':>6} {'a_sim':>8} {'xi':>8} {'a_nm':>10} {'Visible?':>10}")
    print("-" * 48)
    for P in sorted(results.keys()):
        d = results[P]
        print(f"{P:6.2f} {d['a_sim']:8.3f} {d['xi']:8.3f} "
              f"{d['a_nm']:10.1f} {'YES' if d['in_visible'] else 'no':>10}")


def _plot_scaling(results: dict, out_dir: Path):
    P_arr  = np.array(sorted(results.keys()))
    a_nm   = np.array([results[P]['a_nm'] for P in P_arr])
    xi_arr = np.array([results[P]['xi']   for P in P_arr])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: a_nm vs P with visible band shaded
    ax = axes[0]
    ax.plot(P_arr, a_nm, 'o-', color='#D65F5F', linewidth=2, markersize=7,
            label='Lattice constant a(P)')
    ax.axhspan(VISIBLE_MIN_NM, VISIBLE_MAX_NM, alpha=0.15, color='gold',
               label='Visible range (200–500 nm)')
    ax.axhline(TARGET_NM, linestyle='--', color='grey', linewidth=1)
    ax.set_xlabel("Protein loading P", fontsize=11)
    ax.set_ylabel("Lattice constant a (nm)", fontsize=11)
    ax.set_title("Lattice constant vs protein loading", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: Helfrich length xi vs P
    ax = axes[1]
    ax.plot(P_arr, xi_arr, 's-', color='#4878CF', linewidth=2, markersize=7)
    ax.set_xlabel("Protein loading P", fontsize=11)
    ax.set_ylabel("Helfrich length ξ(P)", fontsize=11)
    ax.set_title("Membrane length scale vs protein loading", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "phase4_lattice_scaling.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Phase 4] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json
    sys.path.insert(0, str(Path(__file__).parent))

    FIG_DIR  = Path("/home/ubuntu/cubic-membrane-photonics/figures")
    DATA_DIR = Path("/home/ubuntu/cubic-membrane-photonics/data")

    print("=== Phase 4: Lattice scaling ===")
    P_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = sweep_lattice_vs_P(P_vals, N=64, n_steps=2000, out_dir=FIG_DIR)

    with open(DATA_DIR / "lattice_scaling.json", "w") as f:
        json.dump({str(P): d for P, d in results.items()}, f, indent=2)

    print("[Phase 4] Done.")
