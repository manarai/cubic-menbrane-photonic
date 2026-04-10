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

Expansion models
----------------
In addition to the default Helfrich-based scaling, two alternative
physical models are provided for biomimetic engineering applications:

1. **Osmotic swelling** (``osmotic_lattice``): models the expansion of a
   lyotropic liquid-crystal template when immersed in a solvent of
   osmotic pressure Pi.  The lattice constant scales as:
       a_osm(Pi) = a0 * (Pi0 / Pi)^nu
   where nu ~ 0.3 is the Flory-type swelling exponent.

2. **Intercalation** (``intercalation_lattice``): models the expansion
   of a block-copolymer template when a small-molecule intercalant
   (e.g., a selective solvent or homopolymer) is added at volume
   fraction phi_int.  The lattice constant scales as:
       a_int(phi_int) = a0 * (1 + gamma * phi_int)
   where gamma is the intercalation expansion coefficient.

These models allow engineers to tune the lattice constant post-assembly
without changing the protein loading, enabling fine-grained colour control.
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
    """
    Compute the effective bending rigidity at protein loading P.

    Mirrors the definition in :mod:`phase2_curvature` with the production
    Helfrich parameters used for physical unit conversion.

    Parameters
    ----------
    P : float
        Dimensionless protein loading in [0, 1].

    Returns
    -------
    kappa : float
        Effective bending rigidity (dimensionless).
    """
    return KAPPA0 * (1.0 + BETA * P)


def helfrich_length(P: float, lam: float = LAM) -> float:
    """
    Compute the Helfrich length scale xi(P) = sqrt(kappa_eff / lam).

    The Helfrich length sets the physical length scale of the membrane
    modulations.  It increases with protein loading as the membrane
    stiffens, driving the lattice constant to larger values.

    Parameters
    ----------
    P : float
        Dimensionless protein loading in [0, 1].
    lam : float, optional
        Allen-Cahn interface width parameter.  Default is ``LAM = 1.0``.

    Returns
    -------
    xi : float
        Helfrich length scale (dimensionless simulation units).

    Examples
    --------
    >>> helfrich_length(0.0)
    1.0
    >>> helfrich_length(1.0)
    1.5811388300841898
    """
    return np.sqrt(effective_kappa(P) / lam)


# ─────────────────────────────────────────────────────────────────────────────
# Lattice constant extraction
# ─────────────────────────────────────────────────────────────────────────────

def measure_lattice_constant(phi: np.ndarray) -> tuple:
    """
    Extract the dominant lattice constant from the 3-D power spectrum.

    Computes the spherically averaged power spectrum of the phase-field
    and identifies the dominant peak wavenumber ``k_peak``.  The
    corresponding real-space periodicity is:

        a_sim = 2 * pi / k_peak

    Parameters
    ----------
    phi : ndarray, shape (N, N, N)
        Order-parameter field, as returned by
        :func:`phase2_curvature.run_with_curvature`.

    Returns
    -------
    a_sim : float
        Dimensionless lattice constant in units of ``2*pi / k_grid``.
    k_peak : float
        Dominant wavenumber in grid units (integer wavenumber).

    Notes
    -----
    The DC component (``k = 0``) is excluded from the peak search to
    avoid identifying the mean field as the dominant mode.
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
# Physical unit conversion — Helfrich model
# ─────────────────────────────────────────────────────────────────────────────

def physical_lattice(P: float, a_sim: float,
                     scale_factor: float = None) -> float:
    """
    Convert a dimensionless lattice constant to physical nanometres.

    Uses the Helfrich length scale ``xi(P)`` to convert:

        a_nm = a_sim * xi(P) * scale_factor

    If ``scale_factor`` is ``None``, it is set to 1.0 as a placeholder;
    the calibrated value is computed in :func:`sweep_lattice_vs_P`.

    Parameters
    ----------
    P : float
        Dimensionless protein loading in [0, 1].
    a_sim : float
        Dimensionless lattice constant, as returned by
        :func:`measure_lattice_constant`.
    scale_factor : float or None, optional
        Physical conversion factor (nm per simulation unit).  If ``None``,
        defaults to 1.0.  Default is ``None``.

    Returns
    -------
    a_nm : float
        Physical lattice constant in nanometres.
    """
    xi = helfrich_length(P)
    a_natural = a_sim * xi

    if scale_factor is None:
        scale_factor = 1.0   # placeholder; corrected in sweep

    return a_natural * scale_factor


# ─────────────────────────────────────────────────────────────────────────────
# Alternative expansion models for biomimetic engineering
# ─────────────────────────────────────────────────────────────────────────────

def osmotic_lattice(Pi: float, a0: float = TARGET_NM,
                    Pi0: float = 1.0, nu: float = 0.3) -> float:
    """
    Predict the lattice constant under osmotic swelling.

    Models the expansion of a lyotropic liquid-crystal template when
    immersed in a solvent of osmotic pressure ``Pi``.  The lattice
    constant scales as a power law:

        a_osm(Pi) = a0 * (Pi0 / Pi)^nu

    This is the Flory-type swelling model, where ``nu ~ 0.3`` is the
    swelling exponent for a bicontinuous cubic phase in a good solvent.

    Parameters
    ----------
    Pi : float
        Osmotic pressure of the surrounding solvent (arbitrary units;
        must be consistent with ``Pi0``).
    a0 : float, optional
        Reference lattice constant at reference pressure ``Pi0``
        (in nm).  Default is ``TARGET_NM = 350.0`` nm.
    Pi0 : float, optional
        Reference osmotic pressure.  Default is 1.0.
    nu : float, optional
        Flory swelling exponent.  Typical values are 0.2–0.4 for
        bicontinuous cubic phases.  Default is 0.3.

    Returns
    -------
    a_osm : float
        Predicted lattice constant in nm at osmotic pressure ``Pi``.

    Notes
    -----
    To use this model for colour tuning, set ``a0`` to the lattice
    constant measured at reference conditions and sweep ``Pi`` to find
    the pressure that gives the target wavelength.

    Examples
    --------
    >>> osmotic_lattice(1.0, a0=350.0)
    350.0
    >>> osmotic_lattice(0.5, a0=350.0)   # lower pressure → swelling
    392.3...
    """
    return a0 * (Pi0 / Pi) ** nu


def intercalation_lattice(phi_int: float, a0: float = TARGET_NM,
                          gamma: float = 0.5) -> float:
    """
    Predict the lattice constant upon intercalation of a small molecule.

    Models the expansion of a block-copolymer or lyotropic template when
    a selective small-molecule intercalant (e.g., a homopolymer, selective
    solvent, or surfactant) is added at volume fraction ``phi_int``.
    The lattice constant scales linearly:

        a_int(phi_int) = a0 * (1 + gamma * phi_int)

    This linear model is valid for dilute intercalant concentrations
    (``phi_int < 0.3``).

    Parameters
    ----------
    phi_int : float
        Volume fraction of the intercalant in [0, 1].
    a0 : float, optional
        Lattice constant in the absence of intercalant (in nm).
        Default is ``TARGET_NM = 350.0`` nm.
    gamma : float, optional
        Intercalation expansion coefficient.  A value of ``gamma = 0.5``
        means a 50% expansion at full intercalant loading.  Default is 0.5.

    Returns
    -------
    a_int : float
        Predicted lattice constant in nm at intercalant volume fraction
        ``phi_int``.

    Notes
    -----
    For block-copolymer systems, ``gamma`` can be estimated from the
    ratio of the homopolymer chain length to the block length.  For
    lyotropic systems, ``gamma`` is typically determined experimentally
    by small-angle X-ray scattering (SAXS).

    Examples
    --------
    >>> intercalation_lattice(0.0, a0=350.0)
    350.0
    >>> intercalation_lattice(0.2, a0=350.0, gamma=0.5)
    385.0
    """
    return a0 * (1.0 + gamma * phi_int)


def plot_expansion_models(a0: float = TARGET_NM, out_dir: Path = Path("figures")):
    """
    Plot predicted lattice constant vs control parameter for both expansion models.

    Generates a two-panel figure showing:
    - Left: osmotic swelling model (lattice constant vs osmotic pressure)
    - Right: intercalation model (lattice constant vs intercalant volume fraction)

    Parameters
    ----------
    a0 : float, optional
        Reference lattice constant in nm.  Default is ``TARGET_NM = 350.0`` nm.
    out_dir : Path or str, optional
        Output directory for the saved figure.  Default is ``"figures"``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: osmotic swelling
    ax = axes[0]
    Pi_arr = np.linspace(0.1, 3.0, 200)
    for nu, ls in [(0.2, '--'), (0.3, '-'), (0.4, ':')]:
        a_arr = [osmotic_lattice(Pi, a0=a0, nu=nu) for Pi in Pi_arr]
        ax.plot(Pi_arr, a_arr, ls, linewidth=2, label=f'ν = {nu}')
    ax.axhspan(380, 700, alpha=0.08, color='gold', label='Visible (380–700 nm)')
    ax.axhline(a0, color='grey', linewidth=1, linestyle='--', label=f'a₀ = {a0:.0f} nm')
    ax.set_xlabel("Osmotic pressure Π (arb. units)", fontsize=11)
    ax.set_ylabel("Lattice constant a (nm)", fontsize=11)
    ax.set_title("Osmotic swelling model\na(Π) = a₀ · (Π₀/Π)^ν", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: intercalation
    ax = axes[1]
    phi_arr = np.linspace(0.0, 0.4, 200)
    for gamma, ls in [(0.3, '--'), (0.5, '-'), (0.8, ':')]:
        a_arr = [intercalation_lattice(phi, a0=a0, gamma=gamma) for phi in phi_arr]
        ax.plot(phi_arr, a_arr, ls, linewidth=2, label=f'γ = {gamma}')
    ax.axhspan(380, 700, alpha=0.08, color='gold', label='Visible (380–700 nm)')
    ax.axhline(a0, color='grey', linewidth=1, linestyle='--', label=f'a₀ = {a0:.0f} nm')
    ax.set_xlabel("Intercalant volume fraction φ_int", fontsize=11)
    ax.set_ylabel("Lattice constant a (nm)", fontsize=11)
    ax.set_title("Intercalation model\na(φ) = a₀ · (1 + γ·φ)", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Biomimetic expansion models for lattice constant tuning",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = out_dir / "phase4_expansion_models.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Phase 4] Saved: {path}")


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
    Run simulations for each P value, measure the lattice constant, and
    convert to physical nanometres.

    The scale factor is calibrated so that the maximum P value maps to
    ``TARGET_NM = 350.0`` nm.

    Parameters
    ----------
    P_values : list of float
        Protein loading values to sweep.  Each must be in [0, 1].
    N : int, optional
        Number of grid points per spatial dimension.  Default is 64.
    lam : float, optional
        Allen-Cahn interface width parameter.  Default is 1.0.
    dt : float, optional
        Time step size.  Default is 0.04.
    n_steps : int, optional
        Number of time steps per simulation.  Default is 2000.
    seed : int, optional
        Random seed for all initial conditions.  Default is 42.
    out_dir : Path or str, optional
        Directory in which to save output figures.  Default is ``"figures"``.

    Returns
    -------
    results : dict
        Mapping of ``{P: record}`` where each ``record`` is a dict with keys:

        - ``'a_sim'`` (float): dimensionless lattice constant.
        - ``'k_peak'`` (float): dominant wavenumber in grid units.
        - ``'xi'`` (float): Helfrich length scale.
        - ``'a_nm'`` (float): physical lattice constant in nm.
        - ``'in_visible'`` (bool): whether ``a_nm`` is in [200, 500] nm.
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
    """
    Print a formatted table of lattice scaling results to stdout.

    Parameters
    ----------
    results : dict
        Mapping of ``{P: record}`` as returned by :func:`sweep_lattice_vs_P`.
    """
    print("\n[Phase 4] Lattice scaling table:")
    print(f"{'P':>6} {'a_sim':>8} {'xi':>8} {'a_nm':>10} {'Visible?':>10}")
    print("-" * 48)
    for P in sorted(results.keys()):
        d = results[P]
        print(f"{P:6.2f} {d['a_sim']:8.3f} {d['xi']:8.3f} "
              f"{d['a_nm']:10.1f} {'YES' if d['in_visible'] else 'no':>10}")


def _plot_scaling(results: dict, out_dir: Path):
    """
    Save a two-panel figure of lattice constant and Helfrich length vs P.

    Parameters
    ----------
    results : dict
        Mapping of ``{P: record}`` as returned by :func:`sweep_lattice_vs_P`.
    out_dir : Path
        Output directory for the saved figure.
    """
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

    # Also generate the expansion model comparison figure
    plot_expansion_models(a0=TARGET_NM, out_dir=FIG_DIR)

    print("[Phase 4] Done.")
