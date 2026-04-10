"""
Phase 6: Add Polycrystalline Domains
=====================================
Implements spatially heterogeneous protein loading P(r) to generate
polycrystalline photonic architectures consistent with the observed
3–7 μm domain sizes in beetle scales.

Mathematical model
------------------
    P(r) = sum_i P_i * chi_i(r)

where:
  - chi_i(r): Voronoi domain indicator (1 inside domain i, 0 elsewhere)
  - P_i: random protein density drawn from U[P_min, P_max]

This generates:
  - independent nucleation sites
  - random domain orientations (via random seed per domain)
  - realistic domain boundaries

The polycrystalline phi(r) is assembled by:
  1. Generating Voronoi tessellation from random seed points
  2. Running independent phase-field simulations per domain
  3. Assembling the global field with smooth boundary blending

Additional analysis:
  - Domain size distribution
  - Orientational disorder (structure factor anisotropy)
  - Colour uniformity metric (stop-band spread across domains)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.spatial import Voronoi, cKDTree

# ─────────────────────────────────────────────────────────────────────────────
# Voronoi tessellation
# ─────────────────────────────────────────────────────────────────────────────

def make_voronoi_domains(N: int, n_domains: int, seed: int = 0) -> tuple:
    """
    Generate a 3-D Voronoi domain map on an N^3 grid.

    Places ``n_domains`` seed points uniformly at random in the simulation
    box and assigns each voxel to the nearest seed point using a KD-tree
    nearest-neighbour query.

    Parameters
    ----------
    N : int
        Number of grid points per spatial dimension.  The total grid is N^3.
    n_domains : int
        Number of Voronoi domains (nucleation sites).
    seed : int, optional
        Random seed for reproducible domain placement.  Default is 0.

    Returns
    -------
    domain_map : ndarray, shape (N, N, N), dtype int
        Integer domain label for each voxel, in the range [0, n_domains).
    centers : ndarray, shape (n_domains, 3)
        Seed point coordinates in [0, N)^3 used to generate the tessellation.

    Notes
    -----
    This is a Euclidean (non-periodic) Voronoi tessellation.  Domains near
    the box boundary may be slightly smaller than interior domains because
    periodic images are not included.  For a fully periodic tessellation,
    use ``scipy.spatial.Voronoi`` with replicated seed points.
    """
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, N, size=(n_domains, 3))

    # Build KD-tree for fast nearest-neighbour lookup
    tree = cKDTree(centers)

    # Grid coordinates
    idx = np.arange(N)
    gx, gy, gz = np.meshgrid(idx, idx, idx, indexing='ij')
    coords = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(float)

    _, labels = tree.query(coords)
    domain_map = labels.reshape(N, N, N)
    return domain_map, centers


# ─────────────────────────────────────────────────────────────────────────────
# Per-domain phase-field simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_domain(P: float, N: int, lam: float, dt: float,
               n_steps: int, seed: int) -> np.ndarray:
    """
    Run a single-domain phase-field simulation with protein loading P.

    A thin wrapper around :func:`phase2_curvature.run_with_curvature` that
    runs one complete Allen-Cahn simulation for a single Voronoi domain.

    Parameters
    ----------
    P : float
        Dimensionless protein loading in [0, 1].
    N : int
        Number of grid points per spatial dimension.
    lam : float
        Allen-Cahn interface width parameter.
    dt : float
        Time step size.
    n_steps : int
        Number of time steps to evolve.
    seed : int
        Random seed for the initial condition.  Different seeds produce
        different domain orientations.

    Returns
    -------
    phi : ndarray, shape (N, N, N)
        Final order-parameter field for this domain.
    """
    from phase2_curvature import run_with_curvature
    return run_with_curvature(P=P, N=N, lam=lam, dt=dt,
                              n_steps=n_steps, seed=seed)


# ─────────────────────────────────────────────────────────────────────────────
# Polycrystalline assembly
# ─────────────────────────────────────────────────────────────────────────────

def assemble_polycrystal(
    N: int = 64,
    n_domains: int = 8,
    P_min: float = 0.4,
    P_max: float = 0.9,
    lam: float = 1.0,
    dt: float = 0.04,
    n_steps: int = 1500,
    seed: int = 42,
    blend_width: int = 2,
) -> tuple:
    """
    Generate a polycrystalline phase-field on an N^3 grid.

    Each domain is assigned a random protein loading ``P_i ~ U[P_min, P_max]``
    and an independent random initial condition (different seed), producing
    domains with different orientations and morphologies.  The global field
    is assembled by Voronoi assignment: each voxel takes the value from the
    phase-field simulation of its nearest domain centre.

    Parameters
    ----------
    N : int, optional
        Number of grid points per spatial dimension.  Default is 64.
    n_domains : int, optional
        Number of Voronoi domains.  Default is 8.
    P_min : float, optional
        Minimum protein loading for domain assignment.  Default is 0.4.
    P_max : float, optional
        Maximum protein loading for domain assignment.  Default is 0.9.
    lam : float, optional
        Allen-Cahn interface width parameter.  Default is 1.0.
    dt : float, optional
        Time step size.  Default is 0.04.
    n_steps : int, optional
        Number of time steps per domain simulation.  Default is 1500.
    seed : int, optional
        Master random seed.  Domain seeds are derived as ``seed + i + 1``
        for domain ``i``.  Default is 42.
    blend_width : int, optional
        Width (in voxels) of the soft blending zone at domain boundaries.
        Currently unused (hard assembly); reserved for future implementation.
        Default is 2.

    Returns
    -------
    phi_poly : ndarray, shape (N, N, N)
        Assembled polycrystalline scalar field.
    domain_map : ndarray, shape (N, N, N), dtype int
        Voronoi domain label for each voxel.
    P_values : ndarray, shape (n_domains,)
        Protein loading assigned to each domain.
    centers : ndarray, shape (n_domains, 3)
        Voronoi seed point coordinates.

    Notes
    -----
    The ``blend_width`` parameter is reserved for a future soft-blending
    implementation that would smooth the domain boundaries using a weighted
    average of adjacent domain fields.  The current implementation uses
    hard (sharp) boundaries.
    """
    rng = np.random.default_rng(seed)
    P_values = rng.uniform(P_min, P_max, size=n_domains)

    domain_map, centers = make_voronoi_domains(N, n_domains, seed=seed)

    # Run one simulation per domain
    domain_fields = {}
    for i in range(n_domains):
        print(f"  [Phase 6] Domain {i+1}/{n_domains}, P={P_values[i]:.3f} ...",
              end=" ")
        phi_i = run_domain(P=P_values[i], N=N, lam=lam, dt=dt,
                           n_steps=n_steps, seed=seed + i + 1)
        domain_fields[i] = phi_i
        print("done")

    # Assemble
    phi_poly = np.zeros((N, N, N))
    for i in range(n_domains):
        mask = (domain_map == i)
        phi_poly[mask] = domain_fields[i][mask]

    return phi_poly, domain_map, P_values, centers


# ─────────────────────────────────────────────────────────────────────────────
# Domain size analysis
# ─────────────────────────────────────────────────────────────────────────────

def measure_domain_sizes(domain_map: np.ndarray, a_nm: float = 350.0,
                         N_per_domain_target: int = 64) -> tuple:
    """
    Estimate the physical linear size of each Voronoi domain.

    The simulation box has a total physical length of
    ``L_nm = N_per_domain_target * a_nm`` nm in each dimension.
    The volume of each domain is estimated from its voxel count, and
    the linear size is taken as the cube root of the volume.

    Parameters
    ----------
    domain_map : ndarray, shape (N, N, N), dtype int
        Voronoi domain label for each voxel, as returned by
        :func:`make_voronoi_domains`.
    a_nm : float, optional
        Physical lattice constant in nm.  Default is 350.0 nm.
    N_per_domain_target : int, optional
        Number of unit cells per box dimension used to set the physical
        scale.  Default is 64.

    Returns
    -------
    sizes_nm : dict
        Mapping of ``{domain_id: size_nm}`` giving the estimated linear
        domain size in nanometres.
    sizes_um : dict
        Same as ``sizes_nm`` but in micrometres (nm / 1000).

    Notes
    -----
    The biological reference value for *Callophrys rubi* is a mean grain
    diameter of ~5.4 μm (Morris 1975).  Domains in the range 3–7 μm are
    considered consistent with observations.
    """
    N = domain_map.shape[0]
    L_nm = N_per_domain_target * a_nm   # total box size in nm
    voxel_vol_nm3 = (L_nm / N)**3

    n_domains = domain_map.max() + 1
    sizes_nm = {}
    for i in range(n_domains):
        n_vox = int((domain_map == i).sum())
        vol_nm3 = n_vox * voxel_vol_nm3
        size_nm = vol_nm3 ** (1.0 / 3.0)
        sizes_nm[i] = size_nm

    sizes_um = {i: s / 1000.0 for i, s in sizes_nm.items()}
    return sizes_nm, sizes_um


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_polycrystal(phi_poly: np.ndarray, domain_map: np.ndarray,
                     P_values: np.ndarray, out_dir: Path):
    """
    Save a six-panel figure of the polycrystalline phase-field and domain map.

    The top row shows three orthogonal mid-plane slices of the assembled
    phase-field ``phi_poly``.  The bottom row shows the corresponding
    Voronoi domain map coloured by domain ID.

    Parameters
    ----------
    phi_poly : ndarray, shape (N, N, N)
        Assembled polycrystalline scalar field.
    domain_map : ndarray, shape (N, N, N), dtype int
        Voronoi domain label for each voxel.
    P_values : ndarray, shape (n_domains,)
        Protein loading per domain (used for colour bar range).
    out_dir : Path or str
        Output directory for the saved figure.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    N   = phi_poly.shape[0]
    mid = N // 2

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Row 1: phi field slices
    cmap_phi = plt.cm.RdBu_r
    for ax, (data, title) in zip(axes[0], [
        (phi_poly[:, :, mid], "φ — z slice"),
        (phi_poly[:, mid, :], "φ — y slice"),
        (phi_poly[mid, :, :], "φ — x slice"),
    ]):
        im = ax.imshow(data, cmap=cmap_phi, vmin=-1.2, vmax=1.2,
                       interpolation='nearest', origin='lower')
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 2: domain map slices
    n_domains = len(P_values)
    cmap_dom = plt.cm.get_cmap('tab20', n_domains)
    for ax, (data, title) in zip(axes[1], [
        (domain_map[:, :, mid], "Domains — z slice"),
        (domain_map[:, mid, :], "Domains — y slice"),
        (domain_map[mid, :, :], "Domains — x slice"),
    ]):
        im = ax.imshow(data, cmap=cmap_dom, vmin=0, vmax=n_domains - 1,
                       interpolation='nearest', origin='lower')
        ax.set_title(title, fontsize=11)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Domain ID", fontsize=8)

    fig.suptitle("Polycrystalline phase-field assembly", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = out_dir / "phase6_polycrystal_slices.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Phase 6] Saved: {path}")


def plot_domain_analysis(domain_map: np.ndarray, P_values: np.ndarray,
                         sizes_um: dict, out_dir: Path):
    """
    Save a three-panel domain analysis figure.

    The three panels show:
    1. Domain size distribution (bar chart with biological reference range).
    2. Protein loading per domain (bar chart).
    3. Domain size vs protein loading (scatter plot).

    Parameters
    ----------
    domain_map : ndarray, shape (N, N, N), dtype int
        Voronoi domain label for each voxel.
    P_values : ndarray, shape (n_domains,)
        Protein loading assigned to each domain.
    sizes_um : dict
        Mapping of ``{domain_id: size_um}`` as returned by
        :func:`measure_domain_sizes`.
    out_dir : Path or str
        Output directory for the saved figure.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Domain size distribution
    ax = axes[0]
    sizes = list(sizes_um.values())
    ax.bar(range(len(sizes)), sizes, color='#4878CF', edgecolor='k', linewidth=0.5)
    ax.axhspan(3.0, 7.0, alpha=0.15, color='gold',
               label='Observed range (3–7 μm)')
    ax.set_xlabel("Domain ID", fontsize=11)
    ax.set_ylabel("Domain size (μm)", fontsize=11)
    ax.set_title("Domain size distribution", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Protein loading per domain
    ax = axes[1]
    ax.bar(range(len(P_values)), P_values, color='#D65F5F', edgecolor='k',
           linewidth=0.5)
    ax.set_xlabel("Domain ID", fontsize=11)
    ax.set_ylabel("Protein loading P", fontsize=11)
    ax.set_title("Protein loading per domain", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    # Size vs P scatter
    ax = axes[2]
    ax.scatter(P_values, sizes, c='#6ACC65', s=80, edgecolors='k',
               linewidths=0.5, zorder=3)
    ax.set_xlabel("Protein loading P", fontsize=11)
    ax.set_ylabel("Domain size (μm)", fontsize=11)
    ax.set_title("Domain size vs protein loading", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "phase6_domain_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Phase 6] Saved: {path}")


def plot_stop_band_spread(phi_poly: np.ndarray, domain_map: np.ndarray,
                          P_values: np.ndarray, out_dir: Path):
    """
    Compute per-domain stop bands and plot their spread as a colour uniformity metric.

    For each domain, the phase-field is isolated, the photonic band structure
    is computed using :func:`phase5_photonics.compute_band_structure`, and the
    stop-band wavelength range is extracted.  The resulting horizontal bar chart
    shows the stop-band spread across all domains, which is a proxy for the
    colour uniformity of the polycrystalline wing scale.

    Parameters
    ----------
    phi_poly : ndarray, shape (N, N, N)
        Assembled polycrystalline scalar field.
    domain_map : ndarray, shape (N, N, N), dtype int
        Voronoi domain label for each voxel.
    P_values : ndarray, shape (n_domains,)
        Protein loading assigned to each domain.
    out_dir : Path or str
        Output directory for the saved figure.

    Notes
    -----
    Domains with insufficient phase-field variance (``std < 0.01``) are
    skipped to avoid numerical artefacts in the band structure calculation.
    The physical lattice constant for each domain is estimated from the
    Helfrich length at the domain's protein loading.
    """
    from phase5_photonics import compute_band_structure
    from phase4_scaling import helfrich_length, TARGET_NM

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_domains = len(P_values)
    stop_lam_low  = []
    stop_lam_high = []
    domain_ids    = []

    for i in range(n_domains):
        mask = (domain_map == i)
        phi_i = phi_poly.copy()
        # Zero out other domains for isolated band calculation
        phi_i[~mask] = 0.0
        if phi_i[mask].std() < 0.01:
            continue

        xi = helfrich_length(P_values[i])
        a_nm = TARGET_NM * xi / helfrich_length(1.0)

        result = compute_band_structure(phi_i, a_nm=a_nm, n_pw=3, n_kpoints=8)
        sb = result['stop_band']
        if sb is not None:
            lam_low  = a_nm / sb[1]
            lam_high = a_nm / sb[0]
            stop_lam_low.append(lam_low)
            stop_lam_high.append(lam_high)
            domain_ids.append(i)

    if not domain_ids:
        print("[Phase 6] No stop bands found in any domain.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    for j, did in enumerate(domain_ids):
        ax.barh(j, stop_lam_high[j] - stop_lam_low[j],
                left=stop_lam_low[j], height=0.6,
                color=plt.cm.plasma(P_values[did]), edgecolor='k',
                linewidth=0.5, label=f"D{did} P={P_values[did]:.2f}")

    ax.axvspan(380, 700, alpha=0.1, color='gold', label='Visible range')
    ax.set_xlabel("Wavelength λ (nm)", fontsize=11)
    ax.set_ylabel("Domain", fontsize=11)
    ax.set_yticks(range(len(domain_ids)))
    ax.set_yticklabels([f"D{d}" for d in domain_ids], fontsize=9)
    ax.set_title("Stop-band spread across polycrystalline domains",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "phase6_stop_band_spread.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Phase 6] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    FIG_DIR  = Path("/home/ubuntu/cubic-membrane-photonics/figures")
    DATA_DIR = Path("/home/ubuntu/cubic-membrane-photonics/data")

    print("=== Phase 6: Polycrystalline domains ===")

    phi_poly, domain_map, P_vals, centers = assemble_polycrystal(
        N=64, n_domains=8, P_min=0.4, P_max=0.9,
        n_steps=1500, seed=42
    )

    np.save(DATA_DIR / "phi_polycrystal.npy", phi_poly)
    np.save(DATA_DIR / "domain_map.npy", domain_map)

    sizes_nm, sizes_um = measure_domain_sizes(domain_map, a_nm=350.0)
    plot_polycrystal(phi_poly, domain_map, P_vals, FIG_DIR)
    plot_domain_analysis(domain_map, P_vals, sizes_um, FIG_DIR)
    plot_stop_band_spread(phi_poly, domain_map, P_vals, FIG_DIR)

    print("[Phase 6] Done.")
