"""
Phase 5: Export Geometry to Photonics — Band Structure Calculation
==================================================================
Takes the thresholded phase-field output and computes an approximate
photonic band structure using a plane-wave expansion (PWE) method
implemented in pure NumPy/SciPy (no external MPB/MEEP dependency).

Physical model
--------------
Dielectric mapping:
    epsilon(r) = eps_h * chi(r) + eps_l * (1 - chi(r))
    chi(r) = 1 if phi(r) > 0 else 0

Eigenvalue problem (transverse magnetic, Fourier space):
    sum_G' M(G, G') * H(G') = (omega/c)^2 H(G)

where M(G,G') = (k+G) x (k+G') / epsilon_inv(G-G')

We compute the lowest N_bands eigenvalues along the high-symmetry path
Gamma → X → M → Gamma in the first Brillouin zone of a simple cubic lattice.

This gives:
  - photonic band diagram
  - stop-band identification
  - gap-to-midgap ratio

Physical units:
    Omega = omega * a / (2*pi*c)   (normalised frequency)
    lambda_nm = a_nm / Omega       (physical wavelength)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from itertools import product as iproduct

# ─────────────────────────────────────────────────────────────────────────────
# Dielectric mapping
# ─────────────────────────────────────────────────────────────────────────────

EPS_CHITIN = 2.56   # chitin refractive index ~1.6, eps = n^2
EPS_AIR    = 1.0
N_BANDS    = 8      # number of bands to compute
N_PW       = 2      # plane waves per dimension (2^3 = 125 total); increase to 5+ for production


def make_dielectric(phi: np.ndarray,
                    eps_h: float = EPS_CHITIN,
                    eps_l: float = EPS_AIR,
                    threshold: float = 0.0) -> np.ndarray:
    """Return 3-D dielectric array from phase-field."""
    chi = (phi > threshold).astype(float)
    return eps_h * chi + eps_l * (1.0 - chi)


def dielectric_fourier(eps: np.ndarray) -> np.ndarray:
    """Return Fourier coefficients of the dielectric function."""
    return np.fft.fftn(eps) / eps.size


def inverse_dielectric_fourier(eps: np.ndarray) -> np.ndarray:
    """Return Fourier coefficients of 1/epsilon."""
    inv_eps = 1.0 / eps
    return np.fft.fftn(inv_eps) / inv_eps.size


# ─────────────────────────────────────────────────────────────────────────────
# Plane-wave expansion (scalar approximation)
# ─────────────────────────────────────────────────────────────────────────────

def _pw_indices(n_pw: int):
    """Return list of (gx, gy, gz) integer indices for plane waves."""
    r = range(-n_pw, n_pw + 1)
    return list(iproduct(r, r, r))


def build_hamiltonian(k_vec: np.ndarray, inv_eps_hat: np.ndarray,
                      n_pw: int = N_PW) -> np.ndarray:
    """
    Build the PWE Hamiltonian matrix H(G,G') for wave vector k.

    Scalar (TM-like) approximation:
        H(G,G') = (k+G)·(k+G') * inv_eps_hat[G-G']

    Parameters
    ----------
    k_vec : array (3,)
        Bloch wave vector in units of 2*pi/a.
    inv_eps_hat : ndarray (N, N, N)
        Fourier transform of 1/epsilon (periodic, normalised).
    n_pw : int
        Number of plane waves per dimension.

    Returns
    -------
    H : ndarray (n_G, n_G), real symmetric
    """
    G_list = _pw_indices(n_pw)
    n_G = len(G_list)
    N   = inv_eps_hat.shape[0]

    H = np.zeros((n_G, n_G), dtype=complex)

    for i, G in enumerate(G_list):
        kG_i = k_vec + np.array(G, dtype=float)
        for j, Gp in enumerate(G_list):
            kG_j  = k_vec + np.array(Gp, dtype=float)
            dG    = tuple(int(G[d] - Gp[d]) % N for d in range(3))
            eps_ij = inv_eps_hat[dG]
            H[i, j] = np.dot(kG_i, kG_j) * eps_ij

    # Ensure Hermitian
    H = 0.5 * (H + H.conj().T)
    return H


def compute_bands(k_vec: np.ndarray, inv_eps_hat: np.ndarray,
                  n_bands: int = N_BANDS, n_pw: int = N_PW) -> np.ndarray:
    """
    Return the lowest n_bands normalised frequencies at wave vector k.

    Omega = sqrt(eigenvalue) / (2*pi)  [in units of a/lambda]
    """
    H = build_hamiltonian(k_vec, inv_eps_hat, n_pw)
    eigvals = np.linalg.eigvalsh(H)
    eigvals = np.sort(eigvals.real)
    # Keep only positive eigenvalues (physical)
    eigvals = eigvals[eigvals > 0]
    if len(eigvals) < n_bands:
        eigvals = np.pad(eigvals, (0, n_bands - len(eigvals)),
                         constant_values=np.nan)
    freqs = np.sqrt(eigvals[:n_bands]) / (2 * np.pi)
    return freqs


# ─────────────────────────────────────────────────────────────────────────────
# High-symmetry k-path (simple cubic)
# ─────────────────────────────────────────────────────────────────────────────

def make_kpath(n_points: int = 20):
    """
    Return k-path along Gamma → X → M → Gamma for simple cubic lattice.

    Points in units of 2*pi/a.
    """
    Gamma = np.array([0.0, 0.0, 0.0])
    X     = np.array([0.5, 0.0, 0.0])
    M     = np.array([0.5, 0.5, 0.0])
    R     = np.array([0.5, 0.5, 0.5])

    segments = [
        (Gamma, X, "Γ→X"),
        (X,     M, "X→M"),
        (M,     Gamma, "M→Γ"),
        (Gamma, R, "Γ→R"),
    ]

    k_points = []
    labels   = []
    label_pos = []
    pos = 0

    for start, end, name in segments:
        pts = np.linspace(start, end, n_points, endpoint=False)
        k_points.append(pts)
        label_pos.append(pos)
        labels.append(name.split("→")[0])
        pos += n_points

    k_points.append(end[np.newaxis, :])
    label_pos.append(pos)
    labels.append(name.split("→")[1])

    k_path = np.vstack(k_points)
    return k_path, label_pos, labels


# ─────────────────────────────────────────────────────────────────────────────
# Full band structure calculation
# ─────────────────────────────────────────────────────────────────────────────

def compute_band_structure(
    phi: np.ndarray,
    a_nm: float = 350.0,
    n_bands: int = N_BANDS,
    n_pw: int = N_PW,
    n_kpoints: int = 15,
    eps_h: float = EPS_CHITIN,
    eps_l: float = EPS_AIR,
) -> dict:
    """
    Compute photonic band structure from phase-field output.

    Returns
    -------
    result : dict with keys:
        'bands'      : ndarray (n_k, n_bands)  normalised frequencies
        'k_path'     : ndarray (n_k, 3)
        'label_pos'  : list of ints
        'labels'     : list of str
        'gap_ratio'  : float  (gap-to-midgap ratio, 0 if no gap)
        'stop_band'  : tuple (Omega_low, Omega_high) or None
        'lambda_nm'  : ndarray  physical wavelengths at band edges
        'a_nm'       : float
    """
    eps = make_dielectric(phi, eps_h, eps_l)
    inv_eps_hat = inverse_dielectric_fourier(eps)

    k_path, label_pos, labels = make_kpath(n_kpoints)
    n_k = len(k_path)

    bands = np.zeros((n_k, n_bands))
    for i, k_vec in enumerate(k_path):
        bands[i] = compute_bands(k_vec, inv_eps_hat, n_bands, n_pw)

    # Identify stop band between band n and n+1
    gap_ratio, stop_band = find_stop_band(bands)

    # Convert to physical wavelengths
    with np.errstate(divide='ignore', invalid='ignore'):
        lambda_nm = np.where(bands > 0, a_nm / bands, np.nan)

    return {
        'bands':     bands,
        'k_path':    k_path,
        'label_pos': label_pos,
        'labels':    labels,
        'gap_ratio': gap_ratio,
        'stop_band': stop_band,
        'lambda_nm': lambda_nm,
        'a_nm':      a_nm,
        'eps_h':     eps_h,
        'eps_l':     eps_l,
    }


def find_stop_band(bands: np.ndarray):
    """
    Find the largest photonic stop band (partial gap) in the band structure.

    Returns
    -------
    gap_ratio : float
    stop_band : (Omega_low, Omega_high) or None
    """
    n_bands = bands.shape[1]
    best_ratio = 0.0
    best_band  = None

    for n in range(n_bands - 1):
        omega_max_n   = np.nanmax(bands[:, n])
        omega_min_np1 = np.nanmin(bands[:, n + 1])
        if omega_min_np1 > omega_max_n:
            omega_center = 0.5 * (omega_max_n + omega_min_np1)
            ratio = (omega_min_np1 - omega_max_n) / omega_center
            if ratio > best_ratio:
                best_ratio = ratio
                best_band  = (omega_max_n, omega_min_np1)

    return best_ratio, best_band


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_band_structure(result: dict, out_dir: Path, tag: str = ""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bands     = result['bands']
    label_pos = result['label_pos']
    labels    = result['labels']
    stop_band = result['stop_band']
    gap_ratio = result['gap_ratio']
    a_nm      = result['a_nm']

    n_k, n_bands = bands.shape
    x = np.arange(n_k)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: band diagram (normalised frequency)
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, n_bands))
    for b in range(n_bands):
        ax.plot(x, bands[:, b], color=colors[b], linewidth=1.5)

    if stop_band is not None:
        ax.axhspan(stop_band[0], stop_band[1], alpha=0.2, color='red',
                   label=f"Stop band (Δ={gap_ratio:.3f})")
        ax.legend(fontsize=9)

    for pos in label_pos:
        ax.axvline(pos, color='k', linewidth=0.8, linestyle='--', alpha=0.5)

    ax.set_xticks(label_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Normalised frequency Ω = ωa/2πc", fontsize=10)
    ax.set_title(f"Photonic band structure\n(a = {a_nm:.0f} nm)", fontsize=11,
                 fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, n_k - 1)

    # Right: wavelength axis
    ax2 = axes[1]
    for b in range(n_bands):
        lam = result['lambda_nm'][:, b]
        ax2.plot(x, lam, color=colors[b], linewidth=1.5)

    if stop_band is not None:
        lam_low  = a_nm / stop_band[1]
        lam_high = a_nm / stop_band[0]
        ax2.axhspan(lam_low, lam_high, alpha=0.2, color='red',
                    label=f"Stop band\n{lam_low:.0f}–{lam_high:.0f} nm")
        ax2.legend(fontsize=9)

    ax2.axhspan(380, 700, alpha=0.08, color='gold', label='Visible (380–700 nm)')
    ax2.set_xticks(label_pos)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("Wavelength λ (nm)", fontsize=10)
    ax2.set_title(f"Wavelength map\n(ε_h = {result['eps_h']:.2f})", fontsize=11,
                  fontweight='bold')
    ax2.set_ylim(0, 1200)
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim(0, n_k - 1)
    ax2.legend(fontsize=9)

    for pos in label_pos:
        ax2.axvline(pos, color='k', linewidth=0.8, linestyle='--', alpha=0.5)

    plt.tight_layout()
    fname = f"phase5_band_structure{'_' + tag if tag else ''}.png"
    path = out_dir / fname
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Phase 5] Saved: {path}")


def plot_dielectric_slices(phi: np.ndarray, out_dir: Path, tag: str = ""):
    """Show the thresholded dielectric structure."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eps = make_dielectric(phi)
    N   = phi.shape[0]
    mid = N // 2

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    cmap = plt.cm.binary
    planes = [
        (eps[:, :, mid], "z = N/2"),
        (eps[:, mid, :], "y = N/2"),
        (eps[mid, :, :], "x = N/2"),
    ]
    for ax, (data, title) in zip(axes, planes):
        im = ax.imshow(data, cmap=cmap, vmin=EPS_AIR, vmax=EPS_CHITIN,
                       interpolation='nearest', origin='lower')
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='ε(r)')

    fig.suptitle(f"Dielectric structure ε(r) — {tag}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = f"phase5_dielectric{'_' + tag if tag else ''}.png"
    path = out_dir / fname
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Phase 5] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json
    sys.path.insert(0, str(Path(__file__).parent))

    FIG_DIR  = Path("/home/ubuntu/cubic-membrane-photonics/figures")
    DATA_DIR = Path("/home/ubuntu/cubic-membrane-photonics/data")

    print("=== Phase 5: Photonic band structure ===")

    from phase2_curvature import run_with_curvature

    for P, tag in [(0.5, "gyroid"), (1.0, "diamond")]:
        print(f"  Computing bands for P={P} ({tag}) ...")
        phi = run_with_curvature(P=P, N=64, n_steps=2000)
        plot_dielectric_slices(phi, FIG_DIR, tag=tag)
        result = compute_band_structure(phi, a_nm=350.0, n_pw=4, n_kpoints=12)
        plot_band_structure(result, FIG_DIR, tag=tag)
        print(f"  Gap ratio: {result['gap_ratio']:.4f}")
        if result['stop_band']:
            sb = result['stop_band']
            print(f"  Stop band: Ω = {sb[0]:.4f} – {sb[1]:.4f}")

    print("[Phase 5] Done.")
