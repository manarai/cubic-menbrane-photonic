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

We compute the lowest N_bands eigenvalues along high-symmetry k-paths
for either a simple cubic (SC) or face-centred cubic (FCC) Brillouin zone.

Simple cubic (SC) path:   Gamma → X → M → Gamma → R
FCC path:                 Gamma → X → U|K → Gamma → L → W → X

This gives:
  - photonic band diagram
  - stop-band identification
  - gap-to-midgap ratio

Physical units:
    Omega = omega * a / (2*pi*c)   (normalised frequency)
    lambda_nm = a_nm / Omega       (physical wavelength)

Brillouin zone selection
------------------------
The gyroid structure in *Callophrys rubi* has a body-centred cubic (BCC)
Bravais lattice, which has the same Brillouin zone shape as FCC in
reciprocal space.  For accurate band structure calculations of the gyroid,
use ``lattice='fcc'``.  For the double-diamond (Pn3m), which has a simple
cubic Bravais lattice, use ``lattice='sc'`` (the default).
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
    """
    Build the 3-D dielectric array from a phase-field.

    The dielectric function is defined as:

        epsilon(r) = eps_h  if phi(r) > threshold  (chitin-rich phase)
                   = eps_l  otherwise               (air phase)

    Parameters
    ----------
    phi : ndarray, shape (N, N, N)
        Order-parameter field, as returned by
        :func:`phase2_curvature.run_with_curvature`.
    eps_h : float, optional
        Dielectric constant of the high-index material (chitin).
        Default is ``EPS_CHITIN = 2.56`` (n = 1.6).
    eps_l : float, optional
        Dielectric constant of the low-index material (air).
        Default is ``EPS_AIR = 1.0``.
    threshold : float, optional
        Iso-value used to binarise the phase-field.  Default is 0.0.

    Returns
    -------
    eps : ndarray, shape (N, N, N)
        Dielectric function sampled on the same grid as ``phi``.
    """
    chi = (phi > threshold).astype(float)
    return eps_h * chi + eps_l * (1.0 - chi)


def dielectric_fourier(eps: np.ndarray) -> np.ndarray:
    """
    Return the normalised Fourier coefficients of the dielectric function.

    Parameters
    ----------
    eps : ndarray, shape (N, N, N)
        Real-space dielectric function.

    Returns
    -------
    eps_hat : ndarray, shape (N, N, N), complex
        Fourier coefficients normalised by the number of grid points.
    """
    return np.fft.fftn(eps) / eps.size


def inverse_dielectric_fourier(eps: np.ndarray) -> np.ndarray:
    """
    Return the normalised Fourier coefficients of the inverse dielectric function.

    Parameters
    ----------
    eps : ndarray, shape (N, N, N)
        Real-space dielectric function (must be strictly positive).

    Returns
    -------
    inv_eps_hat : ndarray, shape (N, N, N), complex
        Fourier coefficients of ``1/epsilon``, normalised by grid size.
    """
    inv_eps = 1.0 / eps
    return np.fft.fftn(inv_eps) / inv_eps.size


# ─────────────────────────────────────────────────────────────────────────────
# Plane-wave expansion (scalar approximation)
# ─────────────────────────────────────────────────────────────────────────────

def _pw_indices(n_pw: int) -> list:
    """
    Return a list of integer (gx, gy, gz) reciprocal-lattice vectors.

    Parameters
    ----------
    n_pw : int
        Maximum integer wavenumber in each direction.  The total number
        of plane waves is ``(2*n_pw + 1)^3``.

    Returns
    -------
    G_list : list of tuple
        List of (gx, gy, gz) integer triplets with each component in
        ``[-n_pw, n_pw]``.
    """
    r = range(-n_pw, n_pw + 1)
    return list(iproduct(r, r, r))


def build_hamiltonian(k_vec: np.ndarray, inv_eps_hat: np.ndarray,
                      n_pw: int = N_PW) -> np.ndarray:
    """
    Build the plane-wave expansion Hamiltonian matrix at Bloch vector k.

    Uses the scalar (TM-like) approximation:

        H(G, G') = (k + G) · (k + G') * inv_eps_hat[G - G']

    where ``inv_eps_hat[G - G']`` is the Fourier coefficient of ``1/epsilon``
    at the difference vector ``G - G'``.

    Parameters
    ----------
    k_vec : ndarray, shape (3,)
        Bloch wave vector in units of ``2*pi/a``.
    inv_eps_hat : ndarray, shape (N, N, N), complex
        Fourier transform of ``1/epsilon``, as returned by
        :func:`inverse_dielectric_fourier`.
    n_pw : int, optional
        Number of plane waves per dimension.  Default is ``N_PW = 2``.
        Use ``n_pw >= 5`` for production-quality band structures.

    Returns
    -------
    H : ndarray, shape (n_G, n_G), complex
        Hermitian Hamiltonian matrix.  The number of plane waves is
        ``n_G = (2*n_pw + 1)^3``.

    Notes
    -----
    The matrix is symmetrised as ``H = (H + H†) / 2`` to enforce
    Hermiticity numerically.
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
    Return the lowest ``n_bands`` normalised photonic frequencies at Bloch vector k.

    Solves the Hermitian eigenvalue problem and converts eigenvalues to
    normalised frequencies:

        Omega = sqrt(eigenvalue) / (2*pi)   [units of a/lambda]

    Parameters
    ----------
    k_vec : ndarray, shape (3,)
        Bloch wave vector in units of ``2*pi/a``.
    inv_eps_hat : ndarray, shape (N, N, N), complex
        Fourier transform of ``1/epsilon``.
    n_bands : int, optional
        Number of bands to return.  Default is ``N_BANDS = 8``.
    n_pw : int, optional
        Number of plane waves per dimension.  Default is ``N_PW = 2``.

    Returns
    -------
    freqs : ndarray, shape (n_bands,)
        Normalised photonic frequencies in ascending order.  Bands with
        negative eigenvalues (unphysical) are set to ``nan``.
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
# High-symmetry k-paths
# ─────────────────────────────────────────────────────────────────────────────

def make_kpath_sc(n_points: int = 20) -> tuple:
    """
    Return the high-symmetry k-path for a simple cubic (SC) Brillouin zone.

    Path: Gamma → X → M → Gamma → R

    The high-symmetry points in units of ``2*pi/a`` are:

    +-------+-------------------+------------------------------+
    | Label | Coordinates       | Description                  |
    +=======+===================+==============================+
    | Γ     | (0, 0, 0)         | Zone centre                  |
    +-------+-------------------+------------------------------+
    | X     | (1/2, 0, 0)       | Face centre                  |
    +-------+-------------------+------------------------------+
    | M     | (1/2, 1/2, 0)     | Edge centre                  |
    +-------+-------------------+------------------------------+
    | R     | (1/2, 1/2, 1/2)   | Zone corner                  |
    +-------+-------------------+------------------------------+

    Parameters
    ----------
    n_points : int, optional
        Number of k-points per segment.  Default is 20.

    Returns
    -------
    k_path : ndarray, shape (n_k, 3)
        Bloch wave vectors along the path.
    label_pos : list of int
        Indices of high-symmetry points in ``k_path``.
    labels : list of str
        Labels of the high-symmetry points.
    """
    Gamma = np.array([0.0, 0.0, 0.0])
    X     = np.array([0.5, 0.0, 0.0])
    M     = np.array([0.5, 0.5, 0.0])
    R     = np.array([0.5, 0.5, 0.5])

    segments = [
        (Gamma, X, "Γ", "X"),
        (X,     M, "X", "M"),
        (M,     Gamma, "M", "Γ"),
        (Gamma, R, "Γ", "R"),
    ]

    return _build_kpath(segments, n_points)


def make_kpath_fcc(n_points: int = 20) -> tuple:
    """
    Return the high-symmetry k-path for a face-centred cubic (FCC) Brillouin zone.

    Path: Gamma → X → U|K → Gamma → L → W → X

    The FCC Brillouin zone (a truncated octahedron) is the appropriate
    zone for the gyroid structure in *Callophrys rubi*, which has a
    body-centred cubic (BCC) Bravais lattice (the reciprocal lattice of
    BCC is FCC).

    The high-symmetry points in units of ``2*pi/a`` are:

    +-------+-------------------+------------------------------+
    | Label | Coordinates       | Description                  |
    +=======+===================+==============================+
    | Γ     | (0, 0, 0)         | Zone centre                  |
    +-------+-------------------+------------------------------+
    | X     | (1/2, 0, 1/2)     | Square face centre           |
    +-------+-------------------+------------------------------+
    | U     | (5/8, 1/4, 5/8)   | Edge midpoint (U = K)        |
    +-------+-------------------+------------------------------+
    | K     | (3/8, 3/8, 3/4)   | Hexagonal face edge          |
    +-------+-------------------+------------------------------+
    | L     | (1/2, 1/2, 1/2)   | Hexagonal face centre        |
    +-------+-------------------+------------------------------+
    | W     | (1/2, 1/4, 3/4)   | Zone corner                  |
    +-------+-------------------+------------------------------+

    Parameters
    ----------
    n_points : int, optional
        Number of k-points per segment.  Default is 20.

    Returns
    -------
    k_path : ndarray, shape (n_k, 3)
        Bloch wave vectors along the path.
    label_pos : list of int
        Indices of high-symmetry points in ``k_path``.
    labels : list of str
        Labels of the high-symmetry points.

    Notes
    -----
    The U and K points are degenerate by symmetry in the FCC zone.
    The segment ``X → U|K`` passes through both.  The label ``U|K``
    is used to indicate this degeneracy.

    References
    ----------
    .. [1] Setyawan, W. & Curtarolo, S. (2010). High-throughput electronic
       band structure calculations: Challenges and tools.
       *Computational Materials Science*, 49(2), 299–312.
    """
    Gamma = np.array([0.000, 0.000, 0.000])
    X     = np.array([0.500, 0.000, 0.500])
    U     = np.array([0.625, 0.250, 0.625])   # U = K by symmetry
    L     = np.array([0.500, 0.500, 0.500])
    W     = np.array([0.500, 0.250, 0.750])

    segments = [
        (Gamma, X,     "Γ",   "X"),
        (X,     U,     "X",   "U|K"),
        (U,     Gamma, "U|K", "Γ"),
        (Gamma, L,     "Γ",   "L"),
        (L,     W,     "L",   "W"),
        (W,     X,     "W",   "X"),
    ]

    return _build_kpath(segments, n_points)


def make_kpath(n_points: int = 20, lattice: str = 'sc') -> tuple:
    """
    Return the high-symmetry k-path for the specified Bravais lattice.

    This is a dispatcher that calls either :func:`make_kpath_sc` or
    :func:`make_kpath_fcc` depending on the ``lattice`` argument.

    Parameters
    ----------
    n_points : int, optional
        Number of k-points per segment.  Default is 20.
    lattice : {'sc', 'fcc'}, optional
        Bravais lattice type.  Use ``'sc'`` for the double-diamond (Pn3m)
        structure and ``'fcc'`` for the gyroid (Ia3d) structure.
        Default is ``'sc'``.

    Returns
    -------
    k_path : ndarray, shape (n_k, 3)
        Bloch wave vectors along the path.
    label_pos : list of int
        Indices of high-symmetry points in ``k_path``.
    labels : list of str
        Labels of the high-symmetry points.

    Raises
    ------
    ValueError
        If ``lattice`` is not one of ``'sc'`` or ``'fcc'``.
    """
    if lattice == 'sc':
        return make_kpath_sc(n_points)
    elif lattice == 'fcc':
        return make_kpath_fcc(n_points)
    else:
        raise ValueError(f"Unknown lattice type '{lattice}'. "
                         f"Choose 'sc' (simple cubic) or 'fcc' (face-centred cubic).")


def _build_kpath(segments: list, n_points: int) -> tuple:
    """
    Construct a k-path array from a list of (start, end, start_label, end_label) segments.

    Parameters
    ----------
    segments : list of tuple
        Each element is ``(start, end, start_label, end_label)`` where
        ``start`` and ``end`` are ndarray(3,) k-vectors and the labels
        are strings.
    n_points : int
        Number of k-points per segment (not including the endpoint).

    Returns
    -------
    k_path : ndarray, shape (n_k, 3)
    label_pos : list of int
    labels : list of str
    """
    k_points  = []
    labels    = []
    label_pos = []
    pos = 0

    for i, (start, end, start_lbl, end_lbl) in enumerate(segments):
        pts = np.linspace(start, end, n_points, endpoint=False)
        k_points.append(pts)
        label_pos.append(pos)
        labels.append(start_lbl)
        pos += n_points

    # Append the final endpoint
    k_points.append(end[np.newaxis, :])
    label_pos.append(pos)
    labels.append(end_lbl)

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
    lattice: str = 'sc',
) -> dict:
    """
    Compute the photonic band structure from a phase-field output.

    Converts the phase-field to a dielectric map, constructs the
    plane-wave expansion Hamiltonian at each k-point along the
    high-symmetry path, and solves for the lowest ``n_bands`` eigenvalues.

    Parameters
    ----------
    phi : ndarray, shape (N, N, N)
        Order-parameter field, as returned by
        :func:`phase2_curvature.run_with_curvature`.
    a_nm : float, optional
        Physical lattice constant in nm, used to convert normalised
        frequencies to physical wavelengths.  Default is 350.0 nm.
    n_bands : int, optional
        Number of photonic bands to compute.  Default is ``N_BANDS = 8``.
    n_pw : int, optional
        Number of plane waves per dimension.  Default is ``N_PW = 2``.
        Use ``n_pw >= 5`` for production-quality results.
    n_kpoints : int, optional
        Number of k-points per segment of the high-symmetry path.
        Default is 15.
    eps_h : float, optional
        Dielectric constant of the high-index material.
        Default is ``EPS_CHITIN = 2.56``.
    eps_l : float, optional
        Dielectric constant of the low-index material.
        Default is ``EPS_AIR = 1.0``.
    lattice : {'sc', 'fcc'}, optional
        Brillouin zone type.  Use ``'fcc'`` for the gyroid (Ia3d) structure
        and ``'sc'`` for the double-diamond (Pn3m) structure.
        Default is ``'sc'``.

    Returns
    -------
    result : dict
        Dictionary with the following keys:

        - ``'bands'`` (ndarray, shape (n_k, n_bands)): normalised frequencies.
        - ``'k_path'`` (ndarray, shape (n_k, 3)): Bloch wave vectors.
        - ``'label_pos'`` (list of int): indices of high-symmetry points.
        - ``'labels'`` (list of str): labels of high-symmetry points.
        - ``'gap_ratio'`` (float): gap-to-midgap ratio (0 if no gap).
        - ``'stop_band'`` (tuple or None): (Omega_low, Omega_high) of the
          largest partial stop band, or ``None`` if no gap is found.
        - ``'lambda_nm'`` (ndarray): physical wavelengths at each (k, band).
        - ``'a_nm'`` (float): lattice constant used for conversion.
        - ``'eps_h'`` (float): high-index dielectric constant.
        - ``'eps_l'`` (float): low-index dielectric constant.
        - ``'lattice'`` (str): Brillouin zone type used.

    See Also
    --------
    make_kpath : Returns the k-path for SC or FCC Brillouin zones.
    find_stop_band : Identifies the largest partial stop band.
    """
    eps = make_dielectric(phi, eps_h, eps_l)
    inv_eps_hat = inverse_dielectric_fourier(eps)

    k_path, label_pos, labels = make_kpath(n_kpoints, lattice=lattice)
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
        'lattice':   lattice,
    }


def find_stop_band(bands: np.ndarray) -> tuple:
    """
    Find the largest photonic stop band (partial gap) in the band structure.

    Searches for the band gap with the largest gap-to-midgap ratio:

        delta_Omega / Omega_center = (Omega_{n+1,min} - Omega_{n,max})
                                     / (0.5 * (Omega_{n+1,min} + Omega_{n,max}))

    Parameters
    ----------
    bands : ndarray, shape (n_k, n_bands)
        Normalised photonic frequencies at each k-point and band index.

    Returns
    -------
    gap_ratio : float
        Gap-to-midgap ratio of the largest stop band.  Returns 0.0 if no
        stop band is found.
    stop_band : tuple of float or None
        ``(Omega_low, Omega_high)`` of the largest stop band, or ``None``
        if no gap is found.

    Notes
    -----
    This function identifies *partial* stop bands (gaps that exist for
    some but not all k-directions).  A full photonic bandgap requires
    a gap at every k-point simultaneously, which is much rarer and
    requires a higher refractive index contrast.
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
    """
    Save a two-panel photonic band structure figure.

    The left panel shows the normalised frequency Omega = omega*a/(2*pi*c)
    vs k-path index.  The right panel shows the corresponding physical
    wavelength lambda = a/Omega in nm.  Stop bands are shaded in red and
    the visible range (380–700 nm) is shaded in gold.

    Parameters
    ----------
    result : dict
        Band structure result dict, as returned by
        :func:`compute_band_structure`.
    out_dir : Path or str
        Output directory for the saved figure.
    tag : str, optional
        Label appended to the output filename and figure title.
        Default is ``""``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bands     = result['bands']
    label_pos = result['label_pos']
    labels    = result['labels']
    stop_band = result['stop_band']
    gap_ratio = result['gap_ratio']
    a_nm      = result['a_nm']
    lattice   = result.get('lattice', 'sc')

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
    ax.set_title(f"Photonic band structure ({lattice.upper()} BZ)\n"
                 f"a = {a_nm:.0f} nm{' — ' + tag if tag else ''}",
                 fontsize=11, fontweight='bold')
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
    """
    Save a figure showing three orthogonal mid-plane slices of the dielectric structure.

    Parameters
    ----------
    phi : ndarray, shape (N, N, N)
        Order-parameter field used to construct the dielectric map.
    out_dir : Path or str
        Output directory for the saved figure.
    tag : str, optional
        Label appended to the output filename and figure title.
        Default is ``""``.
    """
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

    # Gyroid (P=0.5) → FCC Brillouin zone
    # Diamond (P=1.0) → SC Brillouin zone
    configs = [
        (0.5, "gyroid",  "fcc", 350.0),
        (1.0, "diamond", "sc",  350.0),
    ]

    for P, tag, lattice, a_nm in configs:
        print(f"  Computing bands for P={P} ({tag}, {lattice.upper()} BZ) ...")
        phi = run_with_curvature(P=P, N=64, n_steps=2000)
        plot_dielectric_slices(phi, FIG_DIR, tag=tag)
        result = compute_band_structure(phi, a_nm=a_nm, n_pw=4,
                                        n_kpoints=12, lattice=lattice)
        plot_band_structure(result, FIG_DIR, tag=tag)
        print(f"  Gap ratio: {result['gap_ratio']:.4f}")
        if result['stop_band']:
            sb = result['stop_band']
            print(f"  Stop band: Ω = {sb[0]:.4f} – {sb[1]:.4f}")

    print("[Phase 5] Done.")
