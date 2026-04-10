"""
Phase 2: Add Spontaneous Curvature Term (Protein Loading P)
============================================================
Extends the baseline Allen-Cahn model with a Helfrich-inspired
spontaneous curvature term controlled by the dimensionless protein
loading parameter P in [0, 1].

Extended free energy:
    F[phi] = integral [
        (kappa/2)(H - H0(P))^2   <- curvature energy
      + (lam/2)|nabla phi|^2     <- surface regularisation
      + V(phi)                   <- double-well
    ] dV

In the phase-field representation, the curvature penalty introduces an
additional term in the chemical potential proportional to H0(P)*|nabla phi|.

Allen-Cahn gradient flow with curvature:
    d phi/dt = lam*nabla^2 phi + phi - phi^3
               + kappa * H0(P) * nabla^2 phi / (|nabla phi| + eps)

For the spectral demo we implement this as an effective modified interface
parameter:
    lam_eff(P) = lam + kappa(P) * H0(P)

which shifts the dominant wavenumber and produces morphology transitions
from lamellar (P=0) through gyroid (P~0.5) to double-diamond (P=1).

Protein coupling:
    H0(P)      = H_lam + alpha * P      (spontaneous curvature)
    kappa(P)   = kappa0 * (1 + beta*P)  (bending rigidity stiffening)
    lam_eff(P) = lam + kappa(P) * H0(P) (effective interface parameter)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Protein coupling parameters
# ─────────────────────────────────────────────────────────────────────────────

H_LAM   = 0.0    # lamellar baseline curvature
ALPHA   = 0.5    # curvature bias per unit protein loading
KAPPA0  = 0.2    # baseline bending rigidity
BETA    = 1.0    # stiffening coefficient


def spontaneous_curvature(P: float) -> float:
    """
    Compute the spontaneous curvature induced by protein loading P.

    The spontaneous curvature is defined as:

        H0(P) = H_lam + alpha * P

    where ``H_lam`` is the baseline (lamellar) curvature and ``alpha``
    is the coupling constant between protein loading and curvature.

    Parameters
    ----------
    P : float
        Dimensionless protein loading in [0, 1].  ``P = 0`` corresponds
        to a protein-free membrane; ``P = 1`` corresponds to maximum
        protein crowding.

    Returns
    -------
    H0 : float
        Spontaneous curvature bias (dimensionless).

    Examples
    --------
    >>> spontaneous_curvature(0.0)
    0.0
    >>> spontaneous_curvature(1.0)
    0.5
    """
    return H_LAM + ALPHA * P


def effective_kappa(P: float) -> float:
    """
    Compute the effective bending rigidity at protein loading P.

    Protein crowding stiffens the membrane via:

        kappa_eff(P) = kappa0 * (1 + beta * P)

    Parameters
    ----------
    P : float
        Dimensionless protein loading in [0, 1].

    Returns
    -------
    kappa : float
        Effective bending rigidity (dimensionless).

    Examples
    --------
    >>> effective_kappa(0.0)
    0.2
    >>> effective_kappa(1.0)
    0.4
    """
    return KAPPA0 * (1.0 + BETA * P)


def effective_lam(P: float, lam: float = 0.1) -> float:
    """
    Compute the effective interface parameter at protein loading P.

    The curvature energy modifies the Allen-Cahn gradient penalty as:

        lam_eff(P) = lam + kappa(P) * H0(P)

    A larger ``lam_eff`` shifts the dominant wavenumber to smaller values,
    driving the morphology from lamellar (small ``lam_eff``) to bicontinuous
    cubic phases (larger ``lam_eff``).

    Parameters
    ----------
    P : float
        Dimensionless protein loading in [0, 1].
    lam : float, optional
        Baseline Allen-Cahn interface width parameter.  Default is 0.1.

    Returns
    -------
    lam_eff : float
        Effective interface parameter incorporating curvature effects.

    See Also
    --------
    spontaneous_curvature : Returns H0(P).
    effective_kappa : Returns kappa_eff(P).
    """
    return lam + effective_kappa(P) * spontaneous_curvature(P)


# ─────────────────────────────────────────────────────────────────────────────
# Grid helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_k2(N: int) -> np.ndarray:
    """
    Return the Laplacian eigenvalue array on an N^3 periodic grid.

    Parameters
    ----------
    N : int
        Number of grid points per spatial dimension.

    Returns
    -------
    k2 : ndarray, shape (N, N, N)
        Laplacian eigenvalue array (``-|k|^2 <= 0``).
    """
    k1d = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY, KZ = np.meshgrid(k1d, k1d, k1d, indexing='ij')
    scale = (2.0 * np.pi / N) ** 2
    return -(KX**2 + KY**2 + KZ**2) * scale


# ─────────────────────────────────────────────────────────────────────────────
# Time stepper with curvature
# ─────────────────────────────────────────────────────────────────────────────

def step_with_curvature(phi_hat: np.ndarray,
                        k2: np.ndarray,
                        lam_eff: float,
                        dt: float) -> np.ndarray:
    """
    Perform one semi-implicit Allen-Cahn step with effective interface parameter.

    Evolves the equation:

        d phi/dt = lam_eff * nabla^2 phi + phi - phi^3

    using the same semi-implicit Euler scheme as Phase 1 but with the
    curvature-modified interface parameter ``lam_eff(P)``.

    Parameters
    ----------
    phi_hat : ndarray, shape (N, N, N), complex
        Fourier transform of the current order-parameter field.
    k2 : ndarray, shape (N, N, N)
        Laplacian eigenvalue array (``-|k|^2 <= 0``).
    lam_eff : float
        Effective interface parameter, as returned by :func:`effective_lam`.
    dt : float
        Time step size.

    Returns
    -------
    phi_hat_new : ndarray, shape (N, N, N), complex
        Fourier transform of the updated field after one time step.
    """
    phi = np.real(np.fft.ifftn(phi_hat))
    nl_hat = np.fft.fftn(-phi**3)
    numerator   = phi_hat + dt * nl_hat
    denominator = 1.0 - dt * (lam_eff * k2 + 1.0)
    return numerator / denominator


# ─────────────────────────────────────────────────────────────────────────────
# Full simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_with_curvature(
    P: float = 0.5,
    N: int = 64,
    lam: float = 0.1,
    dt: float = 0.05,
    n_steps: int = 500,
    seed: int = 42,
    phi_init: np.ndarray = None,
) -> np.ndarray:
    """
    Run the Allen-Cahn phase-field simulation with spontaneous curvature.

    The effective interface parameter ``lam_eff(P)`` is computed from the
    protein loading ``P`` and used throughout the simulation.  A gyroid-seeded
    initial condition is used unless ``phi_init`` is provided.

    Parameters
    ----------
    P : float, optional
        Dimensionless protein loading in [0, 1].  Default is 0.5.
    N : int, optional
        Number of grid points per spatial dimension.  Default is 64.
    lam : float, optional
        Baseline Allen-Cahn interface width parameter.  Default is 0.1.
    dt : float, optional
        Time step size.  Default is 0.05.
    n_steps : int, optional
        Number of time steps to evolve.  Default is 500.
    seed : int, optional
        Random seed for the gyroid initial condition.  Default is 42.
    phi_init : ndarray or None, optional
        If provided, use this array as the initial condition instead of
        generating a gyroid seed.  Must have shape (N, N, N).

    Returns
    -------
    phi : ndarray, shape (N, N, N)
        Final order-parameter field after ``n_steps`` time steps.

    See Also
    --------
    effective_lam : Computes the effective interface parameter lam_eff(P).
    phase1_baseline.run_baseline : Protein-free baseline simulation.
    """
    from phase1_baseline import gyroid_seed

    lam_e = effective_lam(P, lam)
    k2    = _make_k2(N)

    if phi_init is not None:
        phi = phi_init.copy()
    else:
        phi = gyroid_seed(N, A0=0.5, noise=0.02, seed=seed)

    phi_hat = np.fft.fftn(phi)

    for _ in range(n_steps):
        phi_hat = step_with_curvature(phi_hat, k2, lam_e, dt)

    return np.real(np.fft.ifftn(phi_hat))


# ─────────────────────────────────────────────────────────────────────────────
# Protein loading sweep
# ─────────────────────────────────────────────────────────────────────────────

def sweep_protein_loading(
    P_values: list,
    N: int = 64,
    lam: float = 0.1,
    dt: float = 0.05,
    n_steps: int = 500,
    seed: int = 42,
    out_dir: Path = Path("figures"),
) -> dict:
    """
    Run simulations for each value of P and return the final phase fields.

    Parameters
    ----------
    P_values : list of float
        Protein loading values to sweep over.  Each must be in [0, 1].
    N : int, optional
        Number of grid points per spatial dimension.  Default is 64.
    lam : float, optional
        Baseline Allen-Cahn interface width parameter.  Default is 0.1.
    dt : float, optional
        Time step size.  Default is 0.05.
    n_steps : int, optional
        Number of time steps per simulation.  Default is 500.
    seed : int, optional
        Random seed for all initial conditions.  Default is 42.
    out_dir : Path or str, optional
        Directory in which to save the comparison figure.
        Default is ``"figures"``.

    Returns
    -------
    results : dict
        Mapping of ``{P: phi_array}`` for each protein loading value.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for P in P_values:
        print(f"  [Phase 2] Running P = {P:.2f} ...")
        phi = run_with_curvature(P=P, N=N, lam=lam, dt=dt,
                                 n_steps=n_steps, seed=seed)
        results[P] = phi

    _plot_sweep(results, out_dir)
    return results


def _plot_sweep(results: dict, out_dir: Path):
    """
    Save a comparison figure of mid-plane slices for each P value.

    Parameters
    ----------
    results : dict
        Mapping of ``{P: phi_array}`` as returned by :func:`sweep_protein_loading`.
    out_dir : Path
        Output directory for the saved figure.
    """
    P_values = sorted(results.keys())
    n = len(P_values)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    cmap = plt.cm.RdBu_r
    for ax, P in zip(axes, P_values):
        phi = results[P]
        mid = phi.shape[2] // 2
        im = ax.imshow(phi[:, :, mid], cmap=cmap, vmin=-1.2, vmax=1.2,
                       interpolation='nearest', origin='lower')
        ax.set_title(f"P = {P:.2f}\nH\u2080={spontaneous_curvature(P):.2f}  "
                     f"\u03bb_eff={effective_lam(P):.3f}",
                     fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Effect of protein loading P on membrane morphology",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = out_dir / "phase2_curvature_sweep.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Phase 2] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FIG_DIR  = Path("/home/ubuntu/cubic-membrane-photonics/figures")
    DATA_DIR = Path("/home/ubuntu/cubic-membrane-photonics/data")

    print("=== Phase 2: Spontaneous curvature term ===")
    P_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = sweep_protein_loading(P_vals, N=64, n_steps=1000, out_dir=FIG_DIR)

    for P, phi in results.items():
        np.save(DATA_DIR / f"phi_P{int(P*100):03d}.npy", phi)
    print("[Phase 2] Done.")
