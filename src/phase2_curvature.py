"""
Phase 2: Spontaneous Curvature Term (Protein Loading P)
========================================================
Extends the baseline phase-field model with a curvature-biasing term
controlled by the dimensionless protein loading parameter P ∈ [0, 1].

Physical model
--------------
This module implements a **phase-field proxy for curvature-driven membrane
organisation**.  It is *not* a direct numerical solution of the Helfrich
bending energy functional.  The distinction is important and is stated
explicitly in the manuscript.

The Helfrich free energy for a lipid bilayer is:

    F_H[phi] = ∫ [ (κ/2)(H - H₀)² + κ_G K ] dA

where H is the mean curvature, K is the Gaussian curvature, H₀ is the
spontaneous curvature, and κ, κ_G are the bending and saddle-splay moduli.
Solving this exactly requires tracking the membrane surface explicitly, which
is computationally expensive.

Instead, we use a **phase-field proxy** in which the curvature bias is
encoded as a modified interface parameter λ_eff(P) in the Allen–Cahn
gradient-flow equation:

    ∂φ/∂t = λ_eff(P) ∇²φ + φ - φ³

The coupling is:

    H₀(P)      = H_lam + α·P          (spontaneous curvature)
    κ_eff(P)   = κ₀·(1 + β·P)         (bending rigidity stiffening)
    λ_eff(P)   = λ + κ_eff(P)·H₀(P)   (effective interface parameter)

This proxy captures the *topology selection* (lamellar → gyroid → diamond)
driven by increasing spontaneous curvature, but does not explicitly resolve
mean curvature at each interface point.  For a model that includes an
explicit ∇⁴φ bending-like term, see :func:`step_with_bending`.

Limitation (stated explicitly for reviewers)
--------------------------------------------
The governing equation does not include curvature as a geometric quantity.
The model is therefore best described as a minimal phase-field proxy that
reproduces the topology transitions predicted by Helfrich theory, rather
than a direct numerical implementation of that theory.  Future work should
replace λ_eff with an explicit curvature functional derived from the
Canham–Helfrich energy.

Protein coupling
----------------
    H₀(P)      = H_lam + α·P
    κ_eff(P)   = κ₀·(1 + β·P)
    λ_eff(P)   = λ + κ_eff(P)·H₀(P)
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

        H₀(P) = H_lam + α·P

    where ``H_lam`` is the baseline (lamellar) curvature and ``α``
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

    Notes
    -----
    In the Helfrich model, H₀ is the preferred mean curvature of the
    membrane.  Here it is used as a proxy parameter that biases the
    phase-field toward bicontinuous morphologies as P increases.

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

        κ_eff(P) = κ₀·(1 + β·P)

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

    The curvature bias modifies the phase-field gradient penalty as:

        λ_eff(P) = λ + κ_eff(P)·H₀(P)

    A larger ``λ_eff`` shifts the dominant wavenumber to smaller values,
    driving the morphology from lamellar (small ``λ_eff``) to bicontinuous
    cubic phases (larger ``λ_eff``).

    Parameters
    ----------
    P : float
        Dimensionless protein loading in [0, 1].
    lam : float, optional
        Baseline phase-field interface width parameter.  Default is 0.1.

    Returns
    -------
    lam_eff : float
        Effective interface parameter incorporating curvature bias.

    See Also
    --------
    spontaneous_curvature : Returns H₀(P).
    effective_kappa : Returns κ_eff(P).
    step_with_bending : Alternative stepper with explicit ∇⁴φ bending term.
    """
    return lam + effective_kappa(P) * spontaneous_curvature(P)


# ─────────────────────────────────────────────────────────────────────────────
# Grid helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_k2(N: int) -> np.ndarray:
    """
    Return the Laplacian eigenvalue array on an N³ periodic grid.

    Parameters
    ----------
    N : int
        Number of grid points per spatial dimension.

    Returns
    -------
    k2 : ndarray, shape (N, N, N)
        Laplacian eigenvalue array (``-|k|² ≤ 0``).
    """
    k1d = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY, KZ = np.meshgrid(k1d, k1d, k1d, indexing='ij')
    scale = (2.0 * np.pi / N) ** 2
    return -(KX**2 + KY**2 + KZ**2) * scale


# ─────────────────────────────────────────────────────────────────────────────
# Time steppers
# ─────────────────────────────────────────────────────────────────────────────

def step_with_curvature(phi_hat: np.ndarray,
                        k2: np.ndarray,
                        lam_eff: float,
                        dt: float) -> np.ndarray:
    """
    Perform one semi-implicit phase-field step with effective interface parameter.

    Evolves the equation:

        ∂φ/∂t = λ_eff·∇²φ + φ - φ³

    using a semi-implicit Euler scheme.  The linear terms (∇²φ and φ) are
    treated implicitly; the nonlinear term (φ³) is treated explicitly.

    This is the **primary stepper** used in the paper.  It is a phase-field
    proxy for curvature-driven organisation; it does not include an explicit
    curvature term.  See :func:`step_with_bending` for the extended model.

    Parameters
    ----------
    phi_hat : ndarray, shape (N, N, N), complex
        Fourier transform of the current order-parameter field.
    k2 : ndarray, shape (N, N, N)
        Laplacian eigenvalue array (``-|k|² ≤ 0``).
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


def step_with_bending(phi_hat: np.ndarray,
                      k2: np.ndarray,
                      lam: float,
                      kappa_bend: float,
                      dt: float) -> np.ndarray:
    """
    Perform one semi-implicit step with an explicit ∇⁴φ bending-like term.

    Evolves the extended equation:

        ∂φ/∂t = λ·∇²φ - κ_bend·∇⁴φ + φ - φ³

    The ∇⁴φ term is the Fourier-space equivalent of the Cahn–Hilliard
    bending penalty and provides a more physically grounded approximation
    to the Helfrich curvature energy than the λ_eff proxy alone.

    This stepper is provided as an **optional extension** for reviewers who
    require an explicit curvature term.  It is numerically stiffer than
    :func:`step_with_curvature` and requires a smaller time step (dt ≤ 0.01).

    Parameters
    ----------
    phi_hat : ndarray, shape (N, N, N), complex
        Fourier transform of the current order-parameter field.
    k2 : ndarray, shape (N, N, N)
        Laplacian eigenvalue array (``-|k|² ≤ 0``).
    lam : float
        Interface width parameter (coefficient of ∇²φ).
    kappa_bend : float
        Bending stiffness coefficient (coefficient of ∇⁴φ).
        Typical range: 0.001–0.05.  Larger values → more pronounced
        bicontinuous topology selection.
    dt : float
        Time step size.  Must satisfy dt ≤ 1/(κ_bend·|k|⁴_max) for stability.

    Returns
    -------
    phi_hat_new : ndarray, shape (N, N, N), complex
        Fourier transform of the updated field after one time step.

    Notes
    -----
    The ∇⁴φ operator in Fourier space is ``|k|⁴ = (k²)²``, so the
    spectral representation is straightforward.

    See Also
    --------
    step_with_curvature : Simpler proxy stepper used in the main pipeline.
    """
    phi = np.real(np.fft.ifftn(phi_hat))
    nl_hat = np.fft.fftn(-phi**3)
    k4 = k2 ** 2   # ∇⁴ eigenvalue = (∇²)² = k⁴
    numerator   = phi_hat + dt * nl_hat
    denominator = 1.0 - dt * (lam * k2 - kappa_bend * k4 + 1.0)
    return numerator / denominator


def run_with_bending(
    P: float = 0.5,
    N: int = 64,
    lam: float = 0.1,
    kappa_bend: float = 0.01,
    dt: float = 0.005,
    n_steps: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    """
    Run the extended phase-field simulation with explicit ∇⁴φ bending term.

    This is the **Option B (stronger)** model suggested by reviewers.  It
    includes an explicit bending-like term that more closely approximates
    the Helfrich curvature energy.

    Parameters
    ----------
    P : float, optional
        Dimensionless protein loading in [0, 1].  Controls the amplitude
        of the initial gyroid seed.  Default is 0.5.
    N : int, optional
        Number of grid points per spatial dimension.  Default is 64.
    lam : float, optional
        Interface width parameter.  Default is 0.1.
    kappa_bend : float, optional
        Bending stiffness coefficient.  Default is 0.01.
    dt : float, optional
        Time step size.  Must be small (≤ 0.01) for stability.  Default 0.005.
    n_steps : int, optional
        Number of time steps.  Default is 2000.
    seed : int, optional
        Random seed.  Default is 42.

    Returns
    -------
    phi : ndarray, shape (N, N, N)
        Final order-parameter field.
    """
    from phase1_baseline import gyroid_seed
    k2  = _make_k2(N)
    phi = gyroid_seed(N, A0=0.3 + 0.4 * P, noise=0.02, seed=seed)
    phi_hat = np.fft.fftn(phi)
    for _ in range(n_steps):
        phi_hat = step_with_bending(phi_hat, k2, lam, kappa_bend, dt)
    return np.real(np.fft.ifftn(phi_hat))


# ─────────────────────────────────────────────────────────────────────────────
# Full simulation (primary proxy model)
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
    Run the phase-field proxy simulation with spontaneous curvature bias.

    The effective interface parameter ``λ_eff(P)`` is computed from the
    protein loading ``P`` and used throughout the simulation.

    Parameters
    ----------
    P : float, optional
        Dimensionless protein loading in [0, 1].  Default is 0.5.
    N : int, optional
        Number of grid points per spatial dimension.  Default is 64.
    lam : float, optional
        Baseline phase-field interface width parameter.  Default is 0.1.
    dt : float, optional
        Time step size.  Default is 0.05.
    n_steps : int, optional
        Number of time steps to evolve.  Default is 500.
    seed : int, optional
        Random seed for the initial condition.  Default is 42.
    phi_init : ndarray or None, optional
        If provided, use this array as the initial condition.
        Must have shape (N, N, N).

    Returns
    -------
    phi : ndarray, shape (N, N, N)
        Final order-parameter field after ``n_steps`` time steps.

    See Also
    --------
    effective_lam : Computes the effective interface parameter λ_eff(P).
    run_with_bending : Extended model with explicit ∇⁴φ bending term.
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
        Baseline phase-field interface width parameter.  Default is 0.1.
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

    fig.suptitle(
        "Effect of protein loading P on membrane morphology\n"
        "(phase-field proxy for curvature-driven organisation)",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    path = out_dir / "phase2_curvature_sweep.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Phase 2] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FIG_DIR  = Path("figures")
    DATA_DIR = Path("data")
    FIG_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    print("=== Phase 2: Phase-field proxy with spontaneous curvature bias ===")
    print("    (Allen-Cahn proxy for Helfrich curvature-driven organisation)")
    P_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = sweep_protein_loading(P_vals, N=64, n_steps=1000, out_dir=FIG_DIR)

    for P, phi in results.items():
        np.save(DATA_DIR / f"phi_P{int(P*100):03d}.npy", phi)
    print("[Phase 2] Done.")
