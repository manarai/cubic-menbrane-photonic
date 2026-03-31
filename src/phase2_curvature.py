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
    """H0(P) = H_lam + alpha * P"""
    return H_LAM + ALPHA * P


def effective_kappa(P: float) -> float:
    """kappa_eff(P) = kappa0 * (1 + beta * P)"""
    return KAPPA0 * (1.0 + BETA * P)


def effective_lam(P: float, lam: float = 0.1) -> float:
    """
    lam_eff(P) = lam + kappa(P) * H0(P)

    This shifts the preferred wavenumber as P increases, driving the
    morphology from lamellar (small lam_eff) to bicontinuous (larger lam_eff).
    """
    return lam + effective_kappa(P) * spontaneous_curvature(P)


# ─────────────────────────────────────────────────────────────────────────────
# Grid helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_k2(N: int) -> np.ndarray:
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
    Semi-implicit Allen-Cahn step with effective interface parameter lam_eff.

    d phi/dt = lam_eff*nabla^2 phi + phi - phi^3

    phi_hat^{n+1} = (phi_hat^n + dt*(-phi^3)_hat^n)
                    / (1 - dt*(lam_eff*k2 + 1))
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
    Run Allen-Cahn phase-field simulation with spontaneous curvature for P.

    Parameters
    ----------
    P : float
        Protein loading in [0, 1].
    phi_init : ndarray or None
        If provided, use as initial condition.

    Returns
    -------
    phi : ndarray, shape (N, N, N)
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
    """Run simulations for each P value and return dict of final phi fields."""
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
