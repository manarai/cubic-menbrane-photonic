"""
Phase 1: Baseline Phase-Field Model (No Proteins)
==================================================
Implements a non-conserved Allen-Cahn phase-field model on a 3-D periodic
grid using FFT-based spectral derivatives.

Free energy (dimensionless, protein-free):
    F[phi] = integral [ (lam/2)|nabla phi|^2 + V(phi) ] dV

Double-well potential:
    V(phi) = (phi^2 - 1)^2 / 4  →  dV/dphi = phi^3 - phi

Allen-Cahn gradient flow (non-conserved):
    d phi/dt = -delta F/delta phi = lam*nabla^2 phi - phi^3 + phi

In Fourier space (k2 = -|k|^2 <= 0):
    d phi_hat/dt = (lam*k2 + 1)*phi_hat - (phi^3)_hat

Semi-implicit Euler (linear terms implicit, cubic nonlinearity explicit):
    phi_hat^{n+1} = (phi_hat^n + dt * (-phi^3)_hat^n)
                    / (1 - dt*(lam*k2 + 1))

The denominator is always > 0 for the relevant modes, giving unconditional
stability of the linear part.

Initial condition
-----------------
A gyroid-inspired cosine superposition provides immediate bicontinuous
structure at any grid resolution:

    phi_0(r) = A0*[cos(m*x)sin(m*y) + cos(m*y)sin(m*z) + cos(m*z)sin(m*x)]

with m=2 (two periods per box) and small additive noise.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Grid helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_k2(N: int) -> np.ndarray:
    """
    Return k2 = -|k|^2 on an N^3 grid with integer wavenumbers k in [0,N).

    Uses np.fft.fftfreq convention so k in [-N/2, N/2).
    The Laplacian eigenvalue is -(2*pi/N)^2 * |k_int|^2.
    """
    k1d = np.fft.fftfreq(N, d=1.0 / N)          # integer wavenumbers
    KX, KY, KZ = np.meshgrid(k1d, k1d, k1d, indexing='ij')
    scale = (2.0 * np.pi / N) ** 2
    return -(KX**2 + KY**2 + KZ**2) * scale      # <= 0


# ─────────────────────────────────────────────────────────────────────────────
# Initial condition — gyroid-seeded cosine superposition
# ─────────────────────────────────────────────────────────────────────────────

def gyroid_seed(N: int, A0: float = 0.5, noise: float = 0.02,
                seed: int = 42, n_modes: int = 2) -> np.ndarray:
    """
    Generate a gyroid-inspired initial condition.

    phi_0(r) = A0*[cos(m*x)sin(m*y) + cos(m*y)sin(m*z) + cos(m*z)sin(m*x)]
               + noise * randn

    Parameters
    ----------
    n_modes : int
        Spatial frequency multiplier (2 = two periods per box).
    """
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    m = n_modes
    phi = A0 * (np.cos(m * X) * np.sin(m * Y) +
                np.cos(m * Y) * np.sin(m * Z) +
                np.cos(m * Z) * np.sin(m * X))
    rng = np.random.default_rng(seed)
    phi += noise * rng.standard_normal(phi.shape)
    return phi


# ─────────────────────────────────────────────────────────────────────────────
# Allen-Cahn time-stepper (spectral, semi-implicit)
# ─────────────────────────────────────────────────────────────────────────────

def step_allen_cahn(phi_hat: np.ndarray,
                    k2: np.ndarray,
                    lam: float,
                    dt: float) -> np.ndarray:
    """
    One semi-implicit Euler step of the Allen-Cahn equation.

    d phi/dt = lam*nabla^2 phi + phi - phi^3

    Semi-implicit (linear terms implicit, cubic explicit):
        phi_hat^{n+1} = (phi_hat^n + dt*(-phi^3)_hat^n)
                        / (1 - dt*(lam*k2 + 1))

    Parameters
    ----------
    phi_hat : complex ndarray
        Fourier transform of phi.
    k2 : ndarray
        Laplacian eigenvalue array (-|k|^2 <= 0).
    lam : float
        Interface width parameter.
    dt : float
        Time step.
    """
    phi = np.real(np.fft.ifftn(phi_hat))
    nl_hat = np.fft.fftn(-phi**3)

    numerator   = phi_hat + dt * nl_hat
    denominator = 1.0 - dt * (lam * k2 + 1.0)
    return numerator / denominator


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline(
    N: int = 64,
    lam: float = 0.1,
    dt: float = 0.05,
    n_steps: int = 500,
    seed: int = 42,
    save_snapshots: bool = True,
    out_dir: Path = Path("figures"),
) -> np.ndarray:
    """
    Run the baseline (protein-free) Allen-Cahn phase-field simulation.

    Uses a gyroid-seeded initial condition for immediate bicontinuous structure.

    Returns
    -------
    phi : ndarray, shape (N, N, N)
        Final scalar field (values near ±1 in bulk, 0 at interface).
    """
    phi = gyroid_seed(N, A0=0.5, noise=0.02, seed=seed)
    k2  = make_k2(N)
    phi_hat = np.fft.fftn(phi)

    snapshot_steps = sorted({0, n_steps // 4, n_steps // 2, n_steps})
    snapshots = {}

    for step in range(n_steps + 1):
        if step in snapshot_steps:
            snapshots[step] = np.real(np.fft.ifftn(phi_hat)).copy()
        if step < n_steps:
            phi_hat = step_allen_cahn(phi_hat, k2, lam, dt)

    phi_final = snapshots[n_steps]

    if save_snapshots:
        _plot_snapshots(snapshots, out_dir, tag="baseline")

    return phi_final


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _plot_snapshots(snapshots: dict, out_dir: Path, tag: str = ""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = sorted(snapshots.keys())
    fig, axes = plt.subplots(1, len(steps), figsize=(4 * len(steps), 4))
    if len(steps) == 1:
        axes = [axes]

    cmap = plt.cm.RdBu_r
    for ax, step in zip(axes, steps):
        phi = snapshots[step]
        mid = phi.shape[2] // 2
        im = ax.imshow(phi[:, :, mid], cmap=cmap, vmin=-1.2, vmax=1.2,
                       interpolation='nearest', origin='lower')
        ax.set_title(f"t = {step}", fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Phase-field evolution ({tag})", fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = out_dir / f"phase1_{tag}_snapshots.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Phase 1] Saved: {path}")


def plot_isosurface_slice(phi: np.ndarray, out_dir: Path, tag: str = "baseline"):
    """Plot three orthogonal mid-plane slices of the final field."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    N = phi.shape[0]
    mid = N // 2
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    planes = [
        (phi[:, :, mid], "z = N/2", "x", "y"),
        (phi[:, mid, :], "y = N/2", "x", "z"),
        (phi[mid, :, :], "x = N/2", "y", "z"),
    ]
    cmap = plt.cm.RdBu_r
    for ax, (data, title, xl, yl) in zip(axes, planes):
        im = ax.imshow(data, cmap=cmap, vmin=-1.2, vmax=1.2,
                       interpolation='nearest', origin='lower')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Orthogonal slices — {tag}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = out_dir / f"phase1_{tag}_slices.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Phase 1] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FIG_DIR = Path("/home/ubuntu/cubic-membrane-photonics/figures")
    print("=== Phase 1: Baseline phase-field model ===")
    phi = run_baseline(N=64, lam=0.1, dt=0.05, n_steps=1000,
                       out_dir=FIG_DIR)
    plot_isosurface_slice(phi, FIG_DIR, tag="baseline")
    np.save("/home/ubuntu/cubic-membrane-photonics/data/phi_baseline.npy", phi)
    print("[Phase 1] Done. phi saved to data/phi_baseline.npy")
