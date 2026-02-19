"""
barkley_tw_ode.py

Traveling-wave spatial-dynamics ODE associated with the Barkley pipe-flow PDE model.

PDE (Barkley/Engel-Kuehn-de Rijk):
    q_t = D q_xx + (zeta - u) q_x + f(q,u;r)
    u_t = -u u_x + eps g(q,u)

Traveling-wave ansatz (x - s t = xi) + spatial dynamics gives 3D ODE:
    q' = p
    p' = ( (u + mu) p - f(q,u;r) ) / D
    u' = eps * g(q,u) / (u - s)

with mu = -zeta - s.

This script:
  - integrates trajectories with solve_ivp
  - plots 2D projections and a 3D phase portrait (q,p,u)
  - computes equilibria X1 and (optionally) X2 for r>2/3 (via root finding)
  - provides a crude "shooting sweep" utility to scan for candidate connecting orbits

Dependencies: numpy, scipy, matplotlib
"""

from __future__ import annotations

from matplotlib.pylab import eigvals
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, List

from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt


# -----------------------------
# Model: reaction terms f, g
# -----------------------------

def f_reaction(q: float, u: float, r: float) -> float:
    # f(q,u;r) = q( r + u - 2 - (r+0.1)(q-1)^2 )
    return q * (r + u - 2.0 - (r + 0.1) * (q - 1.0) ** 2)


def g_reaction(q: float, u: float) -> float:
    # g(q,u) = 2 - u + 2q(1-u)
    return 2.0 - u + 2.0 * q * (1.0 - u)


# -----------------------------
# ODE system (traveling wave)
# -----------------------------

@dataclass(frozen=True)
class Params:
    D: float        # diffusion coefficient (>0)
    r: float        # Reynolds-like parameter
    eps: float      # time-scale separation (small)
    zeta: float     # advection-offset parameter (paper uses ζ)
    s: float        # wave speed

    @property
    def mu(self) -> float:
        # mu = -zeta - s
        return -self.zeta - self.s


def barkley_tw_rhs(xi: float, y: np.ndarray, par: Params) -> np.ndarray:
    q, p, u = y
    D, r, eps, mu, s = par.D, par.r, par.eps, par.mu, par.s

    dq = p
    dp = ((u + mu) * p - f_reaction(q, u, r)) / D

    denom = (u - s)
    # Avoid blow-up if you land extremely close to u=s.
    # (Geometrically, u=s is a singular surface for this coordinate choice.)
    if abs(denom) < 1e-15:
        du = np.inf if denom >= 0 else -np.inf

    else:
        du = eps * g_reaction(q, u) / denom

    return np.array([dq, dp, du], dtype=float)


# -----------------------------
# Equilibria
# -----------------------------

def X1_equilibrium() -> np.ndarray:
    # laminar equilibrium: (q,p,u)=(0,0,2)
    return np.array([0.0, 0.0, 2.0], dtype=float)


def K_cubic(u: float, r: float) -> float:
    # cubic from paper (used to locate intersections in slow manifold analysis)
    # K(u;r) = 40 u^3 - (50 r + 169) u^2 + (160 r + 224) u - 120 r - 96
    return 40.0 * u**3 - (50.0 * r + 169.0) * u**2 + (160.0 * r + 224.0) * u - 120.0 * r - 96.0


def find_ub_right_branch(r: float) -> Optional[float]:
    """
    For r > 2/3, the paper notes an equilibrium X2 = (qb,+(r), 0, ub(r))
    with ub(r) in (6/5, 4/3). We find a root of K(u;r) in that bracket.

    Returns None if no root is found (e.g., if r not in bistable regime).
    """
    if r <= 2.0 / 3.0:
        return None

    a, b = 6.0/5.0 + 1e-6, 4.0/3.0 - 1e-6
    fa, fb = K_cubic(a, r), K_cubic(b, r)
    if fa * fb > 0:
        # no sign change in bracket; root finding would need a different bracket/strategy
        return None

    sol = root_scalar(lambda u: K_cubic(u, r), bracket=(a, b), method="brentq")
    return float(sol.root) if sol.converged else None


def X2_equilibrium(r: float) -> Optional[np.ndarray]:
    """
    X2 = (qb,+(r), 0, ub(r)), where qb,+(r) = 1 + (r + ub(r) - 2)/(r+0.1).
    """
    ub = find_ub_right_branch(r)
    if ub is None:
        return None
    qb = 1.0 + (r + ub - 2.0) / (r + 0.1)
    return np.array([qb, 0.0, ub], dtype=float)


#------------------------------
# Jacobian + linearization 
#------------------------------

def jacobian_at(y: np.ndarray, par: Params) -> np.ndarray:
    q, p, u = y
    D, r, eps, mu, s = par.D, par.r, par.eps, par.mu, par.s

    # f = q*(A - B*(q-1)^2), A=r+u-2, B=r+0.1
    B = r + 0.1
    A = r + u - 2.0
    h = A - B*(q-1.0)**2
    f_q = h + q*(-2.0*B*(q-1.0))
    f_u = q

    g_q = 2.0*(1.0 - u)
    g_u = -1.0 - 2.0*q
    denom = (u - s)

    # At equilibria g=0, so d/du [g/(u-s)] = g_u/(u-s)
    return np.array([
        [0.0, 1.0, 0.0],
        [-f_q/D, (u+mu)/D, -f_u/D],
        [eps*g_q/denom, 0.0, eps*g_u/denom],
    ], dtype=float)


# -----------------------------
# Integration + plotting
# -----------------------------

def integrate_orbit(
    y0: np.ndarray,
    xi_span: Tuple[float, float],
    par: Params,
    max_step: float = 0.1,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> solve_ivp:
    fun = lambda xi, y: barkley_tw_rhs(xi, y, par)
    sol = solve_ivp(
        fun=fun,
        t_span=xi_span,
        y0=np.array(y0, dtype=float),
        method="DOP853",
        max_step=max_step,
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )
    return sol


def plot_orbits(orbits: List[solve_ivp], show_equilibria: bool = True, r: float = 1.0) -> None:
    fig = plt.figure(figsize=(13, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    for sol in orbits:
        q, p, u = sol.y
        ax1.plot(q, p, linewidth=1.2)
        ax2.plot(q, u, linewidth=1.2)
        ax3.plot(q, p, u, linewidth=1.0)

    ax1.set_xlabel("q")
    ax1.set_ylabel("p")
    ax1.set_title("Projection: (q,p)")

    ax2.set_xlabel("q")
    ax2.set_ylabel("u")
    ax2.set_title("Projection: (q,u)")

    ax3.set_xlabel("q")
    ax3.set_ylabel("p")
    ax3.set_zlabel("u")
    ax3.set_title("3D phase portrait: (q,p,u)")

    if show_equilibria:
        X1 = X1_equilibrium()
        ax1.scatter([X1[0]], [X1[1]], marker="o")
        ax2.scatter([X1[0]], [X1[2]], marker="o")
        ax3.scatter([X1[0]], [X1[1]], [X1[2]], marker="o")

        X2 = X2_equilibrium(r)
        if X2 is not None:
            ax1.scatter([X2[0]], [X2[1]], marker="^")
            ax2.scatter([X2[0]], [X2[2]], marker="^")
            ax3.scatter([X2[0]], [X2[1]], [X2[2]], marker="^")

    plt.tight_layout()
    plt.show()


# -----------------------------
# Crude heteroclinic scan helper
# -----------------------------

def shooting_sweep_from_X1(
    par: Params,
    xi_span: Tuple[float, float],
    perturbations: np.ndarray,
    r: float,
    max_step: float = 0.1,
) -> List[solve_ivp]:
    """
    Very crude first pass: launch orbits from small perturbations of X1 and see where they go.

    Idea:
      - Heteroclinics/fronts/backs correspond to connecting orbits between equilibria in (q,p,u).
      - Start near X1=(0,0,2) and perturb in (q,p,u), integrate forward in xi.

    I can replace this with proper manifold continuation later.
    """
    X1 = X1_equilibrium()
    orbits = []
    for d in perturbations:
        y0 = X1 + d
        sol = integrate_orbit(y0=y0, xi_span=xi_span, par=par, max_step=max_step)
        orbits.append(sol)
    return orbits


def main() -> None:
    # --- Choose parameters (will tune these) ---
    # r > 2/3 puts you in the bistable regime described in the paper.
    par = Params(
        D=0.01,
        r=0.7,
        eps=0.01,
        zeta=0.79,
        s=0.01,   # note mu = -zeta - s
    )

    X1 = X1_equilibrium()
    J1 = jacobian_at(X1, par)
    eigvals, eigvecs = np.linalg.eig(J1)
    iu = np.argmax(eigvals.real)          # unstable eigenvalue index
    vu = eigvecs[:, iu].real
    vu /= np.linalg.norm(vu)

    delta = 1e-7
    y0 = X1 + delta * vu      # try also X1 - delta*vu


    # Print equilibria:
    print("Params:", par)
    print("mu =", par.mu)
    print("X1 =", X1_equilibrium())
    print("X2 =", X2_equilibrium(par.r))

    # -------------------------
    # Single-orbit integration
    # -------------------------
    X1 = X1_equilibrium()

    # Choose ONE perturbation (can tune this direction/magnitude).
    # Keep u-perturbation small so we don't drift toward the singular surface u=s.
    d = np.array([1e-5, 0, 0], dtype=float)   # <-- tweak as needed
    y0 = X1 + d

    # Integrate (choose direction; see notes below)
    sol = integrate_orbit(
        y0=y0,
        xi_span=(0.0, 300.0),   # forward in xi
        # xi_span=(300.0, 0),   # backward in xi
        par=par,
        max_step=0.01,
        rtol=1e-9,
        atol=1e-12,
    )

    # Plot only this orbit
    plot_orbits([sol], show_equilibria=True, r=par.r)

    # -----------------------------------------
    # Keep perturbation sweep code for later
    # -----------------------------------------
    # rng = np.random.default_rng(0)
    # perts = 1e-2 * rng.normal(size=(30, 3))
    # perts[:, 2] *= 0.2
    #
    # orbits = shooting_sweep_from_X1(
    #     par=par,
    #     xi_span=(300.0, 0.0),  # backward in xi
    #     perturbations=perts,
    #     r=par.r,
    #     max_step=0.1,
    # )
    #
    # plot_orbits(orbits, show_equilibria=True, r=par.r)



if __name__ == "__main__":
    main()
