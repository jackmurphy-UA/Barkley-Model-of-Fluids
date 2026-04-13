"""
barkley_tw_ode.py

Traveling-wave spatial-dynamics ODE associated with the Barkley pipe-flow PDE model.

PDE (Barkley/Engel-Kuehn-de Rijk):
    q_t = D q_xx + (zeta - u) q_x + f(q,u;r)
    u_t = -u u_x + eps g(q,u)

Traveling-wave ansatz (x - s t = xi) + spatial dynamics gives 3D ODE:
    q' = p
    p' = ((u + mu) p - f(q,u;r)) / D
    u' = eps * g(q,u) / (u - s)

with mu = -zeta - s.

This script:
  - integrates trajectories with solve_ivp
  - plots 2D projections and a 3D phase portrait (q,p,u)
  - computes equilibria X1 and X2 for r > 2/3
  - launches orbits along unstable eigenvectors (new default)
  - keeps older experimentation blocks commented out below
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List

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
    zeta: float     # advection-offset parameter
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

    denom = u - s
    # singular surface u = s in these coordinates
    if abs(denom) < 1e-8:
        du = np.sign(denom) * np.inf if denom != 0 else np.inf
    else:
        du = eps * g_reaction(q, u) / denom

    return np.array([dq, dp, du], dtype=float)


# -----------------------------
# Equilibria
# -----------------------------

def X1_equilibrium() -> np.ndarray:
    # laminar equilibrium
    return np.array([0.0, 0.0, 2.0], dtype=float)


def K_cubic(u: float, r: float) -> float:
    # cubic used to locate ub(r) on the right branch
    # K(u;r) = 40u^3 - (50r+169)u^2 + (160r+224)u - 120r - 96
    return 40.0 * u**3 - (50.0 * r + 169.0) * u**2 + (160.0 * r + 224.0) * u - 120.0 * r - 96.0


def find_ub_right_branch(r: float) -> Optional[float]:
    """
    For r > 2/3, find ub(r) in (6/5, 4/3).
    """
    if r <= 2.0 / 3.0:
        return None

    a, b = 6.0 / 5.0 + 1e-6, 4.0 / 3.0 - 1e-6
    fa, fb = K_cubic(a, r), K_cubic(b, r)

    if fa * fb > 0:
        return None

    sol = root_scalar(lambda u: K_cubic(u, r), bracket=(a, b), method="brentq")
    return float(sol.root) if sol.converged else None


def X2_equilibrium(r: float) -> Optional[np.ndarray]:
    """
    X2 = (qb,+(r), 0, ub(r)), where
    qb,+(r) = 1 + (r + ub(r) - 2)/(r + 0.1).
    """
    ub = find_ub_right_branch(r)
    if ub is None:
        return None

    qb = 1.0 + (r + ub - 2.0) / (r + 0.1)
    return np.array([qb, 0.0, ub], dtype=float)


# -----------------------------
# Jacobian / linearization
# -----------------------------

def jacobian_at(y: np.ndarray, par: Params) -> np.ndarray:
    q, p, u = y
    D, r, eps, mu, s = par.D, par.r, par.eps, par.mu, par.s

    B = r + 0.1
    A = r + u - 2.0
    h = A - B * (q - 1.0) ** 2
    f_q = h + q * (-2.0 * B * (q - 1.0))
    f_u = q

    g_q = 2.0 * (1.0 - u)
    g_u = -1.0 - 2.0 * q
    denom = u - s

    return np.array([
        [0.0, 1.0, 0.0],
        [-f_q / D, (u + mu) / D, -f_u / D],
        [eps * g_q / denom, 0.0, eps * g_u / denom],
    ], dtype=float)


def unstable_eigenvector_at(eq: np.ndarray, par: Params) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return eigenvalues and the eigenvector corresponding to the eigenvalue
    with largest real part.
    """
    J = jacobian_at(eq, par)
    evals, evecs = np.linalg.eig(J)
    idx = np.argmax(evals.real)
    v = evecs[:, idx].real
    v /= np.linalg.norm(v)
    return evals, v


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
):
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
        events=[blowup_event]
    )
    return sol


def plot_orbits(
    orbits: List,
    show_equilibria: bool = True,
    r: float = 1.0,
    labels: Optional[List[str]] = None,
) -> None:
    fig = plt.figure(figsize=(13, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    if labels is None:
        labels = [None] * len(orbits)

    for sol, label in zip(orbits, labels):
        q, p, u = sol.y
        ax1.plot(q, p, linewidth=1.2, label=label)
        ax2.plot(q, u, linewidth=1.2, label=label)
        ax3.plot(q, p, u, linewidth=1.0, label=label)

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

    if any(label is not None for label in labels):
        ax1.legend()
        ax2.legend()

    plt.tight_layout()
    plt.show()


# -----------------------------
# Old crude shooting helper
# -----------------------------

def shooting_sweep_from_X1(
    par: Params,
    xi_span: Tuple[float, float],
    perturbations: np.ndarray,
    max_step: float = 0.1,
) -> List:
    """
    Old brute-force perturbation sweep from X1.
    Kept here for experimentation.
    """
    X1 = X1_equilibrium()
    orbits = []
    for d in perturbations:
        y0 = X1 + d
        sol = integrate_orbit(y0=y0, xi_span=xi_span, par=par, max_step=max_step)
        orbits.append(sol)
    return orbits


# -----------------------------
# Main experiment
# -----------------------------

def blowup_event(xi, y):
    q, p, u = y
    return 1.0 - max(abs(q)/20.0, abs(p)/50.0, abs(u)/5.0)

blowup_event.terminal = True
blowup_event.direction = -1

def main() -> None:
    # ============================================================
    # NEW DEFAULT: loop-motivated parameter set from the paper
    # ============================================================
    par = Params(
        D=0.12861304646,   # good first test near r=1.2
        r=1.2,
        eps=0.01,
        zeta=0.79,
        s=0.66576108146,
    )

    # ------------------------------------------------------------
    # OLD PARAMETER BLOCKS (kept commented out)
    # ------------------------------------------------------------
    # par = Params(
    #     D=0.5,
    #     r=0.8,
    #     eps=0.01,
    #     zeta=0.56,
    #     s=0.01,
    # )

    # par = Params(
    #     D=0.39356992678,   # rough loop-motivated D for r=0.8, zeta=0.56
    #     r=0.8,
    #     eps=0.01,
    #     zeta=0.56,
    #     s=0.671,           # rough loop-motivated s for r=0.8, zeta=0.56
    # )

    print("Params =", par)
    print("mu     =", par.mu)

    X1 = X1_equilibrium()
    X2 = X2_equilibrium(par.r)

    print("X1 =", X1)
    print("X2 =", X2)

    if X2 is None:
        raise ValueError("X2 does not exist for this r, or root-finding failed.")

    # ============================================================
    # NEW: launch front from X1 along unstable eigendirection
    # ============================================================
    evals1, v1 = unstable_eigenvector_at(X1, par)

    # choose sign so q initially increases
    if v1[0] < 0:
        v1 = -v1

    delta_front = 1e-6
    y0_front = X1 + delta_front * v1

    sol_front = integrate_orbit(
        y0=y0_front,
        xi_span=(0.0, 400.0),
        par=par,
        max_step=0.01,
        rtol=1e-9,
        atol=1e-12,
    )
    print("\nFRONT:")
    print("status =", sol_front.status)
    print("message =", sol_front.message)
    print("final state =", sol_front.y[:, -1])
    print("max |q| =", np.max(np.abs(sol_front.y[0])))
    print("max |p| =", np.max(np.abs(sol_front.y[1])))
    print("max |u| =", np.max(np.abs(sol_front.y[2])))

    # ============================================================
    # NEW: launch back from X2 along unstable eigendirection
    # ============================================================
    evals2, v2 = unstable_eigenvector_at(X2, par)

    # choose sign so q initially decreases toward the laminar branch
    if v2[0] > 0:
        v2 = -v2

    delta_back = 1e-6
    y0_back = X2 + delta_back * v2

    sol_back = integrate_orbit(
        y0=y0_back,
        xi_span=(0.0, 400.0),
        par=par,
        max_step=0.01,
        rtol=1e-9,
        atol=1e-12,
    )

    print("\nBACK:")
    print("status =", sol_back.status)
    print("message =", sol_back.message)
    print("final state =", sol_back.y[:, -1])
    print("max |q| =", np.max(np.abs(sol_back.y[0])))
    print("max |p| =", np.max(np.abs(sol_back.y[1])))
    print("max |u| =", np.max(np.abs(sol_back.y[2])))

    print("\nEigenvalues at X1:")
    print(evals1)
    print("Initial condition for front:", y0_front)

    print("\nEigenvalues at X2:")
    print(evals2)
    print("Initial condition for back:", y0_back)

    plot_orbits(
        [sol_front, sol_back],
        show_equilibria=True,
        r=par.r,
        labels=["front shot from X1", "back shot from X2"],
    )

    # ------------------------------------------------------------
    # OLD SINGLE-ORBIT BLOCK (kept commented out)
    # ------------------------------------------------------------
    # d = np.array([0.0, 0.0, 1e-6], dtype=float)
    # y0 = X1 + d
    #
    # sol = integrate_orbit(
    #     y0=y0,
    #     xi_span=(0.0, 300.0),   # forward in xi
    #     # xi_span=(300.0, 0.0), # backward in xi
    #     par=par,
    #     max_step=0.01,
    #     rtol=1e-9,
    #     atol=1e-12,
    # )
    #
    # plot_orbits([sol], show_equilibria=True, r=par.r, labels=["old single shot"])

    # ------------------------------------------------------------
    # OLD EIGENVECTOR BLOCK (kept commented out)
    # ------------------------------------------------------------
    # J1 = jacobian_at(X1, par)
    # evals_old, evecs_old = np.linalg.eig(J1)
    # iu_old = np.argmax(evals_old.real)
    # vu_old = evecs_old[:, iu_old].real
    # vu_old /= np.linalg.norm(vu_old)
    #
    # delta_old = 1e-7
    # y0_old = X1 - delta_old * vu_old   # try also X1 + delta_old * vu_old

    # ------------------------------------------------------------
    # OLD RANDOM PERTURBATION SWEEP (kept commented out)
    # ------------------------------------------------------------
    # rng = np.random.default_rng(0)
    # perts = 1e-2 * rng.normal(size=(30, 3))
    # perts[:, 2] *= 0.2
    #
    # orbits = shooting_sweep_from_X1(
    #     par=par,
    #     xi_span=(300.0, 0.0),  # backward in xi
    #     perturbations=perts,
    #     max_step=0.1,
    # )
    #
    # plot_orbits(orbits, show_equilibria=True, r=par.r)

    # ------------------------------------------------------------
    # OPTIONAL: try a small delta sweep around the unstable direction
    # ------------------------------------------------------------
    # deltas = [1e-8, 1e-7, 1e-6, 1e-5]
    # front_orbits = []
    # front_labels = []
    # for delta in deltas:
    #     y0 = X1 + delta * v1
    #     sol = integrate_orbit(
    #         y0=y0,
    #         xi_span=(0.0, 400.0),
    #         par=par,
    #         max_step=0.01,
    #         rtol=1e-9,
    #         atol=1e-12,
    #     )
    #     front_orbits.append(sol)
    #     front_labels.append(f"delta={delta:g}")
    #
    # plot_orbits(front_orbits, show_equilibria=True, r=par.r, labels=front_labels)


if __name__ == "__main__":
    main()
