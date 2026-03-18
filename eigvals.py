"""
eigvals.py

Independent eigenvalue / linearization analysis for the Barkley traveling-wave
spatial-dynamics ODE associated with the Barkley pipe-flow PDE model.

System:
    q' = p
    p' = ((u + mu)p - f(q,u;r))/D
    u' = eps*g(q,u)/(u - s)

with
    mu = -zeta - s
    f(q,u;r) = q*(r + u - 2 - (r+0.1)(q-1)^2)
    g(q,u)   = 2 - u + 2q(1-u)

This script:
  1. Computes equilibria X1 and X2 (when it exists)
  2. Builds the analytic Jacobian at an equilibrium
  3. Optionally checks it against a finite-difference Jacobian
  4. Computes eigenvalues/eigenvectors
  5. Classifies the equilibrium by spectral type
  6. Optionally sweeps a parameter (default: s)

Dependencies:
    numpy, scipy
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from scipy.optimize import root_scalar


# ============================================================
# Parameters
# ============================================================

@dataclass(frozen=True)
class Params:
    D: float
    r: float
    eps: float
    zeta: float
    s: float

    @property
    def mu(self) -> float:
        return -self.zeta - self.s


# ============================================================
# Reaction terms
# ============================================================

def f_reaction(q: float, u: float, r: float) -> float:
    return q * (r + u - 2.0 - (r + 0.1) * (q - 1.0) ** 2)


def g_reaction(q: float, u: float) -> float:
    return 2.0 - u + 2.0 * q * (1.0 - u)


# ============================================================
# TW ODE RHS
# ============================================================

def barkley_tw_rhs(y: np.ndarray, par: Params) -> np.ndarray:
    q, p, u = y
    D, r, eps, mu, s = par.D, par.r, par.eps, par.mu, par.s

    dq = p
    dp = ((u + mu) * p - f_reaction(q, u, r)) / D

    denom = u - s
    if abs(denom) < 1e-14:
        raise ZeroDivisionError("Encountered singular surface u = s in u' equation.")

    du = eps * g_reaction(q, u) / denom
    return np.array([dq, dp, du], dtype=float)


# ============================================================
# Equilibria
# ============================================================

def X1_equilibrium() -> np.ndarray:
    return np.array([0.0, 0.0, 2.0], dtype=float)


def K_cubic(u: float, r: float) -> float:
    return (
        40.0 * u**3
        - (50.0 * r + 169.0) * u**2
        + (160.0 * r + 224.0) * u
        - 120.0 * r
        - 96.0
    )


def find_ub_right_branch(r: float) -> Optional[float]:
    if r <= 2.0 / 3.0:
        return None

    a = 6.0 / 5.0 + 1e-8
    b = 4.0 / 3.0 - 1e-8
    fa = K_cubic(a, r)
    fb = K_cubic(b, r)

    if fa * fb > 0:
        return None

    sol = root_scalar(lambda u: K_cubic(u, r), bracket=(a, b), method="brentq")
    return float(sol.root) if sol.converged else None


def X2_equilibrium(r: float) -> Optional[np.ndarray]:
    ub = find_ub_right_branch(r)
    if ub is None:
        return None
    qb = 1.0 + (r + ub - 2.0) / (r + 0.1)
    return np.array([qb, 0.0, ub], dtype=float)


# ============================================================
# Jacobian
# ============================================================

def analytic_jacobian(y: np.ndarray, par: Params) -> np.ndarray:
    q, p, u = y
    D, r, eps, mu, s = par.D, par.r, par.eps, par.mu, par.s

    B = r + 0.1
    A = r + u - 2.0
    h = A - B * (q - 1.0) ** 2

    f_q = h + q * (-2.0 * B * (q - 1.0))
    f_u = q

    g_q = 2.0 * (1.0 - u)
    g_u = -1.0 - 2.0 * q
    g = g_reaction(q, u)

    denom = u - s
    if abs(denom) < 1e-14:
        raise ZeroDivisionError("Jacobian undefined on singular surface u = s.")

    # General derivative of g/(u-s):
    # d/dq [g/(u-s)] = g_q/(u-s)
    # d/du [g/(u-s)] = g_u/(u-s) - g/(u-s)^2
    J = np.array(
        [
            [0.0, 1.0, 0.0],
            [-f_q / D, (u + mu) / D, -f_u / D],
            [eps * g_q / denom, 0.0, eps * (g_u / denom - g / denom**2)],
        ],
        dtype=float,
    )

    return J


def finite_difference_jacobian(y: np.ndarray, par: Params, h: float = 1e-7) -> np.ndarray:
    n = len(y)
    J = np.zeros((n, n), dtype=float)

    for j in range(n):
        ej = np.zeros(n, dtype=float)
        ej[j] = 1.0
        fp = barkley_tw_rhs(y + h * ej, par)
        fm = barkley_tw_rhs(y - h * ej, par)
        J[:, j] = (fp - fm) / (2.0 * h)

    return J


# ============================================================
# Spectral analysis
# ============================================================

def classify_equilibrium(evals: np.ndarray, tol: float = 1e-9) -> str:
    real_parts = np.real(evals)
    npos = np.sum(real_parts > tol)
    nneg = np.sum(real_parts < -tol)
    ncen = len(evals) - npos - nneg

    if ncen > 0:
        return f"nonhyperbolic ({npos} unstable, {nneg} stable, {ncen} center)"

    return f"hyperbolic ({npos} unstable, {nneg} stable)"


def spectral_report(name: str, yeq: np.ndarray, par: Params, fd_check: bool = True) -> None:
    print("=" * 72)
    print(f"{name}")
    print("=" * 72)
    print(f"Equilibrium: {yeq}")
    print(f"Parameters: D={par.D}, r={par.r}, eps={par.eps}, zeta={par.zeta}, s={par.s}, mu={par.mu}")

    J = analytic_jacobian(yeq, par)
    evals, evecs = np.linalg.eig(J)

    idx = np.argsort(np.real(evals))[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    print("\nAnalytic Jacobian:")
    print(J)

    if fd_check:
        Jfd = finite_difference_jacobian(yeq, par)
        err = np.linalg.norm(J - Jfd, ord=np.inf)
        print("\nFinite-difference Jacobian:")
        print(Jfd)
        print(f"\n||J_analytic - J_fd||_inf = {err:.3e}")

    print("\nEigenvalues:")
    for k, lam in enumerate(evals, start=1):
        print(f"  lambda_{k} = {lam.real:+.12e} {lam.imag:+.12e}i")

    print(f"\nClassification: {classify_equilibrium(evals)}")

    print("\nEigenvectors (columns correspond to sorted eigenvalues):")
    print(evecs)

    unstable_idx = np.where(np.real(evals) > 1e-9)[0]
    stable_idx = np.where(np.real(evals) < -1e-9)[0]

    if len(unstable_idx) > 0:
        print("\nUnstable eigendirections:")
        for j in unstable_idx:
            v = np.real_if_close(evecs[:, j])
            vn = v / np.linalg.norm(v)
            print(f"  mode {j+1}: {vn}")

    if len(stable_idx) > 0:
        print("\nStable eigendirections:")
        for j in stable_idx:
            v = np.real_if_close(evecs[:, j])
            vn = v / np.linalg.norm(v)
            print(f"  mode {j+1}: {vn}")

    print()


# ============================================================
# Parameter sweep
# ============================================================

def sweep_parameter_s(
    par: Params,
    s_values: np.ndarray,
    eq: str = "X1",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        s_values
        eigvals_array with shape (len(s_values), 3)
    """
    spectra = []

    for sval in s_values:
        par_s = Params(D=par.D, r=par.r, eps=par.eps, zeta=par.zeta, s=float(sval))

        if eq == "X1":
            yeq = X1_equilibrium()
        elif eq == "X2":
            yeq = X2_equilibrium(par_s.r)
            if yeq is None:
                raise ValueError("X2 does not exist for this r.")
        else:
            raise ValueError("eq must be 'X1' or 'X2'.")

        J = analytic_jacobian(yeq, par_s)
        evals = np.linalg.eigvals(J)
        idx = np.argsort(np.real(evals))[::-1]
        spectra.append(evals[idx])

    return s_values, np.array(spectra)


def print_sweep_table(s_values: np.ndarray, spectra: np.ndarray) -> None:
    print("=" * 72)
    print("Parameter sweep")
    print("=" * 72)
    for sval, ev in zip(s_values, spectra):
        lam_str = ", ".join([f"{z.real:+.5e}{z.imag:+.5e}i" for z in ev])
        print(f"s = {sval:+.6f}   {lam_str}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    par = Params(
        D=0.01,
        r=0.7,
        eps=0.01,
        zeta=0.79,
        s=0.01,
    )

    X1 = X1_equilibrium()
    spectral_report("X1 equilibrium", X1, par, fd_check=True)

    X2 = X2_equilibrium(par.r)
    if X2 is None:
        print("X2 does not exist for this parameter choice.\n")
    else:
        spectral_report("X2 equilibrium", X2, par, fd_check=True)

    s_values = np.linspace(-0.05, 0.10, 8)
    svals, spectra = sweep_parameter_s(par, s_values, eq="X1")
    print_sweep_table(svals, spectra)


if __name__ == "__main__":
    main()
