# Barkley Traveling-Wave ODE (3D Spatial Dynamics)

This repository contains exploratory numerical tools for the *traveling-wave / spatial-dynamics* reduction of the Barkley pipe-flow PDE turbulence model.

## Background

The starting point is the reduced PDE model for turbulence intensity `q(x,t)` and centerline velocity `u(x,t)`:
- `q_t = D q_xx + (ζ - u) q_x + f(q,u;r)`
- `u_t = -u u_x + ε g(q,u)`

with reaction terms
- `f(q,u;r) = q( r + u - 2 - (r+0.1)(q-1)^2 )`
- `g(q,u) = 2 - u + 2q(1-u)`.

A traveling-wave ansatz `(q,u)(x,t) = (q*,u*)(x - s t)` and spatial-dynamics formulation yields a 3D ODE in the traveling coordinate `ξ = x - s t`:
- `q' = p`
- `p' = ((u + μ)p - f(q,u;r))/D`
- `u' = ε g(q,u) / (u - s)`

where `μ = -ζ - s`.

In this viewpoint, bounded orbits of the 3D ODE correspond to traveling waves in the PDE; in particular:
- heteroclinic orbits correspond to traveling fronts/backs,
- homoclinic orbits correspond to localized pulses (“puffs”),
- periodic orbits correspond to wavetrains.

## Goals (research direction)

The long-term goal is to use phase-space geometry to understand transition mechanisms and potentially locate global organizing structures (heteroclinic/homoclinic networks, attractors, etc.) and their perturbations.

Concrete numerical/analytic objectives include:
1. **Equilibria and local linear analysis**
   - compute the laminar equilibrium `X1 = (0,0,2)` and the turbulent equilibrium `X2 = (qb,+(r),0,ub(r))` (for `r > 2/3`);
   - compute eigenvalues/eigendirections to seed stable/unstable manifold computations.

2. **Heteroclinic connections**
   - numerically approximate heteroclinic connections between `X1` and `X2` (fronts/backs);
   - scan parameter space `(r, D, ζ, ε, s)` for regimes exhibiting candidate loops/cycles.

3. **Loop perturbations and complex dynamics**
   - perturb parameter sets near heteroclinic loops and look for nearby complicated invariant sets:
     - long transient “winding” dynamics near the loop,
     - possible chaotic invariant sets in suitable Poincaré sections,
     - families of multi-pulse solutions (k-fronts / k-backs).

4. **Model variations**
   - experiment with simplified 3D ODE surrogates or normal-form style reductions around the loop;
   - compare against analytic predictions (fast–slow structure, Melnikov-style matching, etc.).

## Repo contents

- `barkley_tw_ode.py`
  - Implements the 3D traveling-wave ODE.
  - Provides basic phase-portrait plotting and a crude shooting sweep from `X1`.
  - More to come.
- 'eigvals.py'
  - Computes eigenvalues.

## Quick start

Create an environment and run:

```bash
python -m venv .venv
# activate: Windows PowerShell
.venv\Scripts\Activate.ps1
pip install numpy scipy matplotlib
python barkley_tw_ode.py
