# This file is for simulating the original Barkley PDE model, which is a two-variable reaction-diffusion system.

# In the future I will compare results from this PDE with the reduced ODE model and a Navier-Stokes simulation of the same system. 
 
# The equations are:
# 
# PDE Model:
#    q_t = D q_xx + (zeta - u) q_x + f(q,u;r)
#    u_t = -u u_x + eps g(q,u),
# where
#    f(q,u;r) = q( r + u - 2 - (r+0.1)(q-1)^2 )
#    g(q,u) = 2 - u + 2q(1-u).

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the parameters
D = 0.20      # Diffusion coefficient
zeta = 0.5   # Advection coefficient    
eps = 0.01   # Time scale separation parameter
r = 1.0      # Control parameter

# Define the spatial domain and discretization
L = 100.0                   # Length of the spatial domain
Nx = 200                    # Number of spatial points
dx = L / Nx                 # Spatial step size
x = np.linspace(0, L, Nx)   # Spatial grid

# Define the time domain and discretization
T = 10.0                    # Total simulation time
dt = 0.01                     # Time step size
NT = int(T / dt)             # Number of time steps
t = np.linspace(0, T, NT)    # Time grid

# Define the initial conditions
def initial_conditions(x):
    q0 = np.zeros_like(x)
    u0 = np.zeros_like(x)
    
    # Initialize q with a localized perturbation
    q0[(x > 40) & (x < 60)] = 2.0
    
    return q0, u0

# Define the functions f and g
def f(q, u, r):
    return q * (r + u - 2 - (r + 0.1) * (q - 1) ** 2)

def g(q, u):
    return 2 - u + 2 * q * (1 - u)

# Define the spatial derivatives using finite differences

def spatial_derivatives(q, u):
    q_xx = np.zeros_like(q)
    q_x = np.zeros_like(q)
    u_x = np.zeros_like(u)
    
    # Second derivative (q_xx) using central differences
    q_xx[1:-1] = (q[2:] - 2 * q[1:-1] + q[:-2]) / dx**2
    
    # First derivative (q_x) using central differences
    q_x[1:-1] = (q[2:] - q[:-2]) / (2 * dx)

    # First derivative (u_x) using central differences
    u_x[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
    
    return q_xx, q_x, u_x

# Define the time derivatives
def time_derivatives(t, y):
    q = y[:Nx]
    u = y[Nx:]
    
    q_xx, q_x, u_x = spatial_derivatives(q, u)
    
    dqdt = D * q_xx + (zeta - u) * q_x + f(q, u, r)
    dudt = -u * u_x + eps * g(q, u)
    
    return np.concatenate([dqdt, dudt])


########## fix these time derivatives ################
# I dont't think these are correct right now


# Initialize the state vector
q0, u0 = initial_conditions(x)
y0 = np.concatenate([q0, u0])

# Solve the PDE using solve_ivp
solution = solve_ivp(time_derivatives, [0, T], y0, t_eval=t, method='RK45')

# Extract the solution
q_solution = solution.y[:Nx, :]
u_solution = solution.y[Nx:, :]

# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.imshow(q_solution, aspect='auto', extent=[0, T, 0, L], origin='lower')
plt.colorbar(label='q')
plt.title('Barkley PDE Simulation - q')
plt.xlabel('Time')
plt.ylabel('Space')
plt.subplot(2, 1, 2)
plt.imshow(u_solution, aspect='auto', extent=[0, T, 0, L], origin='lower')
plt.colorbar(label='u')
plt.title('Barkley PDE Simulation - u')
plt.xlabel('Time')
plt.ylabel('Space')
plt.show()

