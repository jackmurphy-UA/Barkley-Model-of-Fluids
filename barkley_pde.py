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
