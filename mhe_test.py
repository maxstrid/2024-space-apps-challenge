import casadi
import numpy as np
import matplotlib.pyplot as plt
import time
import casadi.tools as ctools
from scipy import linalg

#they call me the pendulum the way I integrate

#  Settings of the filter
N = 100 # Horizon length
dt = 0.05; # Time step

# sigma_p = 0.5 # Standard deviation of the position measurements
# sigma_w = 0.1 # Standard deviation for the process noise
# R = casadi.DM(1/sigma_p**2) # resulting weighting matrix for the position measurements
# Q = casadi.DM(1/sigma_w**2) # resulting weighting matrix for the process noise

Nsimulation = 10000 # Lenght of the simulation

# Parameters of the system
l = 1 # The length of the pendulum
c = 0.001 # The damping constant - velocity multiplier

states = ctools.struct_symSX(["theta","dtheta"])
measurements = ctools.struct_symSX(["theta_measured"]) # Measurement vector

shooting = ctools.struct_symSX([(ctools.entry("X",repeat=N,struct=states))])
parameters = ctools.struct_symSX([(ctools.entry("Y",repeat=N,struct=measurements),
                                   ctools.entry("S",shape=(states.size, states.size)),
                                   ctools.entry("x0",shape=(states.size, 1)))])


rhs = ctools.struct_SX(states)
rhs["theta"] = states["dtheta"]
rhs["dtheta"] = -9.8 * casadi.sin(states["theta"]) / l #torque/moi
rhs['dtheta'] = rhs['dtheta'] - rhs['dtheta'] * c
f = function('f', [states], [rhs])

# Build an integrator for this system: Runge Kutta 4 integrator
k1 = f(states)
k2 = f(states+dt/2.0*k1)
k3 = f(states+dt/2.0*k2)
k4 = f(states+dt*k3)

states_1 = states+dt/6.0*(k1+2*k2+2*k3+k4)
phi = casadi.Function('phi', [states], [states_1], ['x'], ['x1'])

# Define the measurement system
h = casadi.Function('h', [states], [states["theta"]], ['x'], ['y']) # We have measurements of the position

#build the objective
obj = 0
# First the arrival cost
obj += casadi.mtimes([(shooting["X",0]-parameters["x0"]).T,parameters["S"],(shooting["X",0]-parameters["x0"])])
#Next the cost for the measurement noise
for i in range(N):
  vm = h(shooting["X",i])-parameters["Y",i]
  obj += mtimes([vm.T,R,vm])
#And also the cost for the process noise
for i in range(N-1):
  obj += mtimes([shooting["W",i].T,Q,shooting["W",i]])