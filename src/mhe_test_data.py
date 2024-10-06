import casadi
import numpy as np
import matplotlib.pylab as plt
import mhe
import data_parsing

data_reader = data_parsing.DataReader()
data = data_reader.read(0, n_max_subsections=50, n_max_sections=1)

mhe = mhe.MHE(50, dt=data.delta,
              ground_spring_constant=30.93,
              ground_damping_constant=0.01,
              force_cost=1,
              rk_error_cost=10000000000,
              measured_error_cost=100000,
              force_variance_cost=10000,
              force_variance_variance_cost=100000)

min_v, max_v = data.max_ranges[0]

mhe.read(data.velocity[min_v:max_v])
mhe.solve_all()
fig, ax = mhe.plot(plot=False)
ax[0].plot(np.linspace(0, -data.time[min_v]+data.time[max_v], mhe.n), data.velocity[min_v:max_v], label="x")
ax[0].legend()
ax[1].legend()

plt.show()
