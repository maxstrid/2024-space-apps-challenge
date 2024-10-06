import casadi
import numpy as np
import matplotlib.pylab as plt
import mhe
import data_parsing

data_reader = data_parsing.DataReader()
data = data_reader.read(0, n_max_subsections=100, n_max_sections=1)

mhe = mhe.MHE(50, dt=data.delta,
              ground_spring_constant=30.93,
              force_cost=1,
              rk_error_cost=10000000000,
              measured_error_cost=10000,
              force_variance_cost=10000,
              force_variance_variance_cost=100000)

min_v, max_v = data.max_ranges[0]

mhe.read(data.velocity[min_v:max_v])
mhe.solve_all()
fig, ax = mhe.plot(plot=False)
ax.plot(np.linspace(0, -data.time[min_v]+data.time[max_v], mhe.n), data.velocity[min_v:max_v], label="x")
ax.legend()

plt.show()
