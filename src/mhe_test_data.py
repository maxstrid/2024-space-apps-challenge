import casadi
import numpy as np
import matplotlib.pylab as plt
import mhe
import data_parsing
import pandas as pd

data_reader = data_parsing.DataReader()
n_data = 0
data = data_reader.read(n_data, n_max_subsections=10, n_max_sections=1)

mhe = mhe.MHE(30, dt=data.delta,
              ground_spring_constant=30.93226074688124,
              ground_damping_constant=0.05,
              force_cost=0.000001,
              rk_error_cost=100000000,
              measured_error_cost=1000,
              force_variance_cost=0.0001,
              force_variance_variance_cost=0)

min_v, max_v = data.max_ranges[0]

mhe.read(data.velocity[min_v:max_v])
mhe.solve_all()

df = pd.DataFrame()
df['x-est'] = pd.Series(mhe.x_est[:,0])
df['y-est'] = pd.Series(mhe.x_est[:,1])
df['f-est'] = pd.Series(mhe.f_est)
df.to_csv('data_' + str(n_data) + '.csv')


fig, ax = mhe.plot(plot=False)
ax[0].plot(np.linspace(0, -data.time[min_v]+data.time[max_v], mhe.n), data.velocity[min_v:max_v], label="x")
ax[0].legend()
ax[1].legend()

plt.show()
