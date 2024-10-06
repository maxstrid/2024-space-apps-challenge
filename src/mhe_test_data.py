import casadi
import numpy as np
import matplotlib.pylab as plt
import mhe
import data_parsing

data_reader = data_parsing.DataReader
data = data_reader.read(0)

mhe = mhe.MHE(50, dt=data.delta)

mhe.read(data.velocity)
mhe.solve_all()
fig, ax = mhe.plot(plot=False)
ax.plot(np.linspace(0, mhe.dt * mhe.n, mhe.n), data.velocity[:, 0], label="x")
ax.legend()

plt.show()
