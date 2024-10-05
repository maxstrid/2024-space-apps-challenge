import casadi
import do_mpc
import numpy as np

model_type = 'continuous'
model = do_mpc.model.Model(model_type)

phi = model.set_variable(var_type="_x", var_name="phi")
dphi = model.set_variable(var_type="_x", var_name="dphi")

#state measurement
phi_meas = model.set_meas('phi_meas', phi, meas_noise=True)

g = model.set_variable('parameter', "g")
l = model.set_variable('parameter', "l")
c = model.set_variable('parameter', "c")

model.set_rhs('phi', dphi)

dphi_next = - (g * casadi.sin(phi) + c * dphi * l) / l

model.set_rhs("dphi", dphi_next, process_noise=False)

model.setup()

mhe = do_mpc.estimator.MHE(model, ['g', 'l', 'c'])

setup_mhe = {
    't_step': 0.01,
    'n_horizon': 10,
    'store_full_solution': True,
    'meas_from_data': True
}
mhe.set_param(**setup_mhe)

Px = np.eye()