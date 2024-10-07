import casadi
import numpy as np
import matplotlib.pylab as plt
import mhe as MHE
import data_parsing
import os
from concurrent.futures import *
import time
import pandas as pd

cpu_count = os.cpu_count()
print(cpu_count)

def run(i):
    try:
        data_reader = data_parsing.DataReader(data_type=data_parsing.DataType.Test)
        data = data_reader.read(i, n_max_subsections=50, n_max_sections=5)
        mhe = MHE.MHE(30, dt=data.delta,
                        ground_spring_constant=30.93226074688124,
                        ground_damping_constant=0.05,
                        force_cost=0.000001,
                        rk_error_cost=100000000,
                        measured_error_cost=1000,
                        force_variance_cost=0.0001,
                        force_variance_variance_cost=0)
        for min_v, max_v in data.max_ranges:
            mhe.read(data.velocity[min_v:max_v])
            mhe.solve_all(i)
            df = pd.DataFrame()
            df['x-est'] = pd.Series(mhe.x_est[:,0])
            df['y-est'] = pd.Series(mhe.x_est[:,1])
            df['f-est'] = pd.Series(mhe.f_est)
            df.to_csv(f'data{min_v}_{max_v}_{i}.csv')


    except Exception as e:
        print(e)
        print(tb=traceback.format_exc())


with ProcessPoolExecutor(max_workers=10) as exe:
    exe.map(run, [i for i in range(45, 55)])
