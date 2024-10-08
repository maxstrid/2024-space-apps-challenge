import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import scipy
import scipy.signal
import os


# data = pd.read_csv("data11448_22896_54.csv")
# data = pd.read_csv("data377784_389232_54.csv")
# data = pd.read_csv("data171720_183168_53.csv")
# data = pd.read_csv("data_10.csv")
files = os.listdir()
for f in files:
    if f[:4] == "data" and f.endswith(".csv"):
        data = pd.read_csv(f)
    else:
        continue
    x = np.linspace(0, 0.15 * len(data['f-est']), len(data['f-est']))
    print(f)

    #take progressive rolling averages
    axes: list[plt.Axes]
    fig, axes = plt.subplots(5)
    axes[0].plot(data['f-est'], label="raw")
    # axes[0].plot(data['y-est'], label="raw")

    #take rolling average
    def roll_avg(data: np.ndarray, rnge: int):
        avgs = np.zeros(len(data))
        for i in range(rnge):
            avgs += np.concatenate((data[i:], np.zeros(i)))
            if i != 0:
                avgs += np.concatenate((np.zeros(i), data[:-i]))
        
        for i in range(len(avgs)):
            n = min(2 * rnge - 1, rnge + i, rnge+len(avgs)-i-1)
            avgs[i]/=n
        

        return avgs


    # axes[1].plot(x, roll_avg(data['f-est'], 10), label="10")
    # axes[2].plot(x, roll_avg(np.sqrt(data['f-est']), 10), label="rt10")
    # axes[3].plot(x, roll_avg(data['f-est'], 100), label="100")
    d = data['f-est'][:-1]
    for i in range(10):
        d = roll_avg(d, 1000)

    axes[1].plot(d, label="short-term", linewidth=5.0)


    d2 = d.copy()
    for i in range(25):
        d2 = roll_avg(d, 4000)

    axes[2].plot(d2, label="long-term")

    ratio = d/d2

    axes[3].plot(ratio, label="ratio")

    axes[4].plot(data['f-est'])
    axes[4].plot(d)
    axes[4].plot(d2)
    axes[4].plot(ratio/25)

plt.show()