import pandas as pd
import numpy as np
import scipy
from enum import Enum
from filtering import butter_bandpass_filter
import random
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

data_folder = "../nasa_space_apps/demo/space_apps_2024_seismic_detection/space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/"


class DataType(Enum):
    Mars = 0
    Lunar = 1


dara_type = DataType.Lunar


def read_catalog() -> pd.DataFrame:
    df = pd.read_csv('./apollo12_catalog_GradeA_final.csv')
    return df


def plot(title, ax, time, vel, line_time):
    ax.plot(time, vel)

    ax.set_xlim([min(time), max(time)])
    ax.set_ylabel('Velocity (m/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'{title}')
    arrival_line = ax.axvline(x=line_time, c='red', label='Arrival')
    ax.legend(handles=[arrival_line])


def plot_catalog_event(event: pd.Series) -> None:
    time_rel = event['time_rel(sec)']
    filename = event['filename']

    event_df = pd.read_csv(f'{data_folder}/{filename}.csv')

    time = np.array(event_df['time_rel(sec)'].tolist())
    velocity = np.array(event_df['velocity(m/s)'].tolist())

    velocity_filtered = butter_bandpass_filter(velocity, 0.5, 1.0, 6.0)

    data_pooled = max_pool_1d(np.vstack((velocity_filtered, time)), 100)

    velocity_pooled = data_pooled[0:1, :].flatten()
    time_pooled = data_pooled[1:2, :].flatten()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 3))
    plt.rcParams['keymap.quit'].append(' ')

    plot(f'{filename} filtered', ax1, time, velocity_filtered, time_rel)
    plot(f'{filename} pooled', ax2, time_pooled, velocity_pooled, time_rel)

    plt.show()


def max_pool_1d(array: np.array, n: int) -> np.array:
    size = array.shape[1] // n

    if len(array) % n != 0:
        size += 1

    result = np.zeros((2, size))

    for i in range(size):
        subarr = array[:, i * n:(i * n) + n]
        max = np.max(subarr, axis=1)
        result[0, i] = max[0]
        result[1, i] = max[1]

    return result


# Implements SA to detect peak(s) in the data
# N is the max iterations
def detect_peak(data: pd.DataFrame, N=1000) -> int:
    temperature = 1.0
    for i in range(0, N):
        temperature = temperature / (i + 1)

    return time


def test_events():
    catalog_df = read_catalog()
    for _, row in catalog_df.iterrows():
        plot_catalog_event(row)


def main():
    test_events()


if __name__ == "__main__":
    main()
