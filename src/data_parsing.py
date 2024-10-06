import pandas as pd
import numpy as np
import scipy
from enum import Enum
from filtering import butter_bandpass_filter
import random
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from dataclasses import dataclass

data_folder = "./data/data/lunar/training/"

@dataclass
class SeismicData:
    velocity: np.array
    time: np.array
    time_of_event: None | float

    def plot(self, title, ax):
        ax.plot(self.time, self.velocity)

        ax.set_xlim([min(self.time), max(self.time)])
        ax.set_ylabel('Velocity (m/s)')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'{title}')

        if self.time_of_event:
            arrival_line = ax.axvline(x=self.time_of_event, c='red', label='Arrival')
            ax.legend(handles=[arrival_line])

class DataReader:
    def __init__(self, filter_data = True, pool_data = False):
        self.catalog_df = pd.read_csv(f'{data_folder}catalogs/apollo12_catalog_GradeA_final.csv')

    def read(self, i: int, filter_data = True, pool_data = False) -> SeismicData:
        return self.__read_event(self.catalog_df.iloc[i], filter_data=filter_data, pool_data=pool_data)

    def __read_event(self, event: pd.Series, filter_data = True, pool_data = False):
        time_of_event = event['time_rel(sec)']
        event_filename = event['filename']

        event_df = pd.read_csv(f'{data_folder}data/S12_GradeA/{event_filename}.csv')

        time = np.array(event_df['time_rel(sec)'].tolist())
        velocity = np.array(event_df['velocity(m/s)'])

        if filter_data:
            velocity = butter_bandpass_filter(velocity, 0.5, 1.0, 6.0)

        if pool_data:
            data_pooled = max_pool_1d(np.vstack((velocity, time)), 100)

            velocity = data_pooled[0:1, :].flatten()
            time = data_pooled[1:2, :].flatten()

            time_index = find_nearest_time_index(time, time_of_event)

        return SeismicData(velocity = velocity, time = time, time_of_event = time_of_event)


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

def find_nearest_time_index(timesteps: np.array, time: float) -> int:
    size = timesteps.shape[0]

    n = size // 2
    max = size
    min = 0
    print("Searching for", time)
    val = 0
    while True:
        val = timesteps[n]

        if max - min <= 1:
            print(n, val, timesteps[n])
            return n

        if time < val:
            max = n
            n = (min + max) // 2
        elif time > val:
            min = n
            n = (max + min) // 2

def main():
    reader = DataReader()

    fig, ax = plt.subplots(1, 1, figsize=(10, 3))

    plt.rcParams['keymap.quit'].append(' ')

    reader.read(0).plot('0', ax)

    plt.show()


if __name__ == "__main__":
    main()
